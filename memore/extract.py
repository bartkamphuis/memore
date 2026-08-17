"""P1 -- extraction (recall-writepath-spec.md §1).

An LLM call is correct here: this runs async, off the response path, after the user
already has their answer. The no-LLM constraint governs recall A-D and the consolidation
decision, not this.

The salience gate (§1.2) is the point of the prompt: returning nothing is the default and
expected outcome. Most turns are transient and yield `[]`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from .config import WritePathConfig
from .llm import LLMConfig, OllamaClient
from .types import CandidateFact, FactType, Message

_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "type": {"type": "string", "enum": [t.value for t in FactType]},
                    "confidence": {"type": "number"},
                    "subject_hint": {"type": "string"},
                    "attribute": {"type": "string"},
                },
                "required": ["fact", "type", "confidence", "subject_hint", "attribute"],
            },
        }
    },
    "required": ["facts"],
}

_SYSTEM = """You extract durable facts from a conversation turn for a long-term memory store.

Returning an empty list is the DEFAULT and EXPECTED outcome. Most turns contain nothing
durable. Only extract a fact if it is still true and still useful in a future session.

DURABLE (extract):
- preferences: "I deploy to prod by default", "I like terse answers"
- identity / stable attributes: "I work in Python", "my main repo is X"
- stable state that persists across sessions: "the auth migration is my current focus"

TRANSIENT (do NOT extract):
- task-local instructions: "run the tests again", "make it shorter"
- one-off computed results, ephemeral values
- pleasantries, meta-conversation, acknowledgements
- anything whose truth expires when the turn ends

A QUESTION IS NEVER A FACT. If the turn is the user asking something -- "what's my
deploy setup?", "remind me where I deploy?" -- return an empty list. The answer may
already be in memory, but the asking teaches you nothing new. Never restate a fact from
the prior turns as if this turn asserted it; extract only what THIS turn asserts.

AN EXPLICIT REQUEST TO REMEMBER OVERRIDES EVERY RULE ABOVE EXCEPT THE QUESTION RULE.
If the user says "remember that X", "don't forget X", "keep in mind X", "note that X",
extract X. Do not apply your own salience judgement to it -- the user has already made
that judgement, and it outranks yours. This holds even when X looks trivial, mundane or
physical: "remember that the chair is against the wall" MUST be extracted. Returning
nothing there is a silent refusal of a direct instruction, which is worse than storing
something dull.

Rules for each extracted fact:
- `fact` MUST be standalone: resolve every pronoun and reference so the sentence is
  meaningful with no surrounding context. "switched it to prod" is WRONG;
  "deploys to prod by default" is RIGHT.
- `subject_hint` is the TOPIC the fact is about. Choose it by these rules IN ORDER:

    1. Name the entity the fact is ABOUT, never the person who mentioned it. "My sister
       Lisa lives in Amsterdam" is a fact about Lisa, so the subject is Lisa. The user is
       the subject only when the fact is about the user.
    2. The person speaking is ALWAYS "the user", even once you learn their name. "My
       name is Bart" is the user's `name` attribute, not a subject called "Bart".
    3. Any OTHER entity with a proper name takes THAT NAME ALONE as its subject.
       RIGHT: "Lisa", "Pixel", "Miso".
       WRONG: "the user's sister Lisa", "the user's dog Pixel", "the cat Miso".
       The relation ("sister", "dog") already appears in `fact`, which is standalone, so
       nothing is lost by leaving it out here -- and putting it in costs you the match,
       because the next turn will just say "Lisa" and its fact will land somewhere else.
       Keep whatever part of the NAME tells two bearers apart: if the conversation has a
       colleague Tom and a neighbour Tom Bakker, the second stays "Tom Bakker".
    4. An entity with no proper name gets the shortest natural noun phrase: "the memory
       system", "the Netherlands", "the user's employer".
    5. A subject is a THING; it is never a thing's PROPERTY. When you are about to write
       "the X of Y", decide which of these X is:
         a VALUE that Y has -> the subject is Y, and X is the attribute. "the capital of
           the Netherlands" is the subject "the Netherlands" with attribute "capital
           city", never a subject of its own. Folding a property into the subject leaves
           the attribute empty and gives the next turn's correction nothing to collide
           with.
         a THING that Y contains, which has properties of its own -> X is its own
           subject, named on its own terms: "the test suite", not "the memory system".
           A test suite has a runtime and a framework and a size; a capital city is a
           single value and has none.

  These five rules pick the name for a subject you have not seen before. A subject already
  listed as in memory outranks every one of them -- reuse that exact string rather than
  renaming it, even where a rule above would have named it differently.
- `attribute` is the single PROPERTY of that subject this fact gives a value for:
  "deploy target", "age", "implementation language", "lookup latency", "capital city".
  This is the test that matters, and it is not the same test as the subject:

    Could BOTH facts be true at the same time?
      YES -> they are different properties -> they MUST get different `attribute`s.
             "the memory system is written in Python" (implementation language) and
             "the memory system takes 70-90ms" (lookup latency) are both true. Storing
             them under one attribute makes the newer one erase the older one.
      NO  -> they are the same property with different values -> they MUST get the
             IDENTICAL `attribute`, character for character.
             "deploys to staging by default" and "the default deployment target is now
             production" are the SAME property (deploy target) and must collide, or the
             store keeps both and answers with the stale one.

  Name the property, never its value: "deploy target", not "staging". Reuse an attribute
  you have already used for that subject rather than rephrasing it -- matching is exact,
  so "deploy target" and "deployment environment" are two different properties and the
  contradiction between them is never noticed.
- `confidence` is your own 0..1 confidence that this is a durable fact.
"""


class Extractor(Protocol):
    async def extract(
        self,
        user_message: str,
        assistant_response: str,
        recent: list[Message],
        known_subjects: list[str] | None = None,
    ) -> list[CandidateFact]: ...


class OllamaExtractor:
    def __init__(self, config: WritePathConfig | None = None, llm: OllamaClient | None = None):
        self.config = config or WritePathConfig()
        self.llm = llm or OllamaClient(LLMConfig(model=self.config.extractor_model))

    async def extract(
        self,
        user_message: str,
        assistant_response: str,
        recent: list[Message],
        known_subjects: list[str] | None = None,
    ) -> list[CandidateFact]:
        window = recent[-self.config.extract_window_turns :] if recent else []
        context = "\n".join(f"{m.role}: {m.content}" for m in window)
        # Showing the subjects already in the store is what lets the model reuse a key
        # instead of coining a synonym. Subject identity is an exact match, so a synonym
        # silently costs a contradiction -- the dominant error mode measured in
        # RESULTS.md §3.
        subjects = ""
        if known_subjects:
            listed = "\n".join(f"- {s}" for s in sorted(known_subjects)[:40])
            subjects = (
                "Already in memory for this session, as `subject -> properties`. If this "
                "turn asserts something about ANY of these, reuse that exact subject "
                "string; and if it gives a new value for a property already listed under "
                "that subject, reuse that exact property string too, so the two collide. "
                "A property NOT listed is a new one -- coin it rather than forcing the "
                "fact into a property it does not belong to:\n"
                f"{listed}\n\n"
            )
        # The assistant's reply is CONTEXT, not part of the turn to extract from, and the
        # distinction is structural rather than stylistic. The reply is generated *after*
        # the user's message, so unlike prior turns it can never be needed to resolve the
        # user's references -- while the reader that produced it holds the entire
        # conversation, against the three messages P1 is shown. Sitting inside "THE TURN
        # TO EXTRACT FROM" with equal standing to the user's own words, it let anything
        # from anywhere in the session launder into "what this turn asserts".
        #
        # Measured, RESULTS.md §17: on the real turn 23 of the console run ("When is my
        # flight to lisbon?"), P1 emitted 2 facts from the reply on both attempts with the
        # reply inside the block, and 0 on both with it demoted here -- while a genuine
        # user assertion carrying a reply still extracted correctly. The hint list was
        # ruled out as a channel by the same experiment.
        #
        # Note where this fix is NOT: `_SYSTEM` is untouched. No rule was added, because
        # the prompt already said "extract only what THIS turn asserts" and the reply was
        # inside "this turn". Restating the rule harder was measured (arm E) and added
        # nothing over moving the text.
        prompt = (
            (f"Prior turns (context only, for resolving references):\n{context}\n\n" if context else "")
            + subjects
            + f"THE TURN TO EXTRACT FROM:\nuser: {user_message}\n"
            + (
                "\nThe assistant's reply to that turn (context only, NOT a source of "
                f"facts):\nassistant: {assistant_response}\n"
                if assistant_response
                else ""
            )
        )
        payload = await self.llm.chat_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            schema=_SCHEMA,
        )
        now = datetime.now(UTC)
        out: list[CandidateFact] = []
        for item in (payload or {}).get("facts", []):
            confidence = float(item.get("confidence", 0.0))
            # §1.3: drop below the floor before the candidate leaves P1.
            if confidence < self.config.min_extract_confidence:
                continue
            fact_text = (item.get("fact") or "").strip()
            subject = (item.get("subject_hint") or "").strip()
            if not fact_text or not subject:
                continue
            try:
                fact_type = FactType(item.get("type", "STATE"))
            except ValueError:
                fact_type = FactType.STATE
            out.append(
                CandidateFact(
                    fact=fact_text,
                    type=fact_type,
                    confidence=confidence,
                    valid_at=now,
                    subject_hint=subject,
                    # Absent or empty is not an error: it means "unspecified", and
                    # `_competing` treats that as colliding with everything, which is
                    # exactly the behaviour of every extractor written before this field.
                    attribute=(item.get("attribute") or "").strip(),
                )
            )
        return out


class ScriptedExtractor:
    """Fake extractor returning pre-scripted candidate lists, so P2/P3 are testable
    without a model (recall-writepath-spec.md §8)."""

    def __init__(self, script: list[list[CandidateFact]]):
        self.script = list(script)
        self.calls = 0

    async def extract(
        self,
        user_message: str,
        assistant_response: str,
        recent: list[Message],
        known_subjects: list[str] | None = None,
    ) -> list[CandidateFact]:
        result = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return result
