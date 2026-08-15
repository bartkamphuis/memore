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

Rules for each extracted fact:
- `fact` MUST be standalone: resolve every pronoun and reference so the sentence is
  meaningful with no surrounding context. "switched it to prod" is WRONG;
  "deploys to prod by default" is RIGHT.
- `subject_hint` is the TOPIC the fact is about: "the user", "the memory system",
  "the Netherlands". Prefer the shortest natural noun phrase, and reuse a subject you
  have already used in this conversation rather than rephrasing it.
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
        prompt = (
            (f"Prior turns (context only, for resolving references):\n{context}\n\n" if context else "")
            + subjects
            + f"THE TURN TO EXTRACT FROM:\nuser: {user_message}\n"
            + (f"assistant: {assistant_response}\n" if assistant_response else "")
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
