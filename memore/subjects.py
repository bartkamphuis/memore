"""Subject admission -- the gate's answer to wrong-subject recall.

The measured failure (RESULTS.md §5). A query about a subject the store does NOT hold,
in a relation it does, clears the similarity floor 68-88% of the time: "Which sport is
<someone we know nothing about> associated with?" against a store full of sport facts
about other people. Cosine measures topical fit and the store genuinely *is* topically
about that, so the gate opens and injects a confidently wrong fact. Off-domain queries,
by contrast, are held out at ~3%. The gate keeps out what the store knows nothing about;
it cannot keep out what the store knows about *someone else*.

No threshold on the score fixes this -- the distributions overlap by construction (bench
positives p05 0.621, hard negatives p95 0.675). The separating signal is not similarity
at all, it is whether the query NAMES the subject. That is decidable deterministically,
with no LLM and no embedding, from the session's own subject-key vocabulary.

The naive version of this rule is a trap, and measuring it first is what stopped it
shipping: requiring the query to share a token with the subject blocks **44% of
conversational positives**, because a chat subject is an abstract label and the query is
a paraphrase -- "what city am I in?" against subject "location" shares nothing. Answering
the hard-negative problem that way would destroy the case the gate exists to serve.

What makes the rule safe is the COMPETITOR precondition. The failure needs a crowded
relation: many subjects differing only by entity. "location" in a personal store has no
competitors, so it is never policed and paraphrase survives untouched. An encyclopedic
corpus has hundreds, and there the query is expected to name its entity.

Measured on the calibration query set, restricted to hits that actually cleared the floor:

    bench hard negatives blocked   58/75  (0.77)
    bench positives blocked         2/99  (0.02)
    chat positives blocked          0/23  (0.00)

and both bench "losses" were inspected: in each the top hit was already the WRONG subject
("Which continent is India located in?" matching a fact about Hyderabad), so the rule
rejected a wrong fact rather than a right one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chain import ChainNode
from .keys import normalize_subject


@dataclass(frozen=True)
class SubjectView:
    """The session's subject vocabulary, as the admission rule needs it."""

    # Fact text -> its normalized subject key. Covers superseded facts too: a superseded
    # hit can still be injected (§6.3 marks it rather than dropping it), so it has to be
    # admissible on the same terms as any other.
    subject_of: dict[str, str] = field(default_factory=dict)
    # Token -> how many DISTINCT live subjects contain it. A relation word ("sport",
    # "capital") is in many; an entity is in one or two. This is the whole entity
    # detector -- no NER, no model, just the store's own statistics.
    df: dict[str, int] = field(default_factory=dict)
    # Subject key -> how many other live subjects share at least one token with it.
    competitors: dict[str, int] = field(default_factory=dict)

    def admits(self, query: str, fact: str, min_competitors: int, df_max: int) -> bool:
        """May this fact be injected for this query?

        True unless the fact's subject is one of a crowd of same-relation subjects AND
        the query names none of what distinguishes it from that crowd.
        """
        subject_key = self.subject_of.get(fact)
        if subject_key is None:
            # Not a fact we have a subject for. Nothing to check against, so do not
            # invent a reason to drop it -- the gate's other rules still apply.
            return True
        if self.competitors.get(subject_key, 0) < min_competitors:
            # Uncontested subject: no sibling it could be confused with. This is the
            # branch that keeps conversational paraphrase working.
            return True
        distinctive = {t for t in subject_key.split() if self.df.get(t, 0) <= df_max}
        if not distinctive:
            # Nothing but relation words -- the subject has no entity to be named.
            return True
        return bool(distinctive & set(normalize_subject(query).split()))


def build_subject_view(nodes: list[ChainNode]) -> SubjectView:
    """Fold a session's facts into the vocabulary the rule reads.

    Document frequency and competitor counts are computed over LIVE subjects only: a
    superseded fact's subject is still live under its newer value, so counting both would
    double-count one subject, and a subject whose every fact is superseded is not
    something the store can currently answer with.
    """
    subject_of = {n.fact: n.subject_key for n in nodes}
    live_keys = {n.subject_key for n in nodes if n.invalid_at is None and n.subject_key}

    df: dict[str, int] = {}
    for key in live_keys:
        for token in set(key.split()):
            df[token] = df.get(token, 0) + 1

    # A subject competes with another when they share any token -- that shared token is
    # the relation ("sport", "capital of") and what differs is the entity.
    by_token: dict[str, set[str]] = {}
    for key in live_keys:
        for token in set(key.split()):
            by_token.setdefault(token, set()).add(key)
    competitors: dict[str, int] = {}
    for key in live_keys:
        siblings: set[str] = set()
        for token in set(key.split()):
            siblings |= by_token.get(token, set())
        competitors[key] = len(siblings - {key})

    return SubjectView(subject_of=subject_of, df=df, competitors=competitors)
