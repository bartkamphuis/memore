"""Subject identity -- the normalization that decides which facts collide.

Lifted out of `store/falkor.py` because it is a domain rule, not a storage concern: the
consolidator uses it to decide subject identity, the multi-hop walk uses it to decide
whether one fact's value names another's subject, and the store merely persists the
result. Keeping it in the store module made `memore.chain` import the store, which is
backwards -- the walk is pure and must stay testable without a database.

`store.falkor` re-exports `normalize_subject`, so existing imports keep working.
"""

from __future__ import annotations

import re

# Characters that are operators in the full-text query language. A user turn is
# arbitrary text, so it must be neutralised before it reaches the index.
FT_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)

# Function words carry no subject identity: "city where X worked" and "city X worked in"
# name the same thing. Dropping them is what lets the two collide.
_SUBJECT_STOPWORDS = frozenset(
    """a an the is are was were of in on at to for by with which that where who whom
    whose and or s did does do""".split()
)


def normalize_subject(subject: str) -> str:
    """Canonical subject key: content words, lowercased, de-duplicated by ORDER.

    Subject identity is decided by exact match on this key -- no threshold, no
    embedding, no LLM -- so this function *is* the identity contract, and how much it
    normalizes is the single biggest lever on accuracy.

    Sorting the tokens is the part that matters. The dominant measured error was P1
    naming one subject two ways, and in practice the two namings are the same content
    words in a different order: `sport associated with tunisia national football team`
    vs `sport tunisia national football team is associated with`. Order-insensitive
    matching collapses those deterministically, with no fuzziness introduced.

    Measured effect (oracle, `use_embedding_comparison=False`):
        sh_6k   95/100 -> 99/100   (314 -> 303 groups)
        sh_32k  87/100 -> 94/100  (1614 -> 1559 groups)
    Inspected: all 55 merges at 32k were word-order or function-word variants of one
    subject. Over-merges stayed at zero.

    The cost is readability -- the key reads as sorted tokens -- so `StoredFact` keeps a
    separate `subject_label` holding the first natural phrasing seen, which is what gets
    shown back to P1 and printed in traces.
    """
    tokens = FT_SPECIAL.sub(" ", subject.lower()).split()
    return " ".join(sorted(t for t in tokens if t not in _SUBJECT_STOPWORDS))
