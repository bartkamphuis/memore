"""Subject aliasing -- the principled version of the subset merge RESULTS.md §3 rejected.

§3 left this open by name. After order-insensitive keys, every remaining consolidation
error at both corpus sizes was an **under-merge**: P1 named one subject two ways, the two
namings never collided, both facts stayed live, and the reader saw a contradiction that
consolidation should have resolved. The residual shape is one naming carrying an extra
content word:

    religion karen armstrong   vs   affiliated karen armstrong religion

Sorting tokens cannot collapse that. Merging on `tokens(A) subset-of tokens(B)` can, and
scores better -- and §3 refused to ship it anyway, because some extra tokens are not
relation words but *entities*, and merging those is unrecoverable:

    buddhism founder  <  buddhism founder shingon      different founders
    england ...       <  england kingdom ...           England is in Europe; the Kingdom
                                                       of England is a different subject

§3 named the missing ingredient: gate the merge on the extra tokens being **generic
relation words rather than rare entity words**, decided by document frequency across
subjects. This module is that gate. No LLM, no embedding, no similarity threshold -- the
same statistic `memore.subjects` already uses on the read side, applied to a different
question.

## The threshold is relative, and that was measured, not assumed

The obvious form -- "generic means df >= N" -- is wrong, and the two corpora show it
outright. `affiliated` is the *same* relation word in both:

    sh_6k    df  9 / 303 subjects   = 0.0297
    sh_32k   df 60 / 1559 subjects  = 0.0385

An absolute cutoff tuned at 32k is inert at 6k; one tuned at 6k merges `rugby union` into
`rugby` at 32k. The ratio is stable across the 5x scale change where the count is not, so
the rule reads `df / n_subjects`.

## Where the default came from

All 68 subset pairs across both corpora were enumerated and hand-labelled. Sorted by the
minimum extra-token ratio, they separate:

    must NOT merge, worst case   kingdom   0.0071   (Kingdom of England vs England)
                                 ireland   0.0045   (UK of GB and Ireland vs Great Britain)
                                 union     0.0038   (rugby union vs rugby)
                                 rock      0.0026   (rock music vs country music)
                                 shingon   0.0006   (Shingon Buddhism vs Buddhism)
    ---- decision boundary ----
    genuine merges               written   0.0083
                                 location  0.0083 / 0.0099
                                 famous    0.0186
                                 current   0.0244
                                 located   0.0334
                                 affiliated 0.0297 / 0.0385
                                 city      0.0673

Two bands separate the labels: (0.0071, 0.0083] captures everything, and (0.0083, 0.0186]
gives up `location` and `written` for a 2.1x margin over the worst unsafe case instead of
1.17x. `ConsolidationConfig` already records why that asymmetry is not a close call -- a
false merge discards a correct fact permanently, a missed merge leaves the status quo -- so
the default sits in the wide gap, not at the edge of the narrow one.

## What this rule can never do, stated so nobody tunes toward it

Two genuine relation words in the 32k corpus have df=1: `employed` (`employer luther
martin` vs `employed employer luther martin`) and `married` (`elvis presley spouse` vs
`elvis married presley spouse`). They are real merges this rule refuses, and no threshold
recovers them -- lowering the bar far enough to catch a df=1 relation word first sweeps in
`shingon`, `gaelic` and `nazi` at the same frequency. A rule reading surface statistics
cannot recognise a relation word that appears once. That is the same wall §9 hits from the
other side with `coffee preference` / `milk`: an identity question, not a frequency one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AliasConfig:
    """Off by default is NOT the setting here -- see `enabled`."""

    enabled: bool = True
    # Fraction of the session's live subjects a token must appear in to count as a relation
    # word rather than an entity. Chosen in the gap between the labelled bands above, not
    # tuned for score: at 0.008 the rule additionally merges 11 correct pairs, and the
    # oracle scores identically, so the extra margin is free.
    df_ratio: float = 0.015
    # Below this many live subjects the rule is inert, and that is the point. Document
    # frequency is a corpus statistic and a small store has no corpus: in a 20-subject
    # personal session, `location` in 2 subjects reads as ratio 0.10 -- seven times the
    # threshold -- so an unguarded relative rule fires hardest exactly where its evidence
    # is weakest. The measured corpora are 303 and 1559 subjects; nothing smaller has been
    # validated, so nothing smaller is policed. Together with `df_ratio` this also implies
    # an absolute floor: a token needs >= 3 subjects at the smallest permitted vocabulary.
    min_subjects: int = 200


class SubjectVocabulary:
    """The session's live subject keys, and the merge decision that reads them.

    Deliberately mutable and incremental. The consolidator sees subjects arriving one at a
    time, so document frequency at decision time covers only what has landed so far --
    `affiliated` is not yet a relation word on its first appearance. The rule is therefore
    conservative when cold and converges as the session fills, which is the correct
    direction for a decision whose false positives are unrecoverable, but it does mean the
    grouping depends on arrival order. Rebuilding the vocabulary from the finished corpus
    would predict merges that ingest never made; `memore.bench.oracle` replays this same
    object in arrival order for exactly that reason.
    """

    def __init__(self, config: AliasConfig | None = None, keys: list[str] | None = None):
        self.config = config or AliasConfig()
        self._keys: set[str] = set()
        self._df: dict[str, int] = {}
        self._postings: dict[str, set[str]] = {}
        # Every merge this vocabulary made, as (arriving key, key it was folded into).
        # Not diagnostics: RESULTS.md §3 rejected the ungated version of this rule *despite*
        # a better aggregate score, because aggregates hide over-merges. A merge log that
        # can be read end to end is the evidence that argument demands, so it is part of
        # the object rather than something a caller has to reconstruct.
        self.merges: list[tuple[str, str]] = []
        for key in keys or []:
            self.add(key)

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str) -> None:
        """Record a subject key. Idempotent -- document frequency counts DISTINCT subjects.

        Keys are only ever added, never removed, because a subject does not stop being live
        when one of its facts is superseded: consolidation writes the newer fact under the
        same key. A key leaves the live set only if every fact on it is superseded, which
        the freshness primitive never does.
        """
        if not key or key in self._keys:
            return
        self._keys.add(key)
        for token in set(key.split()):
            self._df[token] = self._df.get(token, 0) + 1
            self._postings.setdefault(token, set()).add(key)

    def is_generic(self, token: str) -> bool:
        """Does this token name a relation (shared across subjects) or an entity (rare)?"""
        if not self._keys:
            return False
        return self._df.get(token, 0) / len(self._keys) >= self.config.df_ratio

    def resolve(self, key: str) -> str:
        """The key `key`'s facts should actually be stored under.

        Returns `key` itself unless exactly one existing subject differs from it only by
        generic relation words, in which case that subject's key is returned and the two
        collide from here on.

        Returning the INCUMBENT's key rather than a canonical form of the two is what keeps
        this a pure addition to the write path: the facts already stored do not have to be
        rewritten, they simply acquire a new sibling. The consequence, recorded rather than
        hidden, is that which of two equivalent spellings names the group depends on which
        arrived first. Group *membership* -- the only thing consolidation acts on -- does
        not.
        """
        if not self.config.enabled or not key:
            return key
        if key in self._keys:
            # Exact match: already the same subject, nothing to alias.
            return key
        if len(self._keys) < self.config.min_subjects:
            return key

        tokens = set(key.split())
        matches = {c for c in self._candidates(tokens) if self._only_generic_apart(tokens, c)}
        if len(matches) != 1:
            # Zero: a genuinely new subject. More than one: the key sits between two
            # existing subjects and there is no evidence which it belongs to. Refusing an
            # ambiguous merge is the only safe move -- picking the "best" one is a
            # similarity judgement, which is exactly what this design does not make.
            return key
        target = matches.pop()
        self.merges.append((key, target))
        return target

    def _candidates(self, tokens: set[str]) -> set[str]:
        """Existing keys whose token set is a strict subset or superset of `tokens`.

        Every candidate must share at least one token, so the postings lists bound the
        scan; the alternative is comparing against all ~1600 live subjects per candidate.
        """
        pool: set[str] = set()
        for token in tokens:
            pool |= self._postings.get(token, set())
        out = set()
        for other in pool:
            other_tokens = set(other.split())
            if other_tokens < tokens or other_tokens > tokens:
                out.add(other)
        return out

    def _only_generic_apart(self, tokens: set[str], other: str) -> bool:
        extras = tokens.symmetric_difference(set(other.split()))
        # An unseen token has df 0 and is refused, which is right: a word the session has
        # never used elsewhere is the best available evidence of a new entity.
        return bool(extras) and all(self.is_generic(t) for t in extras)
