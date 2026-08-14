"""Consolidation oracle -- validates the decision itself, with no retrieval and no LLM.

`retrieval_hit` in run.py answers "did the top-ranked live fact contain the gold string",
which conflates three things: whether consolidation left the right fact live, whether
retrieval ranked it first, and whether a short gold token ("India", "baseball") happened
to appear somewhere in an unrelated fact.

This module answers the narrower question the thesis actually makes:

    For the subject a question asks about, is the fact we left LIVE the newest one?

Method, deliberately deterministic so it cannot flatter the system under test:

  1. Group the corpus by the subject key consolidation used. Within a group, the
     benchmark's rule says the highest serial number is current.
  2. Match each question to a group by token-F1 between the question text and the
     group's facts -- no model, no embeddings, no reuse of our own retrieval.
  3. Report whether the group's newest fact is the one still live, and whether it
     carries the gold answer.

The two failure modes of exact-match subject keying, which `retrieval_hit` cannot see:

  over-merge   two genuinely different subjects got one key, so a valid fact was
               invalidated by an unrelated one.
  under-merge  one subject got split across two keys ("position Hines Ward plays" vs
               "Hines Ward's position"), so the contradiction was never detected and a
               stale fact is still live.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..aliases import AliasConfig, SubjectVocabulary
from ..consolidate import subject_key
from .data import BenchFact
from .scoring import normalize_answer

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "what", "which", "who", "where", "when", "is", "was", "are", "were", "the", "a", "an",
    "of", "in", "at", "to", "for", "by", "with", "does", "do", "did", "on", "from", "that",
    "this", "it", "its", "his", "her", "their", "s",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _f1(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


@dataclass
class OracleResult:
    n_questions: int = 0
    matched: int = 0
    consolidation_correct: int = 0
    stale_live: int = 0
    gold_fact_superseded: int = 0
    unmatched: int = 0
    n_groups: int = 0
    n_multi_fact_groups: int = 0
    failures: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.consolidation_correct / max(1, self.n_questions)


def build_groups(
    facts: list[BenchFact],
    subjects: dict[int, str],
    alias: AliasConfig | None = None,
) -> dict[str, list[BenchFact]]:
    """Reproduce the grouping ingest actually performed -- including alias resolution.

    Facts are walked in SERIAL order with a vocabulary that grows as it goes, because that
    is what the consolidator sees: document frequency at decision time covers only the
    subjects that have already arrived, so the merges ingest made are a subset of the ones
    the finished corpus would predict. Grouping the completed corpus in one pass would
    credit the store with merges it never made and score a system that was never run.

    Duplicates need no special case: a DUPLICATE candidate is not stored, but its subject
    is already in the vocabulary by definition, so adding its key again is a no-op.
    """
    vocabulary = SubjectVocabulary(alias or AliasConfig())
    groups: dict[str, list[BenchFact]] = defaultdict(list)
    for fact in facts:
        key = vocabulary.resolve(subject_key(subjects.get(fact.serial, fact.text)))
        vocabulary.add(key)
        groups[key].append(fact)
    for key in groups:
        groups[key].sort(key=lambda f: f.serial)
    return dict(groups)


def evaluate(
    facts: list[BenchFact],
    subjects: dict[int, str],
    questions: list[str],
    answers: list[list[str]],
    live_texts: set[str],
    alias: AliasConfig | None = None,
) -> OracleResult:
    """`live_texts` is the set of fact strings still live in the store after ingest.

    `alias` must match the `ConsolidationConfig.alias` the ingest ran with, or the groups
    scored here are not the groups the store built.
    """
    groups = build_groups(facts, subjects, alias)
    result = OracleResult(
        n_questions=len(questions),
        n_groups=len(groups),
        n_multi_fact_groups=sum(1 for g in groups.values() if len(g) > 1),
    )

    # Match against the subject key, not the concatenated fact texts. Joining a group's
    # facts inflates its token count, so token-F1 systematically favours single-fact
    # groups -- exactly the groups that cannot have had a contradiction. Matching on the
    # short canonical key removes that length bias.
    group_tokens = {key: _tokens(key) for key in groups}

    for question, gold in zip(questions, answers, strict=True):
        q_tokens = _tokens(question)
        best_key, best_score = None, 0.0
        for key, tokens in group_tokens.items():
            score = _f1(q_tokens, tokens)
            if score > best_score:
                best_key, best_score = key, score

        if best_key is None or best_score < 0.2:
            result.unmatched += 1
            result.failures.append({"kind": "unmatched", "question": question, "gold": gold})
            continue

        result.matched += 1
        members = groups[best_key]
        newest = members[-1]
        live_members = [f for f in members if f.text in live_texts]
        gold_norm = [normalize_answer(g) for g in gold]

        newest_has_gold = any(g in normalize_answer(newest.text) for g in gold_norm)
        newest_is_live = newest.text in live_texts

        if newest_is_live and newest_has_gold:
            result.consolidation_correct += 1
            continue

        if not newest_is_live and newest_has_gold:
            # We invalidated the fact the benchmark says is current: over-merge.
            result.gold_fact_superseded += 1
            kind = "over_merge"
        elif newest_is_live and not newest_has_gold:
            # The group's newest fact is live but is not the answer -- the question was
            # matched to the wrong group, or the subject was split (under-merge).
            kind = "wrong_group_or_under_merge"
        else:
            kind = "both"
        if len(live_members) > 1:
            result.stale_live += 1

        result.failures.append(
            {
                "kind": kind,
                "question": question,
                "gold": gold,
                "matched_subject": best_key,
                "match_score": round(best_score, 3),
                "group_size": len(members),
                "newest_fact": newest.text,
                "newest_is_live": newest_is_live,
                "live_in_group": [f.text for f in live_members],
            }
        )

    return result
