"""Scoring rules of the two bench instruments added in the wrap-up pass.

Pure functions only -- no store, no embedder, no LLM. What they pin is not arithmetic but
the two discipline rules the harnesses would otherwise silently break:

* a variant that stored NOTHING must not be scored as agreeing with itself
  (`memore.bench.paraphrase`, the §17.4 / §18.8 hole);
* the gate feature dump is computed from values the gate has already finished with, so a
  row can never influence a decision (`memore.bench.calibrate`).
"""

from __future__ import annotations

import math

from memore.bench.calibrate import _entropy
from memore.bench.paraphrase import PARAPHRASES, AxisScore, RunReport, TurnResult, VariantResult
from memore.types import CandidateFact, FactType


def _fact(subject: str, attribute: str = "", single_valued: bool = True) -> CandidateFact:
    return CandidateFact(
        fact=f"a fact about {subject}",
        type=FactType.STATE,
        confidence=1.0,
        valid_at=None,
        subject_hint=subject,
        attribute=attribute,
        single_valued=single_valued,
    )


def _turn(index: int, *variants: list[CandidateFact]) -> TurnResult:
    return TurnResult(
        index=index,
        variants=[VariantResult(text=f"v{i}", facts=list(f)) for i, f in enumerate(variants)],
    )


def test_a_variant_that_stored_nothing_is_excluded_not_passed():
    """Two silent variants and one speaker is NO-DATA, never unanimous.

    The whole point of the harness is agreement, and a turn nothing was extracted from
    agrees with itself trivially. `slots.py` learned this the expensive way: an axis blind
    to the empty case scores highest on the input that destroys the store.
    """
    score = AxisScore()
    score.add(["the user"])
    assert score.turns == 0
    assert score.unanimous == 0
    assert score.no_data == 1


def test_disagreement_is_reported_pairwise_as_well_as_unanimously():
    """One dissenter in four reads differently from four different answers.

    Unanimity alone flattens those to the same 0, and they are not the same failure: the
    first splits a subject in two, the second has no subject at all.
    """
    one_dissenter = AxisScore()
    one_dissenter.add(["a", "a", "a", "b"])
    all_different = AxisScore()
    all_different.add(["a", "b", "c", "d"])

    assert one_dissenter.unanimous == all_different.unanimous == 0
    assert one_dissenter.agreeing_pairs == 3  # the three ways to pick two of the three "a"
    assert all_different.agreeing_pairs == 0


def test_the_axes_are_scored_independently():
    """A turn may agree on the subject and disagree on the arity. Pooling hides that."""
    report = RunReport(run=0)
    report.turns.append(
        _turn(
            0,
            [_fact("the user", "deploy target", True)],
            [_fact("the user", "deploy target", True)],
            [_fact("the user", "deploy target", False)],
            [_fact("the user", "deploy target", True)],
        )
    )
    scores = report.score(PARAPHRASES)
    assert scores["subject"].unanimous == 1
    assert scores["attribute"].unanimous == 1
    assert scores["arity"].unanimous == 0


def test_the_attribute_is_compared_after_normalization():
    """`consolidate()` normalizes the attribute before it becomes half of a slot key, so
    comparing the raw string would report a disagreement the system does not have."""
    report = RunReport(run=0)
    report.turns.append(
        _turn(
            0,
            [_fact("the user", "Deploy Target")],
            [_fact("the user", "Deploy Target")],
            [_fact("the user", "deploy  target")],
            [_fact("the user", "target deploy")],
        )
    )
    assert report.score(PARAPHRASES)["attribute"].unanimous == 1


def test_cardinality_fails_when_one_variant_is_silent():
    """The axis that catches the failure every other axis is blind to."""
    report = RunReport(run=0)
    report.turns.append(
        _turn(0, [_fact("the user")], [_fact("the user")], [], [_fact("the user")])
    )
    scores = report.score(PARAPHRASES)
    assert scores["cardinality"].unanimous == 0
    # ...while the field axes still compare the two variants that DID speak, rather than
    # discarding the turn: agreement among the ones that answered is a real observation.
    assert scores["subject"].unanimous == 1


def test_entropy_is_zero_on_a_single_hit_and_maximal_on_a_flat_block():
    assert _entropy([0.9]) == 0.0
    assert _entropy([]) == 0.0
    assert _entropy([0.0, 0.0]) == 0.0
    flat = _entropy([0.5, 0.5, 0.5, 0.5])
    assert math.isclose(flat, math.log(4), rel_tol=1e-9)
    peaked = _entropy([0.9, 0.05, 0.03, 0.02])
    assert peaked < flat
