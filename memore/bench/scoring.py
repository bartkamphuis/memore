"""MemoryAgentBench's own scorer, ported verbatim.

Source: HUST-AI-HYZ/MemoryAgentBench, `utils/eval_other_utils.py`. Their README states
the metric mapping explicitly:

    | Conflict Resolution | fact_mh, fact_sh | Accuracy | `substring_exact_match` |

so "accuracy" on FactConsolidation -- the ~54% figure this project is measured against --
is normalized substring exact match, maximized over ground truths, and additionally
maximized against the output re-parsed from an "Answer:" prefix.

Do not "improve" the normalization. An absolute number is only comparable to the
published field results if the scoring rule is identical.
"""

from __future__ import annotations

import re
import string


def normalize_answer(answer_text: str) -> str:
    text = answer_text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def drqa_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def _flatten(ground_truths) -> list[str]:
    if isinstance(ground_truths, str):
        return [ground_truths]
    out: list[str] = []
    for item in ground_truths:
        if isinstance(item, str):
            out.append(item)
        else:
            out.extend(_flatten(item))
    return out


def drqa_metric_max_over_ground_truths(metric_function, prediction: str, ground_truths) -> float:
    return max((float(metric_function(prediction, gt)) for gt in _flatten(ground_truths)), default=0.0)


def parse_output(output_text: str, answer_prefix: str = "Answer:") -> str | None:
    """Their `parse_output`: take what follows the answer prefix, if present."""
    if answer_prefix not in output_text:
        return None
    tail = output_text.split(answer_prefix)[-1].strip()
    return tail.split("\n")[0].strip() or None


def score(prediction: str, ground_truths) -> dict[str, float]:
    """The two metrics that matter here. `substring_exact_match` is the headline."""

    def compute(text: str) -> dict[str, float]:
        return {
            "exact_match": drqa_metric_max_over_ground_truths(
                drqa_exact_match_score, text, ground_truths
            ),
            "substring_exact_match": drqa_metric_max_over_ground_truths(
                substring_exact_match_score, text, ground_truths
            ),
        }

    metrics = compute(prediction)
    parsed = parse_output(prediction)
    if parsed is not None:
        for name, value in compute(parsed).items():
            metrics[name] = max(metrics[name], value)
    return metrics
