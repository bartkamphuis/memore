"""Loading the MemoryAgentBench Conflict_Resolution split.

Shape of the data, established by inspection:

  8 rows = {sh, mh} x {6k, 32k, 64k, 262k}, named in `metadata.source` as
  `factconsolidation_<sh|mh>_<len>`. `sh` is single-hop -- the 54%-of-the-field figure.
  Each row's `context` is a numbered list of pre-stated facts ("0. Thomas Kyd was born
  in the city of London."), 455 of them at 6k up to 18332 at 262k, where a later serial
  number overrides an earlier fact about the same subject. Each row carries 100
  questions whose gold answer is the value from the *highest-numbered* matching fact.

  Note the contexts are pre-stated facts rather than dialogue. The benchmark is testing
  the consolidation decision, not extraction -- which is exactly the variable this spike
  is isolating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PARQUET = Path("data/Conflict_Resolution.parquet")
HF_URL = (
    "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/"
    "data/Conflict_Resolution-00000-of-00001.parquet"
)

_NUMBERED = re.compile(r"^(\d+)\.\s+(.*)$")


@dataclass(frozen=True)
class BenchFact:
    serial: int
    text: str


@dataclass(frozen=True)
class BenchSample:
    source: str
    facts: list[BenchFact]
    questions: list[str]
    answers: list[list[str]]


def parse_facts(context: str) -> list[BenchFact]:
    out: list[BenchFact] = []
    for line in context.split("\n"):
        match = _NUMBERED.match(line.strip())
        if match:
            out.append(BenchFact(serial=int(match.group(1)), text=match.group(2).strip()))
    return out


def load(source: str, parquet_path: Path | None = None) -> BenchSample:
    import pyarrow.parquet as pq

    path = parquet_path or DEFAULT_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n  curl -sL -o {path} '{HF_URL}'"
        )
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if row["metadata"]["source"] == source:
            return BenchSample(
                source=source,
                facts=parse_facts(row["context"]),
                questions=list(row["questions"]),
                answers=[list(a) for a in row["answers"]],
            )
    available = sorted(r["metadata"]["source"] for r in rows)
    raise KeyError(f"source {source!r} not in parquet; available: {available}")
