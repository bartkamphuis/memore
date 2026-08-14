"""Bench input adapter: knowledge-pool lines -> `CandidateFact`s.

This is P1's job (turn -> candidates) applied to the benchmark's input shape. The
FactConsolidation contexts are pre-stated facts rather than dialogue, so there is no
salience judgement to make -- every line is a durable fact by construction. What the
extractor must produce is the `subject_hint`: the key that decides which facts collide.

Both spike arms consume the SAME cached candidate list. That is deliberate: extraction
quality then cancels out of the arm-vs-arm comparison, leaving the consolidation decision
as the only variable (the one thing the spike exists to measure).

An LLM is legitimate here -- P1 is off the response path. The consolidation decision that
follows never sees a model.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from ..llm import LLMConfig, OllamaClient
from ..types import CandidateFact, FactType
from .data import BenchFact

# Deliberately NOT `config.DEFAULT_LLM_MODEL`, and the exception is the point: this string
# is a CACHE KEY (`_cache_path`), and every subject cache under `data/cache/` was written by
# this model. Every oracle number in RESULTS.md §3 -- including the subject groupings that
# the under-merge analysis is entirely about -- is that cache. Pointing this at a different
# model does not "upgrade the extractor": it silently misses the cache, spends hours
# re-extracting, and changes how subjects are NAMED, which is the one variable those
# measurements hold fixed. Re-extracting under another model is a separate experiment that
# must re-baseline everything downstream of it, not a default change.
CACHED_EXTRACTOR_MODEL = "gemma4:12b"

_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "subject": {"type": "string"},
                },
                "required": ["i", "subject"],
            },
        }
    },
    "required": ["items"],
}

_SYSTEM = """You split factual statements into SUBJECT and VALUE. You only output the subject.

The SUBJECT is what the statement is ABOUT, excluding the value it asserts. The VALUE is
the specific answer the statement gives.

  "The capital of Germany is Berlin."          subject: "capital of Germany"
  "Hines Ward plays the position of cornerback." subject: "position Hines Ward plays"
  "Chanel was founded by Coco Chanel."          subject: "founder of Chanel"
  "Germany is located in the continent of Asia." subject: "continent Germany is located in"

CRITICAL RULE: two statements that assert DIFFERENT VALUES FOR THE SAME THING must get
the IDENTICAL subject string, character for character. "The capital of Germany is Berlin"
and "The capital of Germany is Bonn" must both yield exactly "capital of Germany".
Always phrase the subject in the same canonical way for the same relation and entity.
Never include the value in the subject.

Return one item per input line, with `i` set to the line's index number.
"""


def _batch_prompt(batch: list[tuple[int, str]]) -> str:
    lines = "\n".join(f"{index}. {text}" for index, text in batch)
    return f"Give the subject for each of these {len(batch)} statements:\n{lines}"


class KnowledgePoolExtractor:
    def __init__(
        self,
        model: str = CACHED_EXTRACTOR_MODEL,
        batch_size: int = 25,
        concurrency: int = 4,
        cache_dir: Path = Path("data/cache"),
    ):
        self.llm = OllamaClient(LLMConfig(model=model, num_ctx=8192))
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.cache_dir = cache_dir
        self.model = model

    def _cache_path(self, source: str) -> Path:
        return self.cache_dir / f"subjects-{source}-{self.model.replace(':', '_')}.json"

    async def _run_batch(self, batch: list[tuple[int, str]], sem: asyncio.Semaphore) -> dict[int, str]:
        async with sem:
            for attempt in range(3):
                try:
                    payload = await self.llm.chat_json(
                        [
                            {"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": _batch_prompt(batch)},
                        ],
                        schema=_SCHEMA,
                    )
                    return {
                        int(item["i"]): str(item["subject"]).strip()
                        for item in (payload or {}).get("items", [])
                        if str(item.get("subject", "")).strip()
                    }
                except Exception:  # noqa: BLE001 -- retry, then fall back below
                    if attempt == 2:
                        return {}
                    await asyncio.sleep(1.0)
            return {}

    async def subjects_for(self, source: str, facts: list[BenchFact]) -> dict[int, str]:
        """Serial -> subject string, cached on disk so both arms reuse one extraction."""
        cache_path = self._cache_path(source)
        if cache_path.exists():
            return {int(k): v for k, v in json.loads(cache_path.read_text()).items()}

        batches = [
            [(f.serial, f.text) for f in facts[i : i + self.batch_size]]
            for i in range(0, len(facts), self.batch_size)
        ]
        sem = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(*(self._run_batch(b, sem) for b in batches))

        subjects: dict[int, str] = {}
        for chunk in results:
            subjects.update(chunk)

        # Any line the model dropped or mangled falls back to the whole sentence as its
        # own subject: it then collides with nothing and is stored as NEW. Recorded in
        # the run report rather than silently patched.
        missing = [f for f in facts if f.serial not in subjects]
        for fact in missing:
            subjects[fact.serial] = fact.text

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({str(k): v for k, v in subjects.items()}, indent=0))
        return subjects

    async def candidates_for(self, source: str, facts: list[BenchFact]) -> list[CandidateFact]:
        subjects = await self.subjects_for(source, facts)
        now = datetime.now(UTC)
        return [
            CandidateFact(
                fact=fact.text,
                type=FactType.STATE,
                confidence=1.0,
                valid_at=now,
                subject_hint=subjects.get(fact.serial, fact.text),
            )
            for fact in facts
        ]

    async def aclose(self) -> None:
        await self.llm.aclose()
