"""The answering step.

MemoryAgentBench measures a memory system end to end: the store retrieves, then an LLM
answers from what was retrieved. The published field numbers (Zep 7%, HippoRAG-v2 54% on
single-hop) are produced that way, so a comparable number needs the same shape.

The query prompt below is MemoryAgentBench's own `factconsolidation` rag_agent template,
copied from `utils/templates.py`, with one deviation recorded here rather than hidden:
their template tells the model to resolve conflicts itself by picking the largest serial
number. Our recalled block carries no serial numbers, because consolidation already
resolved the conflict before the model was asked. Keeping their instruction text intact
while supplying a pre-resolved block is the honest comparison -- the model is given the
same job and simply does not have to do the freshness reasoning.
"""

from __future__ import annotations

from ..config import DEFAULT_LLM_MODEL, DEFAULT_LLM_NUM_CTX
from ..llm import LLMConfig, OllamaClient

QUERY_TEMPLATE = (
    "Pretend you are a knowledge management system. Each fact in the knowledge pool is "
    "provided with a serial number at the beginning, and the newer fact has larger serial "
    "number. \n You need to solve the conflicts of facts in the knowledge pool by finding "
    "the newest fact with larger serial number. You need to answer a question based on "
    "this rule. You should give a very concise answer without saying other words for the "
    "question **only** from the knowledge pool you have memorized rather than the real "
    "facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the "
    "provided Knowledge Pool, what is the name of the current president of Russia? \n"
    "Answer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge "
    "Pool, {question} \nAnswer:"
)

SYSTEM = "You are a helpful assistant."


class Reader:
    """The reader is a measurement instrument, so changing it changes the numbers.

    Every end-to-end figure in RESULTS.md §2-§3 was read by `gemma4:12b` at 8192 ctx. The
    default now follows `config.DEFAULT_LLM_MODEL` instead, because a bench run that
    disagrees with the served pin reloads an 18GB model at every process start. Any
    end-to-end number produced after that change is a different reader's number and is not
    comparable to the recorded ones without a re-run of both. Pass `--reader-model` to
    reproduce the originals. The oracle (`bench.oracle`) is unaffected: it uses no LLM.
    """

    def __init__(self, model: str = DEFAULT_LLM_MODEL, num_ctx: int = DEFAULT_LLM_NUM_CTX):
        self.llm = OllamaClient(LLMConfig(model=model, num_ctx=num_ctx))

    async def answer(self, question: str, recalled_block: str | None) -> str:
        pool = recalled_block or "<recalled_context>\n(nothing recalled)\n</recalled_context>"
        user = f"{pool}\n\n{QUERY_TEMPLATE.format(question=question)}"
        # generation_max_length: 10 in their Factconsolidation_sh_6k.yaml -- answers are
        # short values, and a cap keeps a chatty local model from padding.
        return (
            await self.llm.chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                max_tokens=32,
            )
        ).strip()

    async def aclose(self) -> None:
        await self.llm.aclose()
