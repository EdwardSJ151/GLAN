"""
Minimal generic async vLLM caller — mirrors the SeedDataGen pattern.

Speed-critical pieces kept intact:
  - ONE AsyncOpenAI client, reused for every request.
  - asyncio.gather to fan a batch out concurrently.
  - asyncio.Semaphore(max_concurrent) held only around the HTTP call, so it caps
    the number of simultaneous in-flight requests (this is the throughput knob).
  - extra_body with thinking disabled + stop strings (vLLM correctness).

Edit VLLM_BASE_URL / the prompts, then:  python vllm_async_example.py
Only dependency: `openai`.
"""

import asyncio
from typing import List, Optional

from openai import AsyncOpenAI

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "no-key-needed"
STOP_STRINGS = ["<|im_end|>", "<|end_of_text|>"]

# max_concurrent is the real speed knob: simultaneous in-flight requests.
# Keep batch_size >= max_concurrent so the pipe stays full (batches are processed
# one after another, so a small batch under-utilises the server).
MAX_CONCURRENT = 64
BATCH_SIZE = 64


async def _generate(
    client: AsyncOpenAI,
    model_id: str,
    prompt: str,
    sem: asyncio.Semaphore,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> Optional[str]:
    """One request; the semaphore is held only for the duration of the call."""
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stop": STOP_STRINGS,
                },
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # keep going on a single failure; return None for that item
        print(f"[error] {e}")
        return None


async def run_prompts(
    prompts: List[str],
    *,
    batch_size: int = BATCH_SIZE,
    max_concurrent: int = MAX_CONCURRENT,
) -> List[Optional[str]]:
    """Process *prompts* in batches, return outputs in the same order."""
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    model_id = (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(max_concurrent)
    print(f"model: {model_id}  |  {len(prompts)} prompts, batch={batch_size}, concurrency={max_concurrent}")

    results: List[Optional[str]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results.extend(
            await asyncio.gather(*[_generate(client, model_id, p, sem) for p in batch])
        )
        print(f"  {min(start + batch_size, len(prompts))}/{len(prompts)} done")
    return results


async def main():
    prompts = [
        "Explique em uma frase o que é uma chave fusível.",
        "Liste três cuidados de segurança ao trabalhar com redes de média tensão.",
        "O que significa proteção seletiva?",
    ]
    for prompt, output in zip(prompts, await run_prompts(prompts)):
        print(f"\nPROMPT: {prompt}\nOUTPUT: {output}")


if __name__ == "__main__":
    asyncio.run(main())
