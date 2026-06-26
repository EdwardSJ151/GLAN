"""Async vLLM client — mirrors vllm_async_example.py, supports single and multi-turn calls."""
import asyncio
from typing import Callable, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

_BAR_FMT = "{desc}: {n_fmt}/{total_fmt} | {elapsed} | {rate_fmt}"

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "no-key-needed"
STOP_STRINGS = ["<|im_end|>", "<|end_of_text|>"]


async def _call(
    client: AsyncOpenAI,
    model_id: str,
    messages: List[dict],
    sem: asyncio.Semaphore,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Optional[str]:
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stop": STOP_STRINGS,
                },
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [erro] {e}")
        return None


async def _run_batched(
    client: AsyncOpenAI,
    model_id: str,
    sem: asyncio.Semaphore,
    items: list,
    make_messages: Callable,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    label: str,
) -> List[Optional[str]]:
    results: List[Optional[str]] = []
    with tqdm(total=len(items), desc=label, bar_format=_BAR_FMT, unit=" req") as pbar:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            batch_results = await asyncio.gather(*[
                _call(client, model_id, make_messages(item), sem,
                      temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                for item in batch
            ])
            results.extend(batch_results)
            pbar.update(len(batch))
    return results


async def _make_client() -> tuple[AsyncOpenAI, str]:
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    model_id = (await client.models.list()).data[0].id
    return client, model_id


async def run_prompts(
    prompts: List[str],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    batch_size: int = 32,
    max_concurrent: int = 32,
    label: str = "",
) -> List[Optional[str]]:
    """Run single-turn prompts concurrently in batches."""
    client, model_id = await _make_client()
    sem = asyncio.Semaphore(max_concurrent)
    tqdm.write(f"  [{label}] modelo={model_id} | {len(prompts)} prompts | lote={batch_size} | conc={max_concurrent}")
    return await _run_batched(
        client, model_id, sem, prompts,
        lambda p: [{"role": "user", "content": p}],
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        batch_size=batch_size, label=label,
    )


async def run_conversations(
    conversations: List[List[dict]],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    batch_size: int = 32,
    max_concurrent: int = 32,
    label: str = "",
) -> List[Optional[str]]:
    """Run multi-turn conversations concurrently in batches."""
    client, model_id = await _make_client()
    sem = asyncio.Semaphore(max_concurrent)
    tqdm.write(f"  [{label}] modelo={model_id} | {len(conversations)} conversas | lote={batch_size} | conc={max_concurrent}")
    return await _run_batched(
        client, model_id, sem, conversations,
        lambda msgs: msgs,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        batch_size=batch_size, label=label,
    )
