"""Utilities to extract structured JSONL from LLM responses."""
import json
import re
from typing import Any, Dict, List


def extract_jsonl(text: str) -> List[Dict[str, Any]]:
    """Extract JSONL objects from between ``` fences, or parse text directly."""
    if not text:
        return []
    match = re.search(r"```(?:jsonl|json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    raw = match.group(1) if match else text

    items = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return items
