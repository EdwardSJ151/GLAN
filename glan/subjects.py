"""Stage 2 — Subject Generator.

For each discipline, prompts the LLM to list subjects a Brazilian student should learn.
Uses a two-prompt approach (paper §2.2): content first, then JSONL formatting separately,
because combining both in one prompt degrades subject quality.

Runs each discipline `runs_per_discipline` times (paper uses 10; default 3) for coverage.
Results are deduplicated by subject name, filtered to college level only, and saved
per-discipline to enable resumption.
"""
import json
from pathlib import Path
from typing import Dict, List

from glan.llm import run_conversations, run_prompts
from glan.parse import extract_jsonl

# Only subjects at these levels are kept after generation.
COLLEGE_LEVELS = {"graduação", "pós-graduação", "graduacao", "pos-graduacao", "mestrado", "doutorado"}

_CONTENT_PROMPT = """\
Você é um especialista em educação superior na área de {discipline} no Brasil.

Liste as disciplinas que um estudante universitário brasileiro deveria cursar nessa área, \
cobrindo tanto a graduação quanto a pós-graduação (mestrado e doutorado). Para cada \
disciplina, inclua:
- Nome da disciplina (claro e específico, como apareceria em uma grade curricular de universidade brasileira)
- Nível: obrigatoriamente "graduação" ou "pós-graduação"
- De 3 a 6 subtópicos principais abordados na disciplina

Liste pelo menos 15 disciplinas diferentes. Foque exclusivamente no nível superior — \
não inclua conteúdo de ensino médio, técnico ou fundamental.\
"""

_FORMAT_PROMPT = """\
Ótimo! Converta o texto acima para o formato jsonl para que seja mais fácil de processar \
por um computador. Coloque o jsonl entre tags ``` ```. Para cada linha, use as chaves \
"nome_materia", "nivel" e "subtopicos" (lista de strings).\
"""


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


async def generate_subjects(
    disciplines: List[str],
    output_dir: Path,
    *,
    runs_per_discipline: int = 3,
    batch_size: int = 32,
    max_concurrent: int = 32,
) -> Dict[str, List[dict]]:
    """Return mapping discipline → list of subject dicts with nome_materia/nivel/subtopicos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    done: Dict[str, List[dict]] = {}
    pending: List[str] = []

    for disc in disciplines:
        out_file = output_dir / f"{_slug(disc)}.jsonl"
        if out_file.exists():
            subjects = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            done[disc] = subjects
            print(f"  [matérias] {disc}: {len(subjects)} matérias carregadas do disco")
        else:
            pending.append(disc)

    if not pending:
        return done

    # ── Step 1: content generation (high temperature for diversity) ──────────
    content_prompts = [
        _CONTENT_PROMPT.format(discipline=disc)
        for disc in pending
        for _ in range(runs_per_discipline)
    ]
    print(f"\n[matérias] Gerando conteúdo: {len(pending)} disciplinas × {runs_per_discipline} execuções")
    content_responses = await run_prompts(
        content_prompts,
        temperature=1.0,
        top_p=0.95,
        max_tokens=3000,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        label="matérias/conteúdo",
    )

    # ── Step 2: JSONL conversion (low temperature for reliable formatting) ───
    format_conversations: List[List[dict]] = []
    conv_discipline: List[str] = []
    repeated_disciplines = [d for d in pending for _ in range(runs_per_discipline)]

    for disc, response in zip(repeated_disciplines, content_responses):
        if response is None:
            continue
        format_conversations.append([
            {"role": "user", "content": _CONTENT_PROMPT.format(discipline=disc)},
            {"role": "assistant", "content": response},
            {"role": "user", "content": _FORMAT_PROMPT},
        ])
        conv_discipline.append(disc)

    print(f"\n[matérias] Convertendo {len(format_conversations)} respostas para JSONL...")
    format_responses = await run_conversations(
        format_conversations,
        temperature=0.3,
        top_p=0.95,
        max_tokens=3000,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        label="matérias/formato",
    )

    # ── Aggregate, deduplicate, save ─────────────────────────────────────────
    aggregated: Dict[str, List[dict]] = {d: [] for d in pending}
    for disc, resp in zip(conv_discipline, format_responses):
        if resp:
            aggregated[disc].extend(extract_jsonl(resp))

    for disc in pending:
        seen: set = set()
        unique: List[dict] = []
        for s in aggregated[disc]:
            name = s.get("nome_materia", "").strip().lower()
            nivel = s.get("nivel", "").strip().lower()
            is_college = any(lvl in nivel for lvl in COLLEGE_LEVELS)
            if name and name not in seen and is_college:
                seen.add(name)
                unique.append(s)

        out_file = output_dir / f"{_slug(disc)}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for s in unique:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  [matérias] {disc}: {len(unique)} matérias únicas salvas (nível superior)")
        done[disc] = unique

    return done
