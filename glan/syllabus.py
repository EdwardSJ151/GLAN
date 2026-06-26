"""Stage 3 — Syllabus Generator.

For each subject, generates a detailed syllabus (ementa) with class sessions and key
concepts. Same two-prompt approach as Stage 2: content first, JSONL extraction second.
One query per subject (paper §3.1), saved per-discipline/subject for resumption.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

from glan.llm import run_conversations, run_prompts
from glan.parse import extract_jsonl

_CONTENT_PROMPT = """\
Você é um professor universitário brasileiro especialista em "{nome_materia}" \
({nivel}) na área de {discipline}.

Os principais subtópicos desta disciplina são: {subtopicos}.

Elabore uma ementa detalhada para esta disciplina de nível superior, conforme o \
padrão das universidades brasileiras (seguindo diretrizes do MEC). Divida o conteúdo \
em aulas/sessões e, para cada aula, forneça:
1. Título da aula
2. Breve descrição do conteúdo abordado
3. Lista de conceitos-chave que os alunos devem dominar (de 3 a 7 conceitos por aula)

A ementa deve ter entre 10 e 20 aulas, ter profundidade de nível universitário e \
ser relevante para a realidade brasileira.\
"""

_FORMAT_PROMPT = """\
Ótimo! Converta a ementa acima para o formato jsonl para que seja mais fácil de \
processar por um computador. Coloque o jsonl entre tags ``` ```. Para cada linha \
(aula), use as chaves "aula" (número inteiro), "titulo" (string), "descricao" \
(string) e "conceitos_chave" (lista de strings).\
"""


def _disc_slug(disc: str) -> str:
    return disc.lower().replace(" ", "_").replace("/", "_")


def _subj_slug(name: str) -> str:
    slug = name.lower().replace(" ", "_").replace("/", "_")
    return slug[:60]


def _subject_key(disc: str, nome: str) -> str:
    return f"{disc}::{nome}"


async def generate_syllabi(
    disciplines_subjects: Dict[str, List[dict]],
    output_dir: Path,
    *,
    batch_size: int = 32,
    max_concurrent: int = 32,
) -> Dict[str, List[dict]]:
    """Return mapping 'discipline::subject_name' → list of session dicts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    done: Dict[str, List[dict]] = {}
    pending: List[Tuple[str, dict]] = []

    for disc, subjects in disciplines_subjects.items():
        disc_dir = output_dir / _disc_slug(disc)
        for subj in subjects:
            nome = subj.get("nome_materia", "")
            key = _subject_key(disc, nome)
            out_file = disc_dir / f"{_subj_slug(nome)}.jsonl"
            if out_file.exists():
                sessions = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                done[key] = sessions
            else:
                pending.append((disc, subj))

    if done:
        print(f"  [ementa] {len(done)} ementas carregadas do disco")
    if not pending:
        return done

    print(f"\n[ementa] Gerando ementas para {len(pending)} matérias...")

    def _content_prompt(disc: str, subj: dict) -> str:
        return _CONTENT_PROMPT.format(
            discipline=disc,
            nome_materia=subj.get("nome_materia", ""),
            nivel=subj.get("nivel", "graduação"),
            subtopicos=", ".join(subj.get("subtopicos", [])),
        )

    # ── Step 1: content generation ───────────────────────────────────────────
    content_prompts = [_content_prompt(disc, subj) for disc, subj in pending]
    content_responses = await run_prompts(
        content_prompts,
        temperature=1.0,
        top_p=0.95,
        max_tokens=4000,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        label="ementa/conteúdo",
    )

    # ── Step 2: JSONL extraction ─────────────────────────────────────────────
    format_conversations: List[List[dict]] = []
    valid_pending: List[Tuple[str, dict]] = []

    for (disc, subj), response in zip(pending, content_responses):
        if response is None:
            continue
        format_conversations.append([
            {"role": "user", "content": _content_prompt(disc, subj)},
            {"role": "assistant", "content": response},
            {"role": "user", "content": _FORMAT_PROMPT},
        ])
        valid_pending.append((disc, subj))

    print(f"\n[ementa] Convertendo {len(format_conversations)} ementas para JSONL...")
    format_responses = await run_conversations(
        format_conversations,
        temperature=0.3,
        top_p=0.95,
        max_tokens=4000,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        label="ementa/formato",
    )

    # ── Save per-subject ─────────────────────────────────────────────────────
    for (disc, subj), resp in zip(valid_pending, format_responses):
        nome = subj.get("nome_materia", "")
        key = _subject_key(disc, nome)
        sessions = extract_jsonl(resp) if resp else []
        done[key] = sessions

        disc_dir = output_dir / _disc_slug(disc)
        disc_dir.mkdir(parents=True, exist_ok=True)
        out_file = disc_dir / f"{_subj_slug(nome)}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    saved = sum(len(v) for k, v in done.items() if any(k.startswith(d) for d, _ in pending))
    print(f"  [ementa] {len(valid_pending)} ementas salvas ({saved} aulas no total)")
    return done
