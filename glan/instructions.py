"""Stage 4 — Instruction Generator.

Samples class sessions and key concepts from each syllabus, generates homework questions,
then generates answers separately (paper §2.4). Two sampling strategies are used:

  Strategy A — single session: pick 1 session, sample 1–5 key concepts.
  Strategy B — cross-session: pick 2 sessions, force at least 1 concept from each,
               total 2–5 concepts. Produces questions that integrate knowledge across topics.

Questions use the full syllabus as context (simulating what a student has studied up to
the selected sessions). Questions and answers are generated in separate LLM calls
(paper §3.1: GPT-4 for questions, GPT-3.5 for answers; here one model handles both).

Output is saved per-discipline to enable resumption.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from glan.llm import run_prompts

_QUESTION_PROMPT = """\
Você é um professor universitário brasileiro de {discipline} ministrando a disciplina \
"{nome_materia}" ({nivel}) em uma universidade brasileira.

A ementa completa da disciplina é:
{syllabus_text}

O aluno já estudou até as seguintes aulas: {aulas_estudadas}.

Com base nos conceitos a seguir: {conceitos_selecionados}, elabore uma questão \
dissertativa ou de resolução de problemas para uma prova ou lista de exercícios de \
nível universitário. A questão deve:
- Exigir raciocínio analítico condizente com o nível {nivel}
- Integrar os conceitos selecionados de forma significativa
- Ser contextualizada na realidade brasileira quando aplicável
- Estar escrita em português do Brasil com precisão acadêmica

Escreva apenas a questão, sem incluir a resposta.\
"""

_ANSWER_PROMPT = """\
Você é um professor universitário brasileiro especialista em {discipline}. \
Responda à questão a seguir de forma completa, rigorosa e didática, em português \
do Brasil, no nível de uma correção de prova de graduação ou pós-graduação:

{question}\
"""


def _syllabus_text(sessions: List[dict]) -> str:
    lines = []
    for s in sessions:
        aula = s.get("aula", "?")
        titulo = s.get("titulo", "")
        conceitos = s.get("conceitos_chave", [])
        lines.append(f"Aula {aula}: {titulo} — Conceitos: {', '.join(conceitos)}")
    return "\n".join(lines)


def _strategy_a(sessions: List[dict]) -> Optional[Tuple[List[str], List[str]]]:
    """Pick 1 session, sample 1–5 concepts from it."""
    session = random.choice(sessions)
    concepts = session.get("conceitos_chave", [])
    if not concepts:
        return None
    k = min(random.randint(1, 5), len(concepts))
    return [session.get("titulo", "")], random.sample(concepts, k)


def _strategy_b(sessions: List[dict]) -> Optional[Tuple[List[str], List[str]]]:
    """Pick 2 sessions, combine concepts (at least 1 from each), total 2–5."""
    if len(sessions) < 2:
        return _strategy_a(sessions)
    s1, s2 = random.sample(sessions, 2)
    c1, c2 = s1.get("conceitos_chave", []), s2.get("conceitos_chave", [])
    if not c1 or not c2:
        return _strategy_a(sessions)
    # Guarantee at least one concept from each session
    anchor1 = random.choice(c1)
    anchor2 = random.choice(c2)
    pool = [c for c in (c1 + c2) if c not in (anchor1, anchor2)]
    extra_k = min(random.randint(0, 3), len(pool))
    selected = [anchor1, anchor2] + random.sample(pool, extra_k)
    return [s1.get("titulo", ""), s2.get("titulo", "")], selected


def _build_samples(
    discipline: str,
    subject: dict,
    sessions: List[dict],
    n_samples: int,
) -> List[dict]:
    if not sessions:
        return []
    syllabus = _syllabus_text(sessions)
    samples = []
    for i in range(n_samples):
        strategy = _strategy_a if i % 2 == 0 else _strategy_b
        result = strategy(sessions)
        if result is None:
            continue
        session_titles, concepts = result
        samples.append({
            "discipline": discipline,
            "nome_materia": subject.get("nome_materia", ""),
            "nivel": subject.get("nivel", "graduação"),
            "syllabus_text": syllabus,
            "aulas_estudadas": ", ".join(session_titles),
            "conceitos_selecionados": ", ".join(concepts),
        })
    return samples


def _disc_slug(disc: str) -> str:
    return disc.lower().replace(" ", "_").replace("/", "_")


async def generate_instructions(
    disciplines_subjects: Dict[str, List[dict]],
    syllabi: Dict[str, List[dict]],
    output_dir: Path,
    *,
    samples_per_subject: int = 5,
    batch_size: int = 32,
    max_concurrent: int = 32,
) -> int:
    """Generate Q&A pairs for all subjects. Saves per-discipline JSONL. Returns total pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    pending_discs: List[str] = []

    for disc in disciplines_subjects:
        out_file = output_dir / f"{_disc_slug(disc)}.jsonl"
        if out_file.exists():
            count = sum(1 for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip())
            print(f"  [instruções] {disc}: {count} pares carregados do disco")
            total += count
        else:
            pending_discs.append(disc)

    if not pending_discs:
        return total

    # ── Build all samples across pending disciplines ──────────────────────────
    all_samples: List[dict] = []
    sample_disc: List[str] = []      # which discipline each sample belongs to

    for disc in pending_discs:
        for subj in disciplines_subjects.get(disc, []):
            key = f"{disc}::{subj.get('nome_materia', '')}"
            sessions = syllabi.get(key, [])
            samples = _build_samples(disc, subj, sessions, samples_per_subject)
            all_samples.extend(samples)
            sample_disc.extend([disc] * len(samples))

    if not all_samples:
        print("  [instruções] Nenhuma amostra disponível — verifique se as ementas foram geradas.")
        return total

    print(f"\n[instruções] {len(all_samples)} pares planejados para {len(pending_discs)} disciplinas")

    # ── Step 1: generate questions ────────────────────────────────────────────
    question_prompts = [
        _QUESTION_PROMPT.format(
            discipline=s["discipline"],
            nome_materia=s["nome_materia"],
            nivel=s["nivel"],
            syllabus_text=s["syllabus_text"],
            aulas_estudadas=s["aulas_estudadas"],
            conceitos_selecionados=s["conceitos_selecionados"],
        )
        for s in all_samples
    ]
    print(f"\n[instruções] Gerando {len(question_prompts)} perguntas...")
    questions = await run_prompts(
        question_prompts,
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        label="instruções/pergunta",
    )

    # ── Step 2: generate answers ──────────────────────────────────────────────
    valid: List[Tuple[dict, str, str]] = [
        (s, disc, q)
        for s, disc, q in zip(all_samples, sample_disc, questions)
        if q
    ]
    answer_prompts = [
        _ANSWER_PROMPT.format(discipline=s["discipline"], question=q)
        for s, _, q in valid
    ]
    print(f"\n[instruções] Gerando {len(answer_prompts)} respostas...")
    answers = await run_prompts(
        answer_prompts,
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        label="instruções/resposta",
    )

    # ── Save per-discipline ───────────────────────────────────────────────────
    disc_records: Dict[str, List[dict]] = {d: [] for d in pending_discs}
    for (sample, disc, question), answer in zip(valid, answers):
        if not answer:
            continue
        disc_records[disc].append({
            "instruction": question,
            "output": answer,
            "discipline": sample["discipline"],
            "subject": sample["nome_materia"],
            "level": sample["nivel"],
        })

    new_count = 0
    for disc in pending_discs:
        records = disc_records[disc]
        out_file = output_dir / f"{_disc_slug(disc)}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [instruções] {disc}: {len(records)} pares salvos")
        new_count += len(records)

    total += new_count
    print(f"\n  [instruções] {new_count} novos pares. Total acumulado: {total}")
    return total
