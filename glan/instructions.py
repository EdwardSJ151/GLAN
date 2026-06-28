"""Stage 4 — Instruction Generator.

Samples class sessions and key concepts from each syllabus, generates homework questions,
then generates answers separately (paper §2.4). Two sampling strategies are used:

  Strategy A — single session: pick 1 session, sample 1–5 key concepts.
  Strategy B — cross-session: pick 2 sessions, force at least 1 concept from each,
               total 2–5 concepts. Produces questions that integrate knowledge across topics.

Questions use the full syllabus as context (simulating what a student has studied up to
the selected sessions). Questions and answers are generated in separate LLM calls
(paper §3.1: GPT-4 for questions, GPT-3.5 for answers; here one model handles both).

Q&A pairs stream to ``output_dir/{discipline}.jsonl`` as they are completed.
Checkpoint state (samples, pending questions, skipped ids) lives in a sibling
``instructions_state/`` folder so runs can resume after interruptions without fixing
the random seed — samples are persisted on first use per discipline.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, records: List[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_done_ids(out_file: Path) -> Set[int]:
    done: Set[int] = set()
    for record in _read_jsonl(out_file):
        sample_id = record.get("sample_id")
        if sample_id is not None:
            done.add(int(sample_id))
    return done


def _question_prompt(sample: dict) -> str:
    return _QUESTION_PROMPT.format(
        discipline=sample["discipline"],
        nome_materia=sample["nome_materia"],
        nivel=sample["nivel"],
        syllabus_text=sample["syllabus_text"],
        aulas_estudadas=sample["aulas_estudadas"],
        conceitos_selecionados=sample["conceitos_selecionados"],
    )


def _qa_record(sample: dict, sample_id: int, question: str, answer: str) -> dict:
    return {
        "sample_id": sample_id,
        "instruction": question,
        "output": answer,
        "discipline": sample["discipline"],
        "subject": sample["nome_materia"],
        "level": sample["nivel"],
    }


def _is_discipline_complete(
    sample_ids: Set[int],
    done_ids: Set[int],
    skipped_ids: Set[int],
    pending_ids: Set[int],
) -> bool:
    return not pending_ids and sample_ids <= (done_ids | skipped_ids)


def _count_jsonl_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


async def _generate_answers(
    pending_items: List[Tuple[int, dict, str]],
    *,
    out_file: Path,
    batch_size: int,
    max_concurrent: int,
    label: str,
) -> Tuple[Set[int], Set[int]]:
    """Generate answers for pending (id, sample, question) items. Returns (completed, still_pending)."""
    if not pending_items:
        return set(), set()

    completed: Set[int] = set()
    still_pending: Set[int] = set()
    new_records: List[dict] = []

    for start in range(0, len(pending_items), batch_size):
        batch = pending_items[start : start + batch_size]
        answer_prompts = [
            _ANSWER_PROMPT.format(discipline=sample["discipline"], question=question)
            for _, sample, question in batch
        ]
        answers = await run_prompts(
            answer_prompts,
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            label=label,
        )
        for (sample_id, sample, question), answer in zip(batch, answers):
            if answer:
                new_records.append(_qa_record(sample, sample_id, question, answer))
                completed.add(sample_id)
            else:
                still_pending.add(sample_id)

        _append_jsonl(out_file, new_records)
        new_records.clear()

    return completed, still_pending


async def _process_discipline(
    disc: str,
    samples: List[dict],
    output_dir: Path,
    state_dir: Path,
    *,
    batch_size: int,
    max_concurrent: int,
) -> int:
    slug = _disc_slug(disc)
    disc_state = state_dir / slug
    disc_state.mkdir(parents=True, exist_ok=True)

    samples_file = disc_state / "samples.jsonl"
    pending_file = disc_state / "pending.jsonl"
    skipped_file = disc_state / "skipped.jsonl"
    out_file = output_dir / f"{slug}.jsonl"

    if samples_file.exists():
        stored = _read_jsonl(samples_file)
    else:
        stored = [{"id": i, **s} for i, s in enumerate(samples)]
        _write_jsonl(samples_file, stored)

    sample_by_id = {int(s["id"]): s for s in stored}
    sample_ids = set(sample_by_id)
    done_ids = _load_done_ids(out_file)
    skipped_ids = {int(s["id"]) for s in _read_jsonl(skipped_file)}
    pending_records = _read_jsonl(pending_file)
    pending_by_id = {int(p["id"]): p["question"] for p in pending_records}

    if _is_discipline_complete(sample_ids, done_ids, skipped_ids, set(pending_by_id)):
        count = _count_jsonl_lines(out_file)
        print(f"  [instruções] {disc}: {count} pares (completo)")
        return count

    starting_done = len(done_ids)
    if pending_by_id:
        print(f"  [instruções] {disc}: retomando {len(pending_by_id)} respostas pendentes...")
        pending_items = [
            (sid, sample_by_id[sid], question)
            for sid, question in pending_by_id.items()
            if sid in sample_by_id
        ]
        completed, still_pending = await _generate_answers(
            pending_items,
            out_file=out_file,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            label=f"instruções/resposta/{disc}",
        )
        done_ids |= completed
        pending_by_id = {sid: pending_by_id[sid] for sid in still_pending if sid in pending_by_id}
        _write_jsonl(
            pending_file,
            [{"id": sid, "question": pending_by_id[sid]} for sid in sorted(pending_by_id)],
        )

    remaining_ids = sorted(
        sid for sid in sample_ids
        if sid not in done_ids and sid not in skipped_ids and sid not in pending_by_id
    )
    if remaining_ids:
        print(f"  [instruções] {disc}: {len(remaining_ids)} amostras restantes")

    for start in range(0, len(remaining_ids), batch_size):
        batch_ids = remaining_ids[start : start + batch_size]
        batch_samples = [sample_by_id[sid] for sid in batch_ids]

        questions = await run_prompts(
            [_question_prompt(s) for s in batch_samples],
            temperature=1.0,
            top_p=0.95,
            max_tokens=1024,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            label=f"instruções/pergunta/{disc}",
        )

        batch_pending: List[Tuple[int, dict, str]] = []
        new_skipped: List[dict] = []
        for sample_id, sample, question in zip(batch_ids, batch_samples, questions):
            if question:
                pending_by_id[sample_id] = question
                batch_pending.append((sample_id, sample, question))
            else:
                skipped_ids.add(sample_id)
                new_skipped.append({"id": sample_id, "reason": "pergunta_vazia"})

        if new_skipped:
            _append_jsonl(skipped_file, new_skipped)

        _write_jsonl(
            pending_file,
            [{"id": sid, "question": pending_by_id[sid]} for sid in sorted(pending_by_id)],
        )

        if batch_pending:
            completed, still_pending = await _generate_answers(
                batch_pending,
                out_file=out_file,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                label=f"instruções/resposta/{disc}",
            )
            done_ids |= completed
            for sid in completed:
                pending_by_id.pop(sid, None)
            _write_jsonl(
                pending_file,
                [{"id": sid, "question": pending_by_id[sid]} for sid in sorted(pending_by_id)],
            )

    count = _count_jsonl_lines(out_file)
    new_pairs = count - starting_done
    print(f"  [instruções] {disc}: +{new_pairs} pares → {count} total")
    return count


async def generate_instructions(
    disciplines_subjects: Dict[str, List[dict]],
    syllabi: Dict[str, List[dict]],
    output_dir: Path,
    *,
    state_dir: Optional[Path] = None,
    samples_per_subject: int = 5,
    batch_size: int = 32,
    max_concurrent: int = 32,
) -> int:
    """Generate Q&A pairs for all subjects. Streams to per-discipline JSONL. Returns total pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if state_dir is None:
        state_dir = output_dir.parent / "instructions_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    to_process: List[str] = []

    for disc in disciplines_subjects:
        slug = _disc_slug(disc)
        out_file = output_dir / f"{slug}.jsonl"
        samples_file = state_dir / slug / "samples.jsonl"

        if samples_file.exists():
            to_process.append(disc)
        elif out_file.exists():
            count = _count_jsonl_lines(out_file)
            print(f"  [instruções] {disc}: {count} pares carregados do disco")
            total += count
        else:
            to_process.append(disc)

    if not to_process:
        return total

    print(f"\n[instruções] Processando {len(to_process)} disciplina(s) | estado: {state_dir}")

    for disc in to_process:
        disc_samples: List[dict] = []
        for subj in disciplines_subjects.get(disc, []):
            key = f"{disc}::{subj.get('nome_materia', '')}"
            sessions = syllabi.get(key, [])
            disc_samples.extend(_build_samples(disc, subj, sessions, samples_per_subject))

        if not disc_samples and not (state_dir / _disc_slug(disc) / "samples.jsonl").exists():
            print(f"  [instruções] {disc}: nenhuma amostra — verifique as ementas")
            continue

        count = await _process_discipline(
            disc,
            disc_samples,
            output_dir,
            state_dir,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )
        total += count

    print(f"\n  [instruções] Total acumulado: {total}")
    return total
