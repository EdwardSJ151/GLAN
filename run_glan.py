"""
GLAN — Geração de Dados Sintéticos de Instrução em Português do Brasil
Implementação de: "Synthetic Data (Almost) from Scratch: Generalized Instruction
Tuning for Language Models" (Li et al., 2024 — arXiv 2402.13064)

Pipeline:
  Etapa 2 — Geração de Matérias     (subjects.py)
  Etapa 3 — Geração de Ementas      (syllabus.py)
  Etapa 4 — Geração de Instruções   (instructions.py)

A Etapa 1 (taxonomia) é fornecida diretamente via categories.md.

Uso:
    python run_glan.py
    python run_glan.py --output-dir minha_saida --runs 5 --samples 10
    python run_glan.py --batch 64 --concurrency 64
"""
import argparse
import asyncio
import json
from pathlib import Path

from glan.instructions import generate_instructions
from glan.subjects import generate_subjects
from glan.syllabus import generate_syllabi


def load_disciplines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def merge_instructions(output_dir: Path, instructions_dir: Path) -> Path:
    """Merge all per-discipline instruction files into a single instructions.jsonl."""
    merged = output_dir / "instructions.jsonl"
    total = 0
    with merged.open("w", encoding="utf-8") as out:
        for f in sorted(instructions_dir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1
    return merged, total


async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    categories_file = Path(args.categories)

    disciplines = load_disciplines(categories_file)
    print(f"Disciplinas carregadas ({len(disciplines)}):")
    for d in disciplines:
        print(f"  • {d}")

    # ── Etapa 2: Geração de Matérias ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAPA 2: Geração de Matérias")
    print("=" * 60)
    disciplines_subjects = await generate_subjects(
        disciplines,
        output_dir / "subjects",
        runs_per_discipline=args.runs,
        batch_size=args.batch,
        max_concurrent=args.concurrency,
    )
    total_subjects = sum(len(v) for v in disciplines_subjects.values())
    print(f"\nTotal de matérias: {total_subjects}")

    # ── Etapa 3: Geração de Ementas ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAPA 3: Geração de Ementas")
    print("=" * 60)
    syllabi = await generate_syllabi(
        disciplines_subjects,
        output_dir / "syllabi",
        batch_size=args.batch,
        max_concurrent=args.concurrency,
    )
    total_sessions = sum(len(v) for v in syllabi.values())
    print(f"\nTotal de aulas (sessões): {total_sessions}")

    # ── Etapa 4: Geração de Instruções ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("ETAPA 4: Geração de Instruções (Perguntas e Respostas)")
    print("=" * 60)
    instructions_dir = output_dir / "instructions"
    total_pairs = await generate_instructions(
        disciplines_subjects,
        syllabi,
        instructions_dir,
        samples_per_subject=args.samples,
        batch_size=args.batch,
        max_concurrent=args.concurrency,
    )

    merged_file, merged_count = merge_instructions(output_dir, instructions_dir)

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline concluído!")
    print(f"  Disciplinas : {len(disciplines)}")
    print(f"  Matérias    : {total_subjects}")
    print(f"  Aulas       : {total_sessions}")
    print(f"  Pares Q&A   : {merged_count}")
    print(f"  Saída       : {merged_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline GLAN — geração de dados sintéticos de instrução em português"
    )
    parser.add_argument(
        "--categories", default="categories.md",
        help="Arquivo com categorias/disciplinas (uma por linha)"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Diretório de saída para todos os artefatos gerados"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Execuções por disciplina na geração de matérias (paper usa 10; default 3)"
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Pares Q&A por matéria (default 5)"
    )
    parser.add_argument(
        "--batch", type=int, default=32,
        help="Tamanho do lote de requisições (default 32)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=32,
        help="Máximo de requisições simultâneas ao vLLM (default 32)"
    )
    asyncio.run(main(parser.parse_args()))
