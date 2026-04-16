"""HEPTA — AI HCI Education Performance Test Benchmark.

Command-line interface for running evaluations, computing N-gain scores,
and generating visualisations.

Usage examples::

    python main.py init
    python main.py run --model gpt-4o --phase pre
    python main.py run --model gpt-4o --phase post
    python main.py calc --model gpt-4o
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import click

from src.evaluator import (
    HEPTAEvaluator,
    PhaseResult,
    ScoreRecord,
    load_phase_result,
    load_questions,
    load_rubric,
    save_phase_result,
)
from src.n_gain import (
    DIMENSIONS,
    DimensionGain,
    HEPTACalculator,
    HEPTAResult,
)
from src.visualizer import plot_bar, plot_radar

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")

QUESTIONS_CANDIDATES = ["test_questions.xlsx", "test_questions.json", "test_questions.jsonl"]
RUBRIC_CANDIDATES = ["rubric.xlsx", "rubric.json", "rubric.jsonl"]


def _find_file(directory: Path, candidates: List[str]) -> Path | None:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """HEPTA — AI HCI Education Performance Test Benchmark."""


@cli.command()
def init() -> None:
    """Check that required data files exist and are readable."""
    ok = True

    q_path = _find_file(DATA_DIR, QUESTIONS_CANDIDATES)
    if q_path is None:
        click.secho(f"[ERROR] No test questions file found in {DATA_DIR}/", fg="red")
        ok = False
    else:
        try:
            qs = load_questions(q_path)
            click.echo(f"[OK] {q_path}  ({len(qs)} questions loaded)")
        except Exception as exc:
            click.secho(f"[ERROR] Failed to load {q_path}: {exc}", fg="red")
            ok = False

    r_path = _find_file(DATA_DIR, RUBRIC_CANDIDATES)
    if r_path is None:
        click.secho(f"[ERROR] No rubric file found in {DATA_DIR}/", fg="red")
        ok = False
    else:
        try:
            rb = load_rubric(r_path)
            click.echo(f"[OK] {r_path}  ({len(rb)} rubric items loaded)")
        except Exception as exc:
            click.secho(f"[ERROR] Failed to load {r_path}: {exc}", fg="red")
            ok = False

    kb_path = DATA_DIR / "knowledge_base.xlsx"
    if kb_path.exists():
        click.echo(f"[OK] {kb_path}")
    else:
        click.secho(f"[WARN] {kb_path} not found (optional for training data)", fg="yellow")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    click.echo(f"[OK] results/ and outputs/ directories ready")

    if ok:
        click.secho("\nInitialisation complete.", fg="green")
    else:
        click.secho("\nInitialisation finished with errors.", fg="red")
        sys.exit(1)


@cli.command()
@click.option("--model", required=True, help="Model identifier (e.g. gpt-4o, qwen-max).")
@click.option(
    "--phase",
    required=True,
    type=click.Choice(["pre", "post"]),
    help="Test phase: pre (baseline) or post (after teaching intervention).",
)
@click.option("--mock", is_flag=True, default=False, help="Use mock answers instead of calling an LLM.")
def run(model: str, phase: str, mock: bool) -> None:
    """Run a test phase: load questions, collect answers, score, and save."""
    q_path = _find_file(DATA_DIR, QUESTIONS_CANDIDATES)
    r_path = _find_file(DATA_DIR, RUBRIC_CANDIDATES)
    if q_path is None or r_path is None:
        click.secho("Run 'python main.py init' first to verify data files.", fg="red")
        sys.exit(1)

    questions = load_questions(q_path)
    rubric = load_rubric(r_path)
    evaluator = HEPTAEvaluator(questions=questions, rubric=rubric)

    scores: List[ScoreRecord] = []
    for q in questions:
        if mock:
            import random
            if phase == "pre":
                score_val = round(random.uniform(20, 50), 1)
            else:
                score_val = round(random.uniform(60, 90), 1)
            scores.append(
                ScoreRecord(
                    question_id=q.id,
                    dimension=q.dimension,
                    score=score_val,
                    rationale="[mock]",
                )
            )
            click.echo(f"  Q{q.id} [{q.dimension}] → {score_val:.1f} (mock)")
        else:
            click.secho(
                f"  Q{q.id} [{q.dimension}]: LLM integration not yet wired. "
                "Use --mock for local testing.",
                fg="yellow",
            )

    result = PhaseResult(model=model, phase=phase, scores=scores)
    out_path = RESULTS_DIR / f"{model}_{phase}.json"
    save_phase_result(result, out_path)
    click.echo(f"\nResults saved to {out_path}")


@cli.command()
@click.option("--model", required=True, help="Model identifier.")
def calc(model: str) -> None:
    """Compute N-gain from pre/post results and generate visualisations."""
    pre_path = RESULTS_DIR / f"{model}_pre.json"
    post_path = RESULTS_DIR / f"{model}_post.json"

    if not pre_path.exists() or not post_path.exists():
        click.secho(
            f"Pre/post result files not found for model '{model}'.\n"
            f"  Expected: {pre_path} and {post_path}\n"
            "  Run both phases first.",
            fg="red",
        )
        sys.exit(1)

    pre_result = load_phase_result(pre_path)
    post_result = load_phase_result(post_path)

    pre_avg = pre_result.dimension_averages()
    post_avg = post_result.dimension_averages()

    gains: List[DimensionGain] = []
    click.echo(f"\n{'Dimension':<8} {'Pre':>6} {'Post':>6} {'N-gain':>8}")
    click.echo("-" * 32)
    for dim in DIMENSIONS:
        g = DimensionGain(dimension=dim, pre=pre_avg[dim], post=post_avg[dim])
        gains.append(g)
        click.echo(f"{dim:<8} {g.pre:6.1f} {g.post:6.1f} {g.calculate():7.1f}%")

    hepta_result = HEPTAResult(model=model, gains=gains)
    idx = hepta_result.index
    click.echo(f"\nHEPTA-Index: {idx:.2f}")
    click.echo(f"Classification: {HEPTACalculator.classify(idx)}")

    # Save summary
    summary: Dict = {
        "model": model,
        "hepta_index": round(idx, 2),
        "classification": HEPTACalculator.classify(idx),
        "dimensions": {
            dim: {
                "pre": round(pre_avg[dim], 2),
                "post": round(post_avg[dim], 2),
                "n_gain_pct": round(g.calculate(), 2),
            }
            for dim, g in zip(DIMENSIONS, gains)
        },
    }
    summary_path = OUTPUTS_DIR / f"{model}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"Summary saved to {summary_path}")

    # Visualisations
    radar_path = OUTPUTS_DIR / f"{model}_radar.png"
    plot_radar(gains, title=f"HEPTA N-gain — {model}", save_path=radar_path)
    click.echo(f"Radar chart saved to {radar_path}")

    bar_path = OUTPUTS_DIR / f"{model}_bar.png"
    plot_bar(idx, hepta_result.dimension_gains, title=f"HEPTA-Index — {model}", save_path=bar_path)
    click.echo(f"Bar chart saved to {bar_path}")


if __name__ == "__main__":
    cli()
