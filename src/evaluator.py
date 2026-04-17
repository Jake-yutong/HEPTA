"""Evaluator module for the HEPTA benchmark.

Loads test questions and rubric definitions, dispatches scoring requests to
an LLM-based judge, and produces per-dimension scores that feed into the
N-gain pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.n_gain import DIMENSIONS, DIMENSION_LABELS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Question(BaseModel):
    """A single HCI test question."""

    id: str
    area: str                  # Breadth / Methods / Depth
    dimension: str             # OBJ, CE, TA, HCP, TE, CPI, MA
    text: str
    reference_answer: str = ""
    max_score: float = 100.0


class RubricItem(BaseModel):
    """Scoring rubric for one dimension."""

    dimension: str
    criteria: str
    max_score: float = 100.0


class ScoreRecord(BaseModel):
    """The score assigned to a single answer on one dimension."""

    question_id: str
    dimension: str
    score: float = Field(ge=0, le=100)
    rationale: str = ""


class PhaseResult(BaseModel):
    """All scores for one model in one phase (pre or post)."""

    model: str
    phase: str                          # "pre" or "post"
    scores: List[ScoreRecord] = Field(default_factory=list)

    def dimension_averages(self) -> Dict[str, float]:
        """Return the average score per dimension."""
        dim_scores: Dict[str, List[float]] = {d: [] for d in DIMENSIONS}
        for s in self.scores:
            dim_scores[s.dimension].append(s.score)
        return {
            d: (sum(v) / len(v) if v else 0.0) for d, v in dim_scores.items()
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> pd.DataFrame:
    """Read a .xlsx, .json, .jsonl, or .txt file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in (".txt", ".csv", ".tsv"):
        text = path.read_text(encoding="utf-8-sig")
        first_line = text.split("\n", 1)[0]
        for sep in ("\t", ",", "|"):
            if sep in first_line:
                return pd.read_csv(path, sep=sep, encoding="utf-8-sig")
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    raise ValueError(f"Unsupported file format: {suffix}")


def load_questions(path: Path) -> List[Question]:
    """Load test questions from an xlsx / json / jsonl file."""
    df = _read_file(path)
    questions: List[Question] = []
    for _, row in df.iterrows():
        questions.append(
            Question(
                id=str(row["id"]),
                area=str(row["area"]),
                dimension=str(row["dimension"]),
                text=str(row["text"]),
                reference_answer=str(row.get("reference_answer", "")),
                max_score=float(row.get("max_score", 100)),
            )
        )
    return questions


def load_rubric(path: Path) -> List[RubricItem]:
    """Load rubric items from an xlsx / json / jsonl file."""
    df = _read_file(path)
    items: List[RubricItem] = []
    for _, row in df.iterrows():
        items.append(
            RubricItem(
                dimension=str(row["dimension"]),
                criteria=str(row["criteria"]),
                max_score=float(row.get("max_score", 100)),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Score parsing helpers
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first flat JSON object from *text* (no nested braces)."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def extract_json_object_nested(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from *text*, supporting nested objects."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_score(raw: str) -> Optional[float]:
    """Parse a numeric score from an LLM response string."""
    obj = extract_json_object(raw)
    if obj and "score" in obj:
        try:
            return float(obj["score"])
        except (TypeError, ValueError):
            pass
    match = re.search(r"(\d+(?:\.\d+)?)\s*/?\s*100", raw)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class HEPTAEvaluator:
    """Coordinates the scoring of student answers against the rubric."""

    def __init__(
        self,
        questions: List[Question],
        rubric: List[RubricItem],
    ) -> None:
        self.questions = questions
        self.rubric = {r.dimension: r for r in rubric}

    def build_scoring_prompt(
        self,
        question: Question,
        answer: str,
    ) -> str:
        """Construct the prompt sent to the LLM judge."""
        rubric_item = self.rubric.get(question.dimension)
        criteria_text = rubric_item.criteria if rubric_item else "N/A"
        return (
            "You are an expert HCI examiner.  Score the following student "
            "answer on a 0–100 scale according to the rubric.\n\n"
            f"## Dimension\n{question.dimension} — "
            f"{DIMENSION_LABELS.get(question.dimension, '')}\n\n"
            f"## Question\n{question.text}\n\n"
            f"## Reference answer\n{question.reference_answer}\n\n"
            f"## Rubric criteria\n{criteria_text}\n\n"
            f"## Student answer\n{answer}\n\n"
            "Respond with a JSON object: "
            '{\"score\": <0-100>, \"rationale\": \"<brief justification>\"}'
        )

    def score_answer(
        self,
        question: Question,
        answer: str,
        llm_response: str,
    ) -> ScoreRecord:
        """Parse the LLM judge response into a ``ScoreRecord``."""
        parsed = extract_json_object(llm_response)
        score = 0.0
        rationale = ""
        if parsed:
            score = float(parsed.get("score", 0))
            rationale = str(parsed.get("rationale", ""))
        else:
            s = parse_score(llm_response)
            score = s if s is not None else 0.0
            rationale = llm_response

        score = max(0.0, min(100.0, score))
        return ScoreRecord(
            question_id=question.id,
            dimension=question.dimension,
            score=score,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_phase_result(result: PhaseResult, path: Path) -> None:
    """Persist a ``PhaseResult`` to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_phase_result(path: Path) -> PhaseResult:
    """Load a ``PhaseResult`` from a JSON file."""
    return PhaseResult.model_validate_json(path.read_text(encoding="utf-8"))
