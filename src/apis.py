"""Teacher, Student, and RubricScoring API wrappers for the HEPTA benchmark.

These three classes map directly onto the three agent roles in the evaluation
pipeline:

* :class:`TeacherAPI`        — generates HCI pedagogical guidance.
* :class:`StudentAPI`        — produces student answers (baseline or with guidance).
* :class:`RubricScoringAPI`  — scores answers against the rubric (judge role).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluator import (
    ScoreRecord,
    Question,
    RubricItem,
    extract_json_object,
    parse_score,
)
from src.models import LLMClient
from src.n_gain import DIMENSION_LABELS


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_TEACHER_SYSTEM = (
    "You are an expert HCI educator with comprehensive knowledge of canonical HCI research "
    "literature (Vannevar Bush 1945 to present). Your role is to generate concise, "
    "pedagogically effective guidance to help a student better answer an HCI exam question. "
    "Structure your response as:\n"
    "1. Core concept clarification (1–2 sentences)\n"
    "2. Key papers and arguments to draw upon\n"
    "3. Suggested response structure\n"
    "Keep your guidance under 300 words. Do not write the full answer."
)

_STUDENT_BASELINE_SYSTEM = (
    "You are a graduate student in Computer Science preparing for the HCI Qualifying Examination. "
    "Answer the question based on your knowledge of HCI research. "
    "Be precise, cite specific papers where relevant, and demonstrate depth of understanding."
)

_STUDENT_INTERVENTION_SYSTEM = (
    "You are a graduate student in Computer Science preparing for the HCI Qualifying Examination. "
    "Your instructor has provided pedagogical guidance for this question. "
    "Using the guidance, provide a thorough and well-structured answer."
)

_JUDGE_SYSTEM = (
    "You are an expert HCI examiner. Score the student answer on a 0–100 scale "
    "according to the rubric criteria provided. "
    'Respond ONLY with a JSON object: {"score": <0-100>, "rationale": "<brief justification>"}'
)


# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------

@dataclass
class TeacherAPI:
    """Generates HCI pedagogical guidance for a given exam question."""

    client: LLMClient

    def teach(self, question: Question) -> str:
        """Return teaching guidance tailored to *question*."""
        dimension_label = DIMENSION_LABELS.get(question.dimension, question.dimension)
        prompt = (
            f"Dimension: {question.dimension} — {dimension_label}\n\n"
            f"Question:\n{question.text}"
        )
        return self.client.generate(prompt=prompt, system=_TEACHER_SYSTEM)


@dataclass
class StudentAPI:
    """Generates student answers in baseline and intervention conditions."""

    client: LLMClient

    def answer_baseline(self, question: Question) -> str:
        """Answer *question* without any teaching guidance (pre-test condition)."""
        return self.client.generate(
            prompt=question.text,
            system=_STUDENT_BASELINE_SYSTEM,
        )

    def answer_intervention(self, question: Question, guidance: str) -> str:
        """Answer *question* with *guidance* injected (post-test condition)."""
        prompt = (
            f"[Instructor Guidance]\n{guidance}\n\n"
            f"[Question]\n{question.text}"
        )
        return self.client.generate(prompt=prompt, system=_STUDENT_INTERVENTION_SYSTEM)


@dataclass
class RubricScoringAPI:
    """Scores a student answer against rubric criteria using an LLM judge."""

    client: LLMClient

    def evaluate(
        self,
        question: Question,
        answer: str,
        rubric_item: RubricItem | None,
    ) -> ScoreRecord:
        """Score *answer* and return a :class:`~src.evaluator.ScoreRecord`."""
        criteria = (
            rubric_item.criteria
            if rubric_item
            else "General HCI expertise, accuracy, and depth."
        )
        dimension_label = DIMENSION_LABELS.get(question.dimension, question.dimension)
        prompt = (
            f"Dimension: {question.dimension} — {dimension_label}\n\n"
            f"Question:\n{question.text}\n\n"
            f"Reference answer:\n{question.reference_answer}\n\n"
            f"Rubric criteria:\n{criteria}\n\n"
            f"Student answer:\n{answer}"
        )
        raw = self.client.generate(prompt=prompt, system=_JUDGE_SYSTEM)

        parsed = extract_json_object(raw)
        if parsed:
            score = float(parsed.get("score", 0))
            rationale = str(parsed.get("rationale", ""))
        else:
            s = parse_score(raw)
            score = s if s is not None else 0.0
            rationale = raw

        score = max(0.0, min(100.0, score))
        return ScoreRecord(
            question_id=question.id,
            dimension=question.dimension,
            score=score,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class APIBundle:
    """Container grouping all three API clients."""

    teacher: TeacherAPI
    student: StudentAPI
    judge: RubricScoringAPI
