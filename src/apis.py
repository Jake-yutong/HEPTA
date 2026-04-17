"""Teacher, Student, and RubricScoring API wrappers for the HEPTA benchmark.

These three classes map directly onto the three agent roles in the evaluation
pipeline:

* :class:`TeacherAPI`        — generates HCI pedagogical guidance.
* :class:`StudentAPI`        — produces student answers (baseline or with guidance).
* :class:`RubricScoringAPI`  — scores answers against the rubric (judge role).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.evaluator import (
    ScoreRecord,
    Question,
    RubricItem,
    extract_json_object,
    extract_json_object_nested,
    parse_score,
)
from src.models import LLMClient
from src.n_gain import DIMENSION_LABELS, DIMENSIONS, WEIGHTS


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
    "Be precise, cite specific papers where relevant, and demonstrate depth of understanding.\n\n"
    "OUTPUT RULES (strict):\n"
    "- Do NOT restate or paraphrase the question.\n"
    "- For multiple-choice parts: output ONLY the letter (e.g. 'C'), no explanation unless the question explicitly asks for one.\n"
    "- For short-answer / essay parts: go straight to the answer. No preamble, no summary paragraph at the end.\n"
    "- Do NOT add introductory sentences like 'This is a great question' or 'Let me explain'.\n"
    "- Keep answers concise and information-dense. Every sentence must add new substance."
)

_STUDENT_INTERVENTION_SYSTEM = (
    "You are a graduate student in Computer Science preparing for the HCI Qualifying Examination. "
    "Your instructor has provided pedagogical guidance for this question. "
    "Using the guidance, provide a thorough and well-structured answer.\n\n"
    "OUTPUT RULES (strict):\n"
    "- Do NOT restate or paraphrase the question.\n"
    "- For multiple-choice parts: output ONLY the letter (e.g. 'C'), no explanation unless the question explicitly asks for one.\n"
    "- For short-answer / essay parts: go straight to the answer. No preamble, no summary paragraph at the end.\n"
    "- Do NOT add introductory sentences like 'This is a great question' or 'Let me explain'.\n"
    "- Keep answers concise and information-dense. Every sentence must add new substance."
)

_JUDGE_SYSTEM = (
    "You are an expert HCI examiner. Score the student answer on a 0–100 scale "
    "according to the rubric criteria provided. "
    'Respond ONLY with a JSON object: {"score": <0-100>, "rationale": "<brief justification>"}'
)

# Multi-dimension scoring: each question is assessed on all 7 dimensions.
# Weights: OBJ 10% + CE 15% + TA 15% + HCP 15% + TE 15% + CPI 15% + MA 15% = 100%
_JUDGE_SYSTEM_MULTI = (
    "You are an expert HCI examiner. Score the student answer on ALL SEVEN assessment dimensions. "
    "Each dimension is scored independently on a 0–100 scale. "
    "Weighted question total = OBJ×10% + CE×15% + TA×15% + HCP×15% + TE×15% + CPI×15% + MA×15%.\n"
    "Respond ONLY with a JSON object with exactly this structure (no extra keys):\n"
    '{"OBJ": {"score": <0-100>, "rationale": "..."},'
    ' "CE":  {"score": <0-100>, "rationale": "..."},'
    ' "TA":  {"score": <0-100>, "rationale": "..."},'
    ' "HCP": {"score": <0-100>, "rationale": "..."},'
    ' "TE":  {"score": <0-100>, "rationale": "..."},'
    ' "CPI": {"score": <0-100>, "rationale": "..."},'
    ' "MA":  {"score": <0-100>, "rationale": "..."}}'
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

    def answer_baseline(self, question: Question, pre_exam_constraints: Optional[str] = None) -> str:
        """Answer *question* without any teaching guidance (pre-test condition)."""
        system = _STUDENT_BASELINE_SYSTEM
        if pre_exam_constraints:
            system = f"[Pre-Exam Requirements]\n{pre_exam_constraints}\n\n" + system
        return self.client.generate(
            prompt=question.text,
            system=system,
        )

    def answer_intervention(self, question: Question, guidance: str,
                            pre_exam_constraints: Optional[str] = None) -> str:
        """Answer *question* with *guidance* injected (post-test condition)."""
        system = _STUDENT_INTERVENTION_SYSTEM
        if pre_exam_constraints:
            system = f"[Pre-Exam Requirements]\n{pre_exam_constraints}\n\n" + system
        prompt = (
            f"[Instructor Guidance]\n{guidance}\n\n"
            f"[Question]\n{question.text}"
        )
        return self.client.generate(prompt=prompt, system=system)


@dataclass
class RubricScoringAPI:
    """Scores a student answer against rubric criteria using an LLM judge.

    The rubric is NEVER exposed to the student or teacher agents — it is only
    used inside this class when building prompts for the judge LLM.
    """

    client: LLMClient

    def evaluate(
        self,
        question: Question,
        answer: str,
        rubric_item: RubricItem | None,
    ) -> ScoreRecord:
        """Score *answer* on a single dimension (legacy / single-dim path)."""
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

    def evaluate_all_dimensions(
        self,
        question: Question,
        answer: str,
        rubric: Dict[str, RubricItem],
    ) -> List[ScoreRecord]:
        """Score *answer* on ALL 7 HEPTA dimensions in a single judge call.

        Each question earns one score per dimension (0-100). The weighted sum
        OBJ×10%+CE×15%+TA×15%+HCP×15%+TE×15%+CPI×15%+MA×15% gives the
        100-point question total used for display and N-gain calculation.

        The rubric dict is only visible to this judge method; NEVER passed to
        StudentAPI or TeacherAPI.
        """
        # Build rubric block — judge eyes only
        rubric_parts: List[str] = []
        for dim in DIMENSIONS:
            item = rubric.get(dim)
            criteria = item.criteria if item else "General HCI expertise, accuracy, and depth."
            label = DIMENSION_LABELS.get(dim, dim)
            weight_pct = int(WEIGHTS[dim] * 100)
            rubric_parts.append(
                f"[{dim} — {label} ({weight_pct}%)]\n{criteria}"
            )
        rubric_block = "\n\n".join(rubric_parts)

        prompt = (
            f"Question:\n{question.text}\n\n"
            f"Reference answer:\n{question.reference_answer}\n\n"
            f"Per-dimension rubric criteria (judge reference only):\n{rubric_block}\n\n"
            f"Student answer:\n{answer}"
        )
        raw = self.client.generate(prompt=prompt, system=_JUDGE_SYSTEM_MULTI)

        # Parse multi-dim JSON (nested structure)
        parsed = extract_json_object_nested(raw)

        records: List[ScoreRecord] = []
        for dim in DIMENSIONS:
            if parsed and dim in parsed:
                dim_data = parsed[dim]
                if isinstance(dim_data, dict):
                    score = float(dim_data.get("score", 0))
                    rationale = str(dim_data.get("rationale", ""))
                else:
                    score = float(dim_data)
                    rationale = ""
            else:
                score = 0.0
                rationale = "[parse error — dimension missing from judge response]"
            records.append(ScoreRecord(
                question_id=question.id,
                dimension=dim,
                score=max(0.0, min(100.0, score)),
                rationale=rationale,
            ))
        return records


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class APIBundle:
    """Container grouping all three API clients."""

    teacher: TeacherAPI
    student: StudentAPI
    judge: RubricScoringAPI
