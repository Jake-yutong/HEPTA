"""N-gain calculation module for the HEPTA benchmark.

Implements the Normalized Gain model used to quantify pedagogical
effectiveness across seven HCI assessment dimensions.

Reference:
    Hake, R. R. (1998). Interactive-engagement versus traditional methods:
    A six-thousand-student survey of mechanics test data for introductory
    physics courses. *American Journal of Physics*, 66(1), 64-74.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

DIMENSIONS: List[str] = ["OBJ", "CE", "TA", "HCP", "TE", "CPI", "MA"]

DIMENSION_LABELS: Dict[str, str] = {
    "OBJ": "Objective Questions",
    "CE": "Conceptual Explanation",
    "TA": "Tradeoff Analysis",
    "HCP": "Historical Context & Persistence",
    "TE": "Technical Elucidation",
    "CPI": "Cross-Paper Integration",
    "MA": "Methodological Application",
}

WEIGHTS: Dict[str, float] = {
    "OBJ": 0.10,
    "CE": 0.15,
    "TA": 0.15,
    "HCP": 0.15,
    "TE": 0.15,
    "CPI": 0.15,
    "MA": 0.15,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionGain:
    """Stores pre/post scores for a single dimension and computes N-gain."""

    dimension: str
    pre: float   # 0–100
    post: float  # 0–100

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"Unknown dimension: {self.dimension}")
        if not (0 <= self.pre <= 100):
            raise ValueError(f"pre must be in [0, 100], got {self.pre}")
        if not (0 <= self.post <= 100):
            raise ValueError(f"post must be in [0, 100], got {self.post}")

    def calculate(self) -> float:
        """Return the normalised gain as a percentage in [-100, 100].

        Formula:
            g = (post - pre) / (100 - pre)

        When ``pre >= 100`` (ceiling), returns 0.0 by convention.
        """
        if self.pre >= 100:
            return 0.0
        g = (self.post - self.pre) / (100 - self.pre)
        return max(-1.0, min(1.0, g)) * 100


@dataclass
class HEPTAResult:
    """Aggregated HEPTA evaluation result for one model."""

    model: str
    gains: List[DimensionGain] = field(default_factory=list)

    @property
    def dimension_gains(self) -> Dict[str, float]:
        return {g.dimension: g.calculate() for g in self.gains}

    @property
    def index(self) -> float:
        return HEPTACalculator.index(self.gains)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class HEPTACalculator:
    """Computes the weighted HEPTA-Index from per-dimension N-gain values."""

    @staticmethod
    def index(gains: List[DimensionGain]) -> float:
        """Return the HEPTA-Index (weighted sum of seven N-gain percentages)."""
        return sum(WEIGHTS[g.dimension] * g.calculate() for g in gains)

    @staticmethod
    def classify(n_gain_pct: float) -> str:
        """Classify an N-gain percentage following Hake's thresholds.

        * High gain:   g >= 70 %
        * Medium gain:  30 % <= g < 70 %
        * Low gain:    g < 30 %
        """
        if n_gain_pct >= 70:
            return "High"
        if n_gain_pct >= 30:
            return "Medium"
        return "Low"
