"""Mock data generation and local tests for the HEPTA benchmark.

This module does NOT call any external API.  It generates synthetic pre/post
scores, verifies N-gain arithmetic, HEPTA-Index weighting, and radar-chart
rendering.

Run with::

    python -m pytest tests/mock_data.py -v
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from src.n_gain import (
    DIMENSIONS,
    WEIGHTS,
    DimensionGain,
    HEPTACalculator,
    HEPTAResult,
)
from src.visualizer import plot_bar, plot_radar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_mock_scores(
    seed: int = 42,
    pre_range: tuple[float, float] = (20.0, 50.0),
    post_range: tuple[float, float] = (60.0, 90.0),
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Return ``(pre_scores, post_scores)`` dictionaries keyed by dimension."""
    rng = random.Random(seed)
    pre: Dict[str, float] = {}
    post: Dict[str, float] = {}
    for dim in DIMENSIONS:
        pre[dim] = round(rng.uniform(*pre_range), 1)
        post[dim] = round(rng.uniform(*post_range), 1)
    return pre, post


def make_gains(
    pre: Dict[str, float],
    post: Dict[str, float],
) -> List[DimensionGain]:
    return [DimensionGain(dimension=d, pre=pre[d], post=post[d]) for d in DIMENSIONS]


# ---------------------------------------------------------------------------
# Tests — N-gain calculation
# ---------------------------------------------------------------------------

class TestNGainCalculation:
    """Verify N-gain formula correctness."""

    def test_known_value(self) -> None:
        """pre=30, post=60 → g = (60-30)/(100-30) = 30/70 ≈ 42.86 %."""
        g = DimensionGain(dimension="CE", pre=30.0, post=60.0)
        expected = (60 - 30) / (100 - 30) * 100  # ≈ 42.857
        assert math.isclose(g.calculate(), expected, rel_tol=1e-4)

    def test_ceiling_pre(self) -> None:
        """If pre == 100, N-gain should be 0 by convention."""
        g = DimensionGain(dimension="OBJ", pre=100.0, post=100.0)
        assert g.calculate() == 0.0

    def test_no_improvement(self) -> None:
        """Same pre and post → N-gain == 0."""
        g = DimensionGain(dimension="TA", pre=50.0, post=50.0)
        assert g.calculate() == 0.0

    def test_perfect_improvement(self) -> None:
        """Pre=50, post=100 → N-gain = 100 %."""
        g = DimensionGain(dimension="HCP", pre=50.0, post=100.0)
        assert g.calculate() == 100.0

    def test_negative_gain(self) -> None:
        """Post < pre → negative N-gain, clamped to -100 %."""
        g = DimensionGain(dimension="TE", pre=80.0, post=60.0)
        # (60-80)/(100-80) = -20/20 = -1.0 → -100 %
        assert g.calculate() == -100.0

    def test_clamp_upper(self) -> None:
        """N-gain should not exceed 100 %."""
        g = DimensionGain(dimension="CPI", pre=10.0, post=100.0)
        assert g.calculate() == 100.0


# ---------------------------------------------------------------------------
# Tests — HEPTA-Index
# ---------------------------------------------------------------------------

class TestHEPTAIndex:
    """Verify weighted HEPTA-Index computation."""

    def test_weights_sum_to_one(self) -> None:
        total = sum(WEIGHTS.values())
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_uniform_gain(self) -> None:
        """All dimensions with 50 % N-gain → index = 50."""
        gains = [DimensionGain(dimension=d, pre=0.0, post=50.0) for d in DIMENSIONS]
        # Each N-gain = (50 - 0) / (100 - 0) * 100 = 50
        idx = HEPTACalculator.index(gains)
        assert math.isclose(idx, 50.0, rel_tol=1e-4)

    def test_mock_data_index(self) -> None:
        """Compute index on mock data and ensure it is in valid range."""
        pre, post = generate_mock_scores()
        gains = make_gains(pre, post)
        idx = HEPTACalculator.index(gains)
        assert -100.0 <= idx <= 100.0

    def test_hepta_result_consistency(self) -> None:
        """HEPTAResult.index matches HEPTACalculator.index."""
        pre, post = generate_mock_scores(seed=99)
        gains = make_gains(pre, post)
        result = HEPTAResult(model="test-model", gains=gains)
        assert math.isclose(result.index, HEPTACalculator.index(gains))

    def test_classification(self) -> None:
        assert HEPTACalculator.classify(75) == "High"
        assert HEPTACalculator.classify(50) == "Medium"
        assert HEPTACalculator.classify(10) == "Low"


# ---------------------------------------------------------------------------
# Tests — Visualisation (smoke tests)
# ---------------------------------------------------------------------------

class TestVisualisation:
    """Ensure chart functions run without error and produce files."""

    def test_radar_chart(self) -> None:
        pre, post = generate_mock_scores()
        gains = make_gains(pre, post)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "radar.png"
            fig = plot_radar(gains, title="Test Radar", save_path=path)
            assert path.exists()
            assert path.stat().st_size > 0
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_bar_chart(self) -> None:
        pre, post = generate_mock_scores()
        gains = make_gains(pre, post)
        result = HEPTAResult(model="mock", gains=gains)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bar.png"
            fig = plot_bar(
                result.index,
                result.dimension_gains,
                title="Test Bar",
                save_path=path,
            )
            assert path.exists()
            assert path.stat().st_size > 0
            import matplotlib.pyplot as plt
            plt.close(fig)
