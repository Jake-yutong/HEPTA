"""Visualisation helpers for HEPTA benchmark results.

Produces radar charts (seven-dimension N-gain profiles) and bar charts
(HEPTA-Index with per-dimension weighted contributions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.n_gain import DIMENSIONS, DIMENSION_LABELS, WEIGHTS, DimensionGain


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def plot_radar(
    gains: List[DimensionGain],
    title: str = "HEPTA N-gain Radar",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Draw a seven-axis radar chart of N-gain percentages (0–100 scale).

    Parameters
    ----------
    gains:
        One ``DimensionGain`` per dimension.
    title:
        Chart title.
    save_path:
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = [DIMENSION_LABELS.get(d, d) for d in DIMENSIONS]
    values = []
    for dim in DIMENSIONS:
        matched = [g for g in gains if g.dimension == dim]
        values.append(matched[0].calculate() if matched else 0.0)

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the polygon
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    ax.plot(angles, values, linewidth=2, linestyle="solid", label="N-gain %")
    ax.fill(angles, values, alpha=0.25)
    ax.set_title(title, y=1.08, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_bar(
    index: float,
    dimension_gains: Dict[str, float],
    title: str = "HEPTA-Index Breakdown",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Draw a stacked bar chart showing the weighted contribution of each
    dimension to the overall HEPTA-Index.

    Parameters
    ----------
    index:
        The composite HEPTA-Index value.
    dimension_gains:
        Mapping ``{dimension_code: n_gain_pct}``.
    title:
        Chart title.
    save_path:
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dims = DIMENSIONS
    contributions = [WEIGHTS[d] * dimension_gains.get(d, 0.0) for d in dims]
    labels = [DIMENSION_LABELS.get(d, d) for d in dims]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Stacked horizontal bar
    left = 0.0
    colors = plt.cm.Set2(np.linspace(0, 1, len(dims)))  # type: ignore[attr-defined]
    for label, val, color in zip(labels, contributions, colors):
        ax.barh("HEPTA-Index", val, left=left, color=color, label=label, edgecolor="white")
        left += val

    ax.set_xlim(0, max(index * 1.2, 1))
    ax.axvline(index, color="black", linestyle="--", linewidth=1.2, label=f"Index = {index:.1f}")
    ax.set_xlabel("Weighted N-gain contribution")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
