"""HEPTA — Streamlit Web Interface.

Launch with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

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
    DIMENSION_LABELS,
    WEIGHTS,
    DimensionGain,
    HEPTACalculator,
    HEPTAResult,
)
from src.visualizer import plot_bar, plot_radar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")

Q_FILES = ["test_questions.xlsx", "test_questions.json", "test_questions.jsonl"]
R_FILES = ["rubric.xlsx", "rubric.json", "rubric.jsonl"]


def _find(directory: Path, candidates: list[str]) -> Path | None:
    for n in candidates:
        p = directory / n
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HEPTA Benchmark",
    page_icon="📐",
    layout="wide",
)

# Minimal custom CSS — academic, clean
st.markdown(
    """
    <style>
    /* Reduce top padding */
    .block-container { padding-top: 1.5rem; }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("HEPTA")
st.sidebar.caption("AI HCI Education Performance Test Benchmark")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Run Evaluation", "Compute N-gain", "Results Explorer"],
    label_visibility="collapsed",
)


# ---------------------------------------------------------------------------
# Helper: list available models from results/
# ---------------------------------------------------------------------------

def _available_models() -> list[str]:
    """Scan results/ for models that have at least one phase file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models: set[str] = set()
    for f in RESULTS_DIR.glob("*.json"):
        name = f.stem
        for suffix in ("_pre", "_post"):
            if name.endswith(suffix):
                models.add(name[: -len(suffix)])
    return sorted(models)


def _model_has_both(model: str) -> bool:
    return (RESULTS_DIR / f"{model}_pre.json").exists() and (
        RESULTS_DIR / f"{model}_post.json"
    ).exists()


# ===================================================================
# PAGE: Overview
# ===================================================================

if page == "Overview":
    st.title("HEPTA — AI HCI Education Performance Test Benchmark")
    st.markdown(
        """
        HEPTA evaluates the pedagogical effectiveness of LLM-based HCI teaching
        agents using a **Normalized Gain** model across seven assessment
        dimensions derived from the Stanford HCI Qual structure.
        """
    )

    # --- Data status ---
    st.subheader("Data Status")
    col1, col2, col3 = st.columns(3)

    q_path = _find(DATA_DIR, Q_FILES)
    r_path = _find(DATA_DIR, R_FILES)
    kb_path = DATA_DIR / "knowledge_base.xlsx"

    with col1:
        if q_path and q_path.exists():
            qs = load_questions(q_path)
            st.metric("Test Questions", len(qs))
        else:
            st.metric("Test Questions", "—")
            st.error("File not found")

    with col2:
        if r_path and r_path.exists():
            rb = load_rubric(r_path)
            st.metric("Rubric Dimensions", len(rb))
        else:
            st.metric("Rubric Dimensions", "—")
            st.error("File not found")

    with col3:
        if kb_path.exists():
            kb = pd.read_excel(kb_path, engine="openpyxl")
            st.metric("Knowledge Base Entries", len(kb))
        else:
            st.metric("Knowledge Base", "—")
            st.warning("Optional file not found")

    # --- Dimension table ---
    st.subheader("Assessment Dimensions")
    dim_df = pd.DataFrame(
        [
            {"Code": d, "Dimension": DIMENSION_LABELS[d], "Weight": f"{WEIGHTS[d]:.0%}"}
            for d in DIMENSIONS
        ]
    )
    st.dataframe(dim_df, use_container_width=True, hide_index=True)

    # --- Questions preview ---
    if q_path and q_path.exists():
        st.subheader("Test Questions Preview")
        qs_preview = load_questions(q_path)
        preview_df = pd.DataFrame(
            [{"ID": q.id, "Area": q.area, "Dimension": q.dimension, "Text": q.text[:120] + ("…" if len(q.text) > 120 else "")} for q in qs_preview]
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: Run Evaluation
# ===================================================================

elif page == "Run Evaluation":
    st.title("Run Evaluation")
    st.markdown("Execute a pre-test or post-test phase for a given model. Use **Mock mode** for local testing without API calls.")

    q_path = _find(DATA_DIR, Q_FILES)
    r_path = _find(DATA_DIR, R_FILES)

    if not q_path or not r_path:
        st.error("Data files missing. Place `test_questions.xlsx` and `rubric.xlsx` in `data/`.")
        st.stop()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model_name = st.text_input("Model name", value="mock-agent", help="Identifier for the model being evaluated.")
    with col_b:
        phase = st.selectbox("Phase", ["pre", "post"])
    with col_c:
        mock_mode = st.checkbox("Mock mode", value=True, help="Generate random scores locally instead of calling an LLM.")

    if st.button("Run", type="primary"):
        questions = load_questions(q_path)
        rubric = load_rubric(r_path)
        evaluator = HEPTAEvaluator(questions=questions, rubric=rubric)

        scores: List[ScoreRecord] = []
        progress = st.progress(0, text="Evaluating…")
        for i, q in enumerate(questions):
            if mock_mode:
                rng = random.Random()
                if phase == "pre":
                    s = round(rng.uniform(20, 50), 1)
                else:
                    s = round(rng.uniform(60, 90), 1)
                scores.append(ScoreRecord(question_id=q.id, dimension=q.dimension, score=s, rationale="[mock]"))
            else:
                st.warning(f"LLM API integration pending — skipping Q{q.id}.")
            progress.progress((i + 1) / len(questions), text=f"Q{q.id} [{q.dimension}] done")

        result = PhaseResult(model=model_name, phase=phase, scores=scores)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{model_name}_{phase}.json"
        save_phase_result(result, out_path)

        st.success(f"Saved {len(scores)} scores → `{out_path}`")

        # Show scores table
        st.subheader("Score Summary")
        score_df = pd.DataFrame([{"Question": s.question_id, "Dimension": s.dimension, "Score": s.score} for s in scores])
        st.dataframe(score_df, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: Compute N-gain
# ===================================================================

elif page == "Compute N-gain":
    st.title("Compute N-gain & HEPTA-Index")
    st.markdown("Select a model that has completed both pre-test and post-test phases.")

    models = _available_models()
    complete = [m for m in models if _model_has_both(m)]

    if not complete:
        st.info("No model has both pre and post results yet. Run evaluations first.")
        st.stop()

    model = st.selectbox("Model", complete)

    if st.button("Calculate", type="primary"):
        pre_result = load_phase_result(RESULTS_DIR / f"{model}_pre.json")
        post_result = load_phase_result(RESULTS_DIR / f"{model}_post.json")

        pre_avg = pre_result.dimension_averages()
        post_avg = post_result.dimension_averages()

        gains: List[DimensionGain] = []
        rows = []
        for dim in DIMENSIONS:
            g = DimensionGain(dimension=dim, pre=pre_avg[dim], post=post_avg[dim])
            gains.append(g)
            rows.append({
                "Dimension": dim,
                "Label": DIMENSION_LABELS[dim],
                "Pre": round(pre_avg[dim], 1),
                "Post": round(post_avg[dim], 1),
                "N-gain (%)": round(g.calculate(), 1),
                "Classification": HEPTACalculator.classify(g.calculate()),
            })

        hepta = HEPTAResult(model=model, gains=gains)
        idx = hepta.index

        # --- Metrics row ---
        m1, m2, m3 = st.columns(3)
        m1.metric("HEPTA-Index", f"{idx:.2f}")
        m2.metric("Classification", HEPTACalculator.classify(idx))
        m3.metric("Dimensions Evaluated", len(gains))

        # --- Table ---
        st.subheader("Per-Dimension Results")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Charts side-by-side ---
        st.subheader("Visualisations")
        chart_l, chart_r = st.columns(2)

        with chart_l:
            fig_radar = plot_radar(gains, title=f"N-gain Radar — {model}")
            st.pyplot(fig_radar)
            import matplotlib.pyplot as plt
            plt.close(fig_radar)

        with chart_r:
            fig_bar = plot_bar(idx, hepta.dimension_gains, title=f"HEPTA-Index — {model}")
            st.pyplot(fig_bar)
            import matplotlib.pyplot as plt
            plt.close(fig_bar)

        # --- Save outputs ---
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model,
            "hepta_index": round(idx, 2),
            "classification": HEPTACalculator.classify(idx),
            "dimensions": {
                dim: {"pre": round(pre_avg[dim], 2), "post": round(post_avg[dim], 2), "n_gain_pct": round(g.calculate(), 2)}
                for dim, g in zip(DIMENSIONS, gains)
            },
        }
        summary_path = OUTPUTS_DIR / f"{model}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        radar_path = OUTPUTS_DIR / f"{model}_radar.png"
        plot_radar(gains, title=f"N-gain Radar — {model}", save_path=radar_path)
        bar_path = OUTPUTS_DIR / f"{model}_bar.png"
        plot_bar(idx, hepta.dimension_gains, title=f"HEPTA-Index — {model}", save_path=bar_path)

        st.success(f"Outputs saved to `{OUTPUTS_DIR}/`")


# ===================================================================
# PAGE: Results Explorer
# ===================================================================

elif page == "Results Explorer":
    st.title("Results Explorer")
    st.markdown("Browse and compare saved evaluation results.")

    models = _available_models()
    if not models:
        st.info("No results found. Run evaluations first.")
        st.stop()

    # --- Existing result files ---
    st.subheader("Available Result Files")
    files_data = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        files_data.append({"File": f.name, "Size (KB)": round(f.stat().st_size / 1024, 1)})
    st.dataframe(pd.DataFrame(files_data), use_container_width=True, hide_index=True)

    # --- Compare models ---
    complete = [m for m in models if _model_has_both(m)]
    if len(complete) >= 1:
        st.subheader("Model Comparison")
        selected = st.multiselect("Select models to compare", complete, default=complete[:3])

        if selected:
            comparison_rows = []
            all_gains: Dict[str, List[DimensionGain]] = {}
            for m in selected:
                pre = load_phase_result(RESULTS_DIR / f"{m}_pre.json")
                post = load_phase_result(RESULTS_DIR / f"{m}_post.json")
                pre_avg = pre.dimension_averages()
                post_avg = post.dimension_averages()
                gains = [DimensionGain(dimension=d, pre=pre_avg[d], post=post_avg[d]) for d in DIMENSIONS]
                all_gains[m] = gains
                idx = HEPTACalculator.index(gains)
                row: Dict = {"Model": m, "HEPTA-Index": round(idx, 2), "Class": HEPTACalculator.classify(idx)}
                for dim in DIMENSIONS:
                    g = [x for x in gains if x.dimension == dim][0]
                    row[dim] = round(g.calculate(), 1)
                comparison_rows.append(row)

            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

            # Overlay radar charts
            if len(selected) <= 5:
                st.subheader("Radar Overlay")
                import matplotlib.pyplot as plt
                import numpy as np

                labels = [DIMENSION_LABELS[d] for d in DIMENSIONS]
                num_vars = len(DIMENSIONS)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylim(0, 100)
                ax.set_yticks([20, 40, 60, 80, 100])
                ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

                colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))  # type: ignore[attr-defined]
                for m, color in zip(selected, colors):
                    vals = [g.calculate() for g in all_gains[m]]
                    vals += vals[:1]
                    ax.plot(angles, vals, linewidth=2, label=m, color=color)
                    ax.fill(angles, vals, alpha=0.08, color=color)

                ax.set_title("N-gain Comparison", y=1.08, fontsize=13, fontweight="bold")
                ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # --- Raw JSON viewer ---
    st.subheader("Raw Result Inspector")
    file_choice = st.selectbox("Select file", [f.name for f in sorted(RESULTS_DIR.glob("*.json"))])
    if file_choice:
        data = json.loads((RESULTS_DIR / file_choice).read_text(encoding="utf-8"))
        st.json(data)
