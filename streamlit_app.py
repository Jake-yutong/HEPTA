"""HEPTA — Streamlit Web Interface.

Launch with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.apis import APIBundle, RubricScoringAPI, StudentAPI, TeacherAPI
from src.evaluator import (
    PhaseResult,
    Question,
    RubricItem,
    ScoreRecord,
    load_phase_result,
    load_questions,
    load_rubric,
    save_phase_result,
)
from src.i18n import t
from src.models import (
    LLMClientFactory,
    ModelConfig,
    PROVIDER_BASE_URLS,
    PROVIDER_SUGGESTED_MODELS,
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
# Paths
# ---------------------------------------------------------------------------

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")

Q_CANDIDATES = ["test_questions.xlsx", "test_questions.json", "test_questions.jsonl", "test_questions.txt"]
R_CANDIDATES = ["rubric.xlsx", "rubric.json", "rubric.jsonl", "rubric.txt"]
PROVIDERS    = list(PROVIDER_BASE_URLS.keys())

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HEPTA Benchmark", page_icon="📐", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background:#f8f9fa; border-radius:8px;
        padding:12px 16px; border:1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "page" not in st.session_state:
        st.session_state["page"] = "overview"
    for role, defaults in [
        ("teacher", {"temperature": 0.8}),
        ("student", {"temperature": 0.7}),
        ("judge",   {"temperature": 0.0}),
    ]:
        if f"{role}_config" not in st.session_state:
            st.session_state[f"{role}_config"] = ModelConfig(**defaults)
    # Cached uploaded data (survives page switches)
    for key in ("cached_questions", "cached_rubric", "cached_q_name", "cached_r_name"):
        if key not in st.session_state:
            st.session_state[key] = None

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

lang: str = st.session_state["lang"]

st.sidebar.title(t("app_title", lang))
st.sidebar.caption(t("app_caption", lang))

lang_choice = st.sidebar.selectbox(
    t("language", lang),
    options=["English", "中文"],
    index=0 if lang == "en" else 1,
    key="lang_select",
)
new_lang = "en" if lang_choice == "English" else "zh"
if new_lang != lang:
    st.session_state["lang"] = new_lang
    st.rerun()
lang = st.session_state["lang"]

NAV_KEYS   = ["overview", "api_config", "run", "calc", "explorer"]
nav_labels = [t(f"nav_{k}", lang) for k in NAV_KEYS]

current_idx    = NAV_KEYS.index(st.session_state.get("page", "overview"))
selected_label = st.sidebar.radio("nav", nav_labels, index=current_idx, label_visibility="collapsed")
page = NAV_KEYS[nav_labels.index(selected_label)]
st.session_state["page"] = page

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{t('api_status', lang)}**")
for _role in ("teacher", "student", "judge"):
    _cfg: ModelConfig = st.session_state[f"{_role}_config"]
    _label = t(f"{_role}_api", lang)
    icon = "🟢" if _cfg.api_key else "🔴"
    st.sidebar.markdown(f"{icon} {_label}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find(directory: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


def _read_upload(uploaded) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame, auto-detecting separator for txt/csv."""
    suffix = Path(uploaded.name).suffix.lower()
    raw = uploaded.read()
    data = BytesIO(raw)
    if suffix == ".xlsx":
        return pd.read_excel(data, engine="openpyxl")
    if suffix == ".json":
        return pd.read_json(data)
    if suffix == ".jsonl":
        return pd.read_json(data, lines=True)
    if suffix in (".txt", ".csv", ".tsv"):
        text = raw.decode("utf-8-sig")
        first_line = text.split("\n", 1)[0]
        for sep in ("\t", ",", "|"):
            if sep in first_line:
                return pd.read_csv(BytesIO(raw), sep=sep, encoding="utf-8-sig")
        # Last resort: comma fallback (pandas guess may fail on free-form text)
        raise ValueError("NOT_TABULAR")
    raise ValueError(f"Unsupported format: {suffix}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase & strip column names to tolerate minor formatting differences."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# HEPTA structured-text parsers
# ---------------------------------------------------------------------------

# Standard HEPTA benchmark mappings
_SA_DIM = {1: "CE", 2: "TE", 3: "TA", 4: "CE", 5: "HCP", 6: "CE", 7: "CE", 8: "TE", 9: "CPI"}
_MCQ_ANS = {1: "C", 2: "B", 3: "C", 4: "B", 5: "B", 6: "C", 7: "D", 8: "B", 9: "C"}
_Q_AREA = {1: "Breadth", 2: "Breadth", 3: "Breadth",
           4: "Methods", 5: "Methods", 6: "Methods",
           7: "Depth",   8: "Depth",   9: "Depth"}


def _is_hepta_questions_txt(text: str) -> bool:
    return bool(re.search(r'QUESTION\s+\d+', text) and
                ('\u3010\u9009\u62e9\u9898\u3011' in text or '\u3010\u7b80\u7b54\u9898\u3011' in text))


def _is_hepta_rubric_txt(text: str) -> bool:
    return bool(re.search(r'RUBRIC|SCORING', text, re.I) and
                ('\u3010\u8bc4\u5206\u7ef4\u5ea6' in text or '\u3010\u8bc4\u5206\u7ec6\u5219\u3011' in text
                 or re.search(r'SECTION\s+III', text)))


def _parse_hepta_questions_txt(text: str) -> List[Question]:
    """Parse the HEPTA-Bench structured questions txt file."""
    questions: List[Question] = []
    q_iter = list(re.finditer(r'QUESTION\s+(\d+)\s*:', text))
    for idx, m in enumerate(q_iter):
        qnum = int(m.group(1))
        start = m.end()
        end = q_iter[idx + 1].start() if idx + 1 < len(q_iter) else len(text)
        block = text[start:end]
        area = _Q_AREA.get(qnum, "General")

        # Extract MCQ
        mcq_m = re.search(r'\u3010\u9009\u62e9\u9898\u3011(.*?)(?=\u3010\u7b80\u7b54\u9898\u3011|$)', block, re.DOTALL)
        if mcq_m:
            mcq_text = mcq_m.group(1).strip()
            questions.append(Question(
                id=f"{qnum}_mc", area=area, dimension="OBJ",
                text=mcq_text,
                reference_answer=_MCQ_ANS.get(qnum, ""),
                max_score=100.0,
            ))

        # Extract short answer
        sa_m = re.search(r'\u3010\u7b80\u7b54\u9898\u3011(.*?)(?=-{20,}|={20,}|QUESTION\s+\d+|$)', block, re.DOTALL)
        if sa_m:
            sa_text = sa_m.group(1).strip()
            questions.append(Question(
                id=f"{qnum}_sa", area=area,
                dimension=_SA_DIM.get(qnum, "CE"),
                text=sa_text,
                reference_answer="",
                max_score=100.0,
            ))
    return questions


def _parse_hepta_rubric_txt(text: str) -> List[RubricItem]:
    """Parse the HEPTA-Bench structured rubric txt file."""
    items: List[RubricItem] = []

    # Section III general dimension rubrics: 【N. NAME (CODE) - WEIGHT%】
    dim_iter = list(re.finditer(
        r'\u3010(\d+)\.\s+([^(]+?)\s*\((\w+)\)\s*-\s*(\d+)%\u3011', text))
    for idx, m in enumerate(dim_iter):
        dim_code = m.group(3)
        start = m.end()
        end = dim_iter[idx + 1].start() if idx + 1 < len(dim_iter) else len(text)
        criteria = text[start:end].strip()
        # Trim trailing section separators
        criteria = re.split(r'={40,}', criteria)[0].strip()
        criteria = re.split(r'-{40,}\s*$', criteria)[0].strip()
        items.append(RubricItem(
            dimension=dim_code,
            criteria=criteria,
            max_score=100.0,
        ))

    # Fallback: per-question rubrics from Section II if Section III missing
    if not items:
        pq_iter = list(re.finditer(
            r'\u3010\u8003\u67e5\u7ef4\u5ea6\u3011\s*(\w+)', text))
        for m in pq_iter:
            dim = m.group(1)
            start = m.end()
            sec_end = text.find('\u3010\u8003\u67e5\u7ef4\u5ea6\u3011', start)
            if sec_end == -1:
                sec_end = len(text)
            criteria = text[start:sec_end].strip()[:2000]
            items.append(RubricItem(dimension=dim, criteria=criteria, max_score=100.0))

    return items


def _load_questions_upload(uploaded) -> List[Question]:
    suffix = Path(uploaded.name).suffix.lower()
    # For txt files: try HEPTA structured format first
    if suffix == ".txt":
        raw = uploaded.read()
        text = raw.decode("utf-8-sig")
        if _is_hepta_questions_txt(text):
            qs = _parse_hepta_questions_txt(text)
            if qs:
                return qs
        # Not HEPTA-format, try tabular
        uploaded.seek(0)

    try:
        df = _normalise_columns(_read_upload(uploaded))
    except ValueError:
        raise ValueError(
            "Cannot parse questions file. Expected either:\n"
            "• A tabular file (xlsx/json/jsonl/csv/tsv) with columns: id, area, dimension, text\n"
            "• A HEPTA-Bench structured txt file with QUESTION markers and 【选择题】/【简答题】 tags"
        )
    required = {"id", "area", "dimension", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Questions file missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return [
        Question(
            id=str(row["id"]), area=str(row["area"]), dimension=str(row["dimension"]),
            text=str(row["text"]), reference_answer=str(row.get("reference_answer", "")),
            max_score=float(row.get("max_score", 100)),
        )
        for _, row in df.iterrows()
    ]


def _load_rubric_upload(uploaded) -> List[RubricItem]:
    suffix = Path(uploaded.name).suffix.lower()
    # For txt files: try HEPTA structured format first
    if suffix == ".txt":
        raw = uploaded.read()
        text = raw.decode("utf-8-sig")
        if _is_hepta_rubric_txt(text):
            items = _parse_hepta_rubric_txt(text)
            if items:
                return items
        uploaded.seek(0)

    try:
        df = _normalise_columns(_read_upload(uploaded))
    except ValueError:
        raise ValueError(
            "Cannot parse rubric file. Expected either:\n"
            "• A tabular file (xlsx/json/jsonl/csv/tsv) with columns: dimension, criteria\n"
            "• A HEPTA-Bench structured rubric txt file with dimension rubric blocks"
        )
    required = {"dimension", "criteria"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Rubric file missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return [
        RubricItem(
            dimension=str(row["dimension"]), criteria=str(row["criteria"]),
            max_score=float(row.get("max_score", 100)),
        )
        for _, row in df.iterrows()
    ]


def _available_models() -> list[str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models: set[str] = set()
    for f in RESULTS_DIR.glob("*.json"):
        for sfx in ("_pre", "_post"):
            if f.stem.endswith(sfx):
                models.add(f.stem[: -len(sfx)])
    return sorted(models)


def _model_has_both(model: str) -> bool:
    return (RESULTS_DIR / f"{model}_pre.json").exists() and \
           (RESULTS_DIR / f"{model}_post.json").exists()


def _build_bundle() -> Optional[APIBundle]:
    t_cfg: ModelConfig = st.session_state["teacher_config"]
    s_cfg: ModelConfig = st.session_state["student_config"]
    j_cfg: ModelConfig = st.session_state["judge_config"]
    if not (t_cfg.api_key and s_cfg.api_key and j_cfg.api_key):
        return None
    return APIBundle(
        teacher=TeacherAPI(client=LLMClientFactory.create(t_cfg)),
        student=StudentAPI(client=LLMClientFactory.create(s_cfg)),
        judge=RubricScoringAPI(client=LLMClientFactory.create(j_cfg)),
    )


# ---------------------------------------------------------------------------
# Reusable API config panel
# ---------------------------------------------------------------------------

def _api_panel(role: str) -> None:
    cfg: ModelConfig = st.session_state[f"{role}_config"]

    c1, c2 = st.columns([1, 2])
    with c1:
        provider = st.selectbox(
            t("provider", lang),
            options=PROVIDERS,
            index=PROVIDERS.index(cfg.provider) if cfg.provider in PROVIDERS else 0,
            key=f"{role}_provider",
        )
    with c2:
        suggestions = PROVIDER_SUGGESTED_MODELS.get(provider, [])
        hint = f"{t('suggested_models', lang)}: {', '.join(suggestions)}" if suggestions else ""
        model_name = st.text_input(t("model_name", lang), value=cfg.model_name, help=hint, key=f"{role}_model_name")

    api_key = st.text_input(t("api_key", lang), value=cfg.api_key, type="password", key=f"{role}_api_key")

    if provider == "anthropic":
        base_url: Optional[str] = None
        st.caption("Anthropic SDK — no Base URL required.")
    elif provider == "custom":
        base_url = st.text_input(t("custom_base_url", lang), value=cfg.base_url or "", key=f"{role}_base_url") or None
    else:
        auto = PROVIDER_BASE_URLS.get(provider) or ""
        base_url = st.text_input(t("base_url", lang), value=cfg.base_url if cfg.base_url else auto, key=f"{role}_base_url") or auto

    ct, cm = st.columns(2)
    with ct:
        temperature = st.slider(t("temperature", lang), 0.0, 2.0, value=float(cfg.temperature), step=0.05, key=f"{role}_temp")
    with cm:
        max_tokens = st.number_input(t("max_tokens", lang), 256, 32768, value=int(cfg.max_tokens), step=256, key=f"{role}_max_tokens")

    cs, ct2 = st.columns(2)
    with cs:
        if st.button(t("save_config", lang), key=f"{role}_save"):
            new_cfg = ModelConfig(
                provider=provider, api_key=api_key, model_name=model_name,
                base_url=base_url, temperature=temperature, max_tokens=int(max_tokens),
            )
            st.session_state[f"{role}_config"] = new_cfg
            st.success(t("config_saved", lang))
            # Auto-verify connection after save
            if api_key:
                with st.spinner(t("test_connection", lang) + "…"):
                    try:
                        verify_cfg = ModelConfig(
                            provider=provider, api_key=api_key, model_name=model_name,
                            base_url=base_url, temperature=temperature, max_tokens=64,
                        )
                        LLMClientFactory.create(verify_cfg).generate("Reply OK")
                        st.success(t("conn_ok", lang))
                    except Exception as err:
                        st.error(t("conn_fail", lang, err=str(err)))
    with ct2:
        if st.button(t("test_connection", lang), key=f"{role}_test"):
            if not api_key:
                st.warning(t("api_not_configured", lang))
            else:
                test_cfg = ModelConfig(provider=provider, api_key=api_key, model_name=model_name,
                                       base_url=base_url, temperature=temperature, max_tokens=64)
                try:
                    LLMClientFactory.create(test_cfg).generate("Reply OK")
                    st.success(t("conn_ok", lang))
                except Exception as err:
                    st.error(t("conn_fail", lang, err=str(err)))


# ===========================================================================
# PAGE: Overview
# ===========================================================================

if page == "overview":
    st.title(t("overview_title", lang))
    st.markdown(t("overview_desc", lang))

    st.subheader(t("data_status", lang))
    c1, c2, c3 = st.columns(3)
    q_path = _find(DATA_DIR, Q_CANDIDATES)
    r_path = _find(DATA_DIR, R_CANDIDATES)
    kb_path = DATA_DIR / "knowledge_base.xlsx"

    with c1:
        if q_path:
            st.metric(t("test_questions", lang), len(load_questions(q_path)))
        else:
            st.metric(t("test_questions", lang), "—"); st.error(t("file_not_found", lang))
    with c2:
        if r_path:
            st.metric(t("rubric_dims", lang), len(load_rubric(r_path)))
        else:
            st.metric(t("rubric_dims", lang), "—"); st.error(t("file_not_found", lang))
    with c3:
        if kb_path.exists():
            st.metric(t("kb_entries", lang), len(pd.read_excel(kb_path, engine="openpyxl")))
        else:
            st.metric(t("kb_entries", lang), "—"); st.warning(t("optional_not_found", lang))

    st.subheader(t("assessment_dims", lang))
    st.dataframe(pd.DataFrame([
        {t("code", lang): d, t("dimension", lang): DIMENSION_LABELS[d], t("weight", lang): f"{WEIGHTS[d]:.0%}"}
        for d in DIMENSIONS
    ]), use_container_width=True, hide_index=True)

    if q_path:
        st.subheader(t("questions_preview", lang))
        st.dataframe(pd.DataFrame([
            {"ID": q.id, t("area", lang): q.area, t("dimension", lang): q.dimension,
             t("text", lang): q.text[:120] + ("…" if len(q.text) > 120 else "")}
            for q in load_questions(q_path)
        ]), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: API Configuration
# ===========================================================================

elif page == "api_config":
    st.title(t("api_config_title", lang))
    st.markdown(t("api_config_desc", lang))

    tab_t, tab_s, tab_j = st.tabs([t("teacher_api", lang), t("student_api", lang), t("judge_api", lang)])

    with tab_t:
        _api_panel("teacher")

    with tab_s:
        if st.button(t("copy_config", lang), key="copy_t2s"):
            src: ModelConfig = st.session_state["teacher_config"]
            st.session_state["student_config"] = ModelConfig(**src.model_dump())
            st.success(t("config_saved", lang))
        _api_panel("student")

    with tab_j:
        note = ("Temperature = 0 recommended for deterministic scoring."
                if lang == "en" else "建议将 Temperature 设为 0 以保证评分的确定性。")
        st.info(note)
        _api_panel("judge")


# ===========================================================================
# PAGE: Run Evaluation
# ===========================================================================

elif page == "run":
    st.title(t("run_title", lang))
    st.markdown(t("run_desc", lang))

    with st.expander(t("upload_section", lang), expanded=False):
        up_q = st.file_uploader(t("upload_questions", lang), type=["xlsx","json","jsonl","txt"], key="up_q")
        up_r = st.file_uploader(t("upload_rubric", lang),    type=["xlsx","json","jsonl","txt"], key="up_r")
        if up_q is not None:
            try:
                st.session_state["cached_questions"] = _load_questions_upload(up_q)
                st.session_state["cached_q_name"] = up_q.name
            except Exception as exc:
                st.error(str(exc))
        if up_r is not None:
            try:
                st.session_state["cached_rubric"] = _load_rubric_upload(up_r)
                st.session_state["cached_r_name"] = up_r.name
            except Exception as exc:
                st.error(str(exc))
        q_name = st.session_state.get("cached_q_name")
        r_name = st.session_state.get("cached_r_name")
        if q_name:
            st.caption(f"✅ {t('test_questions', lang)}: {q_name}")
        if r_name:
            st.caption(f"✅ {t('rubric_dims', lang)}: {r_name}")
        if q_name or r_name:
            if st.button(t("clear_cache", lang), key="clear_upload_cache"):
                for k in ("cached_questions", "cached_rubric", "cached_q_name", "cached_r_name"):
                    st.session_state[k] = None
                st.rerun()

    ca, cb, cc = st.columns(3)
    with ca:
        model_name = st.text_input(t("model_id", lang), value="my-agent", help=t("model_id_help", lang))
    with cb:
        phase = st.selectbox(t("phase", lang), ["pre", "post"])
    with cc:
        mock_mode = st.checkbox(t("mock_mode", lang), value=False, help=t("mock_help", lang))

    if st.button(t("run_btn", lang), type="primary"):
        # Load questions (cached upload → data/ directory)
        try:
            if st.session_state.get("cached_questions"):
                questions = st.session_state["cached_questions"]
            elif (p := _find(DATA_DIR, Q_CANDIDATES)):
                questions = load_questions(p)
            else:
                questions = None
            if questions is None:
                st.error(t("data_missing", lang)); st.stop()
        except Exception as exc:
            st.error(str(exc)); st.stop()

        # Load rubric (cached upload → data/ directory)
        try:
            if st.session_state.get("cached_rubric"):
                rubric_list = st.session_state["cached_rubric"]
            elif (rp := _find(DATA_DIR, R_CANDIDATES)):
                rubric_list = load_rubric(rp)
            else:
                rubric_list = None
            if rubric_list is None:
                st.error(t("data_missing", lang)); st.stop()
        except Exception as exc:
            st.error(str(exc)); st.stop()

        rubric_dict: Dict[str, RubricItem] = {r.dimension: r for r in rubric_list}

        bundle: Optional[APIBundle] = None
        if not mock_mode:
            try:
                bundle = _build_bundle()
            except Exception as exc:
                st.error(f"{t('api_not_configured', lang)}: {exc}"); st.stop()
            if bundle is None:
                st.error(t("api_not_configured", lang)); st.stop()

        scores: List[ScoreRecord] = []
        progress = st.progress(0, text=t("evaluating", lang))

        for i, q in enumerate(questions):
            hdr = f"Q{q.id} [{q.dimension}]"

            if mock_mode:
                rng = random.Random(hash(q.id))
                if phase == "pre":
                    sv = round(rng.uniform(20, 50), 1)
                    scores.append(ScoreRecord(question_id=q.id, dimension=q.dimension, score=sv, rationale="[mock]"))
                    with st.expander(f"{hdr} — {sv:.1f}"):
                        st.markdown(f"**{t('question', lang)}:** {q.text[:200]}")
                        st.markdown(f"**{t('student_answer', lang)}:** [mock] {sv:.1f}")
                else:
                    sv_base = round(rng.uniform(20, 50), 1)
                    sv_post = round(rng.uniform(60, 90), 1)
                    improve = sv_post - sv_base
                    sign = "+" if improve >= 0 else ""
                    scores.append(ScoreRecord(question_id=q.id, dimension=q.dimension, score=sv_post, rationale="[mock]"))
                    with st.expander(f"{hdr} — {sv_base:.1f} → {sv_post:.1f}", expanded=False):
                        st.markdown(f"**{t('question', lang)}:** {q.text[:200]}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**{t('step_baseline', lang)}**")
                            st.write("[mock] Student's initial answer before any teaching.")
                            st.caption(f"🔹 {t('baseline_score', lang)}: **{sv_base:.1f}**")
                        with col2:
                            st.markdown(f"**{t('step_teaching', lang)}**")
                            st.info("[mock] Teacher's guidance and explanation.")
                        with col3:
                            st.markdown(f"**{t('step_intervention', lang)}**")
                            st.write("[mock] Student's improved answer after guidance.")
                            st.caption(f"🔸 {t('intervention_score', lang)}: **{sv_post:.1f}**")
                        st.success(f"{t('score_improve', lang)}: {sign}{improve:.1f}")
            else:
                assert bundle is not None
                try:
                    if phase == "pre":
                        with st.expander(hdr, expanded=False):
                            st.markdown(f"**{t('question', lang)}:** {q.text}")
                            with st.spinner(t("student_answer", lang) + "…"):
                                ans = bundle.student.answer_baseline(q)
                            st.markdown(f"**{t('student_answer', lang)}:**")
                            st.write(ans)
                            with st.spinner(t("judge_result", lang) + "…"):
                                rec = bundle.judge.evaluate(q, ans, rubric_dict.get(q.dimension))
                            st.caption(f"🔹 {t('judge_result', lang)}: **{rec.score:.1f}** — {rec.rationale}")
                        scores.append(rec)
                    else:
                        with st.expander(hdr, expanded=True):
                            st.markdown(f"**{t('question', lang)}:** {q.text}")
                            st.divider()
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown(f"**{t('step_baseline', lang)}**")
                                with st.spinner(t("student_baseline_answer", lang) + "…"):
                                    ans_base = bundle.student.answer_baseline(q)
                                st.write(ans_base)
                                with st.spinner(t("judge_result", lang) + "…"):
                                    rec_base = bundle.judge.evaluate(q, ans_base, rubric_dict.get(q.dimension))
                                st.caption(f"🔹 {t('baseline_score', lang)}: **{rec_base.score:.1f}**")
                                st.caption(rec_base.rationale)

                            with col2:
                                st.markdown(f"**{t('step_teaching', lang)}**")
                                with st.spinner(t("teaching_guidance", lang) + "…"):
                                    guidance = bundle.teacher.teach(q)
                                st.info(guidance)

                            with col3:
                                st.markdown(f"**{t('step_intervention', lang)}**")
                                with st.spinner(t("student_intervention_answer", lang) + "…"):
                                    ans_post = bundle.student.answer_intervention(q, guidance)
                                st.write(ans_post)
                                with st.spinner(t("judge_result", lang) + "…"):
                                    rec = bundle.judge.evaluate(q, ans_post, rubric_dict.get(q.dimension))
                                improve = rec.score - rec_base.score
                                sign = "+" if improve >= 0 else ""
                                st.caption(f"🔸 {t('intervention_score', lang)}: **{rec.score:.1f}**")
                                st.caption(rec.rationale)
                                st.success(f"{t('score_improve', lang)}: {sign}{improve:.1f}")
                        scores.append(rec)
                except Exception as exc:
                    with st.expander(f"❌ {hdr}", expanded=True):
                        st.error(str(exc))
                    scores.append(ScoreRecord(question_id=q.id, dimension=q.dimension, score=0.0, rationale=str(exc)))

            progress.progress((i + 1) / len(questions), text=f"{hdr} done")

        result   = PhaseResult(model=model_name, phase=phase, scores=scores)
        out_path = RESULTS_DIR / f"{model_name}_{phase}.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_phase_result(result, out_path)
        st.success(t("saved_to", lang, n=str(len(scores)), path=str(out_path)))

        st.subheader(t("score_summary", lang))
        st.dataframe(pd.DataFrame([
            {t("question", lang): s.question_id, t("dimension", lang): s.dimension,
             t("score", lang): s.score, t("rationale", lang): s.rationale[:80]}
            for s in scores
        ]), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Compute N-gain
# ===========================================================================

elif page == "calc":
    st.title(t("calc_title", lang))
    st.markdown(t("calc_desc", lang))

    complete = [m for m in _available_models() if _model_has_both(m)]
    if not complete:
        st.info(t("no_models", lang)); st.stop()

    model = st.selectbox(t("select_model", lang), complete)

    if st.button(t("calc_btn", lang), type="primary"):
        pre_r  = load_phase_result(RESULTS_DIR / f"{model}_pre.json")
        post_r = load_phase_result(RESULTS_DIR / f"{model}_post.json")
        pre_avg, post_avg = pre_r.dimension_averages(), post_r.dimension_averages()

        gains: List[DimensionGain] = []
        rows = []
        for dim in DIMENSIONS:
            g = DimensionGain(dimension=dim, pre=pre_avg[dim], post=post_avg[dim])
            gains.append(g)
            rows.append({
                t("dimension", lang): dim,
                t("label", lang): DIMENSION_LABELS[dim],
                t("pre_score", lang):  round(pre_avg[dim], 1),
                t("post_score", lang): round(post_avg[dim], 1),
                t("n_gain", lang):     round(g.calculate(), 1),
                t("classification", lang): HEPTACalculator.classify(g.calculate()),
            })

        hepta = HEPTAResult(model=model, gains=gains)
        idx   = hepta.index

        m1, m2, m3 = st.columns(3)
        m1.metric(t("hepta_index", lang), f"{idx:.2f}")
        m2.metric(t("classification", lang), HEPTACalculator.classify(idx))
        m3.metric(t("dims_evaluated", lang), len(gains))

        st.subheader(t("per_dim_results", lang))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader(t("visualisations", lang))
        cl, cr = st.columns(2)
        with cl:
            fig_r = plot_radar(gains, title=f"N-gain Radar — {model}")
            st.pyplot(fig_r); plt.close(fig_r)
        with cr:
            fig_b = plot_bar(idx, hepta.dimension_gains, title=f"HEPTA-Index — {model}")
            st.pyplot(fig_b); plt.close(fig_b)

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model, "hepta_index": round(idx, 2),
            "classification": HEPTACalculator.classify(idx),
            "dimensions": {
                dim: {"pre": round(pre_avg[dim], 2), "post": round(post_avg[dim], 2),
                      "n_gain_pct": round(g.calculate(), 2)}
                for dim, g in zip(DIMENSIONS, gains)
            },
        }
        (OUTPUTS_DIR / f"{model}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        plot_radar(gains, title=f"N-gain Radar — {model}",   save_path=OUTPUTS_DIR / f"{model}_radar.png")
        plot_bar(idx, hepta.dimension_gains, title=f"HEPTA-Index — {model}", save_path=OUTPUTS_DIR / f"{model}_bar.png")
        plt.close("all")
        st.success(t("outputs_saved", lang, path=str(OUTPUTS_DIR)))


# ===========================================================================
# PAGE: Results Explorer
# ===========================================================================

elif page == "explorer":
    st.title(t("explorer_title", lang))
    st.markdown(t("explorer_desc", lang))

    models = _available_models()
    if not models:
        st.info(t("no_results", lang)); st.stop()

    st.subheader(t("available_files", lang))
    st.dataframe(pd.DataFrame([
        {t("file", lang): f.name, t("size_kb", lang): round(f.stat().st_size / 1024, 1)}
        for f in sorted(RESULTS_DIR.glob("*.json"))
    ]), use_container_width=True, hide_index=True)

    complete = [m for m in models if _model_has_both(m)]
    if complete:
        st.subheader(t("model_comparison", lang))
        selected = st.multiselect(t("select_models_compare", lang), complete,
                                  default=complete[:min(3, len(complete))])
        if selected:
            rows, all_gains = [], {}
            for m in selected:
                pre  = load_phase_result(RESULTS_DIR / f"{m}_pre.json")
                post = load_phase_result(RESULTS_DIR / f"{m}_post.json")
                pa, qa = pre.dimension_averages(), post.dimension_averages()
                gs = [DimensionGain(dimension=d, pre=pa[d], post=qa[d]) for d in DIMENSIONS]
                all_gains[m] = gs
                idx = HEPTACalculator.index(gs)
                row: Dict = {"Model": m, t("hepta_index", lang): round(idx, 2),
                             t("classification", lang): HEPTACalculator.classify(idx)}
                for dim in DIMENSIONS:
                    row[dim] = round(next(x for x in gs if x.dimension == dim).calculate(), 1)
                rows.append(row)

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if len(selected) <= 6:
                st.subheader(t("radar_overlay", lang))
                labels = [DIMENSION_LABELS[d] for d in DIMENSIONS]
                angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist() + [0]
                fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
                ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylim(0, 100); ax.set_yticks([20,40,60,80,100])
                ax.set_yticklabels(["20","40","60","80","100"], fontsize=8)
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))  # type: ignore
                for m, color in zip(selected, colors):
                    vals = [g.calculate() for g in all_gains[m]] + [all_gains[m][0].calculate()]
                    ax.plot(angles, vals, lw=2, label=m, color=color)
                    ax.fill(angles, vals, alpha=0.08, color=color)
                ax.set_title(t("n_gain_comparison", lang), y=1.08, fontsize=13, fontweight="bold")
                ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)

    st.subheader(t("raw_inspector", lang))
    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if json_files:
        choice = st.selectbox(t("select_file", lang), [f.name for f in json_files])
        if choice:
            st.json(json.loads((RESULTS_DIR / choice).read_text(encoding="utf-8")))
