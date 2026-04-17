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
    for key in ("cached_questions", "cached_rubric", "cached_q_name", "cached_r_name",
                "cached_pre_exam_constraints"):
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
        try:
            return pd.read_json(data)
        except Exception:
            data.seek(0)
            return pd.read_json(data, lines=True)
    if suffix == ".jsonl":
        return pd.read_json(data, lines=True)
    if suffix in (".txt", ".csv", ".tsv"):
        text = raw.decode("utf-8-sig")
        first_line = text.split("\n", 1)[0]
        for sep in ("\t", ",", "|"):
            if sep in first_line:
                return pd.read_csv(BytesIO(raw), sep=sep, encoding="utf-8-sig")
        raise ValueError("NOT_TABULAR")
    raise ValueError(f"Unsupported format: {suffix}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase & strip column names to tolerate minor formatting differences."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Known HEPTA dimension codes
# ---------------------------------------------------------------------------
_KNOWN_DIMS = set(DIMENSIONS)  # {"OBJ", "CE", "TA", "HCP", "TE", "CPI", "MA"}


# ---------------------------------------------------------------------------
# HEPTA structured-text parsers (multi-strategy, content-agnostic)
# ---------------------------------------------------------------------------

def _is_hepta_questions_txt(text: str) -> bool:
    """Detect whether *text* looks like a HEPTA-style questions file.

    Uses multiple heuristics so format changes don't break detection:
      1. "QUESTION N" + Chinese tags (original format)
      2. "QUESTION N" alone (tags may have changed)
      3. Chinese tags alone (section headers may have changed)
      4. Numbered question blocks like "Q1." / "题目1" / generic numbered items
    """
    has_q_marker = bool(re.search(r'QUESTION\s+\d+', text, re.I))
    has_cn_tag = bool(re.search(r'【(选择题|简答题|单选题|多选题|判断题|填空题|论述题|问答题|问题|题目)】', text))
    has_part = bool(re.search(r'PART\s+[IVX\d]+', text, re.I))
    has_numbered_q = bool(re.search(r'(?:^|\n)\s*(?:Q|题目?)\s*\d+\s*[:.：]', text, re.I | re.M))
    # If any two signals are present, or one strong signal
    if has_q_marker and (has_cn_tag or has_part):
        return True
    if has_q_marker:
        return True
    if has_cn_tag:
        return True
    if has_numbered_q and len(list(re.finditer(r'(?:^|\n)\s*(?:Q|题目?)\s*\d+\s*[:.：]', text, re.I | re.M))) >= 2:
        return True
    return False


def _is_hepta_rubric_txt(text: str) -> bool:
    """Detect whether *text* looks like a HEPTA-style rubric file.

    Checks for:
      1. Classic SECTION III / RUBRIC / SCORING headers + Chinese tags
      2. Known dimension codes (OBJ, CE, TA …) appearing multiple times
      3. Chinese rubric markers: 【评分…】【考查维度】【评分维度】
      4. Generic scoring language: 评分标准 / criteria / rubric
    """
    has_rubric_kw = bool(re.search(r'RUBRIC|SCORING|评分|SECTION\s+III', text, re.I))
    has_cn_rubric_tag = bool(re.search(r'【(评分维度|评分细则|考查维度|评分标准|评分规则|维度)】', text))
    has_dimension_format = bool(re.search(
        r'【\d+\.\s+[^】]+\([A-Z]+\)\s*[-—]\s*\d+%】', text))
    # Count known dimension codes appearing in the text
    dim_hits = sum(1 for d in _KNOWN_DIMS if re.search(rf'\b{d}\b', text))
    has_dims = dim_hits >= 3

    if has_rubric_kw and (has_cn_rubric_tag or has_dimension_format or has_dims):
        return True
    if has_dimension_format:
        return True
    if has_cn_rubric_tag:
        return True
    if has_rubric_kw and has_dims:
        return True
    return False


def _infer_area(qnum: int, total: int) -> str:
    """Infer area from question number position (thirds: Breadth→Methods→Depth)."""
    if total <= 0:
        return "General"
    third = max(total // 3, 1)
    if qnum <= third:
        return "Breadth"
    elif qnum <= 2 * third:
        return "Methods"
    else:
        return "Depth"


def _infer_dimension_from_text(text: str) -> str:
    """Best-effort dimension inference from question text content."""
    t_lower = text.lower()
    # Check for explicit dimension tags in the text
    dim_m = re.search(r'【考查维度】\s*(\w+)', text)
    if dim_m and dim_m.group(1) in _KNOWN_DIMS:
        return dim_m.group(1)
    dim_m2 = re.search(r'\bdimension\s*:\s*(\w+)', text, re.I)
    if dim_m2 and dim_m2.group(1) in _KNOWN_DIMS:
        return dim_m2.group(1)
    # MCQ heuristic
    if re.search(r'\b[A-D]\s*[.)\uff09]', text):
        return "OBJ"
    # Keyword heuristics
    kw_map = [
        ("CPI", ["cross-paper", "integration", "combine", "综合", "跨文献", "整合"]),
        ("HCP", ["historical", "persistence", "history", "历史", "演变", "演进"]),
        ("TA",  ["trade-off", "tradeoff", "compare", "versus", "权衡", "对比"]),
        ("TE",  ["technical", "implementation", "algorithm", "技术", "实现"]),
        ("MA",  ["methodology", "research method", "action research", "方法论", "研究方法"]),
        ("CE",  ["explain", "concept", "define", "definition", "概念", "解释"]),
    ]
    for dim, keywords in kw_map:
        if any(kw in t_lower for kw in keywords):
            return dim
    return "CE"


def _parse_hepta_questions_txt(text: str) -> List[Question]:
    """Parse a HEPTA-style questions txt with multiple fallback strategies.

    Each ``QUESTION N`` block is treated as **one atomic question** (which may
    internally contain both an MCQ part and a short-answer part).  The student
    agent receives the full block; the judge scores it on all 7 dimensions.
    """
    questions: List[Question] = []

    # --- Strategy 1: "QUESTION N:" markers ---
    q_iter = list(re.finditer(r'QUESTION\s+(\d+)\s*[:.：]', text, re.I))
    if q_iter:
        total_q = len(q_iter)
        for idx, m in enumerate(q_iter):
            qnum = int(m.group(1))
            start = m.end()
            end = q_iter[idx + 1].start() if idx + 1 < len(q_iter) else len(text)
            block = text[start:end].strip()
            area = _infer_area(qnum, total_q)

            # Strip trailing separator lines (--- / ===)
            block = re.split(r'\n[-=]{20,}\s*$', block, maxsplit=1)[0].strip()
            if block:
                questions.append(Question(
                    id=str(qnum), area=area, dimension="ALL",
                    text=block, reference_answer="", max_score=100.0))
        if questions:
            return questions

    # --- Strategy 2: "Q1." / "题目1:" / "1." numbered patterns ---
    numbered_patterns = [
        r'(?:^|\n)\s*Q(\d+)\s*[:.：]\s*',
        r'(?:^|\n)\s*题目?\s*(\d+)\s*[:.：]\s*',
        r'(?:^|\n)\s*(\d+)\s*[.)）]\s+',
    ]
    for pat in numbered_patterns:
        n_iter = list(re.finditer(pat, text, re.I | re.M))
        if len(n_iter) >= 2:
            total_q = len(n_iter)
            for idx, m in enumerate(n_iter):
                qnum = int(m.group(1))
                start = m.end()
                end = n_iter[idx + 1].start() if idx + 1 < len(n_iter) else len(text)
                block = text[start:end].strip()
                block = re.split(r'\n[-=]{20,}\s*$', block, maxsplit=1)[0].strip()
                if not block:
                    continue
                area = _infer_area(qnum, total_q)
                questions.append(Question(
                    id=str(qnum), area=area, dimension="ALL",
                    text=block, reference_answer="", max_score=100.0))
            if questions:
                return questions

    # --- Strategy 3: Split by separator lines (=== or ---) ---
    blocks = re.split(r'(?:={20,}|-{20,})', text)
    qnum = 0
    for b in blocks:
        b = b.strip()
        if len(b) < 20:
            continue
        if re.match(r'^(PART|SECTION|HEPTA)', b, re.I):
            continue
        qnum += 1
        questions.append(Question(
            id=str(qnum), area=_infer_area(qnum, len(blocks)),
            dimension="ALL", text=b, reference_answer="", max_score=100.0))
    return questions


def _parse_hepta_rubric_txt(text: str) -> List[RubricItem]:
    """Parse a HEPTA-style rubric txt with multiple fallback strategies."""
    items: List[RubricItem] = []

    # --- Strategy 1: 【N. NAME (CODE) - WEIGHT%】 ---
    dim_iter = list(re.finditer(
        r'【(\d+)\.\s+([^】(]+?)\s*[(\uff08](\w+)[)\uff09]\s*[-—]\s*\d+%】', text))
    if dim_iter:
        for idx, m in enumerate(dim_iter):
            dim_code = m.group(3)
            start = m.end()
            end = dim_iter[idx + 1].start() if idx + 1 < len(dim_iter) else len(text)
            criteria = text[start:end].strip()
            criteria = re.split(r'={40,}', criteria)[0].strip()
            criteria = re.split(r'-{40,}\s*$', criteria)[0].strip()
            if criteria:
                items.append(RubricItem(dimension=dim_code, criteria=criteria, max_score=100.0))
        if items:
            return items

    # --- Strategy 2: 【评分维度】/ 【考查维度】 CODE ---
    tag_patterns = [
        r'【(?:评分维度|考查维度|维度|Dimension)\s*】\s*[:\uff1a]?\s*(\w+)',
        r'【(?:评分维度|考查维度|维度|Dimension)\s*[:\uff1a]\s*(\w+)\s*】',
    ]
    for pat in tag_patterns:
        pq_iter = list(re.finditer(pat, text, re.I))
        if pq_iter:
            for midx, mm in enumerate(pq_iter):
                dim = mm.group(1)
                start = mm.end()
                sec_end = pq_iter[midx + 1].start() if midx + 1 < len(pq_iter) else len(text)
                criteria = text[start:sec_end].strip()[:2000]
                if criteria:
                    items.append(RubricItem(dimension=dim, criteria=criteria, max_score=100.0))
            if items:
                return items

    # --- Strategy 3: Known dimension codes as headers ---
    # e.g. "OBJ:\n..." or "CE —" or "**TA**" or "### CE"
    dim_header_pat = r'(?:^|\n)\s*(?:#{1,4}\s*)?(?:\*{1,2})?(' + '|'.join(_KNOWN_DIMS) + r')(?:\*{1,2})?\s*[-—:：\s]'
    d_iter = list(re.finditer(dim_header_pat, text, re.M))
    if len(d_iter) >= 2:
        for idx, m in enumerate(d_iter):
            dim = m.group(1).upper()
            start = m.end()
            end = d_iter[idx + 1].start() if idx + 1 < len(d_iter) else len(text)
            criteria = text[start:end].strip()
            criteria = re.split(r'={20,}|-{20,}', criteria)[0].strip()[:2000]
            if criteria:
                items.append(RubricItem(dimension=dim, criteria=criteria, max_score=100.0))
        if items:
            return items

    # --- Strategy 4: Dimension code somewhere in block + separator splitting ---
    blocks = re.split(r'(?:={20,}|-{20,})', text)
    for b in blocks:
        b = b.strip()
        if len(b) < 10:
            continue
        for dim in _KNOWN_DIMS:
            if re.search(rf'\b{dim}\b', b):
                criteria = re.sub(rf'^.*?\b{dim}\b\s*[-—:：]?\s*', '', b, count=1)
                if criteria and len(criteria) > 10:
                    items.append(RubricItem(dimension=dim, criteria=criteria[:2000], max_score=100.0))
                    break
    return items


def _parse_pre_exam_constraints(text: str) -> Optional[str]:
    """Extract any pre-exam constraints block from the txt file.

    Recognises multiple marker variants:
      【考前约束要求】 【考前要求】 【约束要求】 【Pre-Exam Requirements】
      As well as markdown-style headers: ## 考前约束要求
    """
    patterns = [
        # Chinese bracket markers (most specific → broadest)
        r'【考前约束要求】(.*?)(?:={10,}|-{10,}|PART\s+[IVX\d]+|QUESTION\s+\d+|【[^考]|$)',
        r'【考前要求】(.*?)(?:={10,}|-{10,}|PART\s+[IVX\d]+|QUESTION\s+\d+|【[^考]|$)',
        r'【约束要求】(.*?)(?:={10,}|-{10,}|PART\s+[IVX\d]+|QUESTION\s+\d+|【[^约]|$)',
        r'【[Pp]re-?[Ee]xam\s+[Rr]equirements?】(.*?)(?:={10,}|-{10,}|PART\s+[IVX\d]+|QUESTION\s+\d+|【|$)',
        # Markdown headers
        r'#{1,4}\s*考前约束要求\s*\n(.*?)(?:={10,}|-{10,}|#{1,4}\s|PART\s+[IVX\d]+|QUESTION\s+\d+|$)',
        r'#{1,4}\s*Pre-?Exam\s+Requirements?\s*\n(.*?)(?:={10,}|-{10,}|#{1,4}\s|PART\s+[IVX\d]+|QUESTION\s+\d+|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.I)
        if m:
            result = m.group(1).strip()
            if result:
                return result
    return None


def _load_questions_upload(uploaded) -> tuple[List[Question], Optional[str]]:
    """Return (questions, pre_exam_constraints).  constraints may be None."""
    suffix = Path(uploaded.name).suffix.lower()

    # --- Branch 1: txt file → try structured parse first ---
    if suffix == ".txt":
        raw = uploaded.read()
        text = raw.decode("utf-8-sig")
        constraints = _parse_pre_exam_constraints(text)

        if _is_hepta_questions_txt(text):
            qs = _parse_hepta_questions_txt(text)
            if qs:
                return qs, constraints

        # Even if detection failed, attempt a lenient parse
        qs = _parse_hepta_questions_txt(text)
        if qs:
            return qs, constraints

        # Fall through to tabular attempt
        uploaded.seek(0)

    # --- Branch 2: Tabular formats (xlsx/json/jsonl/csv/tsv) ---
    try:
        df = _normalise_columns(_read_upload(uploaded))
    except ValueError:
        raise ValueError(
            "Cannot parse questions file. Supported formats:\n"
            "• Tabular (xlsx/json/jsonl/csv/tsv) with columns: id, area, dimension, text\n"
            "• Structured txt with QUESTION markers, numbered questions (Q1/题目1/1.), "
            "or Chinese tags (【选择题】【简答题】 etc.)"
        )
    # Fuzzy column matching: try common aliases
    col_aliases = {
        "id": ["id", "qid", "question_id", "q_id", "编号", "题号", "no"],
        "area": ["area", "category", "section", "类别", "领域", "分区", "part"],
        "dimension": ["dimension", "dim", "维度", "code"],
        "text": ["text", "question", "content", "题目", "问题", "内容"],
    }
    renames = {}
    for target, aliases in col_aliases.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    renames[alias] = target
                    break
    if renames:
        df = df.rename(columns=renames)

    # Auto-fill missing columns with sensible defaults
    if "id" not in df.columns:
        df["id"] = [str(i + 1) for i in range(len(df))]
    if "area" not in df.columns:
        df["area"] = [_infer_area(i + 1, len(df)) for i in range(len(df))]
    if "dimension" not in df.columns and "text" in df.columns:
        df["dimension"] = df["text"].apply(lambda t: _infer_dimension_from_text(str(t)))
    elif "dimension" not in df.columns:
        df["dimension"] = "CE"

    required = {"text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Questions file must contain a 'text' or 'question' column. "
            f"Found: {list(df.columns)}"
        )
    return [
        Question(
            id=str(row["id"]), area=str(row["area"]), dimension=str(row["dimension"]),
            text=str(row["text"]), reference_answer=str(row.get("reference_answer", "")),
            max_score=float(row.get("max_score", 100)),
        )
        for _, row in df.iterrows()
    ], None


def _load_rubric_upload(uploaded) -> List[RubricItem]:
    suffix = Path(uploaded.name).suffix.lower()

    # --- Branch 1: txt file → try structured parse ---
    if suffix == ".txt":
        raw = uploaded.read()
        text = raw.decode("utf-8-sig")
        if _is_hepta_rubric_txt(text):
            items = _parse_hepta_rubric_txt(text)
            if items:
                return items
        # Lenient attempt even if detection didn't fire
        items = _parse_hepta_rubric_txt(text)
        if items:
            return items
        uploaded.seek(0)

    # --- Branch 2: Tabular ---
    try:
        df = _normalise_columns(_read_upload(uploaded))
    except ValueError:
        raise ValueError(
            "Cannot parse rubric file. Supported formats:\n"
            "• Tabular (xlsx/json/jsonl/csv/tsv) with columns: dimension, criteria\n"
            "• Structured txt with dimension rubric blocks (OBJ/CE/TA/…), "
            "Chinese tags (【评分维度】【评分细则】 etc.), or dimension headers"
        )
    # Fuzzy column matching
    col_aliases = {
        "dimension": ["dimension", "dim", "维度", "code", "评分维度"],
        "criteria": ["criteria", "criterion", "description", "评分细则", "标准", "描述", "content", "text"],
    }
    renames = {}
    for target, aliases in col_aliases.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    renames[alias] = target
                    break
    if renames:
        df = df.rename(columns=renames)

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
                _qs, _constraints = _load_questions_upload(up_q)
                st.session_state["cached_questions"] = _qs
                st.session_state["cached_pre_exam_constraints"] = _constraints
                st.session_state["cached_q_name"] = up_q.name
                if _constraints:
                    st.info(f"📋 已识别【考前约束要求】，将在学生答题前注入。\n\n> {_constraints[:200]}{'…' if len(_constraints) > 200 else ''}")
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
                for k in ("cached_questions", "cached_rubric", "cached_q_name", "cached_r_name",
                           "cached_pre_exam_constraints"):
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

        # Pre-exam constraints (extracted from uploaded questions txt, if any)
        pre_exam_constraints: Optional[str] = st.session_state.get("cached_pre_exam_constraints")
        if pre_exam_constraints:
            st.info(f"📋 已识别【考前约束要求】，已注入学生答题系统提示词。")

        bundle: Optional[APIBundle] = None
        if not mock_mode:
            try:
                bundle = _build_bundle()
            except Exception as exc:
                st.error(f"{t('api_not_configured', lang)}: {exc}"); st.stop()
            if bundle is None:
                st.error(t("api_not_configured", lang)); st.stop()

        # ── Helpers for multi-dimension scoring display ──────────────────
        def _weighted_total(recs: List[ScoreRecord]) -> float:
            """Weighted sum across all 7 dimensions → 0-100 question total."""
            return sum(WEIGHTS.get(r.dimension, 0.0) * r.score for r in recs)

        def _dim_table(recs: List[ScoreRecord]) -> pd.DataFrame:
            """Build a per-dimension score table with weighted contribution."""
            rows = []
            total_w = 0.0
            for r in recs:
                w = WEIGHTS.get(r.dimension, 0.0)
                wt = round(r.score * w, 1)
                total_w += wt
                rows.append({
                    t("dimension", lang): r.dimension,
                    t("label", lang):     DIMENSION_LABELS.get(r.dimension, r.dimension),
                    t("weight", lang):    f"{int(w * 100)}%",
                    t("score", lang):     round(r.score, 1),
                    t("weighted", lang):  wt,
                    t("rationale", lang): r.rationale[:120],
                })
            rows.append({
                t("dimension", lang): "TOTAL",
                t("label", lang):     "",
                t("weight", lang):    "100%",
                t("score", lang):     "—",
                t("weighted", lang):  round(total_w, 1),
                t("rationale", lang): "",
            })
            return pd.DataFrame(rows)

        scores: List[ScoreRecord] = []
        progress = st.progress(0, text=t("evaluating", lang))

        for i, q in enumerate(questions):
            hdr = f"Q{q.id}"

            if mock_mode:
                rng = random.Random(hash(q.id))
                if phase == "pre":
                    mock_recs = [
                        ScoreRecord(question_id=q.id, dimension=dim,
                                    score=round(rng.uniform(20, 50), 1), rationale="[mock]")
                        for dim in DIMENSIONS
                    ]
                    total = _weighted_total(mock_recs)
                    scores.extend(mock_recs)
                    with st.expander(f"{hdr} — {t('weighted_total', lang)}: {total:.1f}"):
                        st.markdown(f"**{t('question', lang)}:** {q.text[:200]}")
                        st.markdown(f"**{t('student_answer', lang)}:** [mock]")
                        st.dataframe(_dim_table(mock_recs), use_container_width=True, hide_index=True)
                else:
                    rng2 = random.Random(hash(q.id) + 1)
                    mock_base = [
                        ScoreRecord(question_id=q.id, dimension=dim,
                                    score=round(rng.uniform(20, 50), 1), rationale="[mock]")
                        for dim in DIMENSIONS
                    ]
                    mock_post = [
                        ScoreRecord(question_id=q.id, dimension=dim,
                                    score=round(rng2.uniform(60, 90), 1), rationale="[mock]")
                        for dim in DIMENSIONS
                    ]
                    base_total = _weighted_total(mock_base)
                    post_total = _weighted_total(mock_post)
                    improve = post_total - base_total
                    sign = "+" if improve >= 0 else ""
                    scores.extend(mock_post)
                    with st.expander(f"{hdr} — {base_total:.1f} → {post_total:.1f}", expanded=False):
                        st.markdown(f"**{t('question', lang)}:** {q.text[:200]}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**{t('step_baseline', lang)}**")
                            st.write("[mock] Student's initial answer.")
                            st.caption(f"🔹 {t('baseline_score', lang)}: **{base_total:.1f}**")
                            st.dataframe(_dim_table(mock_base), use_container_width=True, hide_index=True)
                        with col2:
                            st.markdown(f"**{t('step_teaching', lang)}**")
                            st.info("[mock] Teacher's guidance.")
                        with col3:
                            st.markdown(f"**{t('step_intervention', lang)}**")
                            st.write("[mock] Student's improved answer.")
                            st.caption(f"🔸 {t('intervention_score', lang)}: **{post_total:.1f}**")
                            st.dataframe(_dim_table(mock_post), use_container_width=True, hide_index=True)
                        st.success(f"{t('score_improve', lang)}: {sign}{improve:.1f}")
            else:
                assert bundle is not None
                try:
                    if phase == "pre":
                        with st.expander(hdr, expanded=False):
                            st.markdown(f"**{t('question', lang)}:** {q.text}")
                            with st.spinner(t("student_answer", lang) + "…"):
                                # ⚠️ Rubric is NEVER passed to the student agent
                                ans = bundle.student.answer_baseline(
                                    q, pre_exam_constraints=pre_exam_constraints)
                            st.markdown(f"**{t('student_answer', lang)}:**")
                            st.write(ans)
                            with st.spinner(t("judge_result", lang) + "…"):
                                # Judge receives the full rubric; student answer is already fixed
                                recs = bundle.judge.evaluate_all_dimensions(q, ans, rubric_dict)
                            total = _weighted_total(recs)
                            st.caption(f"🔹 {t('weighted_total', lang)}: **{total:.1f} / 100**")
                            st.dataframe(_dim_table(recs), use_container_width=True, hide_index=True)
                        scores.extend(recs)
                    else:
                        with st.expander(hdr, expanded=True):
                            st.markdown(f"**{t('question', lang)}:** {q.text}")
                            st.divider()
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown(f"**{t('step_baseline', lang)}**")
                                with st.spinner(t("student_baseline_answer", lang) + "…"):
                                    # ⚠️ Rubric NOT shown to student
                                    ans_base = bundle.student.answer_baseline(
                                        q, pre_exam_constraints=pre_exam_constraints)
                                st.write(ans_base)
                                with st.spinner(t("judge_result", lang) + "…"):
                                    recs_base = bundle.judge.evaluate_all_dimensions(
                                        q, ans_base, rubric_dict)
                                base_total = _weighted_total(recs_base)
                                st.caption(f"🔹 {t('baseline_score', lang)}: **{base_total:.1f}**")
                                st.dataframe(_dim_table(recs_base), use_container_width=True, hide_index=True)

                            with col2:
                                st.markdown(f"**{t('step_teaching', lang)}**")
                                with st.spinner(t("teaching_guidance", lang) + "…"):
                                    # ⚠️ Teacher also does NOT receive the rubric
                                    guidance = bundle.teacher.teach(q)
                                st.info(guidance)

                            with col3:
                                st.markdown(f"**{t('step_intervention', lang)}**")
                                with st.spinner(t("student_intervention_answer", lang) + "…"):
                                    # ⚠️ Rubric NOT shown to student
                                    ans_post = bundle.student.answer_intervention(
                                        q, guidance, pre_exam_constraints=pre_exam_constraints)
                                st.write(ans_post)
                                with st.spinner(t("judge_result", lang) + "…"):
                                    recs = bundle.judge.evaluate_all_dimensions(
                                        q, ans_post, rubric_dict)
                                post_total = _weighted_total(recs)
                                improve = post_total - base_total
                                sign = "+" if improve >= 0 else ""
                                st.caption(f"🔸 {t('intervention_score', lang)}: **{post_total:.1f}**")
                                st.dataframe(_dim_table(recs), use_container_width=True, hide_index=True)
                                st.success(f"{t('score_improve', lang)}: {sign}{improve:.1f}")
                        scores.extend(recs)
                except Exception as exc:
                    with st.expander(f"❌ {hdr}", expanded=True):
                        st.error(str(exc))
                    # Ensure all 7 dimensions still get a zero record so N-gain
                    # aggregation won't silently omit this question
                    scores.extend([
                        ScoreRecord(question_id=q.id, dimension=dim,
                                    score=0.0, rationale=str(exc))
                        for dim in DIMENSIONS
                    ])

            progress.progress((i + 1) / len(questions), text=f"{hdr} done")

        result   = PhaseResult(model=model_name, phase=phase, scores=scores)
        out_path = RESULTS_DIR / f"{model_name}_{phase}.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_phase_result(result, out_path)
        st.success(t("saved_to", lang, n=str(len(questions)), path=str(out_path)))

        st.subheader(t("score_summary", lang))
        st.dataframe(pd.DataFrame([
            {
                t("question", lang):   s.question_id,
                t("dimension", lang):  s.dimension,
                t("weight", lang):     f"{int(WEIGHTS.get(s.dimension, 0) * 100)}%",
                t("score", lang):      round(s.score, 1),
                t("weighted", lang):   round(WEIGHTS.get(s.dimension, 0) * s.score, 1),
                t("rationale", lang):  s.rationale[:80],
            }
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
