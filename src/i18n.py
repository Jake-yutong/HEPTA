"""Internationalisation strings for the HEPTA Streamlit UI.

Usage::

    from src.i18n import t
    label = t("run_btn", lang="zh")
"""

from __future__ import annotations

from typing import Dict

# Key → {lang: text}
STRINGS: Dict[str, Dict[str, str]] = {
    # ── Sidebar ────────────────────────────────────────────────────────────
    "app_title":        {"en": "HEPTA",        "zh": "HEPTA"},
    "app_caption":      {"en": "AI HCI Education Performance Test Benchmark",
                         "zh": "AI HCI 教育绩效测试基准"},
    "language":         {"en": "Language",     "zh": "语言"},
    "api_status":       {"en": "API Status",   "zh": "API 状态"},
    "configured":       {"en": "Configured",   "zh": "已配置"},
    "not_configured":   {"en": "Not configured", "zh": "未配置"},

    # ── Navigation ─────────────────────────────────────────────────────────
    "nav_overview":     {"en": "Overview",          "zh": "概览"},
    "nav_api_config":   {"en": "API Configuration", "zh": "API 配置"},
    "nav_run":          {"en": "Run Evaluation",    "zh": "运行评测"},
    "nav_calc":         {"en": "Compute N-gain",    "zh": "计算 N-gain"},
    "nav_explorer":     {"en": "Results Explorer",  "zh": "结果浏览"},

    # ── Overview ───────────────────────────────────────────────────────────
    "overview_title":   {"en": "HEPTA — AI HCI Education Performance Test Benchmark",
                         "zh": "HEPTA — AI HCI 教育绩效测试基准"},
    "overview_desc":    {
        "en": "HEPTA evaluates the pedagogical effectiveness of LLM-based HCI teaching agents "
              "using a **Normalized Gain** model across seven assessment dimensions derived from "
              "the Stanford HCI Qual structure.",
        "zh": "HEPTA 通过**归一化增益（N-gain）**模型，在七个源自斯坦福 HCI 资格考试结构的"
              "评估维度上，评估基于 LLM 的 HCI 教学 Agent 的教学有效性。",
    },
    "data_status":       {"en": "Data Status",              "zh": "数据状态"},
    "test_questions":    {"en": "Test Questions",            "zh": "测试题目"},
    "rubric_dims":       {"en": "Rubric Dimensions",         "zh": "评分维度"},
    "kb_entries":        {"en": "Knowledge Base Entries",    "zh": "知识库条目"},
    "file_not_found":    {"en": "File not found",            "zh": "文件未找到"},
    "optional_not_found":{"en": "Optional — file not found","zh": "可选 — 文件未找到"},
    "assessment_dims":   {"en": "Assessment Dimensions",     "zh": "评估维度"},
    "questions_preview": {"en": "Test Questions Preview",    "zh": "测试题目预览"},
    "code":              {"en": "Code",      "zh": "代码"},
    "dimension":         {"en": "Dimension", "zh": "维度"},
    "weight":            {"en": "Weight",    "zh": "权重"},
    "area":              {"en": "Area",      "zh": "领域"},
    "text":              {"en": "Text",      "zh": "题目"},

    # ── API Configuration ──────────────────────────────────────────────────
    "api_config_title": {"en": "API Configuration",
                         "zh": "API 配置"},
    "api_config_desc":  {
        "en": "Configure LLM endpoints for Teacher, Student, and Judge roles. "
              "Settings are stored for the current session.",
        "zh": "为教师、学生和评分员角色配置 LLM 端点。设置将保存在当前会话中。",
    },
    "teacher_api":       {"en": "Teacher API",                     "zh": "教师 API"},
    "student_api":       {"en": "Student API",                     "zh": "学生 API"},
    "judge_api":         {"en": "Judge (Rubric Scoring) API",       "zh": "评分员（Rubric 评分）API"},
    "provider":          {"en": "Provider",                        "zh": "服务商"},
    "model_name":        {"en": "Model Name",                      "zh": "模型名称"},
    "api_key":           {"en": "API Key",                         "zh": "API 密钥"},
    "base_url":          {"en": "Base URL (auto-filled, editable)", "zh": "Base URL（自动填充，可修改）"},
    "custom_base_url":   {"en": "Base URL",                        "zh": "Base URL"},
    "temperature":       {"en": "Temperature",                     "zh": "Temperature"},
    "max_tokens":        {"en": "Max Tokens",                      "zh": "最大 Token 数"},
    "save_config":       {"en": "Save Configuration",              "zh": "保存配置"},
    "config_saved":      {"en": "Configuration saved.",            "zh": "配置已保存。"},
    "suggested_models":  {"en": "Suggested models",                "zh": "推荐模型"},
    "test_connection":   {"en": "Test Connection",                 "zh": "测试连接"},
    "conn_ok":           {"en": "Connection successful.",          "zh": "连接成功。"},
    "conn_fail":         {"en": "Connection failed: {err}",        "zh": "连接失败：{err}"},
    "copy_config":       {"en": "Copy from Teacher",               "zh": "从教师配置复制"},

    # ── Run Evaluation ─────────────────────────────────────────────────────
    "run_title":         {"en": "Run Evaluation",    "zh": "运行评测"},
    "run_desc":          {
        "en": "Execute a pre-test or post-test phase. Enable **Mock mode** for local testing "
              "without API calls.",
        "zh": "执行预测试或后测试阶段。启用**模拟模式**可在不调用 API 的情况下进行本地测试。",
    },
    "upload_section":    {"en": "Data Files",        "zh": "数据文件"},
    "upload_questions":  {"en": "Upload questions file (xlsx / json / jsonl) — overrides data/",
                         "zh": "上传题目文件（xlsx / json / jsonl）— 将覆盖 data/ 目录"},
    "upload_rubric":     {"en": "Upload rubric file (xlsx / json / jsonl) — overrides data/",
                         "zh": "上传 Rubric 文件（xlsx / json / jsonl）— 将覆盖 data/ 目录"},
    "model_id":          {"en": "Model identifier",  "zh": "模型标识符"},
    "model_id_help":     {"en": "Label used to name result files.",
                         "zh": "用于命名结果文件的标签。"},
    "phase":             {"en": "Phase",             "zh": "测试阶段"},
    "mock_mode":         {"en": "Mock mode",         "zh": "模拟模式"},
    "mock_help":         {"en": "Generate random scores locally (no API calls).",
                         "zh": "在本地生成随机分数，不调用任何 API。"},
    "run_btn":           {"en": "Run",               "zh": "运行"},
    "evaluating":        {"en": "Evaluating…",       "zh": "评测中…"},
    "data_missing":      {"en": "Data files missing. Upload files above or place them in `data/`.",
                         "zh": "数据文件缺失，请在上方上传或将文件放置于 `data/` 目录。"},
    "api_not_configured":{"en": "API not configured. Go to **API Configuration** or enable Mock mode.",
                         "zh": "API 尚未配置，请前往 **API 配置** 页面或启用模拟模式。"},
    "score_summary":     {"en": "Score Summary",          "zh": "评分摘要"},
    "saved_to":          {"en": "Saved {n} scores → `{path}`",
                         "zh": "已保存 {n} 条评分 → `{path}`"},
    "teaching_guidance": {"en": "Teaching Guidance",      "zh": "教学辅导"},
    "student_answer":    {"en": "Student Answer",         "zh": "学生答案"},
    "student_baseline_answer":    {"en": "Student Answer (Before Teaching)",  "zh": "学生初次作答（教学前）"},
    "student_intervention_answer":{"en": "Student Answer (After Teaching)",   "zh": "学生再次作答（教学后）"},
    "step_baseline":     {"en": "Step 1 · Student answers directly",          "zh": "第 1 步 · 学生直接答题"},
    "step_teaching":     {"en": "Step 2 · Teacher generates guidance",         "zh": "第 2 步 · 教师生成教学内容"},
    "step_intervention": {"en": "Step 3 · Student answers with guidance",      "zh": "第 3 步 · 学生结合辅导再次作答"},
    "baseline_score":    {"en": "Score Before Teaching",  "zh": "教学前得分"},
    "intervention_score":{"en": "Score After Teaching",   "zh": "教学后得分"},
    "score_improve":     {"en": "Improvement",            "zh": "提升"},
    "judge_result":      {"en": "Judge Result",           "zh": "评分结果"},
    "question":          {"en": "Question",    "zh": "题目"},
    "score":             {"en": "Score",       "zh": "得分"},
    "rationale":         {"en": "Rationale",   "zh": "评分理由"},

    # ── Compute N-gain ─────────────────────────────────────────────────────
    "calc_title":        {"en": "Compute N-gain & HEPTA-Index",    "zh": "计算 N-gain 与 HEPTA-Index"},
    "calc_desc":         {"en": "Select a model that has completed both pre and post phases.",
                         "zh": "选择已完成预测试和后测试的模型。"},
    "no_models":         {"en": "No model has both pre and post results yet.",
                         "zh": "尚无模型同时完成预测试和后测试，请先运行评测。"},
    "select_model":      {"en": "Model",       "zh": "模型"},
    "calc_btn":          {"en": "Calculate",   "zh": "计算"},
    "hepta_index":       {"en": "HEPTA-Index", "zh": "HEPTA 指数"},
    "classification":    {"en": "Classification",        "zh": "分类"},
    "dims_evaluated":    {"en": "Dimensions Evaluated",  "zh": "评估维度数"},
    "per_dim_results":   {"en": "Per-Dimension Results", "zh": "各维度结果"},
    "visualisations":    {"en": "Visualisations",        "zh": "可视化图表"},
    "outputs_saved":     {"en": "Outputs saved to `{path}/`",
                         "zh": "输出已保存至 `{path}/`"},
    "label":             {"en": "Label",     "zh": "名称"},
    "pre_score":         {"en": "Pre",       "zh": "前测"},
    "post_score":        {"en": "Post",      "zh": "后测"},
    "n_gain":            {"en": "N-gain (%)", "zh": "N-gain (%)"},

    # ── Results Explorer ───────────────────────────────────────────────────
    "explorer_title":    {"en": "Results Explorer",     "zh": "结果浏览"},
    "explorer_desc":     {"en": "Browse and compare saved evaluation results.",
                         "zh": "浏览并比较已保存的评测结果。"},
    "no_results":        {"en": "No results found. Run evaluations first.",
                         "zh": "未找到结果，请先运行评测。"},
    "available_files":   {"en": "Available Result Files", "zh": "可用结果文件"},
    "file":              {"en": "File",      "zh": "文件"},
    "size_kb":           {"en": "Size (KB)", "zh": "大小 (KB)"},
    "model_comparison":  {"en": "Model Comparison",  "zh": "模型对比"},
    "select_models_compare": {"en": "Select models to compare",
                              "zh": "选择要对比的模型"},
    "radar_overlay":     {"en": "Radar Overlay",      "zh": "雷达图叠加"},
    "n_gain_comparison": {"en": "N-gain Comparison",  "zh": "N-gain 对比"},
    "raw_inspector":     {"en": "Raw Result Inspector", "zh": "原始结果查看"},
    "select_file":       {"en": "Select file",        "zh": "选择文件"},
}


def t(key: str, lang: str = "en", **kwargs: str) -> str:
    """Return the translated string for *key* in *lang*.

    Falls back to the English string if the key or language is missing.
    """
    entry = STRINGS.get(key, {})
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        text = text.format(**kwargs)
    return text
