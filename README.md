# HEPTA — AI HCI Education Performance Test Benchmark

**HEPTA** is an automated benchmarking framework for evaluating the pedagogical effectiveness of Large Language Models (LLMs) serving as HCI teaching agents. It operationalises a Normalized Gain (N-gain) model across seven assessment dimensions derived from the Stanford HCI Qual examination structure.

## Motivation

HCI education faces persistent challenges in resource allocation and pedagogical knowledge transfer. Instructors with strong domain expertise often struggle to translate that expertise into effective teaching strategies — a gap well documented in the educational literature. The disparity between elite and under-resourced institutions further compounds the problem.

LLMs, when fine-tuned on domain-specific datasets, have the potential to serve as HCI teaching agents that convert HCI theory and methods into pedagogical knowledge. HEPTA provides a rigorous, reproducible benchmark for measuring how effectively different LLM-based agents achieve this goal.

## Assessment Model

HEPTA replaces the simple Gain Score ($S_{\text{knowledge}} - S_{\text{base}}$) used in prior work with the **Normalized Gain** model (Hake, 1998):

$$
g = \frac{S_{\text{post}} - S_{\text{pre}}}{100 - S_{\text{pre}}}
$$

Scores are evaluated across **seven dimensions** aligned with the HCI Qual competency areas:

| Code | Dimension | Weight |
|------|-----------|--------|
| OBJ  | Objective Questions | 10 % |
| MA   | Methodological Application | 15 % |
| CE   | Conceptual Explanation | 15 % |
| TA   | Tradeoff Analysis | 15 % |
| HCP  | Historical Context & Persistence | 15 % |
| TE   | Technical Elucidation | 15 % |
| CPI  | Cross-Paper Integration | 15 % |

The composite **HEPTA-Index** is the weighted sum of per-dimension N-gain percentages. Classifications follow Hake's thresholds: High ($g \geq 70\%$), Medium ($30\% \leq g < 70\%$), Low ($g < 30\%$).

## Test Instrument

The test set consists of **9 questions** covering three areas of the Stanford HCI Qual:

- **Breadth Areas** (3 questions): human-centred design foundations
- **Methods Areas** (3 questions): statistical methods and research design
- **Depth Areas** (3 questions): social computing, design, and Human-AI interaction

A companion **knowledge base** (50+ entries) provides training data sourced from canonical HCI readings.

## Project Structure

```
HEPTA/
├── data/
│   ├── knowledge_base.xlsx      # 50+ training entries
│   ├── test_questions.xlsx      # 9 test questions (Breadth / Methods / Depth)
│   └── rubric.xlsx              # 7-dimension scoring rubric
├── src/
│   ├── n_gain.py                # N-gain model & HEPTA-Index calculator
│   ├── evaluator.py             # Question/rubric loading, LLM judge scoring
│   └── visualizer.py            # Radar chart & bar chart generation
├── tests/
│   └── mock_data.py             # Local mock tests (no API calls)
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Jake-yutong/HEPTA.git
cd HEPTA
pip install -r requirements.txt
```

## Usage

### 1. Initialise and validate data files

```bash
python main.py init
```

### 2. Run a test phase

```bash
# Pre-test (baseline, no teaching intervention)
python main.py run --model <model_name> --phase pre

# Post-test (after teaching intervention)
python main.py run --model <model_name> --phase post

# Local mock run (no LLM API required)
python main.py run --model mock-agent --phase pre --mock
python main.py run --model mock-agent --phase post --mock
```

### 3. Compute N-gain and generate visualisations

```bash
python main.py calc --model <model_name>
```

This produces:
- `outputs/<model>_summary.json` — per-dimension N-gain and composite HEPTA-Index
- `outputs/<model>_radar.png` — seven-axis radar chart
- `outputs/<model>_bar.png` — weighted contribution bar chart

## Testing

```bash
python -m pytest tests/mock_data.py -v
```

All tests run locally without external API calls.

## Evaluation Pipeline

```
Input: test_questions.xlsx (9 questions)
           │
           ▼
  [Student LLM]  pre-test answers  →  [Judge LLM]  →  S_pre per dimension
           │
  [Teacher Agent]  generates HCI pedagogical guidance
           │
  [Student LLM]  post-test answers  →  [Judge LLM]  →  S_post per dimension
           │
           ▼
  N-gain = (S_post − S_pre) / (100 − S_pre)
  HEPTA-Index = Σ (weight_d × N-gain_d)
```

## License

MIT
