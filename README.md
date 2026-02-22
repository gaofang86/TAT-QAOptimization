# Multi-Agent Financial QA Optimization Engine

[![HuggingFace Space](https://huggingface.co/spaces/fang8/Multi-Agent_Financial_QA_Optimization_Engine)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)

A multi-agent evaluation framework for financial question answering on the [TAT-QA](https://arxiv.org/abs/2105.07624) dataset (ACL 2021). Compares three inference strategies — Zero-shot, Chain-of-Thought, and a Router Agent — across two model scales, with full cost, latency, and failure analysis.

**→ [Live Dashboard](https://huggingface.co/spaces/fang8/Multi-Agent_Financial_QA_Optimization_Engine)**

![Dashboard Preview](docs/title.png)

---

## What This Project Does

TAT-QA questions span four answer types: arithmetic, span extraction, multi-span, and counting. A naive one-size-fits-all prompting strategy leaves performance on the table. This project:

1. **Benchmarks** three strategies (Zero-shot, CoT, Router Agent) on 40 TAT-QA questions across `openai-gpt-oss-120b` and `openai-gpt-oss-20b`
2. **Routes** each question to the most appropriate sub-strategy based on predicted answer type
3. **Analyzes** failure patterns systematically (scale contamination, span truncation, format mismatch, unit leakage)
4. **Visualizes** cost-performance trade-offs including a Pareto frontier and risk-weighted F1

---

## Key Results

| Strategy | F1 | EM | Cost/Correct | Avg Latency |
|---|---|---|---|---|
| **Zero-shot** | **0.740** | **0.550** | **$0.0039** | **2.69s** |
| Router Agent | 0.716 | 0.475 | $0.0069 | 4.82s |
| Chain-of-Thought | 0.661 | 0.425 | $0.0058 | 4.74s |

Zero-shot outperforms CoT by **+7.9% F1** while running **43% faster** and costing **13% less**. The Router Agent improves multi-span F1 by **+14.5pp** over CoT despite higher total cost, justified when format-sensitive accuracy matters.

---

## Router Dispatch Logic

The Router Agent classifies each question's answer type before inference and dispatches accordingly:

| Answer Type | → Zero-shot | → CoT |
|---|---|---|
| arithmetic | 20% | 80% |
| span | 50% | 50% |
| multi-span | 90% | 10% |
| count | 10% | 90% |

Arithmetic and count questions are almost always routed to CoT (calculation-heavy). Multi-span questions default to Zero-shot (direct extraction), yet still suffer from format mismatch — indicating the bottleneck is **post-processing normalization**, not strategy selection.

---

## Failure Taxonomy

| Pattern | Affects | Frequency | Fix |
|---|---|---|---|
| Scale/Unit Contamination | arithmetic | High | Strip units in post-processing |
| Span Truncation | span | Medium | Prompt: copy verbatim from source |
| Multi-span Format Mismatch | multi-span | High | Normalize: sort, lowercase, strip punctuation |
| Count Unit Leakage | count | Low | Extract first numeric token |

---

## Project Structure

```
├── agent_eval/
│   ├── strategies.py          # Zero-shot, CoT, Router agent implementations
│   ├── evaluator.py            # F1 / EM scoring
│   ├── reporting.py       # Aggregates run logs → report.json
│   └── orchestrator.py          # Experiment runner
├── experiments/
│   ├── report.json                          # Aggregated results (all runs)
│   ├── router_openai-gpt-oss-120b_log.json  # Router dispatch decisions
│   └── router_openai-gpt-oss-20b_log.json
├── dashboard.py           # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/gaofang86/TAT-QAOptimization.git
cd TAT-QA
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Run Experiments

```bash
# Run all strategies and models
python -m agent_eval.orchestrator

# Generate report.json
python -m agent_eval.reporting
```

---

## Tech Stack

- **Evaluation**: TAT-QA dataset, F1/EM scoring, MLflow tracking
- **Agents**: Zero-shot, Chain-of-Thought, Router with answer-type classification
- **Dashboard**: Streamlit, Plotly (Pareto frontier, heatmaps, failure taxonomy)
- **Models**: `openai-gpt-oss-120b`, `openai-gpt-oss-20b`

---

*Based on [TAT-QA](https://github.com/NExTplusplus/TAT-QA?tab=readme-ov-file)*