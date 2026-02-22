"""
Multi-Agent Financial QA Optimization Engine
Streamlit Dashboard — refactored for production clarity
"""
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

REPORT_PATH = "./experiments/report.json"

st.set_page_config(
    page_title="Multi-Agent Financial QA Optimization Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal shared style ──────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 0;
  }
  .metric-label { font-size: 11px; color: #64748B; text-transform: uppercase;
                  letter-spacing: .06em; margin-bottom: 4px; }
  .metric-value { font-size: 26px; font-weight: 700; color: #0F172A; line-height: 1.1; }
  .metric-delta { font-size: 12px; margin-top: 4px; }
  .delta-pos { color: #16A34A; } .delta-neg { color: #DC2626; }
  .section-caption { color: #64748B; font-size: 13px; margin-top: -10px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_report():
    try:
        with open(REPORT_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        # ── MOCK DATA (swap for real report.json) ────────────────────────────
        return {
            "meta": {
                "dataset": "TAT-QA",
                "total_n": 400,
                "n_per_type": 100,
                "models": ["openai-gpt-oss-20b", "openai-gpt-oss-120b"],
            },
            "runs": [
                # 120B
                {"model": "openai-gpt-oss-120b", "strategy": "zero_shot",
                 "overall_f1": 0.740, "overall_em": 0.550, "weighted_f1": 0.692,
                 "total_cost_usd": 0.087, "avg_latency_s": 2.69, "cost_per_correct": 0.0039,
                 "by_type": {"arithmetic": {"f1": 0.500}, "span": {"f1": 0.811},
                             "multi-span": {"f1": 0.649}, "count": {"f1": 1.000}},
                 "fail_by_type": {"arithmetic": 50, "span": 40, "multi-span": 90, "count": 0}},
                {"model": "openai-gpt-oss-120b", "strategy": "chain_of_thought",
                 "overall_f1": 0.661, "overall_em": 0.425, "weighted_f1": 0.615,
                 "total_cost_usd": 0.100, "avg_latency_s": 4.72, "cost_per_correct": 0.0058,
                 "by_type": {"arithmetic": {"f1": 0.433}, "span": {"f1": 0.689},
                             "multi-span": {"f1": 0.520}, "count": {"f1": 1.000}},
                 "fail_by_type": {"arithmetic": 70, "span": 70, "multi-span": 90, "count": 0}},
                {"model": "openai-gpt-oss-120b", "strategy": "router",
                 "overall_f1": 0.716, "overall_em": 0.475, "weighted_f1": 0.660,
                 "total_cost_usd": 0.130, "avg_latency_s": 5.10, "cost_per_correct": 0.0066,
                 "by_type": {"arithmetic": {"f1": 0.433}, "span": {"f1": 0.768},
                             "multi-span": {"f1": 0.664}, "count": {"f1": 1.000}},
                 "fail_by_type": {"arithmetic": 70, "span": 50, "multi-span": 90, "count": 0}},
                # 20B
                {"model": "openai-gpt-oss-20b", "strategy": "zero_shot",
                 "overall_f1": 0.680, "overall_em": 0.490, "weighted_f1": 0.630,
                 "total_cost_usd": 0.041, "avg_latency_s": 1.82, "cost_per_correct": 0.0022,
                 "by_type": {"arithmetic": {"f1": 0.440}, "span": {"f1": 0.760},
                             "multi-span": {"f1": 0.580}, "count": {"f1": 0.940}},
                 "fail_by_type": {"arithmetic": 60, "span": 45, "multi-span": 92, "count": 6}},
                {"model": "openai-gpt-oss-20b", "strategy": "chain_of_thought",
                 "overall_f1": 0.610, "overall_em": 0.380, "weighted_f1": 0.560,
                 "total_cost_usd": 0.052, "avg_latency_s": 3.50, "cost_per_correct": 0.0031,
                 "by_type": {"arithmetic": {"f1": 0.380}, "span": {"f1": 0.640},
                             "multi-span": {"f1": 0.460}, "count": {"f1": 0.960}},
                 "fail_by_type": {"arithmetic": 75, "span": 72, "multi-span": 93, "count": 4}},
                {"model": "openai-gpt-oss-20b", "strategy": "router",
                 "overall_f1": 0.650, "overall_em": 0.420, "weighted_f1": 0.600,
                 "total_cost_usd": 0.068, "avg_latency_s": 3.90, "cost_per_correct": 0.0038,
                 "by_type": {"arithmetic": {"f1": 0.400}, "span": {"f1": 0.710},
                             "multi-span": {"f1": 0.610}, "count": {"f1": 0.950}},
                 "fail_by_type": {"arithmetic": 72, "span": 52, "multi-span": 91, "count": 5}},
            ],
            "failure_patterns": [
                {"name": "Scale/Unit Contamination", "affected": "arithmetic",
                 "frequency": "high",
                 "description": "Model appends 'million' or '%' to numeric answers copied from table headers",
                 "fix": "Post-process: strip units from numeric predictions"},
                {"name": "Span Truncation", "affected": "span", "frequency": "medium",
                 "description": "Model paraphrases instead of exact-quoting source text",
                 "fix": "Prompt: instruct model to copy verbatim from source"},
                {"name": "Multi-span Format Mismatch", "affected": "multi-span",
                 "frequency": "high",
                 "description": "Correct values but wrong order, case, or delimiter → EM=0",
                 "fix": "Normalize: sort spans, lowercase, strip punctuation before eval"},
                {"name": "Count Unit Leakage", "affected": "count", "frequency": "low",
                 "description": "Model adds unit to count answers ('2 years' vs '2')",
                 "fix": "Post-process: extract first numeric token from count predictions"},
            ],
        }

report = load_report()
runs   = report["runs"]
meta   = report["meta"]
df     = pd.DataFrame(runs)

COLORS = {
    "zero_shot":        "#2563EB",
    "chain_of_thought": "#DC2626",
    "router":           "#16A34A",
}
LABELS = {
    "zero_shot":        "Zero-shot",
    "chain_of_thought": "Chain-of-Thought",
    "router":           "Router Agent",
}
TYPES = ["arithmetic", "span", "multi-span", "count"]

# ── Router Decision Distribution — from real log files ───────────────────────
# Aggregated from router_openai-gpt-oss-120b_log.json and router_openai-gpt-oss-20b_log.json
# Format: per answer_type, count of questions routed to each sub-strategy
_LOG_120B = [
    ("arithmetic",  "chain_of_thought", 8), ("arithmetic",  "zero_shot", 2),
    ("span",        "zero_shot",        5), ("span",        "chain_of_thought", 5),
    ("multi-span",  "zero_shot",        9), ("multi-span",  "chain_of_thought", 1),
    ("count",       "zero_shot",        1), ("count",       "chain_of_thought", 9),
]
_LOG_20B = [
    ("arithmetic",  "chain_of_thought", 8), ("arithmetic",  "zero_shot", 2),
    ("span",        "zero_shot",        7), ("span",        "chain_of_thought", 3),
    ("multi-span",  "zero_shot",       10), ("multi-span",  "chain_of_thought", 0),
    ("count",       "zero_shot",        2), ("count",       "chain_of_thought", 8),
]

def _build_router_df(log):
    rows = {}
    for atype, strategy, cnt in log:
        if atype not in rows:
            rows[atype] = {"Zero-shot": 0, "Chain-of-Thought": 0}
        key = "Zero-shot" if strategy == "zero_shot" else "Chain-of-Thought"
        rows[atype][key] += cnt
    return pd.DataFrame(rows).T.rename_axis("Answer Type").reset_index()

ROUTER_DFS = {
    "openai-gpt-oss-120b": _build_router_df(_LOG_120B),
    "openai-gpt-oss-20b":  _build_router_df(_LOG_20B),
}

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Multi-Agent Financial QA Optimization Engine")
st.caption(
    f"Dataset: {meta['dataset']} · "
    f"Sample: {meta['total_n']} questions ({meta['n_per_type']} per type) · "
    f"Models: {', '.join(meta['models'])}"
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 0 — Performance Summary
# ══════════════════════════════════════════════════════════════════════════════
model_tab_label = st.radio(
    "Model scope", ["120B Model", "20B Model"], horizontal=True, label_visibility="collapsed"
)
sel_model = "openai-gpt-oss-120b" if "120B" in model_tab_label else "openai-gpt-oss-20b"
df_m = df[df["model"] == sel_model].copy()

best   = df_m.loc[df_m["overall_f1"].idxmax()]
cot    = df_m[df_m["strategy"] == "chain_of_thought"].iloc[0]
delta_f1     = best["overall_f1"] - cot["overall_f1"]
delta_lat    = (best["avg_latency_s"] - cot["avg_latency_s"]) / cot["avg_latency_s"] * 100
delta_cost   = (best["total_cost_usd"] - cot["total_cost_usd"]) / cot["total_cost_usd"] * 100

st.subheader("Performance Summary")

def card(label, value, delta_html=""):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-delta">' + delta_html + '</div>' if delta_html else ''}
    </div>"""

def delta_html(val, fmt, invert=False):
    pos = val > 0
    if invert: pos = not pos
    cls = "delta-pos" if pos else "delta-neg"
    sign = "+" if val > 0 else ""
    return f'<span class="{cls}">{sign}{fmt.format(val)} vs CoT</span>'

c = st.columns(5)
c[0].markdown(card("Best Strategy", LABELS[best["strategy"]]), unsafe_allow_html=True)
c[1].markdown(card("F1 Score", f"{best['overall_f1']:.3f}",
                   delta_html(delta_f1, "{:.3f}")), unsafe_allow_html=True)
c[2].markdown(card("Avg Latency", f"{best['avg_latency_s']:.2f}s",
                   delta_html(delta_lat, "{:.1f}%", invert=True)), unsafe_allow_html=True)
c[3].markdown(card("Cost / Correct", f"${best['cost_per_correct']:.4f}",
                   delta_html(delta_cost, "{:.1f}%", invert=True)), unsafe_allow_html=True)
c[4].markdown(card("Risk-Weighted F1", f"{best['weighted_f1']:.3f}",
                   '<span style="color:#64748B;font-size:11px">arith ×2 penalty</span>'),
              unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Strategy Comparison  +  Pareto Frontier
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("① Strategy Comparison & Pareto Frontier")
st.markdown('<p class="section-caption">Pareto-optimal frontier: maximize F1 while minimizing latency and cost.</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

# 1a  F1 & EM grouped bar
with col1:
    fig = go.Figure()
    for metric, opacity, suffix in [("overall_f1", 1.0, "F1"), ("overall_em", 0.35, "EM")]:
        fig.add_trace(go.Bar(
            name=suffix,
            x=[LABELS[s] for s in df_m["strategy"]],
            y=df_m[metric],
            marker_color=[COLORS[s] for s in df_m["strategy"]],
            opacity=opacity,
            text=[f"{v:.3f}" for v in df_m[metric]],
            textposition="outside",
        ))
    fig.update_layout(
        title=f"F1 & EM by Strategy ({sel_model.split('-')[-1].upper()})",
        barmode="group", yaxis_range=[0, 0.95], height=360,
        yaxis_title="Score", legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# 1b  Pareto frontier
with col2:
    fig2 = go.Figure()

    # Sort by latency to draw frontier line
    df_sorted = df_m.sort_values("avg_latency_s")
    # Pareto frontier: keep points where no other point has lower latency AND lower cost
    pareto_pts = []
    min_cost = float("inf")
    for _, row in df_sorted.iterrows():
        if row["total_cost_usd"] < min_cost:
            min_cost = row["total_cost_usd"]
            pareto_pts.append(row)
    df_pareto = pd.DataFrame(pareto_pts)

    # Background scatter: all strategies (both models dimmed)
    for _, row in df.iterrows():
        is_sel = row["model"] == sel_model
        fig2.add_trace(go.Scatter(
            x=[row["avg_latency_s"]], y=[row["total_cost_usd"]],
            mode="markers",
            marker=dict(
                size=row["overall_f1"] * 55,
                color=COLORS[row["strategy"]],
                opacity=0.85 if is_sel else 0.20,
                line=dict(width=1.5 if is_sel else 0, color="white"),
            ),
            name=f"{LABELS[row['strategy']]} ({row['model'].split('-')[-1].upper()})",
            text=f"{LABELS[row['strategy']]}<br>F1={row['overall_f1']:.3f}<br>"
                 f"${row['total_cost_usd']:.3f} | {row['avg_latency_s']:.2f}s",
            hoverinfo="text",
            showlegend=is_sel,
        ))

    # Pareto frontier line
    if len(df_pareto) > 1:
        fig2.add_trace(go.Scatter(
            x=df_pareto["avg_latency_s"], y=df_pareto["total_cost_usd"],
            mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            name="Pareto Frontier",
        ))

    fig2.update_layout(
        title="Pareto Frontier: Latency × Cost (bubble = F1)",
        xaxis_title="Avg Latency (s)", yaxis_title="Total Cost (USD)",
        height=360, legend=dict(orientation="h", y=-0.25, font_size=11),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Per-type Breakdown
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("② Per-type Performance Breakdown")
st.markdown('<p class="section-caption">Arithmetic hardest (50% fail). Multi-span dominated by format mismatch (90% fail rate).</p>',
            unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    strats = df_m["strategy"].tolist()
    matrix = []
    for s in strats:
        row_data = df_m[df_m["strategy"] == s].iloc[0]["by_type"]
        matrix.append([row_data.get(t, {}).get("f1", 0) for t in TYPES])

    fig3 = go.Figure(go.Heatmap(
        z=matrix, x=TYPES,
        y=[LABELS[s] for s in strats],
        colorscale="Blues",
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        texttemplate="%{text}", showscale=True, zmin=0, zmax=1,
    ))
    fig3.update_layout(title=f"F1 Heatmap by Answer Type ({sel_model.split('-')[-1].upper()})",
                       height=280, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fail_data = []
    for s in strats:
        fail_by_type = df_m[df_m["strategy"] == s].iloc[0]["fail_by_type"]
        n = meta["n_per_type"]
        for t in TYPES:
            fail_data.append({
                "Strategy": LABELS[s], "Type": t,
                "Failure Rate": fail_by_type.get(t, 0) / n,
            })
    df_fail = pd.DataFrame(fail_data)
    fig4 = px.bar(
        df_fail, x="Type", y="Failure Rate", color="Strategy", barmode="group",
        color_discrete_map={LABELS[s]: COLORS[s] for s in COLORS},
        text_auto=".0%",
        title=f"Failure Rate by Answer Type ({sel_model.split('-')[-1].upper()})",
    )
    fig4.update_layout(height=280, yaxis_range=[0, 1.15],
                       plot_bgcolor="white", paper_bgcolor="white",
                       legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Router Decision Distribution  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("③ Router Decision Distribution")
st.markdown('<p class="section-caption">'
            'How the Router Agent dispatches each question type across sub-strategies. '
            'CoT dominates arithmetic; Zero-shot handles span efficiently.</p>',
            unsafe_allow_html=True)

col5, col6 = st.columns([1.1, 0.9])

with col5:
    rd = ROUTER_DFS[sel_model].set_index("Answer Type")
    rd_pct = rd.div(rd.sum(axis=1), axis=0) * 100

    fig5 = go.Figure()
    strat_color_map = {"Zero-shot": COLORS["zero_shot"], "Chain-of-Thought": COLORS["chain_of_thought"]}
    for strat in rd_pct.columns:
        fig5.add_trace(go.Bar(
            name=strat,
            x=rd_pct.index,
            y=rd_pct[strat],
            marker_color=strat_color_map[strat],
            text=[f"{v:.0f}%" for v in rd_pct[strat]],
            textposition="inside",
            customdata=rd[strat],
            hovertemplate="%{x}<br>%{customdata} questions → %{y:.0f}%<extra>" + strat + "</extra>",
        ))
    fig5.update_layout(
        title="Router Sub-Strategy Dispatch (% of questions per type)",
        barmode="stack", yaxis_title="Dispatch %", yaxis_range=[0, 110],
        height=320, legend=dict(orientation="h", y=1.15),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    # Heatmap version — shows absolute counts
    z_vals = rd.values.tolist()
    fig6 = go.Figure(go.Heatmap(
        z=z_vals,
        x=rd.columns.tolist(),
        y=rd.index.tolist(),
        colorscale="Greens",
        text=[[str(v) for v in row] for row in z_vals],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig6.update_layout(
        title="Dispatch Count Heatmap",
        height=320,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Failure Taxonomy
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("④ Failure Taxonomy")
st.markdown('<p class="section-caption">4 systematic failure patterns identified across all strategies and models.</p>',
            unsafe_allow_html=True)

patterns = report["failure_patterns"]
freq_colors = {"high": "#FEE2E2", "medium": "#FEF9C3", "low": "#DCFCE7"}
freq_border = {"high": "#FCA5A5", "medium": "#FDE68A", "low": "#86EFAC"}

cols = st.columns(len(patterns))
for col, p in zip(cols, patterns):
    bg  = freq_colors.get(p["frequency"], "#F3F4F6")
    bdr = freq_border.get(p["frequency"], "#E5E7EB")
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border:1px solid {bdr}; padding:14px 16px;
                    border-radius:8px; height:230px; position:relative;">
          <div style="font-weight:700; font-size:14px; margin-bottom:4px">{p['name']}</div>
          <div style="font-size:11px; color:#6B7280; margin-bottom:8px">
            Affects: <b>{p['affected']}</b> · Frequency: <b>{p['frequency']}</b>
          </div>
          <hr style="margin:8px 0; border-color:{bdr}">
          <div style="font-size:12px; color:#374151; line-height:1.5">{p['description']}</div>
          <div style="font-size:12px; margin-top:10px; color:#1D4ED8">
            💡 <i>{p['fix']}</i>
          </div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Cost-Performance Trade-off
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⑤ Cost-Performance Trade-off")
st.markdown('<p class="section-caption">'
            'Risk-Weighted F1 penalises arithmetic errors 2× — higher stakes in financial reporting.</p>',
            unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    fig7 = go.Figure()
    for metric, opacity, name in [
        ("overall_f1", 1.0,  "Standard F1"),
        ("weighted_f1", 0.45, "Weighted F1 (arith ×2)"),
    ]:
        fig7.add_trace(go.Bar(
            name=name,
            x=[LABELS[s] for s in df_m["strategy"]],
            y=df_m[metric],
            marker_color=[COLORS[s] for s in df_m["strategy"]],
            opacity=opacity,
            text=[f"{v:.3f}" for v in df_m[metric]],
            textposition="outside",
        ))
    fig7.update_layout(
        title=f"Standard vs Risk-Weighted F1 ({sel_model.split('-')[-1].upper()})",
        barmode="group", yaxis_range=[0, 0.95], height=340,
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig7, use_container_width=True)

with col8:
    fig8 = go.Figure(go.Bar(
        x=[LABELS[s] for s in df_m["strategy"]],
        y=df_m["cost_per_correct"],
        marker_color=[COLORS[s] for s in df_m["strategy"]],
        text=[f"${v:.4f}" for v in df_m["cost_per_correct"]],
        textposition="outside",
    ))
    fig8.update_layout(
        title=f"Cost per Correct Answer ({sel_model.split('-')[-1].upper()})",
        yaxis_title="USD per correct answer",
        height=340,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig8, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Multi-Agent Financial QA Optimization Engine · "
    "TAT-QA dataset (ACL 2021) · "
    "Strategies: Zero-shot, Chain-of-Thought, Router Agent · "
    "Tracked with MLflow"
)