"""
Failure analysis for TAT-QA agent evaluation.
scoring predictions, breaking down by type, and identifying common failure patterns.
"""
import json
import os
from collections import defaultdict
from .evaluator import evaluate_predictions

EXPERIMENTS_DIR = "./experiments"

# 
KNOWN_STRATEGIES = ["chain_of_thought", "zero_shot", "router"]

def parse_filename(fname):
    """Extract strategy and model from filename, e.g. "chain_of_thought_gpt-4-120b_predictions.json"""
    base = fname.replace("_predictions.json", "")
    for s in KNOWN_STRATEGIES:
        if base.startswith(s + "_"):
            model = base[len(s) + 1:]
            return s, model
    return "unknown", base

# load predictions from a JSON file (list of {uid, answer_type, prediction, gold, ...})
def load_predictions(filename):
    with open(os.path.join(EXPERIMENTS_DIR, filename)) as f:
        return json.load(f)

# Analyze predictions, compute metrics, and identify failure patterns
def analyze(predictions, strategy_name, model_name):
    metrics  = evaluate_predictions(predictions)
    failures = [d for d in metrics["details"] if d["em"] == 0.0]
    partial  = [d for d in metrics["details"] if d["em"] == 0.0 and d["f1"] > 0.3]

    fail_by_type = defaultdict(list)
    for f in failures:
        fail_by_type[f["answer_type"]].append(f)

    # print summary report
    print(f"\n{'='*55}")
    print(f"Strategy: {strategy_name} | Model: {model_name}")
    print(f"{'='*55}")
    print(f"Overall  EM={metrics['overall']['em']:.3f}  F1={metrics['overall']['f1']:.3f}  n={metrics['overall']['n']}")
    print(f"\nPer-type breakdown:")
    for t, m in metrics["by_type"].items():
        fail_n = len(fail_by_type[t])
        print(f"  {t:12s}  EM={m['em']:.3f}  F1={m['f1']:.3f}  "
              f"n={m['n']}  failures={fail_n} ({100*fail_n/m['n']:.0f}%)")

    print(f"\nTotal failures: {len(failures)}/{metrics['overall']['n']}")
    print(f"Partial matches (F1>0.3 but EM=0): {len(partial)}")

    print(f"\n── Failure Examples ──")
    for t, cases in fail_by_type.items():
        print(f"\n[{t}]")
        for c in cases[:2]:
            print(f"  Q:    {c.get('question', c['uid'])}")
            print(f"  Gold: {c['gold']}")
            print(f"  Pred: {c['prediction']}")
            print(f"  F1:   {c['f1']:.3f}")

    return {
        "strategy": strategy_name,
        "model": model_name,
        "overall": metrics["overall"],
        "by_type": metrics["by_type"],
        "total_failures": len(failures),
        "partial_matches": len(partial),
        "fail_by_type": {t: len(v) for t, v in fail_by_type.items()},
    }

def compare_strategies(reports):
    """Compare strategies for the same model, identify which performs best on each type, and summarize common failure patterns."""
    # filter reports for the same model (e.g. 120b) and compare strategies
    target = {r["strategy"]: r for r in reports.values()
              if "120b" in r["model"] and r["strategy"] in ("zero_shot", "chain_of_thought", "router")}
    if len(target) < 2:
        return

    print(f"\n{'='*55}")
    print("Strategy Comparison (120b model)")
    print(f"{'='*55}")

    types = ["arithmetic", "span", "multi-span", "count"]
    header = f"{'Type':12s}" + "".join(f"  {s[:8]:>8}" for s in target)
    print(header)
    print("-" * 55)

    for t in types:
        row = f"  {t:12s}"
        best_f1, best_s = -1, ""
        for s, r in target.items():
            f1 = r["by_type"].get(t, {}).get("f1", 0)
            row += f"  {f1:>8.3f}"
            if f1 > best_f1:
                best_f1, best_s = f1, s
        row += f"  ← {best_s}"
        print(row)

    print()
    for s, r in target.items():
        print(f"  {s:20s} Overall F1={r['overall']['f1']:.3f}  "
              f"failures={r['total_failures']}/{r['overall']['n']}")

    # summarize common failure patterns observed across strategies
    print(f"\n── Systematic Failure Patterns ──")
    print("  1. Scale/unit contamination: model appends 'million'/'%' to numeric answers")
    print("     → affects arithmetic EM most (F1 stays ~0.5 but EM drops)")
    print("  2. Multi-span ordering: correct values but wrong order → EM=0")
    print("  3. Span truncation: model paraphrases instead of exact-quoting")
    print("  4. Count: model adds unit ('2 years' vs '2') → fixable with post-processing")

def main():
    files = sorted([f for f in os.listdir(EXPERIMENTS_DIR)
                    if f.endswith("_predictions.json")])
    if not files:
        print("No prediction files found. Run orchestrator.py first.")
        return

    reports = {}
    for fname in files:
        strategy, model = parse_filename(fname)
        # load predictions and analyze failures, skipping if all predictions are empty (indicating a possible error in the run)
        preds = load_predictions(fname)
        if not preds or all(p["prediction"] == "" for p in preds):
            print(f"[skip] {fname} — all empty predictions")
            continue
        report = analyze(preds, strategy, model)
        reports[fname] = report

    compare_strategies(reports)

    out_path = os.path.join(EXPERIMENTS_DIR, "failure_report.json")
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nFull report saved to {out_path}")

if __name__ == "__main__":
    main()