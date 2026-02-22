from dotenv import load_dotenv
load_dotenv()

import time
import json
import mlflow
import os
from groq import Groq
from .strategies import zero_shot, chain_of_thought
from .strategies.router import route
from .evaluator import evaluate_predictions
from .utils import parse_answer

# ---------- LLM CLIENT ----------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_model(prompt, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {
        "text": response.choices[0].message.content,
        "usage": response.usage
    }

# ---------- CONFIG ----------
DATA_PATH  = "./dataset_raw/tatqa_dataset_dev.json"
N_PER_TYPE = 10   # each type question count for evaluation (keep small for quick testing)

MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

STRATEGIES = ["zero_shot", "chain_of_thought", "router"]

# risk adjustment of different question types: arithmetic questions are more important and should be weighted higher in the final evaluation, since they are more challenging and more indicative of reasoning ability.
TYPE_WEIGHTS = {
    "arithmetic": 2.0,
    "span":       1.0,
    "multi-span": 1.0,
    "count":      1.0,
}

# ---------- DATA ----------
def load_sample(path, n_per_type):
    with open(path) as f:
        data = json.load(f)
    buckets = {"arithmetic": [], "span": [], "multi-span": [], "count": []}
    for entry in data:
        for q in entry["questions"]:
            t = q.get("answer_type")
            if t in buckets and len(buckets[t]) < n_per_type:
                buckets[t].append((entry, q))
        if all(len(v) == n_per_type for v in buckets.values()):
            break
    samples = []
    for items in buckets.values():
        samples.extend(items)
    return samples

# ---------- WEIGHTED F1 ----------
def compute_weighted_f1(metrics):
    """
    based on per-type F1 and counts, compute a weighted average F1 that reflects the higher importance of arithmetic questions.
    """
    total_weight, weighted_f1 = 0.0, 0.0
    for t, m in metrics["by_type"].items():
        w = TYPE_WEIGHTS.get(t, 1.0)
        weighted_f1  += w * m["f1"] * m["n"]
        total_weight += w * m["n"]
    return weighted_f1 / total_weight if total_weight > 0 else 0.0

# ---------- STRATEGY RUNNER ----------
def run_strategy(samples, strategy_name, model_name):
    predictions  = []
    total_tokens = 0
    total_latency = 0.0
    router_log   = []

    for i, (entry, q) in enumerate(samples):
        try:
            fn = lambda p: call_model(p, model_name)

            if strategy_name == "router":
                # Step 1: Router decides which strategy to use based on the question
                routing = route(q["question"], fn)
                chosen  = routing["strategy"]
                total_tokens += routing["tokens"]
                router_log.append({
                    "uid":         q["uid"],
                    "answer_type": q["answer_type"],
                    "routed_to":   chosen,
                    "reason":      routing["reason"],
                })
                strat = zero_shot if chosen == "zero_shot" else chain_of_thought
            else:
                strat = zero_shot if strategy_name == "zero_shot" else chain_of_thought

            # ── time count ──────────────────────────
            t_start = time.time()
            result  = strat.run(entry, q, fn)
            latency = time.time() - t_start
            total_latency += latency
            # ──────────────────────────────────

            text  = result["text"]
            total_tokens += result["usage"].total_tokens
            pred  = parse_answer(text)

        except Exception as e:
            print(f"  [!] q{i} error: {e}")
            pred    = ""
            latency = 0.0

        predictions.append({
            "uid":         q["uid"],
            "answer_type": q["answer_type"],
            "prediction":  pred,
            "gold":        q["answer"],
            "derivation":  q.get("derivation", ""),
            "scale":       q.get("scale", ""),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{strategy_name}] {i+1}/{len(samples)}")

    avg_latency = total_latency / len(samples) if samples else 0
    return predictions, total_tokens, avg_latency, router_log

# ---------- MAIN ----------
def main():
    print("Loading data...")
    run_log = []
    samples = load_sample(DATA_PATH, N_PER_TYPE)
    print(f"Samples: {len(samples)} ({N_PER_TYPE} per type)\n")

    os.makedirs("experiments", exist_ok=True)
    mlflow.set_experiment("TAT-QA-Strategy-Comparison")

    for model_name in MODELS:
        for strategy_name in STRATEGIES:
            print(f"Running [{strategy_name}] | [{model_name}]")

            with mlflow.start_run(run_name=f"{strategy_name}__{model_name}"):
                mlflow.log_param("strategy",   strategy_name)
                mlflow.log_param("model",      model_name)
                mlflow.log_param("n_per_type", N_PER_TYPE)

                # 1. run strategy and get predictions + tokens/latency
                predictions, total_tokens, avg_latency, router_log = run_strategy(
                    samples, strategy_name, model_name
                )

                # 2. keep predictions and router log (if any) for failure analysis, and also log to MLflow
                safe_model = model_name.replace("/", "-")
                pred_path  = f"experiments/{strategy_name}_{safe_model}_predictions.json"
                with open(pred_path, "w") as f:
                    json.dump(predictions, f, indent=2)
                mlflow.log_artifact(pred_path)

                if router_log:
                    log_path = f"experiments/router_{safe_model}_log.json"
                    with open(log_path, "w") as f:
                        json.dump(router_log, f, indent=2)
                    mlflow.log_artifact(log_path)

                # 3. evaluate predictions, compute weighted F1, and cost analysis
                metrics      = evaluate_predictions(predictions)
                weighted_f1  = compute_weighted_f1(metrics)

                cost_per_1m  = 3.0
                total_cost   = total_tokens * cost_per_1m / 1_000_000
                n_correct    = sum(1 for d in metrics["details"] if d["em"] == 1.0)
                cost_per_correct = total_cost / n_correct if n_correct > 0 else float("inf")

                # 4. print summary
                print(f"  Overall     EM={metrics['overall']['em']:.3f}  F1={metrics['overall']['f1']:.3f}")
                print(f"  Weighted F1={weighted_f1:.3f}  (arithmetic×2 penalty)")
                print(f"  Tokens={total_tokens}  Cost=${total_cost:.4f}  "
                      f"Cost/correct=${cost_per_correct:.4f}  Avg latency={avg_latency:.2f}s")
                for t, m in metrics["by_type"].items():
                    w = TYPE_WEIGHTS.get(t, 1.0)
                    print(f"  {t:12s} EM={m['em']:.3f}  F1={m['f1']:.3f}  n={m['n']}  weight={w}")

                # 5. MLflow
                mlflow.log_metric("overall_em",       metrics["overall"]["em"])
                mlflow.log_metric("overall_f1",       metrics["overall"]["f1"])
                mlflow.log_metric("weighted_f1",      weighted_f1)
                mlflow.log_metric("total_tokens",     total_tokens)
                mlflow.log_metric("total_cost_usd",   total_cost)
                mlflow.log_metric("cost_per_correct", cost_per_correct)
                mlflow.log_metric("avg_latency_s",    avg_latency)
                for t, m in metrics["by_type"].items():
                    mlflow.log_metric(f"{t}_em", m["em"])
                    mlflow.log_metric(f"{t}_f1", m["f1"])

                # 6. run_log
                run_log.append({
                    "strategy":         strategy_name,
                    "model":            model_name,
                    "weighted_f1":      weighted_f1,
                    "total_tokens":     total_tokens,
                    "total_cost_usd":   total_cost,
                    "cost_per_correct": cost_per_correct,
                    "avg_latency_s":    avg_latency,
                })

            print()

    # 7. save run_log for reporting
    run_log_path = "experiments/run_log.json"
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Run log saved to {run_log_path}")
    print("Done. Run `mlflow ui` to see results.")

if __name__ == "__main__":
    main()