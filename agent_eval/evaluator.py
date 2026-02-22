"""
Evaluator module for scoring predictions against gold answers.
"""
import string
from collections import Counter

print("EVALUATOR LOADING")

# ── Scale normalize ───────────────────────────────
SCALE_MAP = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 1, "": 1, None: 1}

def normalize_number(val, scale=""):
    try:
        v = float(str(val).replace(",", "").replace("%", "").strip())
        return v * SCALE_MAP.get(scale, 1)
    except:
        return None

def normalize_text(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

# ── Token F1 ─────────────────────────────────────
def token_f1(pred, gold):
    p_tok = normalize_text(pred).split()
    g_tok = normalize_text(gold).split()
    common = Counter(p_tok) & Counter(g_tok)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(p_tok) if p_tok else 0
    r = n / len(g_tok) if g_tok else 0
    return 2 * p * r / (p + r)

# ── Single prediction scoring ─────────────────────
def score_one(pred_str, gold, scale=""):
    if isinstance(gold, list):
        gold_joined = " ".join(str(g) for g in gold)
        em = 1.0 if normalize_text(pred_str) == normalize_text(gold_joined) else 0.0
        return em, token_f1(pred_str, gold_joined)

    pred_num = normalize_number(pred_str, scale)
    gold_num = normalize_number(gold, scale)
    if pred_num is not None and gold_num is not None:
        em = 1.0 if abs(pred_num - gold_num) < 1e-4 else 0.0
        return em, em  # for numeric answers, EM and F1 are the same

    em = 1.0 if normalize_text(pred_str) == normalize_text(str(gold)) else 0.0
    return em, token_f1(pred_str, str(gold))

# ── Main evaluate function ────────────────────────
def evaluate_predictions(predictions):
    """
    returns {:
    {
      "overall": {"em", "f1", "n"},
      "by_type": {"arithmetic": {"em", "f1", "n"}, ...},
      "details":  each prediction's em/f1 results (for failure analysis)
    }
    """
    type_results = {}
    details = []

    # score each prediction, accumulate results by answer_type
    for p in predictions:
        t = p["answer_type"]
        em, f1 = score_one(p["prediction"], p["gold"], p.get("scale", ""))

        details.append({**p, "em": em, "f1": f1})

        if t not in type_results:
            type_results[t] = {"em": [], "f1": []}
        type_results[t]["em"].append(em)
        type_results[t]["f1"].append(f1)

    all_em, all_f1 = [], []
    by_type = {}
    # compute average EM/F1 per type and overall
    for t, vals in type_results.items():
        avg_em = sum(vals["em"]) / len(vals["em"])
        avg_f1 = sum(vals["f1"]) / len(vals["f1"])
        by_type[t] = {"em": avg_em, "f1": avg_f1, "n": len(vals["em"])}
        all_em.extend(vals["em"])
        all_f1.extend(vals["f1"])

    return {
        "overall": {"em": sum(all_em)/len(all_em), "f1": sum(all_f1)/len(all_f1), "n": len(all_em)},
        "by_type": by_type,
        "details": details,
    }