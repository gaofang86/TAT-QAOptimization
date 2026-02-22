# Zero-shot prompting strategy for TAT-QA agent evaluation.
# table/text + question -> direct answer (no explanation, no chain of thought)
def build_prompt(entry, q):
    table = entry.get("table", {}).get("table", [])
    paragraphs = [
        p.get("text", "") if isinstance(p, dict) else str(p)
        for p in entry.get("paragraphs", [])
    ]

    table_str = "\n".join([" | ".join(row) for row in table])
    para_str = "\n".join(paragraphs)

    return f"""You are a financial analyst. Answer the question using ONLY the table and text below.
Return ONLY the answer value (number or phrase). No explanation.

TABLE:
{table_str}

TEXT:
{para_str}

QUESTION: {q["question"]}
ANSWER:"""


def run(entry, q, call_model):
    prompt = build_prompt(entry, q)
    result = call_model(prompt)  
    return result
