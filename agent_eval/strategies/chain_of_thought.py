# Chain of Thought strategy for TAT-QA agent evaluation.
# table/text + question -> step-by-step reasoning + final answer
def build_prompt(entry, q):
    table = entry.get("table", {}).get("table", [])
    paragraphs = [
        p.get("text", "") if isinstance(p, dict) else str(p)
        for p in entry.get("paragraphs", [])
    ]

    table_str = "\n".join([" | ".join(row) for row in table])
    para_str = "\n".join(paragraphs)

    return f"""You are a financial analyst. Answer the question using ONLY the table and text below.
Think step by step, then return your final answer in this exact format:

REASONING: <your step-by-step reasoning>
ANSWER: <final answer only, number or phrase>

TABLE:
{table_str}

TEXT:
{para_str}

QUESTION: {q["question"]}"""


def parse_answer(text):
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip().split("\n")[-1].strip()


def run(entry, q, call_model):
    prompt = build_prompt(entry, q)
    result = call_model(prompt)
    return result