"""
Router module for TAT-QA agent evaluation.
Implements a routing strategy that classifies questions into "zero_shot" or "chain_of
thought" categories, allowing the agent to choose the most efficient answering method.
"""
import json

ROUTER_PROMPT = """You are an AI system that routes financial questions to the most efficient answering strategy.

Analyze the question and classify it:
- "zero_shot": question asks to directly extract a value, name, or count from the table/text. No calculation needed.
- "chain_of_thought": question requires arithmetic (add, subtract, multiply, divide, percentage change) or multi-step reasoning.

Respond ONLY with valid JSON, no explanation:
{{"strategy": "zero_shot" | "chain_of_thought", "reason": "<one sentence>"}}

QUESTION: {question}"""


def route(question_text, call_model):
    """
    Returns:
        {
          "strategy": "zero_shot" | "chain_of_thought",
          "reason": str,
          "raw": str   # LLM原始输出，用于debug
        }
    """
    prompt = ROUTER_PROMPT.format(question=question_text)
    result = call_model(prompt)
    raw = result["text"].strip()

    try:
        # clean the output to extract JSON, allowing for some formatting issues (like code blocks)
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        strategy = parsed.get("strategy", "chain_of_thought")
        reason   = parsed.get("reason", "")

        # validate strategy field to avoid misclassification due to parsing errors
        if strategy not in ("zero_shot", "chain_of_thought"):
            strategy = "chain_of_thought"

        return {
            "strategy": strategy,
            "reason": reason,
            "raw": raw,
            "tokens": result["usage"].total_tokens
        }

    except (json.JSONDecodeError, AttributeError):
        # parsing failed, default to chain_of_thought and log the raw output for analysis
        return {
            "strategy": "chain_of_thought",
            "reason": "parse_failed",
            "raw": raw,
            "tokens": result.get("usage", {}).total_tokens if hasattr(result.get("usage", {}), "total_tokens") else 0
        }