"""Calling the Claude API on complaint narratives (section 4 of index.ipynb).

The API key is read from the .env file (ANTHROPIC_API_KEY) and is never printed.
Every answer is saved to a CSV in Data/, so the notebook re-runs without a key
and without paying again.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
from dotenv import load_dotenv

# The two Claude models we compare: a small low-cost one and a larger, stronger one
CLAUDE_MODELS = {"Haiku 5.5": "claude-haiku-5-5", "Sonnet 5.5": "claude-sonnet-5-5"}

# USD per million (input, output) tokens, from the model pages (November 2026):
# https://platform.claude.com/docs/en/models/haiku-5-5/overview  (rate for prompts under 100,000 tokens)
# https://www.anthropic.com/claude/sonnet
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}

PARALLEL_REQUESTS = 8   # requests sent at the same time; lower this if the API reports rate limits

PRODUCTS = [
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Credit card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Vehicle loan or lease",
    "Student loan",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Debt or credit management",
]
_CATEGORY_LIST = "\n".join(f"- {p}" for p in PRODUCTS)

# Prompt 1: zero-shot classification (compared with our trained model)
CLASSIFY_PROMPT = """You route consumer financial complaints to the right product team.

Product categories:
{categories}

Complaint:
<complaint>
{complaint}
</complaint>

Which ONE product category is this complaint about? Reply with the category name exactly as written above, and nothing else."""

# Prompt 2: error analysis on complaints our model got wrong.
# Claude is NOT told the consumer's label or the model's prediction, so it cannot just agree with either.
EXPLAIN_PROMPT = """Read the following consumer financial complaint and identify the most appropriate financial product category from the 11 categories provided.

Product categories:
{categories}

Complaint:
<complaint>
{complaint}
</complaint>

Answer on three lines:
CATEGORY: the category name exactly as written above
AMBIGUOUS: YES if the complaint could reasonably belong to more than one category, otherwise NO
CLUES: one or two sentences naming the words or context in the complaint that point to this category"""


def get_client():
    """Create an API client using the key in .env (stops with a clear message if it is missing).

    max_retries: if the API is busy or rate-limits us, the SDK waits and tries again.
    """
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not found. Add it to the .env file in the repo root.")
    return anthropic.Anthropic(max_retries=6)


def ask_claude(client, model, prompt, max_tokens=100):
    """Send one prompt; return Claude's reply text and the tokens used.

    Note: since API version 2026-10-21 the API no longer accepts `temperature`, so
    answers can vary slightly between runs. That is why every answer is saved.
    """
    response = client.messages.create(model=model, max_tokens=max_tokens,
                                      messages=[{"role": "user", "content": prompt}])
    return response.content[0].text.strip(), response.usage.input_tokens, response.usage.output_tokens


def check_access(models=CLAUDE_MODELS):
    """Send one tiny request to each model to confirm the key works with it (costs well under a cent)."""
    client = get_client()
    rows = []
    for name, model in models.items():
        try:
            reply, _, _ = ask_claude(client, model, "Reply with the word OK.", max_tokens=5)
            rows.append({"model": name, "model_id": model, "works": True, "message": reply})
        except Exception as error:                      # e.g. model not enabled for this key
            rows.append({"model": name, "model_id": model, "works": False, "message": str(error)[:200]})
    return pd.DataFrame(rows).set_index("model")


def _run_parallel(task, rows):
    """Run `task` on every row, several requests at a time; results come back in the original order."""
    rows = list(rows)
    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as pool:
        results = list(pool.map(task, rows))
    return results


def _match_category(answer):
    """Return the official category name, or flag an answer that is not on the list."""
    answer = answer.strip().strip(".").strip()
    for product in PRODUCTS:
        if answer.lower() == product.lower():
            return product
    return f"UNMATCHED: {answer}"


def _fields(answer):
    """Turn 'KEY: value' lines into a dictionary."""
    return {line.split(":", 1)[0].strip().upper(): line.split(":", 1)[1].strip()
            for line in answer.splitlines() if ":" in line}


def classify_sample(sample, model, text_col="Consumer complaint narrative"):
    """Zero-shot classification of every complaint in `sample` with one Claude model."""
    client = get_client()

    def task(row):
        prompt = CLASSIFY_PROMPT.format(categories=_CATEGORY_LIST, complaint=row[text_col])
        try:
            answer, tokens_in, tokens_out = ask_claude(client, model, prompt, max_tokens=30)
            product = _match_category(answer)
        except Exception as error:                      # keep going if one request fails
            product, tokens_in, tokens_out = f"ERROR: {str(error)[:100]}", 0, 0
        return {"Complaint ID": row["Complaint ID"], "true_product": row["Product"],
                "claude_product": product, "model": model,
                "input_tokens": tokens_in, "output_tokens": tokens_out}

    return pd.DataFrame(_run_parallel(task, (row for _, row in sample.iterrows())))


def explain_errors(errors, model, text_col="Consumer complaint narrative"):
    """For each misclassified complaint: Claude's category, an ambiguity flag and its clues.

    `errors` needs the columns Complaint ID, Product (the consumer's label) and model_product.
    """
    client = get_client()

    def task(row):
        prompt = EXPLAIN_PROMPT.format(categories=_CATEGORY_LIST, complaint=row[text_col])
        try:
            answer, tokens_in, tokens_out = ask_claude(client, model, prompt, max_tokens=150)
        except Exception as error:
            answer, tokens_in, tokens_out = f"CATEGORY: ERROR\nCLUES: {str(error)[:100]}", 0, 0
        fields = _fields(answer)
        return {"Complaint ID": row["Complaint ID"], "consumer_product": row["Product"],
                "model_product": row["model_product"],
                "claude_product": _match_category(fields.get("CATEGORY", "")),
                "claude_ambiguous": fields.get("AMBIGUOUS", "?").upper(),
                "claude_clues": fields.get("CLUES", answer), "model": model,
                "input_tokens": tokens_in, "output_tokens": tokens_out}

    return pd.DataFrame(_run_parallel(task, (row for _, row in errors.iterrows())))


def error_verdict(row):
    """Compare Claude's reading with the consumer's label and our model's prediction."""
    if row["claude_product"] == row["consumer_product"]:
        return "Model error"            # Claude agrees with the consumer: our model got it wrong
    if row["claude_product"] == row["model_product"]:
        return "Possible mislabel"      # Claude agrees with our model: the consumer's choice may be off
    return "Unclear"                    # Claude picks a third category


def load_or_run(path, run):
    """Load saved Claude results from `path`; if there are none, call `run()` and save them.

    Without saved results AND without an API key, print a message and return None,
    so the rest of the notebook still runs.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"No saved results at {path} and no ANTHROPIC_API_KEY in .env: skipping this step.")
        return None
    results = run()
    results.to_csv(path, index=False)
    return results


def cost_usd(input_tokens, output_tokens, model):
    """Cost of a number of tokens for one model, at the prices above."""
    price_in, price_out = PRICES[model]
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
