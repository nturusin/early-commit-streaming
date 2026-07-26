"""Simulated stream: where does the commit point land in each field order?

This is a MECHANISM demo, not a benchmark. It replays a synthetic response at a
fixed token rate so you can see how field order moves time-to-act. Real numbers
depend on your model, provider, and the ratio of verdict tokens to essay tokens.

Run:  python3 demo.py
"""

from __future__ import annotations

import json
import time

from early_commit import AbortEarlyCommit, DecisionParser

TIME_TO_FIRST_TOKEN = 0.30  # seconds of fixed latency before any output
TOKENS_PER_SECOND = 300.0
CHARS_PER_TOKEN = 4  # rough; only the ratio matters here

ESSAY = {
    "customer_friendly_explanation": (
        "Lunch was a working meal at the cafe near the office, so it is "
        "recorded against meals and entertainment for the trading period."
    ),
    "internal_explanation": (
        "Counterparty and amount are consistent with a working meal. "
        "Allowable where the expense is incurred wholly for the trade."
    ),
    "citation": "Internal bookkeeping guidance, meals and subsistence.",
}
VERDICT = {"category": "04_meals", "confidence": 87}

SCHEMA_A = {**{"category": VERDICT["category"]}, **ESSAY, "confidence": VERDICT["confidence"]}
SCHEMA_B = {**VERDICT, **ESSAY}


def stream(payload: dict, chunk_chars: int = 7):
    """Yield (elapsed_seconds, chunk) as if the response were being generated."""
    text = json.dumps(payload)
    per_char = 1.0 / (TOKENS_PER_SECOND * CHARS_PER_TOKEN)
    for i in range(0, len(text), chunk_chars):
        chunk = text[i : i + chunk_chars]
        elapsed = TIME_TO_FIRST_TOKEN + (i + len(chunk)) * per_char
        yield elapsed, chunk


def run(label: str, payload: dict) -> None:
    parser = DecisionParser()
    committed_at = None
    aborted = None
    complete_at = 0.0

    for elapsed, chunk in stream(payload):
        complete_at = elapsed
        if committed_at is not None or aborted is not None:
            continue
        try:
            if parser.feed(chunk) is not None:
                committed_at = elapsed
        except AbortEarlyCommit as exc:
            # No immutable prefix exists in this field order. The safe fallback
            # is to wait for the complete response, which is what the original
            # implementation did by default.
            aborted = str(exc)

    total_tokens = len(json.dumps(payload)) / CHARS_PER_TOKEN
    acted_at = complete_at if committed_at is None else committed_at
    saved = (complete_at - acted_at) / complete_at * 100

    print(f"{label}")
    print(f"  tokens in response        ~{total_tokens:.0f}")
    if aborted:
        print(f"  early commit              not possible ({aborted})")
    print(f"  time to act                {acted_at:.2f}s")
    print(f"  time to complete           {complete_at:.2f}s")
    print(f"  removed from critical path {saved:.0f}%\n")


if __name__ == "__main__":
    print(
        f"\nsimulation: {TOKENS_PER_SECOND:.0f} tok/s, "
        f"{TIME_TO_FIRST_TOKEN:.2f}s time-to-first-token\n"
    )
    run("schema A - verdict split around the essay (confidence last)", SCHEMA_A)
    run("schema B - verdict first", SCHEMA_B)
    print("Same bytes, same token count. Only the field order changed.")
