"""Does early commit actually work on your provider, and what does it buy?

The technique rests on one assumption nobody guarantees: that the provider emits
fields in the order your schema declares them, and streams them incrementally.
This probe checks that against a real model and reports:

    order held   did the verdict fields arrive first, as declared?
    agreed       did the early verdict match the completed object?
    time to act  when did the verdict become structurally final?

Run it:

    pip install google-genai
    gcloud auth application-default login
    python3 probe.py --project YOUR_PROJECT --runs 20

Gemini on Vertex AI is implemented because it is the stack the article's numbers
came from. Another provider is one function: an async generator of
(seconds_since_request, text_fragment) events. Nothing else changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Iterable, Optional

from early_commit import AbortEarlyCommit, CATEGORY_ENUM, Decision, DecisionParser

Event = tuple[float, str]

# The verdict fields come first. That ordering is the whole intervention.
SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": sorted(CATEGORY_ENUM)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "customer_friendly_explanation": {"type": "string"},
        "internal_explanation": {"type": "string"},
        "citation": {"type": "string"},
    },
    "required": [
        "category",
        "confidence",
        "customer_friendly_explanation",
        "internal_explanation",
        "citation",
    ],
}
SCHEMA_FIELD_ORDER = list(SCHEMA["properties"])

PROMPT = """You categorise bank transactions for a small business.

Return the category and a confidence score from 0 to 100, then explain the
decision for the customer, then explain it for an internal accountant, then
cite the guidance you relied on. Write two or three sentences per explanation.

Transaction: {transaction}
"""

DEFAULT_TRANSACTION = "Card payment, 41.98 GBP, online marketplace"


# --- reading a real stream --------------------------------------------------


async def stream_gemini_vertex(
    project: str,
    location: str,
    model: str,
    transaction: str,
) -> AsyncIterator[Event]:
    """Stream a structured response from Gemini on Vertex AI, timing each fragment.

    Uses application default credentials. The clock starts before the request is
    sent, so the first event includes time-to-first-token.
    """
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(vertexai=True, project=project, location=location)
    config = GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA)

    started = time.perf_counter()
    stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=PROMPT.format(transaction=transaction),
        config=config,
    )
    async for chunk in stream:
        if chunk.text:
            yield time.perf_counter() - started, chunk.text


# --- what the stream tells us -----------------------------------------------


@dataclass(frozen=True)
class Report:
    observed_order: list[str]
    early_verdict: Optional[Decision]
    final_verdict: Optional[Decision]
    verdict_at: Optional[float]
    complete_at: float
    aborted: Optional[str]

    @property
    def order_held(self) -> bool:
        return self.observed_order == SCHEMA_FIELD_ORDER

    @property
    def agreed(self) -> bool:
        return self.early_verdict is not None and self.early_verdict == self.final_verdict

    @property
    def seconds_saved(self) -> float:
        return 0.0 if self.verdict_at is None else self.complete_at - self.verdict_at

    @property
    def fraction_saved(self) -> float:
        if self.verdict_at is None or self.complete_at <= 0:
            return 0.0
        return self.seconds_saved / self.complete_at


def analyse(events: Iterable[Event]) -> Report:
    """Replay timed stream events and report what early commit would have bought.

    Pure function, so the reporting is testable without a network.
    """
    parser = DecisionParser()
    early: Optional[Decision] = None
    aborted: Optional[str] = None
    verdict_at: Optional[float] = None
    fragments: list[str] = []
    complete_at = 0.0

    for elapsed, fragment in events:
        complete_at = elapsed
        fragments.append(fragment)

        if early is not None or aborted is not None:
            continue
        try:
            early = parser.feed(fragment)
        except AbortEarlyCommit as exc:
            aborted = str(exc)
            continue
        if early is not None:
            verdict_at = elapsed

    observed_order: list[str] = []
    final: Optional[Decision] = None
    try:
        body = json.loads("".join(fragments))
        observed_order = list(body)
        final = Decision(body["category"], float(body["confidence"]))
    except (ValueError, KeyError, TypeError):
        pass

    return Report(observed_order, early, final, verdict_at, complete_at, aborted)


def format_report(report: Report, model: str) -> str:
    lines = [
        f"\nmodel: {model}\n",
        f"  field order declared   {SCHEMA_FIELD_ORDER}",
        f"  field order observed   {report.observed_order or '(response was not valid JSON)'}",
        f"  order held             {'yes' if report.order_held else 'NO'}\n",
    ]

    if report.aborted:
        lines.append(f"  early commit           aborted: {report.aborted}")
    elif report.early_verdict is None:
        lines.append("  early commit           never became structurally final")
    else:
        lines.append(f"  early verdict          {report.early_verdict.category} @ {report.early_verdict.confidence}")
        lines.append(f"  agreed with final      {'yes' if report.agreed else 'NO - DO NOT SHIP THIS'}\n")
        lines.append(f"  time to act            {report.verdict_at:.2f}s")
        lines.append(f"  time to complete       {report.complete_at:.2f}s")
        lines.append(f"  removed from path      {report.seconds_saved:.2f}s ({report.fraction_saved * 100:.0f}%)")

    return "\n".join(lines) + "\n"


# --- cli --------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure early commit against a real provider.")
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--location", default="europe-west2")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--transaction", default=DEFAULT_TRANSACTION)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    reports = []
    for run in range(args.runs):
        events = [
            event
            async for event in stream_gemini_vertex(
                args.project, args.location, args.model, args.transaction
            )
        ]
        report = analyse(events)
        reports.append(report)

        if args.runs == 1:
            print(format_report(report, args.model))
        else:
            ok = report.order_held and report.agreed
            print(
                f"run {run + 1}/{args.runs}: {'ok' if ok else 'PROBLEM'}  "
                f"complete {report.complete_at:.2f}s  saved {report.fraction_saved * 100:.0f}%"
            )

    if args.runs > 1:
        acted = [r.verdict_at for r in reports if r.verdict_at is not None]
        print(f"\n  order held             {sum(r.order_held for r in reports)}/{len(reports)}")
        print(f"  early verdict agreed   {sum(r.agreed for r in reports)}/{len(reports)}")
        if acted:
            print(f"  median time to act     {statistics.median(acted):.2f}s")
        print(f"  median time to complete {statistics.median(r.complete_at for r in reports):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
