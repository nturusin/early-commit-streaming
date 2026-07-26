"""Does early commit actually work on your provider, and what does it buy?

The technique rests on one assumption nobody guarantees: that the provider emits
fields in the order your schema declares them, and streams them incrementally.
This probe checks that against a real model and reports:

    order held   did the verdict fields arrive first, as declared?
    agreed       did the early verdict match the completed object?
    time to act  when did the verdict become structurally final?

Run it:

    pip install -e '.[probe]'
    gcloud auth application-default login
    python3 probe.py --project YOUR_PROJECT --runs 20

Gemini on Vertex AI is implemented because it is the stack the article's numbers
came from. Another provider means writing one more `stream_*` function: an async
generator of (seconds_since_request, text_fragment) events. Nothing else changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from early_commit import CATEGORY_ENUM, AbortEarlyCommit, Decision, DecisionParser

# (seconds since the request was sent, fragment of generated text)
Event = tuple[float, str]


# --- what we ask for --------------------------------------------------------

# The verdict fields come first. That ordering is the whole intervention.
SCHEMA: dict[str, Any] = {
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
SCHEMA_FIELD_ORDER: list[str] = list(SCHEMA["properties"])

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
    """One run, seen from the outside."""

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
        if self.verdict_at is None:
            return 0.0
        return self.complete_at - self.verdict_at

    @property
    def fraction_saved(self) -> float:
        if self.complete_at <= 0:
            return 0.0
        return self.seconds_saved / self.complete_at


def analyse(events: Iterable[Event]) -> Report:
    """Replay timed events and report what early commit would have bought.

    A pure function, so the reporting is testable without a network.
    """
    parser = DecisionParser()
    fragments: list[str] = []
    commit: Optional[tuple[Decision, float]] = None  # (verdict, seconds)
    aborted: Optional[str] = None
    complete_at = 0.0

    for elapsed, fragment in events:
        complete_at = elapsed
        fragments.append(fragment)

        if commit is not None or aborted is not None:
            continue  # already decided; keep reading to time the full response

        try:
            verdict = parser.feed(fragment)
        except AbortEarlyCommit as exc:
            aborted = str(exc)
        else:
            if verdict is not None:
                commit = (verdict, elapsed)

    observed_order, final_verdict = _parse_completed("".join(fragments))

    return Report(
        observed_order=observed_order,
        early_verdict=commit[0] if commit else None,
        final_verdict=final_verdict,
        verdict_at=commit[1] if commit else None,
        complete_at=complete_at,
        aborted=aborted,
    )


def _parse_completed(body: str) -> tuple[list[str], Optional[Decision]]:
    """Field order and verdict of the finished response, if it parsed at all."""
    try:
        parsed = json.loads(body)
        return list(parsed), Decision(parsed["category"], float(parsed["confidence"]))
    except (ValueError, KeyError, TypeError):
        return [], None


# --- printing ---------------------------------------------------------------


def _rows_to_text(rows: Sequence[tuple[str, object]]) -> str:
    """Render (label, value) pairs as an aligned block."""
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label:<{width}}   {value}" for label, value in rows)


def format_report(report: Report, model: str) -> str:
    """The single-run view."""
    rows: list[tuple[str, object]] = [
        ("field order declared", ", ".join(SCHEMA_FIELD_ORDER)),
        ("field order observed", ", ".join(report.observed_order) or "(not valid JSON)"),
        ("order held", "yes" if report.order_held else "NO"),
    ]

    if report.aborted:
        rows.append(("early commit", f"aborted: {report.aborted}"))
    elif report.early_verdict is None:
        rows.append(("early commit", "never became structurally final"))
    else:
        verdict = report.early_verdict
        rows += [
            ("early verdict", f"{verdict.category} @ {verdict.confidence}"),
            ("agreed with final", "yes" if report.agreed else "NO - DO NOT SHIP THIS"),
            ("time to act", f"{report.verdict_at:.2f}s"),
            ("time to complete", f"{report.complete_at:.2f}s"),
            (
                "removed from path",
                f"{report.seconds_saved:.2f}s ({report.fraction_saved * 100:.0f}%)",
            ),
        ]

    return f"\nmodel: {model}\n\n{_rows_to_text(rows)}\n"


def format_progress(run: int, total: int, report: Report) -> str:
    """One line per run, for multi-run mode."""
    verdict = "ok" if report.order_held and report.agreed else "PROBLEM"
    return (
        f"run {run}/{total}: {verdict:<7} "
        f"complete {report.complete_at:.2f}s  "
        f"saved {report.fraction_saved * 100:.0f}%"
    )


def format_summary(reports: Sequence[Report]) -> str:
    """The aggregate view. The two counts matter more than the timings."""
    committed = [r.verdict_at for r in reports if r.verdict_at is not None]
    rows: list[tuple[str, object]] = [
        ("runs", len(reports)),
        ("order held", f"{sum(r.order_held for r in reports)}/{len(reports)}"),
        ("early verdict agreed", f"{sum(r.agreed for r in reports)}/{len(reports)}"),
    ]
    if committed:
        rows.append(("median time to act", f"{statistics.median(committed):.2f}s"))
    rows.append(
        ("median time to complete", f"{statistics.median(r.complete_at for r in reports):.2f}s")
    )
    return "\n" + _rows_to_text(rows) + "\n"


# --- cli --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure early commit against a real provider.")
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--location", default="europe-west2")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--transaction", default=DEFAULT_TRANSACTION)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


async def run_once(args: argparse.Namespace) -> Report:
    events = [
        event
        async for event in stream_gemini_vertex(
            args.project, args.location, args.model, args.transaction
        )
    ]
    return analyse(events)


async def main() -> None:
    args = parse_args()
    reports: list[Report] = []

    for run in range(1, args.runs + 1):
        report = await run_once(args)
        reports.append(report)
        if args.runs == 1:
            print(format_report(report, args.model))
        else:
            print(format_progress(run, args.runs, report))

    if args.runs > 1:
        print(format_summary(reports))


if __name__ == "__main__":
    asyncio.run(main())
