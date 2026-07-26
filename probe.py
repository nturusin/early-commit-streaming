"""Measure whether early commit works on your provider, and what it buys.

The technique rests on one empirical assumption: that the provider emits fields
in the order your schema declares them, and streams them incrementally. That is
not guaranteed anywhere — it has to be checked against the exact model and API
path you ship.

This probe answers four questions for a real stream:

1. Did the fields arrive in the order the schema declared?
2. When did the verdict become structurally final?
3. When did the whole response arrive?
4. Did the early verdict match the completed object?

`analyse` is a pure function over (elapsed_seconds, text_fragment) events, so
the reporting is testable offline. Talking to a provider is a separate, small
adapter — see `providers.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional

from early_commit import AbortEarlyCommit, Decision, DecisionParser

Event = tuple[float, str]  # (seconds since request start, text fragment)


@dataclass(frozen=True)
class ProbeReport:
    expected_order: list[str]
    observed_order: list[str]
    early_verdict: Optional[Decision]
    final_verdict: Optional[Decision]
    verdict_at: Optional[float]
    complete_at: float
    verdict_after_chars: Optional[int]
    total_chars: int
    aborted: Optional[str]

    @property
    def order_held(self) -> bool:
        """Did the verdict fields arrive before everything else, as declared?"""
        return self.observed_order[: len(self.expected_order)] == self.expected_order

    @property
    def agreed(self) -> bool:
        """Did the early verdict match the completed object?"""
        return self.early_verdict is not None and self.early_verdict == self.final_verdict

    @property
    def seconds_saved(self) -> float:
        return 0.0 if self.verdict_at is None else self.complete_at - self.verdict_at

    @property
    def fraction_saved(self) -> float:
        if self.verdict_at is None or self.complete_at <= 0:
            return 0.0
        return self.seconds_saved / self.complete_at


def analyse(events: Iterable[Event], expected_order: list[str]) -> ProbeReport:
    """Replay a stream's events and report what early commit would have bought."""
    parser = DecisionParser()
    early: Optional[Decision] = None
    aborted: Optional[str] = None
    verdict_at: Optional[float] = None
    verdict_after_chars: Optional[int] = None
    text: list[str] = []
    complete_at = 0.0

    for elapsed, fragment in events:
        complete_at = elapsed
        text.append(fragment)

        if early is not None or aborted is not None:
            continue
        try:
            early = parser.feed(fragment)
        except AbortEarlyCommit as exc:
            aborted = str(exc)
            continue
        if early is not None:
            verdict_at = elapsed
            verdict_after_chars = len("".join(text))

    body = "".join(text)
    observed_order: list[str] = []
    final: Optional[Decision] = None
    try:
        parsed = json.loads(body)
        observed_order = list(parsed.keys())
        final = Decision(parsed["category"], float(parsed["confidence"]))
    except (ValueError, KeyError, TypeError):
        pass

    return ProbeReport(
        expected_order=expected_order,
        observed_order=observed_order,
        early_verdict=early,
        final_verdict=final,
        verdict_at=verdict_at,
        complete_at=complete_at,
        verdict_after_chars=verdict_after_chars,
        total_chars=len(body),
        aborted=aborted,
    )


def format_report(report: ProbeReport, model: str) -> str:
    lines = [f"\nmodel: {model}", ""]

    lines.append(f"  field order declared   {report.expected_order}")
    lines.append(f"  field order observed   {report.observed_order or '(response was not valid JSON)'}")
    lines.append(f"  order held             {'yes' if report.order_held else 'NO'}")
    lines.append("")

    if report.aborted:
        lines.append(f"  early commit           aborted: {report.aborted}")
    elif report.early_verdict is None:
        lines.append("  early commit           never became structurally final")
    else:
        lines.append(f"  early verdict          {report.early_verdict.category} @ {report.early_verdict.confidence}")
        lines.append(f"  final verdict          {report.final_verdict.category} @ {report.final_verdict.confidence}"
                     if report.final_verdict else "  final verdict          (unparsed)")
        lines.append(f"  agreed                 {'yes' if report.agreed else 'NO — DO NOT SHIP THIS'}")
        lines.append("")
        lines.append(f"  time to act            {report.verdict_at:.2f}s"
                     f"  (after {report.verdict_after_chars} of {report.total_chars} chars)")
        lines.append(f"  time to complete       {report.complete_at:.2f}s")
        lines.append(f"  removed from path      {report.seconds_saved:.2f}s"
                     f"  ({report.fraction_saved * 100:.0f}%)")

    lines.append("")
    return "\n".join(lines)
