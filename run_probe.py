"""Run the probe against a real provider.

    pip install google-genai
    gcloud auth application-default login
    python3 run_probe.py --project YOUR_PROJECT

Repeat with --runs to see the spread; a single call tells you whether the
technique is viable, not what its median saving is.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics

from probe import analyse, format_report
from providers import SCHEMA_FIELD_ORDER, stream_gemini_vertex


async def one_run(args: argparse.Namespace):
    events = [
        event
        async for event in stream_gemini_vertex(
            project=args.project,
            location=args.location,
            model=args.model,
            transaction=args.transaction,
        )
    ]
    return analyse(events, expected_order=SCHEMA_FIELD_ORDER)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--location", default="europe-west2")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--transaction", default=None)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    if args.transaction is None:
        from providers import DEFAULT_TRANSACTION

        args.transaction = DEFAULT_TRANSACTION

    reports = []
    for run in range(args.runs):
        report = await one_run(args)
        reports.append(report)
        if args.runs == 1:
            print(format_report(report, args.model))
        else:
            status = "ok" if report.order_held and report.agreed else "PROBLEM"
            print(
                f"run {run + 1}/{args.runs}: {status}  "
                f"act {report.verdict_at or float('nan'):.2f}s  "
                f"complete {report.complete_at:.2f}s  "
                f"saved {report.fraction_saved * 100:.0f}%"
            )

    if args.runs > 1:
        acted = [r.verdict_at for r in reports if r.verdict_at is not None]
        complete = [r.complete_at for r in reports]
        print(f"\n  runs                   {len(reports)}")
        print(f"  order held             {sum(r.order_held for r in reports)}/{len(reports)}")
        print(f"  early verdict agreed   {sum(r.agreed for r in reports)}/{len(reports)}")
        if acted:
            print(f"  median time to act     {statistics.median(acted):.2f}s")
        print(f"  median time to complete{statistics.median(complete):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
