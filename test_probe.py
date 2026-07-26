"""Tests for the probe's reporting, using recorded-looking event lists.

No network: `analyse` is a pure function over (elapsed, fragment) events, which
is exactly what a provider adapter produces.

Run:  python3 test_probe.py
"""

from __future__ import annotations

from early_commit import Decision
from probe import analyse, format_report

SCHEMA_ORDER = ["category", "confidence", "customer_friendly_explanation"]


def verdict_first_events():
    return [
        (0.30, '{"category": "04_me'),
        (0.34, 'als", "confidence": 8'),
        (0.38, '7, "customer_friendly_explanation": "A wor'),
        (0.90, 'king lunch."}'),
    ]


def essay_first_events():
    return [
        (0.30, '{"customer_friendly_explanation": "A wor'),
        (0.80, 'king lunch.", "category": "04_meals", "confidence": 87}'),
    ]


def test_reports_the_saving_when_the_verdict_comes_first():
    report = analyse(verdict_first_events(), SCHEMA_ORDER)

    assert report.early_verdict == Decision("04_meals", 87.0)
    assert report.verdict_at == 0.38
    assert report.complete_at == 0.90
    assert round(report.seconds_saved, 2) == 0.52
    assert 0.57 < report.fraction_saved < 0.59


def test_confirms_order_and_agreement():
    report = analyse(verdict_first_events(), SCHEMA_ORDER)

    assert report.observed_order == SCHEMA_ORDER
    assert report.order_held
    assert report.agreed


def test_flags_a_provider_that_reorders_fields():
    """The failure this probe exists to catch."""
    report = analyse(essay_first_events(), SCHEMA_ORDER)

    assert not report.order_held
    assert report.observed_order[0] == "customer_friendly_explanation"
    assert report.early_verdict is None
    assert report.aborted is not None
    assert report.fraction_saved == 0.0


def test_verdict_position_is_reported_in_characters():
    report = analyse(verdict_first_events(), SCHEMA_ORDER)

    assert report.verdict_after_chars is not None
    assert report.verdict_after_chars < report.total_chars


def test_unparseable_response_does_not_crash_the_report():
    report = analyse([(0.2, '{"category": "04_meals", "confid')], SCHEMA_ORDER)

    assert report.observed_order == []
    assert report.final_verdict is None
    assert not report.agreed


def test_report_renders():
    for events in (verdict_first_events(), essay_first_events()):
        text = format_report(analyse(events, SCHEMA_ORDER), model="test-model")
        assert "field order" in text and "order held" in text


def test_parser_enum_matches_the_schema():
    """The parser's enum and schema.json must not drift apart.

    If the schema offers a category the parser does not know, a perfectly good
    response gets rejected and the probe reports a failure that is not there.
    """
    import json
    from pathlib import Path

    from early_commit import CATEGORY_ENUM

    schema = json.loads(Path(__file__).with_name("schema.json").read_text())
    assert set(schema["properties"]["category"]["enum"]) == set(CATEGORY_ENUM)


if __name__ == "__main__":
    passed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
