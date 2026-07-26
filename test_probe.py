"""Tests for the probe's reporting, using recorded-looking event lists.

No network: `analyse` is a pure function over (elapsed, fragment) events, which
is exactly what the provider adapter produces.

Run:  python3 test_probe.py     (or: pytest)
"""

from __future__ import annotations

from early_commit import Decision
from probe import SCHEMA, SCHEMA_FIELD_ORDER, analyse, format_report


def verdict_first_events():
    """A well-behaved stream: verdict fields first, then the essay."""
    return [
        (0.30, '{"category": "04_me'),
        (0.34, 'als", "confidence": 8'),
        (0.38, '7, "customer_friendly_explanation": "A working lunch.",'),
        (0.70, ' "internal_explanation": "Allowable.",'),
        (0.90, ' "citation": "Guidance 1.2"}'),
    ]


def essay_first_events():
    """The failure this probe exists to catch: the provider reordered fields."""
    return [
        (0.30, '{"customer_friendly_explanation": "A working lunch.",'),
        (0.60, ' "internal_explanation": "Allowable.", "citation": "Guidance 1.2",'),
        (0.80, ' "category": "04_meals", "confidence": 87}'),
    ]


def test_reports_the_saving_when_the_verdict_comes_first():
    report = analyse(verdict_first_events())

    assert report.early_verdict == Decision("04_meals", 87.0)
    assert report.verdict_at == 0.38
    assert report.complete_at == 0.90
    assert round(report.seconds_saved, 2) == 0.52
    assert 0.57 < report.fraction_saved < 0.59


def test_confirms_order_and_agreement():
    report = analyse(verdict_first_events())

    assert report.observed_order == SCHEMA_FIELD_ORDER
    assert report.order_held
    assert report.agreed


def test_flags_a_provider_that_reorders_fields():
    report = analyse(essay_first_events())

    assert not report.order_held
    assert report.early_verdict is None
    assert report.aborted is not None
    assert report.fraction_saved == 0.0


def test_unparseable_response_does_not_crash_the_report():
    report = analyse([(0.2, '{"category": "04_meals", "confid')])

    assert report.observed_order == []
    assert report.final_verdict is None
    assert not report.agreed


def test_report_renders_for_both_outcomes():
    for events in (verdict_first_events(), essay_first_events()):
        text = format_report(analyse(events), model="test-model")
        assert "order held" in text


def test_schema_declares_the_verdict_fields_first():
    """If this ever stops being true, the technique stops working."""
    assert SCHEMA_FIELD_ORDER[:2] == ["category", "confidence"]
    assert SCHEMA["properties"]["category"]["enum"], "category must be a closed set"


if __name__ == "__main__":
    passed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
