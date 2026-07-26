"""Structural guarantees of the early-commit parser.

Run:  python3 -m pytest test_early_commit.py -q
      (or: python3 test_early_commit.py  for a dependency-free run)
"""

from __future__ import annotations

from early_commit import AbortEarlyCommit, Decision, DecisionParser


def feed_all(chunks):
    """Feed chunks until a Decision appears; return (decision, chunks_consumed)."""
    parser = DecisionParser()
    for i, chunk in enumerate(chunks, start=1):
        decision = parser.feed(chunk)
        if decision is not None:
            return decision, i
    return None, len(chunks)


def test_commits_once_the_number_is_terminated():
    decision, consumed = feed_all(
        ['{"category": "04_meals", "confidence": 87', ", "]
    )
    assert decision == Decision("04_meals", 87.0)
    assert consumed == 2, "must wait for the terminator before committing"


def test_unterminated_number_does_not_commit():
    parser = DecisionParser()
    assert parser.feed('{"category": "04_meals", "confidence": 8') is None


def test_the_8_that_becomes_87():
    """The core hazard: an unterminated 8 must never be committed as 8."""
    parser = DecisionParser()
    assert parser.feed('{"category": "04_meals", "confidence": 8') is None
    decision = parser.feed("7, ")
    assert decision == Decision("04_meals", 87.0)


def test_chunk_boundaries_are_irrelevant():
    """Same bytes, split one character at a time, must yield the same result."""
    payload = '{"category": "04_meals", "confidence": 87, "citation": "x"}'
    decision, _ = feed_all(list(payload))
    assert decision == Decision("04_meals", 87.0)


def test_prefix_of_a_longer_enum_value_is_not_accepted_early():
    """04_meals is a prefix of 04_meals_entertainment: the quote decides."""
    parser = DecisionParser()
    assert parser.feed('{"category": "04_meals') is None
    decision = parser.feed('_entertainment", "confidence": 12, ')
    assert decision == Decision("04_meals_entertainment", 12.0)


def test_unknown_category_aborts():
    parser = DecisionParser()
    try:
        parser.feed('{"category": "99_not_a_category", "confidence": 50, ')
    except AbortEarlyCommit:
        return
    raise AssertionError("expected AbortEarlyCommit for an out-of-enum category")


def test_essay_before_verdict_aborts():
    parser = DecisionParser()
    try:
        parser.feed('{"customer_friendly_explanation": "This tr')
    except AbortEarlyCommit:
        return
    raise AssertionError("expected AbortEarlyCommit when order is violated")


def test_verdict_and_essay_in_one_chunk_still_commits():
    """Order check must not reject a valid stream that arrives in one frame."""
    parser = DecisionParser()
    decision = parser.feed(
        '{"category": "08_personal", "confidence": 91, '
        '"customer_friendly_explanation": "Groceries'
    )
    assert decision == Decision("08_personal", 91.0)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
