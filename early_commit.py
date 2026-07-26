"""Early commit on a streamed structured LLM response.

The decision ("verdict") fields are placed first in the schema. This parser
reads the accumulated buffer and commits only when the verdict is provably
final, which requires three structural proofs:

  1. completion  - the value is terminated (closing quote / JSON terminator)
  2. membership  - the value belongs to the closed set declared in the schema
  3. order       - no later ("essay") field arrived before the verdict closed

A missing early verdict is recoverable: the caller falls back to the complete
response or to a deterministic default. A misparsed verdict is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CATEGORY_ENUM = frozenset(
    {
        "01_income_employment",
        "02_income_self_employment",
        "03_income_property",
        "04_meals",
        "04_meals_entertainment",
        "05_travel",
        "06_office",
        "07_professional_services",
        "08_personal",
        "09_general_purchase",
    }
)

# Field names that must never appear before the verdict is complete.
ESSAY_MARKERS = (
    '"customer_friendly_explanation"',
    '"internal_explanation"',
    '"citation"',
)

# The trailing [,}\s] is check 1: a number is only final once terminated.
DECISION_RE = re.compile(
    r'"category"\s*:\s*"(?P<category>[A-Za-z0-9_]+)"\s*,\s*'
    r'"confidence"\s*:\s*(?P<confidence>\d+(?:\.\d+)?)'
    r"\s*[,}\s]"
)


class AbortEarlyCommit(Exception):
    """The early path is unsafe; fall back to the deterministic pipeline."""


@dataclass(frozen=True)
class Decision:
    category: str
    confidence: float


class DecisionParser:
    """Feed stream chunks in; get a Decision as soon as it is provably final."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, chunk: str) -> Optional[Decision]:
        self.buffer += chunk

        match = DECISION_RE.search(self.buffer)
        if match:
            category = match.group("category")
            if category not in CATEGORY_ENUM:  # check 2: membership
                raise AbortEarlyCommit(f"category not in enum: {category!r}")
            confidence = float(match.group("confidence"))
            if not 0 <= confidence <= 100:
                raise AbortEarlyCommit(f"confidence out of range: {confidence}")
            return Decision(category, confidence)

        # Order is checked only after the match attempt: a single chunk may
        # carry both the end of the verdict and the start of the essay.
        if any(marker in self.buffer for marker in ESSAY_MARKERS):
            raise AbortEarlyCommit("essay field arrived before the verdict")

        return None
