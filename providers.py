"""Talking to a real provider, and timing what comes back.

Each adapter is an async generator of `(seconds_since_request, text_fragment)`
events, which is all `probe.analyse` needs. Adding a provider means writing one
of these; nothing else in the repo changes.

Gemini on Vertex AI is implemented, because that is the stack the article's
measurements came from.

    pip install google-genai
    gcloud auth application-default login
    python3 run_probe.py --project YOUR_PROJECT
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.json")

# Verdict fields first. That ordering is the whole intervention.
RESPONSE_SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
SCHEMA_FIELD_ORDER: list[str] = list(RESPONSE_SCHEMA["properties"])

PROMPT = """You categorise bank transactions for a small business.

Return the category and a confidence score from 0 to 100, then explain the
decision for the customer, then explain it for an internal accountant, then
cite the guidance you relied on. Write two or three sentences for each
explanation.

Transaction: {transaction}
"""

DEFAULT_TRANSACTION = "Card payment, 41.98 GBP, online marketplace"


async def stream_gemini_vertex(
    *,
    project: str,
    location: str = "europe-west2",
    model: str = "gemini-flash-latest",
    transaction: str = DEFAULT_TRANSACTION,
) -> AsyncIterator[tuple[float, str]]:
    """Stream a structured response from Gemini on Vertex AI, timing each fragment.

    Uses application default credentials. The clock starts before the request is
    sent, so the first event includes time-to-first-token.
    """
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(vertexai=True, project=project, location=location)
    config = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    started = time.perf_counter()
    stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=PROMPT.format(transaction=transaction),
        config=config,
    )
    async for chunk in stream:
        if chunk.text:
            yield time.perf_counter() - started, chunk.text
