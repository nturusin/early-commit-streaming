"""Network-shaped tests: SSE reassembly and early commit over a stream.

No network and no aiohttp needed. The transport is a fake async iterator of
byte chunks, deliberately cut in awkward places.

Run:  python3 test_stream_client.py
"""

from __future__ import annotations

import asyncio
import json

from early_commit import Decision
from stream_client import iter_sse_data, read_verdict


# --- building fake streams -------------------------------------------------


def sse_lines(*payloads: str) -> bytes:
    """Wrap each payload as an SSE `data:` line."""
    return "".join(f"data: {payload}\n\n" for payload in payloads).encode()


def response_stream(*deltas: str) -> bytes:
    """A full response: one delta frame per fragment, then [DONE]."""
    frames = [json.dumps({"type": "delta", "delta": delta}) for delta in deltas]
    return sse_lines(*frames, "[DONE]")


async def read_in_chunks_of(raw: bytes, size: int):
    """Replay raw bytes `size` at a time, ignoring line boundaries."""
    for start in range(0, len(raw), size):
        yield raw[start : start + size]


def sse_payloads(raw: bytes, chunk_size: int) -> list[str]:
    async def collect() -> list[str]:
        return [payload async for payload in iter_sse_data(read_in_chunks_of(raw, chunk_size))]

    return asyncio.run(collect())


def read(raw: bytes, chunk_size: int):
    return asyncio.run(read_verdict(read_in_chunks_of(raw, chunk_size)))


# --- reassembling the transport --------------------------------------------


def test_sse_line_split_across_reads():
    """A data: line cut in half by the transport is still one event."""
    raw = sse_lines('{"delta": "hello"}')

    for chunk_size in (1, 3, 7, 11, len(raw)):
        assert sse_payloads(raw, chunk_size) == ['{"delta": "hello"}'], f"chunk_size={chunk_size}"


def test_keepalives_and_blank_lines_are_skipped():
    raw = b": keepalive\n\n" + sse_lines('{"delta": "x"}') + b"\n\n: keepalive\n\n"

    assert sse_payloads(raw, chunk_size=5) == ['{"delta": "x"}']


# --- committing early ------------------------------------------------------


def test_verdict_lands_before_the_stream_ends():
    """The whole point: a usable decision well before [DONE]."""
    raw = response_stream(
        '{"category": "04_m',
        'eals", "confidence": 8',
        '7, "customer_friendly_explanation": "Lunch',
        ' was a working meal."}',
    )

    result = read(raw, chunk_size=9)

    assert result.verdict == Decision("04_meals", 87.0)
    assert json.loads(result.text)["confidence"] == 87


def test_verdict_survives_byte_level_fragmentation():
    """One byte per read, so every value straddles several frames."""
    raw = response_stream('{"category": "08_personal", "confidence": 91, "citation": "x"}')

    result = read(raw, chunk_size=1)

    assert result.verdict == Decision("08_personal", 91.0)


def test_essay_first_stream_gives_no_verdict_but_still_drains():
    """Order violated: no early commit, but the full text still arrives."""
    raw = response_stream('{"customer_friendly_explanation": "This tr', 'ansaction", "category": "04_meals"}')

    result = read(raw, chunk_size=6)

    assert not result.committed_early
    assert "customer_friendly_explanation" in result.text


def test_stream_ending_mid_number_commits_nothing():
    raw = sse_lines(json.dumps({"delta": '{"category": "04_meals", "confidence": 8'}))

    result = read(raw, chunk_size=4)

    assert not result.committed_early


if __name__ == "__main__":
    passed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
