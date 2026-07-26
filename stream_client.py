"""Reading the verdict off a real network stream.

`demo.py` replays a string. Over a socket there are two independent framings
between you and a JSON value, and neither respects the other:

1. Bytes to SSE events. TCP reads land wherever they land, so a single
   `data: {...}` line can arrive split across two reads. Lines have to be
   reassembled from a byte buffer before anything can parse them.

2. SSE events to JSON values. Each event carries a delta: a fragment of
   generated text. A JSON string, number, or key can straddle any number
   of deltas.

That is the practical reason the parser accumulates a buffer rather than
trusting a frame. By the time a chunk reaches your code it has been re-cut
twice, and neither cut has anything to do with where values begin and end.

`iter_sse_data` takes any async iterator of bytes, so the reframing can be
tested without a network or an HTTP library. `stream_decision` is the thin
aiohttp binding on top of it.

    pip install aiohttp    # only needed for stream_decision
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional

from early_commit import AbortEarlyCommit, Decision, DecisionParser

DONE = "[DONE]"


@dataclass(frozen=True)
class StreamResult:
    """What a completed stream gave us."""

    verdict: Optional[Decision]
    text: str

    @property
    def committed_early(self) -> bool:
        return self.verdict is not None


async def iter_sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Yield the payload of each SSE `data:` line, reassembling across reads.

    Blank lines and `:` keepalive comments are skipped.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            raw_line, _, buffer = buffer.partition(b"\n")
            line = raw_line.strip()
            if line.startswith(b"data:"):
                yield line[len(b"data:") :].strip().decode("utf-8")

    leftover = buffer.strip()
    if leftover.startswith(b"data:"):
        yield leftover[len(b"data:") :].strip().decode("utf-8")


def extract_delta(frame: dict[str, Any]) -> Optional[str]:
    """Pull the text fragment out of one decoded frame.

    Frame shapes differ between providers; adapt this to yours. Terminal
    frames, usage frames, and keepalives have no delta and return None.
    """
    if "error" in frame:
        raise RuntimeError(f"stream error frame: {frame['error']}")

    delta = frame.get("delta")
    if isinstance(delta, str) and delta:
        return delta
    return None


async def read_verdict(chunks: AsyncIterator[bytes]) -> StreamResult:
    """Consume a stream, capturing the verdict the moment it becomes final.

    The loop keeps draining afterwards so the caller still receives the whole
    response. In production the drain belongs in a background task, so the
    request path can return as soon as the verdict lands.
    """
    parser = DecisionParser()
    verdict: Optional[Decision] = None
    early_commit_possible = True
    received: list[str] = []

    async for data in iter_sse_data(chunks):
        if data == DONE:
            break

        delta = extract_delta(json.loads(data))
        if delta is None:
            continue
        received.append(delta)

        if early_commit_possible:
            try:
                verdict = parser.feed(delta)
            except AbortEarlyCommit:
                # No safe prefix in this stream. Stop trying, keep reading, and
                # let the caller fall back to the completed object.
                early_commit_possible = False
            if verdict is not None:
                early_commit_possible = False

    return StreamResult(verdict=verdict, text="".join(received))


async def stream_decision(session: Any, url: str, payload: dict[str, Any]) -> StreamResult:
    """POST a request and read the verdict off the SSE response, using aiohttp.

    Two timeout choices matter in production:

    * Bound the socket read, not the total request. The generation is not
      time-bounded; a stalled connection is the failure you care about.
    * Keep that bound well above the server's keepalive interval, or a normal
      idle gap looks like a stall.

    Separately, put a hard deadline on the verdict itself (we used two seconds)
    and fall back to the deterministic path when it expires.
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=None, sock_read=30.0)
    async with session.post(url, json=payload, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"stream request failed: HTTP {response.status}")
        return await read_verdict(response.content.iter_any())
