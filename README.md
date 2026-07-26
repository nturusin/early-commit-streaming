# early-commit-streaming

Act on a streamed LLM response as soon as the decision fields are provably
final, and drain the explanation in the background.

Put the decision fields first in your structured-output schema and a classifier
can commit after ~30 tokens instead of ~250. In the system this came from, median
time-to-act fell from 1.33s to 0.65s — same model, same prompt, same tokens, only
the field order and the moment of commit changed.

Reference implementation for [Act on the Verdict. Stream the Rest.](https://nturusin.github.io/act-on-the-verdict.html).

## Requirements

Python 3.9+. The parser, tests, and demo use the standard library only.
`aiohttp` is needed only for `stream_decision`.

## Usage

```python
from early_commit import AbortEarlyCommit, DecisionParser

parser = DecisionParser()

for chunk in stream:                  # whatever your provider yields
    try:
        decision = parser.feed(chunk)
    except AbortEarlyCommit:
        decision = None               # fall back to the deterministic path
        break

    if decision is not None:
        act_on(decision.category, decision.confidence)
        break                         # then drain the rest in the background
```

`feed()` returns `None` until the verdict is provably final, a `Decision` the
moment it is, and raises `AbortEarlyCommit` if the stream cannot be trusted.

Over a network, `stream_client.py` wraps this:

```python
from stream_client import stream_decision

result = await stream_decision(session, url, payload)
if result.committed_early:
    act_on(result.verdict)
```

## How it works

Committing early is only safe with three structural proofs:

| Proof | Question | Evidence required |
|---|---|---|
| Completion | Has the value finished arriving? | Closing quote for a string; `,` `}` or whitespace after a number |
| Membership | Is it a legal value? | Present in the enum declared in the schema |
| Order | Did the verdict arrive first? | No later field seen before the verdict closed |

Completion is the one that bites. A partial-JSON parser will happily tell you
`confidence` has arrived here:

```
{"category": "04_meals", "confidence": 8
```

The next chunk may be `7,`, making the real value `87`. Rendering `8` for a
moment is harmless; writing it to a database or routing on it is not.

Hence two rules: **parse the accumulated buffer, never an individual frame**
(chunk boundaries are a transport detail), and **check order only after
attempting the match** (one frame can carry both the end of the verdict and the
start of the explanation).

If any proof fails, abort and fall back. A missing prediction is recoverable; a
confidently misparsed one is not.

## Over the network

A socket re-cuts the data twice: TCP reads split `data:` lines, and SSE events
carry text fragments that split JSON values. Neither boundary relates to where
values begin and end, which is why the parser buffers. `iter_sse_data` accepts
any async iterator of bytes, so the reassembly is testable without a network.

Bound the socket read rather than the total request — a stalled connection is
the real failure — and keep that bound above the server's keepalive interval.
Put a separate hard deadline on the verdict itself.

## Files

| File | What it is |
|---|---|
| `early_commit.py` | The parser: three proofs |
| `stream_client.py` | SSE reassembly, draining, and the aiohttp binding |
| `demo.py` | Simulated stream comparing both field orders |
| `schema.json` | Example schema with the verdict fields first |
| `test_early_commit.py` | Structural tests, including the `8` → `87` case |
| `test_stream_client.py` | Stream tests over a faked transport, down to one byte per read |

## Tests

```bash
python3 test_early_commit.py    # 8 tests
python3 test_stream_client.py   # 6 tests
python3 demo.py                 # field order vs. time-to-act
```

`demo.py` replays a synthetic response at a fixed token rate. Same bytes and
same token count either way: the verdict-first schema removes about half the
wait, the other cannot commit early at all. Your own ratio depends mostly on
how many explanation tokens follow the verdict.

## When not to use it

- The verdict cannot be made self-contained and structurally final.
- Your provider cannot give you constrained, ordered streaming.
- There is no deterministic fallback, or the full response already fits inside
  your latency budget.

## Provider documentation

Field ordering and streaming behaviour vary by provider and change over time.
Verify both on the exact model and API path you ship.

- OpenAI — [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) · [Streaming Responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- Anthropic — [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) · [Streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- Google Gemini — [Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- Amazon Bedrock — [Structured Outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html)
- vLLM — [Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)

## Licence

MIT — see [LICENSE](LICENSE).
