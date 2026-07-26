# early-commit-streaming

Act on a streamed LLM response as soon as the decision fields are provably
final, and drain the explanation in the background.

Put the decision fields first in your structured-output schema and a classifier
can commit after ~30 tokens instead of ~250. In the system this came from, median
time-to-act fell from 1.33s to 0.65s — same model, same prompt, same tokens, only
the field order and the moment of commit changed.

Reference implementation for [Act on the Verdict. Stream the Rest.](https://nturusin.github.io/act-on-the-verdict.html).

## Requirements

Python 3.9+. The parser, the tests, and the demo use the standard library only.
Optional: `google-genai` to run the probe against Vertex AI, `aiohttp` for
`stream_decision`.

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

This is a worked example rather than a library: `DECISION_RE` hardcodes the two
field names, so adapting it to your schema means editing that regex and the
enum. It is about ninety lines — read it before you use it.

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

## Measure it on your own stack

The technique rests on an assumption nobody guarantees: that your provider emits
fields in the order your schema declares them, and streams them incrementally.
`run_probe.py` checks that against a real model and reports what early commit
would buy you.

```bash
pip install google-genai
gcloud auth application-default login
python3 run_probe.py --project YOUR_PROJECT --runs 20
```

```
  field order declared   ['category', 'confidence', 'customer_friendly_explanation', ...]
  field order observed   ['category', 'confidence', 'customer_friendly_explanation', ...]
  order held             yes

  early verdict          04_meals @ 87.0
  final verdict          04_meals @ 87.0
  agreed                 yes

  time to act            0.38s  (after 82 of 95 chars)
  time to complete       0.90s
  removed from path      0.52s  (58%)
```

Three of those lines matter more than the timings. **order held** tells you the
technique is viable at all. **agreed** tells you the early verdict matched the
completed object — the check that has to hold every time, not on average. A
single run tells you whether it works; use `--runs` for a sense of the spread.

Gemini on Vertex AI is implemented because it is the stack the article's numbers
came from. A different provider is one async generator of
`(elapsed, text_fragment)` events in `providers.py`; nothing else changes.

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
| `probe.py` | Turns a timed stream into a report: order, agreement, saving |
| `providers.py` | Provider adapters; Gemini on Vertex AI is implemented |
| `run_probe.py` | CLI for the probe |
| `demo.py` | Simulated stream comparing both field orders |
| `schema.json` | The schema used by the probe, verdict fields first |
| `test_early_commit.py` | Structural tests, including the `8` → `87` case |
| `test_stream_client.py` | Stream tests over a faked transport, down to one byte per read |
| `test_probe.py` | Reporting tests, including a provider that reorders fields |

## Tests

```bash
python3 test_early_commit.py    # 8 tests
python3 test_stream_client.py   # 6 tests
python3 test_probe.py           # 6 tests
python3 demo.py                 # field order vs. time-to-act
```

`demo.py` replays a synthetic response at a fixed token rate. It is an
illustration, not a measurement — the numbers it prints follow from constants
at the top of the file. For evidence, run the probe against a real model.

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
