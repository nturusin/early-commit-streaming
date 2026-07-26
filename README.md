# llm-streaming-early-commit

**Act on a streamed LLM response as soon as the decision fields are provably
final, instead of waiting for the closing brace.**

In the classifier this came from, median time-to-act fell from **1.33s to 0.65s**
— same model, same prompt, same number of tokens. Only the field order and the
moment of commit changed.

## The problem

A structured response usually serves two audiences at once. Here is one from a
transaction classifier:

```json
{
  "category": "04_meals",
  "confidence": 87,
  "customer_friendly_explanation": "Lunch was a working meal, so it is recorded against meals and entertainment.",
  "internal_explanation": "Counterparty and amount are consistent with a working meal, allowable where incurred wholly for the trade.",
  "citation": "Internal bookkeeping guidance, meals and subsistence."
}
```

The first two fields are the **verdict**: `category` is written to the
transaction, `confidence` decides whether a human reviews it. They arrive in
roughly the first 30 tokens. The other three are the **essay** — another ~220
tokens of prose that a person might read later, or never.

Treat the response as one atomic result and the application waits for all 250
tokens before it can do anything with the first 30. The fix is not a second
model call or a shorter response. It is to put the verdict first in the schema
and act the moment those fields are provably final, letting the essay finish in
the background.

Two things make that safe rather than reckless, and this repo covers both:
**what "provably final" means in code**, and **whether your provider actually
behaves the way the technique requires**.

Background reading, not required: [Act on the Verdict. Stream the Rest.](https://nturusin.github.io/act-on-the-verdict.html)

## Install

Python 3.9+. The parser and the tests need no dependencies at all.

```bash
git clone https://github.com/nturusin/llm-streaming-early-commit
cd llm-streaming-early-commit

pip install -e .            # parser only, no dependencies
pip install -e '.[probe]'   # adds google-genai==1.66.0 to run the probe
```

## 1. What "provably final" means

```python
from early_commit import AbortEarlyCommit, DecisionParser

parser = DecisionParser()

for chunk in stream:                  # whatever your provider yields
    try:
        decision = parser.feed(chunk)
    except AbortEarlyCommit:
        break                         # fall back to the deterministic path

    if decision is not None:
        act_on(decision.category, decision.confidence)
        break                         # then drain the rest in the background
```

`feed()` returns `None` until the verdict is provably final, a `Decision` the
moment it is, and raises `AbortEarlyCommit` if the stream cannot be trusted.
It demands three structural proofs before committing:

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

Two rules follow: **parse the accumulated buffer, never an individual frame**
(chunk boundaries are a transport detail), and **check order only after
attempting the match** (one frame can carry the end of the verdict and the start
of the essay). If any proof fails, abort and fall back — a missing prediction is
recoverable, a confidently misparsed one is not.

`early_commit.py` is about ninety lines and hardcodes the two field names in
`DECISION_RE`. It is a worked example to adapt, not a library to depend on.

## 2. Whether your provider allows it

Nothing guarantees that a provider emits fields in the order your schema
declares them, or streams them incrementally. The probe checks against a real
model and reports what early commit would have bought:

```bash
gcloud auth application-default login
python3 probe.py --project YOUR_PROJECT --runs 20
```

```
  field order declared   ['category', 'confidence', 'customer_friendly_explanation', ...]
  field order observed   ['category', 'confidence', 'customer_friendly_explanation', ...]
  order held             yes

  early verdict          04_meals @ 87.0
  agreed with final      yes

  time to act            0.38s
  time to complete       0.90s
  removed from path      0.52s (58%)
```

`order held` says whether the technique is viable at all. `agreed with final`
says the early verdict matched the completed object — that must hold every time,
not on average. The timings matter least: your ratio depends mostly on how many
essay tokens follow the verdict.

Gemini on Vertex AI is implemented because it is the stack the measurements came
from. Another provider is one function — an async generator of
`(seconds_since_request, text_fragment)` events.

## Tests

```bash
python3 test_early_commit.py   # the parser's structural claims
python3 test_probe.py          # the probe's reporting
pytest                         # if you prefer
```

They run offline. The one piece with no coverage is the network call itself,
`stream_gemini_vertex` — running the probe is what exercises it.

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
