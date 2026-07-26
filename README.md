# Early commit on a streamed structured LLM response

Reference implementation for **[Act on the Verdict. Stream the Rest.](https://nturusin.github.io/act-on-the-verdict.html)**

A structured LLM response can become useful before it becomes complete. If the
decision fields come first in the schema, an application can act as soon as
those fields are **provably final** and drain the explanation in the background.

In the system this came from, median time-to-act fell from **1.33s to 0.65s**
with no change to the model, the prompt, or the number of tokens generated.
Only the field order and the moment of commit changed.

![Two token bars of equal length. In schema A the fields sit in the order they were added and the commit flag points at the closing brace. In schema B both verdict fields sit first and the commit flag points at token 30.](figures/fig2.png)

## The idea

Most structured responses serve two audiences:

- the **verdict** — the short decision the application acts on;
- the **essay** — explanations, citations, audit notes a human may read later.

Schemas usually grow in the order product requests arrive, not in order of
urgency. That is how a second decision field ends up *after* a long explanation,
which pushes the earliest safe commit point all the way to the closing brace.

![Four cumulative bars, one per schema version. The commit marker stays just after category for v1 to v3, then jumps to the far right in v4 when confidence is appended last.](figures/fig1.png)

## Presence is not finality

A partial-JSON parser can tell you a field has appeared. That does not mean its
value is complete:

```
{
  "category": "04_meals",
  "confidence": 8
```

`confidence` is visible, but the next chunk may be `7,` — the real value is
`87`. Rendering `8` for a moment is harmless. Writing it to a database, routing
a payment, or triggering an external action is not.

## Three proofs before commit

| Proof | Question | Evidence required |
|---|---|---|
| **Completion** | Has the value finished arriving? | Closing quote for a string; a terminator (`,` `}` whitespace) for a number |
| **Membership** | Is it a legal value? | Present in the enum declared in the schema |
| **Order** | Did the verdict arrive first? | No later field seen before the verdict closed |

![Three buffers with their structural evidence and outcome: a terminated verdict commits, an unterminated number waits, an essay field before the verdict aborts.](figures/fig3.png)

Two rules make the difference between a parse and a guess:

1. **Parse the accumulated buffer, never an individual frame.** Chunk
   boundaries are a transport detail. A frame may end inside a string, after the
   first digit of a number, or just before the comma that proves it finished.
2. **Check order only after attempting the match.** One frame can carry both the
   end of the verdict and the start of the essay; checking order first would
   reject a perfectly valid stream.

If any proof fails, abort the early path and fall back to the deterministic
pipeline. A missing prediction is recoverable; a confidently misparsed one is not.

## Run it

No dependencies beyond the standard library.

```bash
python3 test_early_commit.py   # structural tests, including the 8 -> 87 case
python3 demo.py                # field order vs. time-to-act
```

`demo.py` replays a synthetic response at a fixed token rate. Same bytes, same
token count — only the field order differs:

```
schema A - verdict split around the essay (confidence last)
  early commit              not possible (essay field arrived before the verdict)
  time to act                0.65s
  removed from critical path 0%

schema B - verdict first
  time to act                0.33s
  time to complete           0.65s
  removed from critical path 49%
```

This is a **mechanism demo, not a benchmark**. Your ratio depends on the model,
the provider, and above all the ratio of verdict tokens to essay tokens.

## Files

| File | What it is |
|---|---|
| `early_commit.py` | The parser: three proofs, ~90 lines |
| `test_early_commit.py` | Structural tests, including one that feeds the payload a character at a time |
| `demo.py` | Simulated stream comparing both field orders |
| `schema.json` | Example structured-output schema with the verdict fields first |

## Applying it

1. Split your fields into a verdict and an essay.
2. Put the verdict first in the schema, and confirm your provider preserves
   field order while streaming.
3. Parse the accumulated buffer; require completion, membership, and order.
4. Commit, then drain the remainder in a bounded background worker.
5. Verify the finished object still agrees with what you committed.
6. Keep a deterministic fallback and a hard deadline on the verdict.
7. Re-check application state before storing the essay — a human may have
   overridden the decision while the stream was still open.

![Timeline: at 0.65s the model commits, at 0.90s a human overrides it, at 1.33s the explanation finishes, at 1.34s the stale explanation is discarded.](figures/fig4-override.png)

## When not to bother

- The verdict cannot be made self-contained and structurally final.
- Your provider cannot give you constrained, ordered streaming.
- There is no deterministic fallback, or the full response already fits inside
  your latency budget.

## Provider documentation

Verify schema enforcement, field ordering, and chunk behaviour on the exact
model and API path you ship — these details vary and change.

- OpenAI — [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) · [Streaming Responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- Anthropic — [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) · [Streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- Google Gemini — [Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- Amazon Bedrock — [Structured Outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html)
- vLLM — [Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)

## Licence

MIT — see [LICENSE](LICENSE).
