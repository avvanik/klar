# ElevenAgents Q&A over briefs: realtime design (not built)

*A voice agent you ask "what did we commit to Acme last week?" and it answers in speech,
grounded in Klar's stored briefs. This is design, not code. The point is to show I know
what realtime actually needs, and why I'd keep the batch core underneath.*

## Why this is next
Klar already builds the hard part: a clean knowledge base of `Brief` JSON per call. A Q&A
agent is a thin layer on top. It turns "read the brief" into "ask the briefs". Much
stronger demo.

## Batch vs realtime
Klar today is batch: call ends, transcribe the file, one extraction pass. Right for
post-call work.

A voice agent is realtime, and that's a different problem. It's a turn-taking loop with a
latency budget, not a file job. Retrieval runs over briefs I already extracted, so that
stays offline. The conversation (listen, retrieve, answer in voice) has to feel live.

## Architecture
```
  caller speech ─▶ STT (streaming) ─▶ intent + query
                                          │
                                          ▼
                         retrieval over stored Brief JSON
                    (filter by customer/date/entity → top-k)
                                          │
                                          ▼
                        LLM answer, grounded + cited to run_id
                                          │
                                          ▼
                            TTS (streamed) ─▶ caller hears answer
```
* Retrieval, not RAG over audio. Filter the briefs as data (customer, date, speaker,
  entity), then semantic search over `searchable_text()` for the top-k. Briefs are small,
  so it's precise and cheap.
* Grounding. The LLM sees only the retrieved briefs and cites the `run_id` and date ("On
  Aug 6 you committed to send the order form by Friday"). No brief, no answer. Same
  fail-loud rule as extraction.
* Reuse. The `LLMClient` protocol and the cost and latency logging carry over. Only STT
  and TTS go streaming.

## Latency budget
Target: under 1.5 s from end of speech to first spoken word.

| stage | budget | how |
|---|---|---|
| endpointing (end of turn) | ~200 ms | VAD / streaming STT finals |
| retrieval | ~150 ms | pre-embedded briefs, in-memory or pgvector top-k |
| LLM first token | ~600 ms | small model, tight grounded prompt, stream out |
| TTS first audio | ~300 ms | streaming synthesis, start speaking on first sentence |

The unlock is streaming end to end and overlap: start TTS on the first sentence while the
LLM is still going, and pre-warm retrieval on partial transcripts.

## Eval
* Grounded-answer eval (offline, reuses the harness): question, brief-set, expected fact.
  Assert the answer has the fact and cites the right `run_id`, and refuses when there's no
  support. No audio needed.
* Latency eval: p50 and p95 per stage against the budget, under load.
* Barge-in: caller interrupts, agent stops and re-listens.

## What I'd ship first
1. Text-only Q&A over the briefs (retrieval plus grounded, cited answers). Proves the
   value with the offline eval, zero realtime risk.
2. Wrap it in ElevenAgents with streaming STT and TTS. Enforce the latency budget.
3. Live agent-assist: surface open questions and next-best-action to a human agent during
   the call. Highest-value realtime use once the loop works.

Step 1 is a few days on what exists. Steps 2 and 3 are the real realtime work (latency,
streaming eval, barge-in). I'd scope them, not pretend a flag turns batch into realtime.
