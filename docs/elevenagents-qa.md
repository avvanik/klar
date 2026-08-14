# ElevenAgents Q&A over briefs: realtime design (not built)

*A voice agent you ask "what did we commit to Acme last week?" and it answers in speech,
grounded in Klar's stored briefs. Design only.*

## What it does
Takes a spoken question, retrieves the relevant stored briefs, and answers in voice,
grounded in and cited to those briefs. Klar already produces the knowledge base: `Brief`
JSON per call (summary, action items, decisions, open questions, sentiment).

## Batch and realtime
Klar's extraction stays batch and offline: call ends, transcribe the file, one
extraction pass. Retrieval runs over those already-extracted briefs. The conversation
loop (listen, retrieve, answer in voice) is the realtime part: a turn-taking loop with a
latency budget, not a file job.

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
  Aug 6 you committed to send the order form by Friday"). No brief, no answer.
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

Streaming end to end plus overlap: start TTS on the first sentence while the LLM is still
generating, and pre-warm retrieval on partial transcripts.

## Eval
* Grounded-answer eval (offline, reuses the harness): question, brief-set, expected fact.
  Assert the answer has the fact and cites the right `run_id`, and refuses when there's no
  support.
* Latency eval: p50 and p95 per stage against the budget, under load.
* Barge-in: caller interrupts, agent stops and re-listens.

## Build order
1. Text-only Q&A over the briefs (retrieval plus grounded, cited answers). Validated by
   the offline grounded-answer eval.
2. Wrap it in ElevenAgents with streaming STT and TTS. Enforce the latency budget.
3. Live agent-assist: surface open questions and next-best-action to a human agent during
   the call.

Step 1 runs on what exists. Steps 2 and 3 add the realtime work: latency, streaming eval,
barge-in.
