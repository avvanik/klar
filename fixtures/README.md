# Golden fixtures: real pipeline output, committed

This directory holds **real** Klar output so a reviewer without API keys can see
genuine transcripts and briefs immediately, and so `tests/test_fixtures.py` can guard
extraction quality against ground truth.

It's empty until you run the pipeline once with real keys:

```bash
# needs ELEVENLABS_API_KEY + ANTHROPIC_API_KEY and generated sample audio
python scripts/generate_samples.py
python -m klar run samples/en_sales_call.mp3 --customer acme_sales --save-fixtures
python -m klar run samples/de_support_call.mp3 --customer medtech_support --save-fixtures
python -m klar run samples/es_project_sync.mp3 --save-fixtures
git add fixtures/
```

Each run writes `fixtures/<audio-stem>/brief.json` (the strict-schema brief) and
`transcript.txt` (the human-readable diarized transcript).

`de_support_call/summary.mp3` is a real Text-to-Speech render of that brief's spoken
summary (from `klar run --speak`). Play it to hear the voice output without keys.

`tests/test_fixtures.py` then checks every committed `brief.json` is schema-valid,
non-empty, and in the expected language. It does not assert on wording: extraction
varies run to run, so content quality is gated live by `python -m klar eval` instead.
Until fixtures exist the test **skips**, so the suite stays green either way.
