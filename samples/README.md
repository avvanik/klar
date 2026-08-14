# samples/

The eval set. Audio is **not** committed (see `.gitignore`); regenerate it.

## Generate the sample audio

```bash
export ELEVENLABS_API_KEY=...        # or put it in .env
python scripts/generate_samples.py
```

This writes four short two-speaker clips that match `expectations.yaml`:

| file                 | language | customer          | tests |
|----------------------|----------|-------------------|-------|
| `en_sales_call.mp3`  | English  | `acme_sales`      | action items, "Friday" due date, brand keyterms |
| `de_support_call.mp3`| German   | `medtech_support` | German STT, clinical keyterms, open questions |
| `es_project_sync.mp3`| Spanish  | (default)         | Spanish STT, budget/design keywords |
| `fr_kickoff_call.mp3`| French   | (default)         | French STT, budget/calendar keywords |

## Add your own

Drop any supported audio/video file here (`.mp3 .wav .m4a .flac .mp4 ...`) and add
an entry to `expectations.yaml` with the keywords a correct brief must contain:

```yaml
- file: my_call.mp3
  language: eng
  customer: acme_sales      # optional
  expect_keywords: ["renewal", "Q3"]
  min_action_items: 1
```

Then: `python -m klar run samples/my_call.mp3 --customer acme_sales`
