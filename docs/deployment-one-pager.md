# Deploying Klar for a large CX org

*What deploying Klar for a Deutsche-Telekom-style CX org looks like: high call volume,
German and English, regulated, existing contact-centre stack.*

## What it delivers
Every call leaves a structured, searchable brief (commitments, decisions, open
questions, sentiment) without adding handle time or a second tool for agents.

## Configuration per team
Each team is one `customers/<team>.yaml` (`acme_sales.yaml` and `medtech_support.yaml`
are the templates): language, keyterms, speaker roles, verbatim cleanup, emphasised
fields, glossary. One file changes both the transcript request and the brief, with no
code change, and the team's own ops lead can edit it.

Setup per team:
* Pull 20 to 30 real calls and listen.
* Collect the vocabulary that breaks generic STT (product names, tariffs, device models,
  acronyms) into `keyterms` and the `glossary`.
* Set `emphasis` to the fields that team needs. Sales: commitments and next steps.
  Retention: open questions and sentiment.
* Fix the data boundaries: PII, retention, EU residency, who can read a brief.

## Rollout: phased and gated
1. Shadow. Recordings only, agents see nothing. Tune configs until the briefs are
   trustworthy.
2. Assist. The brief goes to the agent after the call, one click to paste into the CRM.
3. Integrate. Push action items and decisions into the CRM or ticketing.

Each phase gates on the eval staying green on that team's samples before it advances.

## Metrics
* Extraction quality: `klar eval` keyword and entity recall on a labelled sample set.
  The acceptance gate.
* Editing rate: how often an agent changes a brief before saving.
* Coverage and cost: share of calls processed, plus the per-call cost from the run log.
* Latency: p95 on transcribe and extract, from the run log.
* Adoption: briefs used downstream.

## Data and compliance
Transcripts and briefs are personal data, and support calls are often special-category.
Before real audio: EU-region processing, encryption at rest, a retention and redaction
policy, access control on the store. Klar writes plaintext JSON locally today, which is
fine for a demo and not for regulated data.

## Batch and realtime
Klar is batch: call ends, transcribe the file, one extraction pass (map-reduced for long
calls), brief out. Post-call intelligence with full context.

Realtime is separate work: streaming transcripts, sub-second turns, incremental
extraction. The next surfaces are realtime agent-assist (live open-question and
next-best-action hints during the call) and an ElevenAgents voice agent over the stored
briefs. Design in `docs/elevenagents-qa.md`.

## Risks
1. STT accuracy on domain vocab and accents. Handled by keyterms and a per-team eval
   gate.
2. Compliance scope. Handled up front.
3. Agent trust. Handled by shadow-first rollout and the editing-rate metric.

## Timeline
Week 1 discovery and first configs. Weeks 2 to 3 shadow and tune to a green eval. Week 4
assist pilot with one team. Then integrate and widen.
