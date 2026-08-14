# Deploying Klar for a large CX org

*How I'd take Klar to production for a Deutsche-Telekom-style CX org: high call volume,
German and English, regulated, existing contact-centre stack.*

## The problem
Thousands of support and sales calls a day. They want every call to leave a structured,
searchable brief (commitments, decisions, open questions, sentiment) without adding
handle time or a second tool for agents to babysit.

## Why Klar fits
Klar isn't a transcription product. It's a per-account extraction pipeline. One YAML file
per team changes both the transcript request and the brief. No fork, no redeploy. That's
why this is an FDE job, not a SaaS signup.

## Week 1: discovery, not config
* Pull 20 to 30 real calls per team and listen.
* Collect the words that break generic STT: product names, tariffs, device models,
  acronyms. These become `keyterms` and the `glossary`.
* Agree what decision the brief drives. Sales wants commitments and next steps. Retention
  wants open questions and sentiment. That sets `emphasis`.
* Fix the data boundaries now: PII, retention, EU residency, who can read a brief.
  Everything below depends on it.

## Configuration: the file they own
Each team is one `customers/<team>.yaml` (`acme_sales.yaml` and `medtech_support.yaml`
are the templates): language, keyterms, speaker roles, verbatim cleanup, emphasised
fields, glossary. Their own ops lead edits it. That's what makes the deployment stick
after I leave.

## Rollout: phased and gated
1. Shadow. Recordings only, agents see nothing. Tune configs until the briefs are
   trustworthy.
2. Assist. The brief goes to the agent after the call, one click to paste into the CRM.
3. Integrate. Push action items and decisions into the CRM or ticketing.

Each phase has a gate: the eval stays green on that team's samples before it moves on.

## What I'd measure
* Extraction quality: `klar eval` keyword and entity recall on a labelled sample set.
  This is the acceptance gate.
* Editing rate: how often an agent changes a brief before saving. The real quality signal
  once live. I want it dropping.
* Coverage and cost: share of calls processed, and the per-call cost already in the run
  log.
* Latency: p95 on transcribe and extract, already recorded.
* Adoption: briefs actually used downstream.

## Data and compliance
Transcripts and briefs are personal data, and support calls are often special-category.
Before any real audio: EU-region processing, encryption at rest, a retention and
redaction policy, access control on the store. Klar writes plaintext JSON locally today.
Fine for a demo, not fine for regulated data. I'd scope this in week 1, not find it in
the pilot.

## Batch vs realtime
Klar is batch, on purpose. Call ends, transcribe the file, one extraction pass
(map-reduced for long calls), brief out. That's the right shape for post-call
intelligence: full context, cheaper, more accurate. It covers most of the value.

Realtime is a different problem: streaming transcripts, sub-second turns, incremental
extraction, no second look at the audio. The next surface here is realtime agent-assist
(live open-question and next-best-action hints during the call) and an ElevenAgents voice
agent over the stored briefs ("what did we commit to Kunde X last week?", answered in
voice). The `Brief` files are already the knowledge base for it. Batch is the foundation,
realtime is the roadmap, and I'd call it new engineering, not a config flag. Design in
`docs/elevenagents-qa.md`.

## Risks I'd name on day one
1. STT accuracy on domain vocab and accents. Keyterms plus a per-team eval gate, measured
   not assumed.
2. Compliance scope creep. Handled up front.
3. Agent trust. Shadow-first rollout and the editing-rate metric earn it.

## Timeline
Week 1 discovery and first configs. Weeks 2 to 3 shadow and tune to a green eval. Week 4
assist pilot with one team. Then integrate and widen.
