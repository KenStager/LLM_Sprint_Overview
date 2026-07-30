# LLM-Guided Development: a sprint on the SNAPPED pipeline

A record of how I run development work through LLMs, using one real sprint as the worked
example. The documents in this repo are the actual artifacts from that sprint, not
illustrations written afterward.

The task: SNAPPED, a news-processing pipeline, had two clustering engines that couldn't
talk to each other — different weight configurations, incompatible JSON contracts, and no
multi-session support, so a run that stopped halfway had to start over. They needed to
become one service without re-clustering the 183 clusters already in the database.

The work took about 90 minutes auditing the codebase and writing down exactly what had to
change, then roughly three hours implementing against that plan. Around 1,700 lines. The
ratio is the point: the planning pass is what makes the implementation pass fast, and it's
the part people skip.

## What actually made it work

**Documents as shared memory.** Every session reads from and writes to the same markdown
files. A model has no memory between sessions, so the documents are the memory, which
means they have to be maintained as deliberately as the code. The continuation prompt
exists so a new session starts where the last one stopped instead of re-deriving context.

**Zoom in without losing the map.** Working a specific bug and reasoning about the
architecture need different amounts of context. Moving between them on purpose, rather
than dumping the whole repo into every prompt, is most of the skill.

**Constraints written down before implementation.** Cost ceilings, format contracts, and
the no-re-clustering rule were fixed in the strategy document first. A model will
cheerfully violate a constraint it was never told about.

The honest limitation: this worked because I could review every line it produced. On a
codebase I didn't know, the audit phase would have taken far longer than 90 minutes and
the plan would have been worse. The speedup comes from the reviewer, not the model.

## The documents

| Document | What it's for |
|---|---|
| [`CLUSTERING_ENGINE_STRATEGY_V3.md`](CLUSTERING_ENGINE_STRATEGY_V3.md) | The technical plan — architecture, contracts, constraints (1,707 lines) |
| [`CLUSTERING_V3_IMPLEMENTATION_CHECKLIST.md`](CLUSTERING_V3_IMPLEMENTATION_CHECKLIST.md) | Task tracking, updated continuously during the sprint |
| [`CLUSTERING_V3_QUICK_REF.md`](CLUSTERING_V3_QUICK_REF.md) | Summary for anyone who won't read the strategy doc |
| [`CLUSTERING_V3_DELIVERY_OVERVIEW.md`](CLUSTERING_V3_DELIVERY_OVERVIEW.md) | What shipped, at sprint close |
| [`SNAPPED Clustering Session Completion Log.md`](SNAPPED%20Clustering%20Session%20Completion%20Log.md) | Session-end record of work and technical notes |
| [`SNAPPED Clustering Continuation Prompt.md`](SNAPPED%20Clustering%20Continuation%20Prompt.md) | Handoff prompt that boots the next session with full context |

## Outcome

Two clustering engines unified behind one service, with no duplicate clusters created.
Multi-session support added, so interrupted runs resume. Selective LLM enrichment kept API
spend under $5/day. Format adapters absorbed the JSON incompatibility instead of forcing a
migration.

The SNAPPED pipeline itself is a private repository; these planning documents are the part
that's shareable. Infrastructure identifiers have been redacted.
