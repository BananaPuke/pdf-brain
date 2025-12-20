---
"pdf-brain": minor
---

Auto-enrich documents by default on add

- `pdf-brain add` now runs LLM enrichment automatically (title, summary, tags, concepts)
- Use `--no-enrich` flag to skip enrichment for faster ingestion
- Enrichment uses configured provider (ollama or gateway) from config
