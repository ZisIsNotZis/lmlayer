# lmlayer project status

## Classification

Standalone tooling: an OpenAI-compatible enhancement layer (transparent proxy)
that sits in front of any LLM server speaking the OpenAI API.

## Status

Not under active development as of 2026-10-19. The last functional change was
2026-01-07 (`git log -1 --date=short --format='%h %ad %s' -- app.py core.py
tool.py util.py db/common.py` → `805f072` 2026-01-07; `core.py` last changed in
`57ddeab`, same day); the 2026-09-30 commits (`498e298`, `6f5644c`) touched only
README, Dockerfile, compose, smoke test and docs, and the latest commit
`69497e9` (2026-10-15) is repository metadata.

Implemented by then and not re-verified since: full-lifecycle safety checks
(pre/post/tool, regex + model), server-side session & history management with
cache/PP/TG token metrics, Auto RAG + pgvector retrieval + embedding/rerank,
sandboxed python/bash tool execution, token quota accounting, fair scheduling
(resource semaphore + sticky session), OpenAI-compatible streaming.

## Evidence

- `app.py` — `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, plus
  transparent pass-through for unregistered paths.
- `core.py` — chat / embedding / rerank pipelines, `safeChk`, `addCost`.
- `db/common.py` — SQLAlchemy models incl. a `pgvector.sqlalchemy.Vector`
  column (`EMBEDDING_DIM`) for retrieval.
- `tests/smoke_import.py` — asserts the OpenAI-compatible surface without a DB
  or model server (`uv run ... python tests/smoke_import.py`).
- `docker-compose.yml` — pgvector + service; `.env.example` documents every
  environment variable.

## Boundaries

No tests cover the retrieval/safety pipelines end-to-end yet (the smoke test
only asserts the HTTP surface). Tool execution isolation depends on the
deployment (dedicated Docker network recommended). AGPL-3.0.
