# lmlayer project status

This is the single status document for the repository. The README covers what
the project is and how to run it; this file covers where it stands, what the
code actually does, and what is known to be missing.

## Classification

Standalone tooling: an OpenAI-compatible enhancement layer (transparent proxy)
that sits in front of any LLM server speaking the OpenAI API.

## Status

No functional change since 2026-01-07, and nothing has been re-verified by
execution since then. The last functional commit is `805f072` (2026-01-07) for
`app.py`, `core.py`, `tool.py`, `util.py` and `db/common.py` (`core.py` last
changed in `57ddeab`, same day); the 2026-09-30 commits (`498e298`, `6f5644c`)
touched only README, Dockerfile, compose, the smoke test and docs; later commits
(`69497e9` and the README rebuild) are repository metadata and documentation
only.

Implemented and not changed since:

- OpenAI-compatible `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`,
  plus transparent pass-through for every other path.
- Safety checks (regex `SAFE_RE`, optional model endpoint `SAFE_MODEL`) on the
  user prompt, the final response, and every streamed chunk.
- Server-side sessions and history keyed by user and session, with
  `sessionFork`, and cache / prompt / generation metrics stored per message.
- Auto RAG: embed the query, retrieve candidates from pgvector (or the local
  NumPy store), rerank them, and inject the result into the prompt.
- Tool execution through a subprocess with a hard 60-second timeout.
- Cost accounting with exponential decay, and quota enforcement per user.
- Priority lock per model, and least-busy instance selection per request.

## Evidence

- `app.py` — the three OpenAI endpoints and the catch-all pass-through route
  (only `content-length` and `content-encoding` are dropped when forwarding).
- `core.py` — chat / embedding / rerank pipelines, `safeChk`, `addCost`, the
  `PriorityLock` and cost-decay logic.
- `db/common.py` — SQLAlchemy models including a `pgvector.sqlalchemy.Vector`
  column sized by `EMBEDDING_DIM` (default 1024); `db/pg.py` and `db/local.py`
  expose the same interface.
- `tool.py` — tool schemas and `doTool` (60-second `wait_for`, `'timeout'` exit
  code).
- `tests/smoke_import.py` — asserts the OpenAI-compatible surface without a
  database or model server.
- `docker-compose.yml` / `.env.example` — pgvector + service stack and the full
  documented environment.
- `docs/findings.md` — a ledger of individually verified facts with file:line
  references.

## Boundaries

- No end-to-end tests for the retrieval or safety pipelines; the only automated
  check is `tests/smoke_import.py`, which asserts the HTTP surface exists.
- Tool execution is not isolated by the code. The default `python`/`bash` run
  inside `docker run --rm ipython/ipython`; `docker/python` and `docker/bash`
  run directly in the service's own environment. Isolation depends on the
  deployment (a dedicated Docker network is recommended), and the two tool-name
  families read backwards relative to what they actually do.
- Web search is not implemented.
- Sticky session routing is not implemented: `_STICK` in `core.py` is read when
  choosing an instance but never written.
- AGPL-3.0.

## Next steps

Only work the repository itself records:

- Web search tool integration (README roadmap).
- End-to-end tests for retrieval and safety pipelines.
- Sticky session routing.
- Safety checks on tool arguments and tool output.
- Clarify the `python`/`bash` versus `docker/python`/`docker/bash` naming.
- Document a hardened deployment profile for tool isolation.

No schedule, milestone, or date exists in the repository; none is invented here.
