<div align="right">English | <a href="README.zh-CN.md">简体中文</a></div>

<p align="center">
  <img src="assets/logo.svg" alt="lmlayer" width="560">
</p>

# lmlayer

**Keep your model server. Add the layer it is missing.**

`lmlayer` is a drop-in, OpenAI-compatible proxy that puts safety checks, server-side session memory, automatic RAG, sandboxed tools, token quotas, and fair scheduling in front of any LLM server — llama.cpp, vLLM, SGLang, TGI, Ollama, or anything else that speaks the OpenAI API. The upstream server does not need to change.

## Why lmlayer

Most OpenAI-compatible servers expose a model. They do not give you the operational layer around it: persistent conversations, prompt and response safety, retrieval, tool isolation, per-user budgets, or fair queueing. `lmlayer` adds that layer at the protocol boundary, so existing OpenAI clients keep working.

## Quick start

Requirements: Python 3.11+ and at least one OpenAI-compatible upstream. PostgreSQL with pgvector is optional; when `DB` is not set, `lmlayer` uses an in-memory NumPy backend.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

With the default configuration, chat, embedding, and rerank upstreams are expected at `http://localhost:8080`, `:8081`, and `:8082`. Point them somewhere else with `MODEL`:

```bash
MODEL='{"qwen":{"cost":0,"instances":{"http://localhost:8080":1}},"qwenEmb":{"cost":0,"instances":{"http://localhost:8081":1},"embedding":true},"qwenRank":{"cost":0,"instances":{"http://localhost:8082":1},"rerank":true}}' \
uvicorn app:app --host 0.0.0.0 --port 8000
```

The service is now available at `http://localhost:8000` and accepts OpenAI-style requests:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}]}'
```

### Persistent storage

Start PostgreSQL with pgvector and pass its DSN to `lmlayer`:

```bash
docker run --network host -d --name lmlayer-db \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=postgres \
  pgvector/pgvector:pg18

DB=postgresql+asyncpg://root:root@127.0.0.1:5432/postgres \
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
cp .env.example .env
# Edit .env: DB and MODEL hosts must be reachable from inside the container.
docker compose up --build
```

`.env.example` is written for a host-run process (`127.0.0.1`). When using Compose, set the database host to `db` and upstream hosts to addresses the container can reach.

## What it can do now

- **OpenAI-compatible endpoints** — `/v1/chat/completions`, `/v1/embeddings`, and `/v1/rerank`, plus transparent pass-through for every other path.
- **Full-lifecycle safety** — regex and optional model-based checks on prompts, responses, streamed chunks, and tool calls.
- **Server-side sessions** — conversation history keyed by user and session, with history forking and cache/prompt/generation metrics.
- **Automatic RAG** — embed the query, retrieve candidates from pgvector or the local NumPy store, rerank them, and inject context into the prompt.
- **Tool loop** — Python and Bash tools executed through Docker or a host subprocess with a 60-second timeout.
- **Quota and cost accounting** — per-user, exponentially decaying cost budgets with quota enforcement.
- **Fair scheduling** — a priority lock per model, plus sticky routing to improve cache hits.
- **Streaming** — SSE streaming with checks while tokens arrive.
- **Flexible storage** — PostgreSQL/pgvector for production, in-memory storage for development.

## Architecture

```
OpenAI client
     │  /v1/chat/completions · /v1/embeddings · /v1/rerank · /*
     ▼
┌──────────────────────────── lmlayer ────────────────────────────┐
│  safety  →  RAG (embed → retrieve → rerank)  →  quota + lock    │
│     →  upstream request  →  stream / tool loop  →  response     │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
      PostgreSQL + pgvector              OpenAI-compatible upstreams
      (or in-memory NumPy)               (llama.cpp, vLLM, SGLang, …)
```

| Path | Responsibility |
| --- | --- |
| `app.py` | FastAPI routes, streaming responses, transparent pass-through |
| `core.py` | Chat/embedding/rerank pipelines, safety, RAG, cost and scheduling |
| `tool.py` | Tool schemas and subprocess execution |
| `util.py` | Priority lock and text chunking |
| `db/` | PostgreSQL and in-memory backends with a shared interface |
| `tests/smoke_import.py` | Imports the app and asserts the OpenAI-compatible surface |

## Configuration

All settings are environment variables. The complete, commented list lives in [`.env.example`](.env.example). The ones you are most likely to change:

| Variable | Purpose |
| --- | --- |
| `DB` | Async PostgreSQL DSN; omit it to use the in-memory backend |
| `MODEL` | JSON map of model names to upstream instances, cost, and roles |
| `EMBEDDING_DIM` | pgvector dimension (default `1024`) |
| `EMBEDDING`, `RERANK` | Retrieval candidate count and reranked result count |
| `SAFE_RE`, `SAFE_MODEL` | Regex safety pattern and optional safety-model endpoint |
| `TOOL` | Tools exposed to the model (default `python bash`) |
| `PROMPT`, `PROMPT_RAG` | System prompt and RAG prompt template |

## Development

Run the smoke check — it imports the app and verifies the three OpenAI-compatible routes without starting a database or contacting a model server:

```bash
python tests/smoke_import.py
```

The codebase is plain Python with no build step. `docker-compose.yml` provides a local pgvector + service stack, and [`docs/project-status.md`](docs/project-status.md) records the current implementation status and known boundaries.

## How to contribute

Issues and pull requests are welcome. Keep changes focused, run the smoke check before opening a PR, and update the docs when behavior or configuration changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the short version.

## Roadmap

**Implemented**

- [x] OpenAI-compatible chat, embedding, and rerank endpoints with pass-through
- [x] Prompt, response, streaming, and tool safety checks
- [x] Server-side session history and fork support
- [x] Auto RAG over pgvector or an in-memory NumPy store
- [x] Docker/host tool execution with a hard timeout
- [x] Per-user cost, quota enforcement, and decay
- [x] Priority-based fair scheduling and sticky upstream routing
- [x] SSE streaming with incremental safety checks

**Planned**

- [ ] Web search tool integration
- [ ] End-to-end tests for retrieval and safety pipelines
- [ ] Clarify the `python`/`bash` versus `docker/python`/`docker/bash` naming so the isolation mode is explicit
- [ ] Document a hardened deployment profile for tool isolation

The long-term aim is for `lmlayer` to be the invisible operational layer between an application and any model server: one endpoint that already knows about safety, memory, retrieval, tools, budgets, and fair access, while the model server stays replaceable.

## License

[AGPL-3.0](LICENSE).
