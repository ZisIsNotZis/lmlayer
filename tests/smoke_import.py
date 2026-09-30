"""Smoke test: import the app and assert the OpenAI-compatible surface exists.

No database or model server is contacted (the SQLAlchemy engine is lazy).

    uv run --with fastapi --with httpx --with httpx_sse \
       --with 'sqlalchemy[asyncio]' --with asyncpg --with pgvector \
       --with numpy --with python-multipart --with uvicorn \
       python tests/smoke_import.py

Exit code 0 = the service imports and serves the three OpenAI-compatible
endpoints this layer exists to provide.
"""
import os
import sys

os.environ.setdefault("DB", "postgresql+asyncpg://root:root@127.0.0.1:5432/postgres")

import app  # noqa: E402

REQUIRED = {"/v1/chat/completions", "/v1/embeddings", "/v1/rerank"}
paths = {getattr(r, "path", "") for r in app.app.routes}
missing = REQUIRED - paths
if missing:
    sys.exit(f"missing required routes: {sorted(missing)}")
print("OK: OpenAI-compatible surface present ->", sorted(REQUIRED))
