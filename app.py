"""HTTP API endpoints for the service.

Provides FastAPI routes for chat, embeddings, reranking and a simple passthrough handler.
Each route converts incoming requests, calls business logic in `core` and returns JSON or streaming SSE.
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncIterable
from random import choice
from core import *
app = FastAPI()


def clean(headers) -> dict[str, str]:
    """Normalize and sanitize headers for internal requests.

    Removes hop-by-hop and dynamic headers that should not be forwarded.
    Returns a plain dict of headers suitable for httpx and FastAPI responses.
    """
    headers = dict(headers)
    # Remove headers that may interfere with content handling or cause spurious differences
    headers.pop('content-length', None)
    headers.pop('content-encoding', None)
    return headers


async def sse(rsp: AsyncIterable) -> AsyncIterable[str]:
    """Turn an async iterable of JSON-able objects into SSE (text) frames.

    Yields server-sent events (SSE) encoded as text frames.
    """
    async for i in rsp:
        yield 'data: '+dumps(i, separators=(',', ':'), ensure_ascii=False)+'\n\n'


@app.on_event('startup')
async def startup():
    """Application startup hook.

    Initializes the storage backend (database or local store).
    """
    await init()


@app.post('/v1/chat/completions')
async def doChat(httpReq: Request) -> Response:
    """HTTP endpoint to handle chat completion requests.

    Accepts a `ChatRequest` body, delegates to `core.chat`, and returns either
    a streaming SSE response (if `stream` is enabled) or a single JSON response.
    The `x-user-id` header is used to identify the user.
    """
    req: ChatRequest = await httpReq.json()
    response = chat(httpReq.headers.get('x-user-id', ''), clean(httpReq.headers), req)
    headers = clean(await anext(response))
    return StreamingResponse(sse(response), 200, headers) if req.get('stream') else JSONResponse(await anext(response), 200, headers)


@app.post('/v1/embeddings')
async def doEmbedding(httpReq: Request) -> JSONResponse:
    """HTTP endpoint to create embeddings for input text.

    Delegates to `core.embedding` and returns the embedding response as JSON.
    """
    headers, rsp = await embedding(httpReq.headers.get('x-user-id', ''), clean(httpReq.headers), await httpReq.json())
    return JSONResponse(rsp, 200, clean(headers))


@app.post('/v1/rerank')
async def doRerank(httpReq: Request) -> JSONResponse:
    """HTTP endpoint to rerank a list of documents for a query.

    Delegates to `core.rerank` and returns the ranked results as JSON.
    """
    headers, rsp = await rerank(httpReq.headers.get('x-user-id', ''), clean(httpReq.headers), await httpReq.json())
    return JSONResponse(rsp, 200, clean(headers))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def forward_unregistered_requests(req: Request, path: str) -> StreamingResponse:
    """Pass-through for any unregistered route to a downstream model instance.

    Forwards the request method, path and body to a chosen model instance and streams back the response.
    """
    rsp = await CLIENT.request(req.method, choice(list(choice(list(MODEL.values()))['instances']))+'/'+path, params=req.query_params, headers=clean(req.headers), content=await req.body())
    return StreamingResponse(rsp.aiter_bytes(), rsp.status_code, clean(rsp.headers))
