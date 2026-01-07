from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncIterable
from random import choice
from core import *
app = FastAPI()


def clean(headers) -> dict[str, str]:
    headers = dict(headers)
    headers.pop('content-length', None)
    headers.pop('content-encoding', None)
    return headers


async def sse(rsp: AsyncIterable) -> AsyncIterable[str]:
    async for i in rsp:
        yield 'data: '+dumps(i, separators=(',', ':'), ensure_ascii=False)+'\n\n'


@app.on_event('startup')
async def startup():
    await init()


@app.post('/v1/chat/completions')
async def doChat(httpReq: Request) -> Response:
    req: ChatRequest = await httpReq.json()
    response = chat(httpReq.headers.get('x-user-id', ''), clean(httpReq.headers), req)
    headers = clean(await anext(response))
    return StreamingResponse(sse(response), 200, headers) if req.get('stream') else JSONResponse(await anext(response), 200, headers)


@app.post('/v1/embeddings')
async def doEmbedding(httpReq: Request) -> JSONResponse:
    headers, rsp = await embedding(httpReq.headers.get('x-user-id', ''), clean(httpReq.headers), await httpReq.json())
    return JSONResponse(rsp, 200, clean(headers))


@app.post('/v1/rerank')
async def doRerank(httpReq: Request) -> JSONResponse:
    headers, rsp = await rerank(httpReq.headers.get('x-user-id', ''), clean(httpReq.headers), await httpReq.json())
    return JSONResponse(rsp, 200, clean(headers))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def forward_unregistered_requests(req: Request, path: str) -> StreamingResponse:
    rsp = await CLIENT.request(req.method, choice(list(choice(list(MODEL.values()))['instances']))+'/'+path, params=req.query_params, headers=clean(req.headers), content=await req.body())
    return StreamingResponse(rsp.aiter_bytes(), rsp.status_code, clean(rsp.headers))
