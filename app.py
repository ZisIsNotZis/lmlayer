from fastapi import FastAPI, HTTPException, Request, Response
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


@app.post('/v1/chat/completions')
async def doChat(req: Request) -> Response:
    reqJson: ChatRequest = await req.json()
    response = chat(req.headers.get('x-user-id') or '', clean(req.headers), reqJson)
    headers = clean(await anext(response))
    try:
        return StreamingResponse(sse(response), 200, headers) if reqJson.get('stream') else JSONResponse(await anext(response), 200, headers)
    except AssertionError as e:
        raise HTTPException(422, e.args[0])


@app.post('/v1/embeddings')
async def doEmbedding(req: Request) -> Response:
    try:
        headers, rsp = await embedding(req.headers.get('x-user-id') or '', clean(req.headers), await req.json())
        return JSONResponse(rsp, 200, clean(headers))
    except AssertionError as e:
        raise HTTPException(422, e.args[0])


@app.post('/v1/rerank')
async def doRerank(req: Request) -> Response:
    try:
        headers, rsp = await rerank(req.headers.get('x-user-id') or '', clean(req.headers), await req.json())
        return JSONResponse(rsp, 200, clean(headers))
    except AssertionError as e:
        raise HTTPException(422, e.args[0])


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def forward_unregistered_requests(req: Request, path: str) -> Response:
    rsp = await CLIENT.request(req.method, choice(list(choice(list(MODEL.values()))['instances']))+'/'+path, params=req.query_params, headers=clean(req.headers), content=await req.body())
    return StreamingResponse(rsp.aiter_bytes(), rsp.status_code, clean(rsp.headers))
