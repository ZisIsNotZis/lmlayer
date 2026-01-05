from fastapi import FastAPI, Response as AppResponse, Header
from fastapi.responses import StreamingResponse
from typing import AsyncIterable, Annotated
from core import *
app = FastAPI()


async def sse(gen: AsyncIterable[BaseModel]) -> AsyncIterable[str]:
    async for i in gen:
        yield 'data:'+i.model_dump_json()+'\n\n'


@app.post('/v1/chat/completions', response_model=None)
async def chat(request: Request, authorization: Annotated[str | None, Header()]) -> Response | AppResponse | StreamingResponse:
    response = run(authorization or '', request)
    try:
        return StreamingResponse(sse(response)) if request.stream else await anext(aiter(response))
    except AssertionError as e:
        return AppResponse(e.args[0], 422)
