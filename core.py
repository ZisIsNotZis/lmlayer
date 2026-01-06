from time import time
from typing import AsyncIterator
from asyncio import Task, Future
from db import *
from httpx import AsyncClient, Request as HttpRequest
from httpx_sse import aconnect_sse
from re import compile
from pydantic import BaseModel, Field
from json import loads, dumps, JSONDecodeError
from traceback import format_exc
from os import getenv
from tool import *
from util import *


class Instance(BaseModel):
    num: int = 1


class Model(BaseModel):
    instances: dict[str, Instance]
    prompt: str = 'You are a helpful assistant'
    cost: float = 1


class User(BaseModel):
    _cost: float = 0
    _costTime: float = 0

    @property
    def cost(self):
        return self._cost*TIME_FAC**(self._costTime-time())

    @cost.setter
    def cost(self, cost):
        self._cost = cost
        self._costTime = time()


class Message(BaseModel):
    role: str = ''
    content: str | None = ''
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[Tool] | None = None


class Request(BaseModel):
    model: str
    messages: list[Message]
    sessionId: int = 0
    stream: bool = False
    tools: list[Tool] = list(TOOL.values())


class Choice(BaseModel):
    delta: Message = Message()
    message: Message = Message()


class Timings(BaseModel):
    cache_n: int = 0
    prompt_n: int = 0
    predicted_n: int = 0
    prompt_ms: float = 0
    predicted_ms: float = 0


class Response(BaseModel):
    choices: list[Choice] = Field(default_factory=lambda: [Choice()])
    timings: Timings = Timings()


async def logRequest(request: HttpRequest):
    print(request.method, request.url, request.content and request.content.decode())


MODEL = {str(i): Model(**j) for i, j in loads(getenv('MODELS', '{"qwen":{"instances":{"http://localhost:8080/v1/chat/completions":{}},"cost":0}}')).items()}
LOCK = {i: PriorityLock(sum(i.num for i in j.instances.values()))for i, j in MODEL.items()}
USER = dict[str, User]()
JOB = dict[str, list[tuple[float, str, Request, Future[AsyncIterator[Response]]]]]()
STICK = dict[tuple[str, str, str, int], float]()
PP_FAC = float(getenv('PP_FAC', 1))
TIME_FAC: float = float(getenv('TIME_FAC', .5**(1/-3600)))
CLIENT = AsyncClient(timeout=300, event_hooks={'request': [logRequest]})
SAFE_RE = compile(getenv('SAFE_RE', '炸弹'))
SAFE_MODEL = getenv('SAFE_MODE', '')
SAFE_PERIOD = int(getenv('SAFE_PERIOD', 20))


async def run(user: str, request: Request) -> AsyncIterator[Response]:
    lock = LOCK[request.model].acquire(USER.get(user, User()).cost)
    if lock:
        await lock
    url, instance = max([(i, j)for i, j in MODEL[request.model].instances.items()if j.num], key=lambda i: STICK.get((request.model, i[0], user, request.sessionId), 0)+i[1].num)
    instance.num -= 1
    try:
        safe = [Task(hasBalance(user)), safeChk(request.messages[-1].content or '')]
        role = 'user'
        request.messages = (
            [Message(role='system', content=MODEL[request.model].prompt)] +
            sum(([Message(role=i, content=j), Message(role='assistant', content=j)]for i, j, k in await getSession(user, request.sessionId)), []) +
            [Message(role=role, content=request.messages[-1].content)]
        )
        while role:
            response = Response()
            tool = ''
            arguments = ''
            if request.stream:
                message = Message()
                reasoning_content = ''
                content = ''
                async with aconnect_sse(CLIENT, 'POST', url, json=request.model_dump(exclude_none=True)) as sses:
                    try:
                        async for sse in sses.aiter_sse():
                            response = Response(**sse.json())
                            message = response.choices[0].delta
                            reasoning_content += message.reasoning_content or ''
                            content += message.content or ''
                            if len(content)/SAFE_PERIOD > len(safe):
                                safe.append(safeChk(content))
                            if message.tool_calls:
                                tool += message.tool_calls[0].function.name
                                arguments += message.tool_calls[0].function.arguments or ''
                            assert all(not i.done() or i.result()for i in safe), 'no money/unsafe'
                            yield response
                    except JSONDecodeError:
                        pass
            else:
                responseObj = (await CLIENT.post(url, json=request.model_dump(exclude_none=True))).json()
                response = Response(**responseObj)
                message = response.choices[0].message
                reasoning_content = message.reasoning_content
                content = message.content or ''
                safe.append(safeChk(content))
                if message.tool_calls:
                    tool = message.tool_calls[0].function.name
                    arguments = message.tool_calls[0].function.arguments or ''
            print(reasoning_content, content, tool, arguments)
            assert await safe[0] and await safe[1] and await safe[-1], 'no money/unsafe'
            cost = MODEL[request.model].cost*(response.timings.prompt_n*PP_FAC + response.timings.predicted_n)
            USER.setdefault(user, User()).cost += cost
            Task(addWallet(user, -cost))
            Task(addSession(user, request.sessionId, Session(
                role=role,
                user=request.messages[-1].content or '',
                reasoning_content=reasoning_content or '',
                content=content,
                tool=tool,
                arguments=arguments,
                **response.timings.model_dump()
            )))
            role = ''
            if tool:
                role = 'tool'
                try:
                    result = dumps(await TOOL[tool].function.run(**loads(arguments)), separators=(',', ':'), ensure_ascii=False)
                except Exception:
                    result = format_exc()
                request.messages += message, Message(role='tool', name=tool, content=result)
            elif not request.stream:
                yield response
    finally:
        instance.num += 1
        LOCK[request.model].release()


def safeChk(s: str) -> Future[bool]:
    assert not SAFE_RE.findall(s), 'unsafe'
    future = Future[bool]()
    if s and SAFE_MODEL:
        Task(CLIENT.post(SAFE_MODEL, json={'text': s})).add_done_callback(lambda task: getattr(future, 'set_'+('exception'if task.exception()else 'result'))(task.exception() or task.result()))
    else:
        future.set_result(True)
    return future
