from time import time
from typing import AsyncIterator
from asyncio import Task, gather
from httpx import AsyncClient, Headers
from httpx_sse import aconnect_sse
from re import compile
from typing import TypedDict, NotRequired, Annotated
from json import loads, dumps, JSONDecodeError
from traceback import format_exc
from os import getenv
from tool import *
from util import *
from db import *


class Model(TypedDict):
    instances: dict[str, int]
    cost: NotRequired[float]
    prompt: NotRequired[str]
    chat: NotRequired[Annotated[bool, True]]
    embedding: NotRequired[bool]
    rerank: NotRequired[bool]


class Message(TypedDict):
    role: str
    content: NotRequired[str | None]
    reasoning_content: NotRequired[str]
    tool_calls: NotRequired[list[Tool]]
    name: NotRequired[str]


class ChatRequest(TypedDict):
    model: NotRequired[str]
    messages: list[Message]
    session: NotRequired[str]
    stream: NotRequired[bool]
    tools: NotRequired[list[Tool]]
    sessionFork: NotRequired[str]
    embeddingModel: NotRequired[str]
    embedding: NotRequired[int]
    rerankModel: NotRequired[str]
    rerank: NotRequired[int]


class Choice(TypedDict):
    delta: Message
    message: Message


class Timings(TypedDict):
    cache_n: NotRequired[int]
    prompt_n: NotRequired[int]
    predicted_n: NotRequired[int]
    prompt_ms: NotRequired[float]
    predicted_ms: NotRequired[float]


class Usage(TypedDict):
    completion_tokens: NotRequired[int]
    prompt_tokens: NotRequired[int]
    total_tokens: NotRequired[int]


class ChatResponse(TypedDict):
    model: NotRequired[str]
    choices: list[Choice]
    usage: NotRequired[Usage]
    timings: NotRequired[Timings]
    messages: NotRequired[list[Message]]
    cost: NotRequired[float]


class EmbeddingRequest(TypedDict):
    model: NotRequired[str]
    input: str | list[str]
    document: NotRequired[str]
    chunk: NotRequired[int]


class EmbeddingData(TypedDict):
    embedding: list[float]


class EmbeddingResponse(TypedDict):
    model: str
    data: list[EmbeddingData]
    usage: NotRequired[Usage]
    cost: float


class RerankRequest(TypedDict):
    model: NotRequired[str]
    documents: list[str]
    top_n: int


class RerankResult(TypedDict):
    index: int


class RerankResponse(TypedDict):
    model: str
    results: list[RerankResult]
    usage: NotRequired[Usage]
    cost: float


def addCost(user: str, cost: float = 0.) -> float:
    v, t = _COST.get(user, (0., 0.))
    v = v*_TIME_FAC**(t-time())+cost
    _COST[user] = v, t
    return v


MODEL: dict[str, Model] = loads(getenv('MODEL', '{"qwen":{"cost":0,"instances":{"http://localhost:8080":1}}}'))
_LOCK = {i: PriorityLock(sum(i for i in j['instances'].values()))for i, j in MODEL.items()}
_COST = dict[str, tuple[float, float]]()
_STICK = dict[tuple[str, str, str, str], float]()
# You are a professional, safety-focused assistant. You must follow these rules:
# 1) First, read the user query from the User Query. Do not assume any prior context.
# 2) If the query is underspecified and you cannot generate a valid answer without clarification, ask a follow-up question. Do not generate an answer until the user clarifies.
# 3) Determine whether the RAG contains retrieved text.
#    • If RAG contains text, you should prioritize information from that text.
#    • Only use outside knowledge if the RAG text does not contain relevant content and the user has explicitly accepted generation without RAG.
# 4) You must never hallucinate.
#    • If relying on RAG, answer only with facts supported by the provided text.
#    • If details are missing in RAG and the query is about specific facts/events not covered, say:
#      “I cannot find verified information in the provided RAG content about that specific item.”
#    • If no RAG is present, you may answer from general public knowledge, but always add a brief statement about the source — e.g., “Based on general knowledge ...”.
# 5) When using RAG content:
#    • Quote exact spans from RAG that support your facts.
#    • Provide a “Source” line listing the passage identifiers.
# 6) If multiple interpretations are possible, ask a clarifying question.
_PROMPT = getenv('PROMPT', 'You are a helpful assistant')
# User Query:
# ```
# {user}
# ```

# Retrieved Evidence (if any):
# ```
# {RAG}
# ```
_PROMPT_RAG = getenv('PROMPT_RAG', "{user}\n\nHere's some possibly related information: {RAG}")
_PP_FAC = float(getenv('PP_FAC', 1.))
_TIME_FAC: float = float(getenv('TIME_FAC', .5**(1/-3600)))
CLIENT = AsyncClient(timeout=300)
_SAFE_RE = compile(getenv('SAFE_RE', '炸弹'))
_SAFE_MODEL = getenv('SAFE_MODEL', '')
_SAFE_PERIOD = int(getenv('SAFE_PERIOD', 50))
_CHUNK = int(getenv('CHUNK', 1024))
_EMBEDDING = int(getenv('EMBEDDING', 0))
_RERANK = int(getenv('RERANK', 0))
_TOOL = getenv('TOOL', '').split() or []


async def chat(userId: str, headers: dict[str, str], req: ChatRequest) -> AsyncIterator[Headers | ChatResponse]:
    session = req.get('session', '')
    if not req['messages']:
        yield Headers()
        yield ChatResponse(choices=[], messages=sum(([Message(role=i.role, content=i.user), Message(role='assistant', content=i.content)] for i in await getMessages(userId, session)), []))
        return
    messages = await getMessages(userId, session)
    if sessionFork := req.get('sessionFork'):
        session = sessionFork
        for i in messages:
            i.session = sessionFork
        Task(addMessages(*messages))
    content = _SAFE_RE.sub('', req['messages'][-1].get('content') or '')
    cost = 0.
    await safeChk(content)
    if nembedding := req.get('embedding', _EMBEDDING):
        try:
            _, emb = await embedding(userId, headers, EmbeddingRequest(input=content, model=req.get('embeddingModel') or ''), False)
            documents = await queryEmbeddings(emb['model'], emb['data'][0]['embedding'], nembedding)
            cost += emb['cost']
            if len(documents) > (nrerank := req.get('rerank', _RERANK)):
                try:
                    _, rank = await rerank(userId, headers, RerankRequest(documents=documents, top_n=nrerank, model=req.get('rerankModel', '')))
                    documents = [documents[i['index']] for i in rank['results']]
                    cost += rank['cost']
                except AssertionError:
                    pass
            if len(documents):
                content = _PROMPT_RAG.format(user=messages, RAG='\n\n'.join(documents))
        except AssertionError:
            pass
    model = req.get('model', next(i for i, j in MODEL.items()if j.get('chat', True)))
    if lock := _LOCK[model].acquire(addCost(userId)):
        await lock
    url = max([i for i, j in MODEL[model]['instances'].items()if j], key=lambda i: _STICK.get((model, i, userId, session), 0)+MODEL[model]['instances'][i])
    MODEL[model]['instances'][url] -= 1
    try:
        user = await getUser(userId)
        assert user.quota >= 0, 'no balance'
        req['messages'] = [Message(role='system', content=MODEL[model].get('prompt', _PROMPT))] + sum(([Message(role=i.role, content=i.userMsg), Message(role='assistant', content=i.content)]for i in messages), []) + [{'role': 'user', 'content': content}]
        req['tools'] = [TOOL[i] for i in _TOOL]
        while True:
            rsp: ChatResponse = ChatResponse(choices=[])
            tool = ''
            arguments = ''
            print(url, req)
            if req.get('stream'):
                message = Message(role='')
                reasoning_content = ''
                content = ''
                task = list[Task]()
                async with aconnect_sse(CLIENT, 'POST', url+'/v1/chat/completions', headers=headers, json=req) as sses:
                    if req['messages'][-1]['role'] == 'user':
                        yield sses.response.headers
                    try:
                        async for sse in sses.aiter_sse():
                            rsp: ChatResponse = sse.json()
                            message = rsp['choices'][0]['delta']
                            reasoning_content += message.get('reasoning_content', '')
                            content += message.get('content') or ''
                            if len(content)/_SAFE_PERIOD > len(task):
                                assert not _SAFE_RE.findall(content), 'unsafe'
                                task.append(Task(safeChk(content)))
                            if tool_calls := message.get('tool_calls'):
                                tool += tool_calls[0]['function']['name']
                                arguments += tool_calls[0]['function']['arguments']
                            for i in task:
                                if i.done() and i.exception():
                                    await i
                            yield rsp
                    except JSONDecodeError:
                        pass
                await gather(*task)
                print(reasoning_content, content, tool, arguments)
            else:
                httpRsp = await CLIENT.post(url+'/v1/chat/completions', headers=headers, json=req)
                if req['messages'][-1]['role'] == 'user':
                    yield httpRsp.headers
                rsp: ChatResponse = httpRsp.json()
                print(rsp)
                message = rsp['choices'][0]['message']
                reasoning_content = message.get('reasoning_content', '')
                content = _SAFE_RE.sub('', message.get('content') or '')
                await safeChk(content)
                if tool_calls := message.get('tool_calls'):
                    tool = tool_calls[0]['function']['name']
                    arguments = tool_calls[0]['function']['arguments']
            timings = rsp.get('timings', Timings())
            usage = rsp.get('usage', Usage())
            predicted_n = timings.get('predicted_n', usage.get('completion_tokens', 0))
            prompt_n = timings.get('prompt_n', usage.get('prompt_tokens', usage.get('total_tokens', 0)-predicted_n))
            cost = (prompt_n*_PP_FAC + predicted_n)*MODEL[model].get('cost', 1.)
            addCost(userId, cost)
            user.quota -= cost
            Task(setUser(user))
            Task(addMessages(DbMessage(
                user=userId,
                session=session,
                role=req['messages'][-1]['role'],
                userMsg=req['messages'][-1].get('content') or '',
                reasoning_content=reasoning_content,
                content=content,
                cache_n=timings.get('cache_n', 0),
                prompt_n=prompt_n,
                predicted_n=predicted_n,
                cost=cost,
                prompt_ms=timings.get('prompt_ms', 0.),
                predicted_ms=timings.get('predicted_ms', 0.),
                tool=tool,
                arguments=arguments,
                time=time()
            )))
            if tool:
                try:
                    result = dumps(await doTool(tool, **loads(arguments)), separators=(',', ':'), ensure_ascii=False)
                except Exception:
                    result = format_exc()
                req['messages'].append(message)
                req['messages'].append(Message(role='tool', name=tool, content=result))
            else:
                if not req.get('stream'):
                    yield rsp
                break
    finally:
        MODEL[model]['instances'][url] += 1
        _LOCK[model].release()


async def embedding(userId: str, headers: dict[str, str], req: EmbeddingRequest, add=True) -> tuple[Headers, EmbeddingResponse]:
    model = req.get('model') or next((i for i, j in MODEL.items()if j.get('embedding')), '')
    assert model in MODEL, 'no support'
    if lock := _LOCK[model].acquire(addCost(userId)):
        await lock
    url = max([i for i, j in MODEL[model].items()if j], key=lambda i: MODEL[model][i])
    MODEL[model]['instances'][url] -= 1
    try:
        user = await getUser(userId)
        assert user.quota >= 0, 'no balance'
        input = req['input']
        await safeChk(input if isinstance(input, str)else ''.join(input))
        input = req['input'] = list(chunkit(input, req.get('chunk', _CHUNK if add else 99999)))if isinstance(input, str)else input
        print(url, req)
        httpRsp = await CLIENT.post(url+'/v1/embeddings', headers=headers, json=req)
        rsp: EmbeddingResponse = httpRsp.json()
        print(rsp)
        rsp['model'] = model
        usage = rsp.get('usage', Usage())
        prompt_n = usage.get('prompt_tokens', usage.get('total_tokens', 0))
        cost = prompt_n*_PP_FAC*MODEL[model].get('cost', 1.)
        user.quota -= cost
        Task(setUser(user))
        if add:
            Task(addEmbeddings([Embedding(
                model=model,
                document=req.get('document', ''),
                embedding=i['embedding'],
                text=j,
                prompt_n=prompt_n,
                cost=cost,
                time=time()
            )for i, j in zip(rsp['data'], input)]))
        return httpRsp.headers, rsp
    finally:
        MODEL[model]['instances'][url] += 1
        _LOCK[model].release()


async def rerank(userId: str, headers: dict[str, str], req: RerankRequest) -> tuple[Headers, RerankResponse]:
    model = req.get('model') or next((i for i, j in MODEL.items()if j.get('rerank')), '')
    assert model in MODEL, 'no support'
    if lock := _LOCK[model].acquire(addCost(userId)):
        await lock
    url = max([i for i, j in MODEL[model].items()if j], key=lambda i: MODEL[model][i])
    MODEL[model]['instances'][url] -= 1
    try:
        user = await getUser(userId)
        assert user.quota >= 0, 'no balance'
        print(url, req)
        httpRsp = await CLIENT.post(url+'/v1/rerank', headers=headers, json=req)
        rsp: RerankResponse = httpRsp.json()
        print(rsp)
        usage = rsp.get('usage', Usage())
        prompt_n = usage.get('prompt_tokens', usage.get('total_tokens', 0))
        cost = prompt_n*_PP_FAC*MODEL[model].get('cost', 1.)
        user.quota -= cost
        Task(setUser(user))
        return httpRsp.headers, rsp
    finally:
        MODEL[model]['instances'][url] += 1
        _LOCK[model].release()


async def safeChk(s: str):
    assert not _SAFE_MODEL or (await CLIENT.post(_SAFE_MODEL, json={'text': s})).json()['safe'], 'unsafe'
