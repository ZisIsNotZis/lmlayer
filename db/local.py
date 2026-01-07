from .common import *
import numpy as np

_USER = dict[str, User]()
_MESSAGE = dict[str, dict[str, list[DbMessage]]]()
_EMBEDDING = dict[str, list[Embedding]]()


async def init():
    pass


async def getUser(id: str) -> User:
    return _USER.get(id, User(quota=0))


async def setUser(user: User) -> None:
    _USER[user.id] = user


async def getMessages(user: str, session: str) -> list[DbMessage]:
    return _MESSAGE.get(user, {}).get(session, [])


async def addMessages(*messages: DbMessage) -> None:
    _MESSAGE.setdefault(messages[0].user, {}).setdefault(messages[0].session, []).extend(messages)


async def getDocuments(model: str) -> set[str]:
    return {i.document for i in _EMBEDDING.setdefault(model, [])}


async def queryEmbeddings(model: str, embedding: list[float], n: int = 5) -> list[str]:
    text = _EMBEDDING.setdefault(model, [])
    if len(text) > n:
        text = [text[i]for i in (np.array([i.embedding for i in text], 'f')@np.array(embedding, 'f')).argpartition(-n)[-n:]]
    return [i.text for i in text]


async def addEmbeddings(embeddings: list[Embedding]) -> None:
    _EMBEDDING.setdefault(embeddings[0].model, []).extend(embeddings)


async def delEmbeddings(model: str, document: str) -> None:
    _EMBEDDING[model] = [i for i in _EMBEDDING.get(model, [])if i != document]
