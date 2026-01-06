from .common import *
import numpy as np

_USER = dict[str, User]()
_MESSAGE = dict[str, dict[str, list[DbMessage]]]()
_DOCUMENT = dict[str, dict[str, list[Embedding]]]()


async def getUser(id: str) -> User:
    return _USER.get(id, User(quota=0))


async def setUser(user: User) -> None:
    _USER[user.id] = user


async def getMessages(user: str, session: str) -> list[DbMessage]:
    return _MESSAGE.get(user, {}).get(session, [])


async def addMessages(*messages: DbMessage) -> None:
    _MESSAGE.setdefault(messages[0].user, {}).setdefault(messages[0].session, []).extend(messages)


async def getDocuments(model: str) -> list[str]:
    return list(_DOCUMENT.setdefault(model, {}))


async def queryEmbeddings(model: str, embedding: list[float], n: int = 5) -> list[str]:
    score = np.stack([np.array(i.embedding, 'f') for i in _DOCUMENT[model].values()for i in i])@np.array(embedding, 'f')
    text = [i.text for i in _DOCUMENT[model].values()for i in i]
    return [text[i] for i in score.argpartition(-n-1)[-n:]]


async def addEmbeddings(embeddings: list[Embedding]) -> None:
    _DOCUMENT.setdefault(embeddings[0].model, {})[embeddings[0].document] = embeddings


async def delEmbeddings(model: str, document: str) -> None:
    _DOCUMENT.setdefault(model, {}).pop(document, None)
