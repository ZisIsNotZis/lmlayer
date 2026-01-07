"""In-memory local storage used as a fallback or for testing.

Provides the same API as the Postgres backend but stores everything in memory.
This is useful for simple local development or when a database is unavailable.
"""

from .common import *
import numpy as np

_USER = dict[str, User]()
_MESSAGE = dict[str, dict[str, list[DbMessage]]]()
_EMBEDDING = dict[str, list[Embedding]]()


async def init():
    """No-op initializer for the in-memory backend."""
    pass


async def getUser(id: str) -> User:
    """Return the `User` for `id` or a default user with zero quota."""
    return _USER.get(id, User(quota=0))


async def setUser(user: User) -> None:
    """Persist `user` in the in-memory store."""
    _USER[user.id] = user


async def getMessages(user: str, session: str) -> list[DbMessage]:
    """Return stored messages for `user` and `session`."""
    return _MESSAGE.get(user, {}).get(session, [])


async def addMessages(*messages: DbMessage) -> None:
    """Append messages to the in-memory session history."""
    _MESSAGE.setdefault(messages[0].user, {}).setdefault(messages[0].session, []).extend(messages)


async def getDocuments(model: str) -> set[str]:
    """Return the set of documents stored for a model."""
    return {i.document for i in _EMBEDDING.setdefault(model, [])}


async def queryEmbeddings(model: str, embedding: list[float], n: int = 5) -> list[str]:
    """Return top-`n` documents by cosine-similarity in-memory using numpy."""
    text = _EMBEDDING.setdefault(model, [])
    if len(text) > n:
        # Use vectorized dot product and argpartition to efficiently select top-n
        text = [text[i]for i in (np.array([i.embedding for i in text], 'f')@np.array(embedding, 'f')).argpartition(-n)[-n:]]
    return [i.text for i in text]


async def addEmbeddings(embeddings: list[Embedding]) -> None:
    """Add embeddings to the in-memory store for the corresponding model."""
    _EMBEDDING.setdefault(embeddings[0].model, []).extend(embeddings)


async def delEmbeddings(model: str, document: str) -> None:
    """Delete embeddings that match `document` for `model`.

    Note: this implementation compares stored objects with `document` directly; if
    `document` is a string it may need to compare against `i.document` instead.
    """
    _EMBEDDING[model] = [i for i in _EMBEDDING.get(model, [])if i != document]
