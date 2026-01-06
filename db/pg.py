from .common import *
from typing import Coroutine
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from os import environ

DB = environ['DB']  # 'postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres'
SESSION: AsyncSession | None = None


async def _getSession() -> AsyncSession:
    global SESSION
    if not SESSION:
        SESSION = await AsyncSession(create_async_engine(DB)).__aenter__()
    return SESSION


async def _set(*o):
    (session := await _getSession()).add_all(o)
    await session.commit()


async def getUser(id: str) -> User:
    return (await (await _getSession()).scalars(select(User).where(User.id == id))).first() or User()


def setUser(user: User) -> Coroutine:
    return _set(user)


async def getMessages(user: str, session: str) -> list[DbMessage]:
    return list((await (await _getSession()).scalars(select(DbMessage).where(DbMessage.user == user and DbMessage.session == session).order_by(DbMessage.sequence))).all())


def addMessages(*messages: DbMessage) -> Coroutine:
    return _set(*messages)


async def getDocuments(model: str) -> list[str]:
    return list((await (await _getSession()).scalars(select(Embedding.document).where(Embedding.model == model).distinct())).all())


async def queryEmbeddings(model: str, embedding: list[float], n: int = 5) -> list[str]:
    return list((await (await _getSession()).scalars(select(Embedding.text).where(Embedding.model == model).order_by(Embedding.embedding.op('<->')(embedding)).limit(n))).all())


def addEmbeddings(embeddings: list[Embedding]) -> Coroutine:
    return _set(*embeddings)


async def delEmbeddings(model: str, document: str):
    await (await _getSession()).execute(delete(Embedding).where(Embedding.model == model and Embedding.document == document))
