from .common import *
from typing import Coroutine
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from os import environ

_ENGINE = create_async_engine(environ['DB'])
_MAKER = async_sessionmaker(_ENGINE)


async def init():
    async with _ENGINE.begin() as connection:
        await connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS vector;')
        await connection.run_sync(Base.metadata.create_all)
        await connection.exec_driver_sql('CREATE INDEX ON embedding USING hnsw(embedding vector_ip_ops);')


async def _set(*o):
    async with _MAKER() as session:
        session.add_all(o)
        await session.commit()


async def getUser(id: str) -> User:
    async with _MAKER() as session:
        return (await session.scalars(select(User).where(User.id == id))).first() or User(id=id, quota=0)


def setUser(user: User) -> Coroutine:
    return _set(user)


async def getMessages(user: str, sessionId: str) -> list[DbMessage]:
    async with _MAKER() as session:
        return list((await session.scalars(select(DbMessage).where(DbMessage.user == user and DbMessage.session == sessionId).order_by(DbMessage.create_at))).all())


def addMessages(*messages: DbMessage) -> Coroutine:
    return _set(*messages)


async def getDocuments(model: str) -> list[str]:
    async with _MAKER() as session:
        return list((await session.scalars(select(Embedding.document).where(Embedding.model == model).distinct())).all())


async def queryEmbeddings(model: str, embedding: list[float], n: int = 5) -> list[str]:
    async with _MAKER() as session:
        return list((await session.scalars(select(Embedding.text).where(Embedding.model == model).order_by(Embedding.embedding.op('<#>')(embedding)).limit(n))).all())


def addEmbeddings(embeddings: list[Embedding]) -> Coroutine:
    return _set(*embeddings)


async def delEmbeddings(model: str, document: str):
    async with _MAKER() as session:
        await session.execute(delete(Embedding).where(Embedding.model == model and Embedding.document == document))
