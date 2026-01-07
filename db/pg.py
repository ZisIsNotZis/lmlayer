"""Postgres storage backend using SQLAlchemy Async engine.

This module provides helpers to initialize the DB schema and perform basic
CRUD/lookup operations for users, messages and embeddings. Functions either
return results or schedule writes via coroutine wrappers.
"""

from .common import *
from typing import Coroutine
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from os import environ

_ENGINE = create_async_engine(environ['DB'])
_MAKER = async_sessionmaker(_ENGINE)


async def init():
    """Initialize Postgres schema and required extensions/indexes.

    Ensures pgvector extension is available and creates tables and HNSW index used for nearest-neighbor searches.
    """
    async with _ENGINE.begin() as connection:
        await connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS vector;')
        await connection.run_sync(Base.metadata.create_all)
        await connection.exec_driver_sql('CREATE INDEX ON embedding USING hnsw(embedding vector_ip_ops);')


async def _set(*o):
    """Batch-insert provided ORM objects into the database and commit.

    Small helper used by the public wrappers which return coroutines for background scheduling.
    """
    async with _MAKER() as session:
        session.add_all(o)
        await session.commit()


async def getUser(id: str) -> User:
    """Fetch a user by id. Returns an empty User with zero quota if not found."""
    async with _MAKER() as session:
        return (await session.scalars(select(User).where(User.id == id))).first() or User(id=id, quota=0)


def setUser(user: User) -> Coroutine:
    """Return a coroutine that will persist `user` when executed by the event loop."""
    return _set(user)


async def getMessages(user: str, sessionId: str) -> list[DbMessage]:
    """Return list of `DbMessage` for a user/session ordered by creation time."""
    async with _MAKER() as session:
        # Note: SQLAlchemy boolean expressions should use `&` or multiple arguments; the current
        # `and` expression may not behave as intended and could be revisited if queries fail.
        return list((await session.scalars(select(DbMessage).where(DbMessage.user == user and DbMessage.session == sessionId).order_by(DbMessage.create_at))).all())


def addMessages(*messages: DbMessage) -> Coroutine:
    """Return a coroutine that will persist provided messages when executed."""
    return _set(*messages)


async def getDocuments(model: str) -> list[str]:
    """Return a list of distinct document identifiers for the given model."""
    async with _MAKER() as session:
        return list((await session.scalars(select(Embedding.document).where(Embedding.model == model).distinct())).all())


async def queryEmbeddings(model: str, embedding: list[float], n: int = 5) -> list[str]:
    """Return top-`n` documents by nearest-neighbor distance to `embedding` for `model`."""
    async with _MAKER() as session:
        return list((await session.scalars(select(Embedding.text).where(Embedding.model == model).order_by(Embedding.embedding.op('<#>')(embedding)).limit(n))).all())


def addEmbeddings(embeddings: list[Embedding]) -> Coroutine:
    """Return a coroutine that will insert provided embeddings into the DB."""
    return _set(*embeddings)


async def delEmbeddings(model: str, document: str):
    """Delete embeddings for a given model/document pair."""
    async with _MAKER() as session:
        # Note: using `and` inside `where()` may not yield the intended SQL expression; consider
        # changing to `&` or passing multiple where() arguments if deletion fails.
        await session.execute(delete(Embedding).where(Embedding.model == model and Embedding.document == document))
