from sqlalchemy import func
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from os import getenv

EMBEDDING_DIM = int(getenv('EMBEDDING_DIM', 1024))


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'user'
    id: Mapped[str] = mapped_column(primary_key=True)
    quota: Mapped[float] = mapped_column(nullable=False)
    create_at: Mapped[datetime] = mapped_column(nullable=False, default=func.now())
    update_at: Mapped[datetime] = mapped_column(nullable=False, default=func.now(), onupdate=func.now())


class DbMessage(Base):
    __tablename__ = 'message'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user: Mapped[str] = mapped_column(nullable=False)
    session: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(nullable=False)
    userMsg: Mapped[str] = mapped_column(nullable=False)
    reasoning_content: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    cache_n: Mapped[int] = mapped_column(nullable=False)
    prompt_n: Mapped[int] = mapped_column(nullable=False)
    predicted_n: Mapped[int] = mapped_column(nullable=False)
    cost: Mapped[float] = mapped_column(nullable=False)
    prompt_ms: Mapped[float] = mapped_column(nullable=False)
    predicted_ms: Mapped[float] = mapped_column(nullable=False)
    tool: Mapped[str] = mapped_column(nullable=False)
    arguments: Mapped[str] = mapped_column(nullable=False)
    create_at: Mapped[datetime] = mapped_column(nullable=False, default=func.now())


class Embedding(Base):
    __tablename__ = f'embedding{EMBEDDING_DIM}'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document: Mapped[str] = mapped_column(nullable=False)
    model: Mapped[str] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    prompt_n: Mapped[int] = mapped_column(nullable=False)
    cost: Mapped[float] = mapped_column(nullable=False)
    create_at: Mapped[datetime] = mapped_column(nullable=False, default=func.now())
