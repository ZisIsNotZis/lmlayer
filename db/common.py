from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'user'
    id: Mapped[str] = mapped_column(primary_key=True)
    quota: Mapped[float] = mapped_column(nullable=False)


class DbMessage(Base):
    __tablename__ = 'message'
    id: Mapped[int] = mapped_column(primary_key=True)
    user: Mapped[str] = mapped_column(nullable=False)
    session: Mapped[str] = mapped_column(nullable=False)
    sequence: Mapped[int] = mapped_column(nullable=False)
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
    time: Mapped[float] = mapped_column(nullable=False)


class Embedding(Base):
    __tablename__ = 'embedding'
    id: Mapped[int] = mapped_column(primary_key=True)
    document: Mapped[str] = mapped_column(nullable=False)
    model: Mapped[str] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    __table_args__ = Index('ix_embedding', 'embedding', postgresql_using='hnsw(embedding vector_ip_ops)'),
