"""SQLite persistence (MVP). Postgres-ready via DATABASE_URL later."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from apps.api.app.config import REPO_ROOT

DATABASE_URL = f"sqlite:///{(REPO_ROOT / 'artifacts' / 'crohnstracker.db').as_posix()}"


class Base(DeclarativeBase):
    pass


class ImageCaseRow(Base):
    __tablename__ = "image_cases"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64))
    video_id: Mapped[str] = mapped_column(String(128), index=True)
    frame_idx: Mapped[int] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(String(512))
    gt_label: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    split: Mapped[str] = mapped_column(String(16), default="demo")


class PredictionRow(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    pred_label: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64), default="0.1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewFlagRow(Base):
    __tablename__ = "review_flags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="open")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    novel: Mapped[bool] = mapped_column(Boolean, default=False)


def get_engine():
    return create_engine(DATABASE_URL, future=True)


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)
