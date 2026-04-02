from sqlalchemy import Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import JSON
from .db import Base


class ReviewAnalysis(Base):
    __tablename__ = "review_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    product_id: Mapped[int] = mapped_column(Integer, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    review_fix_skipped: Mapped[bool] = mapped_column(Boolean, default=False)

    original_review: Mapped[str] = mapped_column(Text)
    fixed_review: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(32), nullable=True)

    good_points: Mapped[dict] = mapped_column(JSON, default=dict)
    bad_points: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())