from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False)

    original_dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    sheet_name: Mapped[str] = mapped_column(String, nullable=False)

    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    dtypes: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    bucket: Mapped[str] = mapped_column(String, nullable=False)
    raw_key: Mapped[str] = mapped_column(String, nullable=False)

    raw_parquet_key: Mapped[str] = mapped_column(String, nullable=False)
    current_parquet_key: Mapped[str] = mapped_column(String, nullable=False)

    clean_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    last_cleaning_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cleaned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    forecast_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    last_forecast_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    forecasted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    viz_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    last_viz_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    visualized_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    report_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    last_report_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reported_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Profile(Base):
    __tablename__ = "profiles"

    profile_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.dataset_id"), nullable=False)

    bucket: Mapped[str] = mapped_column(String, nullable=False, default="local")
    report_key: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CleaningRun(Base):
    __tablename__ = "cleaning_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.dataset_id"), nullable=False, index=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    bucket: Mapped[str] = mapped_column(String, nullable=False)
    report_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cleaned_parquet_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cleaned_xlsx_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.dataset_id"), nullable=False, index=True)

    run_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)

    parent_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)

    kind: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False, default="application/octet-stream")
    bucket: Mapped[str] = mapped_column(String, nullable=False, default="local")
    storage_key: Mapped[str] = mapped_column(String, nullable=False)

    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    forecast_run_id: Mapped[str] = mapped_column(String, primary_key=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.dataset_id"), nullable=False, index=True)

    run_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("cleaning_runs.run_id"), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    model: Mapped[str] = mapped_column(String, nullable=False, default="auto")
    horizon: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    datetime_column: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    targets: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)

    plan_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    result_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    forecast_parquet_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    forecast_json_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class VisualizationRun(Base):
    __tablename__ = "visualization_runs"

    viz_run_id: Mapped[str] = mapped_column(String, primary_key=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.dataset_id"), nullable=False, index=True)

    run_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("cleaning_runs.run_id"), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    mode: Mapped[str] = mapped_column(String, nullable=False, default="auto")
    meta_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReportRun(Base):
    __tablename__ = "report_runs"

    report_run_id: Mapped[str] = mapped_column(String, primary_key=True)

    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.dataset_id"), nullable=False, index=True)

    run_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("cleaning_runs.run_id"), nullable=True, index=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    include_cleaning: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    include_forecast: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    include_viz: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    template: Mapped[str] = mapped_column(String, nullable=False, default="default")
    options_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    pdf_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)