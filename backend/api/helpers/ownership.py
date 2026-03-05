# backend/api/helpers/ownership.py

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.security import decode_token
from backend.database.models import (
    Dataset,
    CleaningRun,
    User,
    VisualizationRun,
    ForecastRun,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/auth/login")

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme), ) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user: Optional[User] = (
        db.query(User)
        .filter(User.user_id == user_id)
        .first()
    )

    if not user or not getattr(user, "is_active", True):
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user


def get_owned_dataset_or_404(db: Session, dataset_id: str, user_id: str, ) -> Dataset:
    row = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == user_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row


def get_owned_clean_run_or_404( db: Session, run_id: str, user_id: str, ) -> CleaningRun:
    row = (
        db.query(CleaningRun)
        .filter(
            CleaningRun.run_id == run_id,
            CleaningRun.user_id == user_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Cleaning run not found")
    return row


def get_owned_forecast_run_or_404(db: Session, forecast_run_id: str, user_id: str,) -> ForecastRun:
    row = (
        db.query(ForecastRun)
        .filter(
            ForecastRun.forecast_run_id == forecast_run_id,
            ForecastRun.user_id == user_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return row


def get_owned_visualization_run_or_404(
    db: Session,
    viz_run_id: str,
    user_id: str,
) -> VisualizationRun:
    row = (
        db.query(VisualizationRun)
        .filter(
            VisualizationRun.viz_run_id == viz_run_id,
            VisualizationRun.user_id == user_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Visualization run not found")
    return row