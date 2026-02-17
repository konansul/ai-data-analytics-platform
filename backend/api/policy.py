# backend/api/policy.py
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.models import PolicySuggestRequest, PolicySuggestResponse
from backend.database.db import get_db
from backend.database.models import Dataset
from backend.database.storage import get_bytes

from backend.app.profiling.profiling import profile_dataframe
from backend.app.cleaning.cleaning_agent.cleaning_policy_agent import build_cleaning_plan

router = APIRouter()

@router.post("/policy/suggest", response_model=PolicySuggestResponse)
def suggest_policy(req: PolicySuggestRequest, db: Session = Depends(get_db)):
    row: Dataset | None = db.get(Dataset, req.dataset_id)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        parquet_bytes = get_bytes(row.current_parquet_key)
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset parquet not found in local storage")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet: {e}")

    pre_profile = profile_dataframe(df)
    use_llm = (req.mode == "llm")

    try:
        plan = build_cleaning_plan(pre_profile, use_llm=use_llm, model=req.llm_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy agent failed: {e}")

    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {
        "enabled_steps": getattr(plan, "enabled_steps", {}),
        "params": getattr(plan, "params", {}),
        "notes": getattr(plan, "notes", []),
        "source": getattr(plan, "source", "unknown"),
        "version": getattr(plan, "version", "unknown"),
    }

    return {
        "policy": plan_dict,
        "source": str(plan_dict.get("source", "unknown")),
        "notes": list(plan_dict.get("notes", []))
        if isinstance(plan_dict.get("notes", []), list)
        else [],
    }