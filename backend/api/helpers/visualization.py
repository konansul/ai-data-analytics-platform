# backend/api/helpers/visualization.py

import re
import numpy as np

from typing import Dict, Any


def enrich_profile(profile_data: Dict[str, Any], df) -> Dict[str, Any]:

    enriched = dict(profile_data)
    if not enriched.get("numeric_columns"):
        enriched["numeric_columns"] = df.select_dtypes(include=[np.number]).columns.tolist()
    if not enriched.get("categorical_columns"):
        enriched["categorical_columns"] = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    if not enriched.get("date_columns"):
        enriched["date_columns"] = df.select_dtypes(
            include=["datetime64", "datetimetz"]
        ).columns.tolist()
    if not any(enriched.get(k) for k in ["columns", "column_stats", "fields", "schema", "variables", "features"]):
        enriched["columns"] = {col: {"dtype": str(df[col].dtype)} for col in df.columns}
    return enriched

def safe_filename(name: str, default: str = "plot") -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return (name or default) + ".png"