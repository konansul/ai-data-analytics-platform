# backend/app/cleaning/_01_normalize.py
from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names:
      - cast to str
      - strip
      - lowercase
      - replace whitespace with underscore
      - remove duplicated underscores
      - keep alnum + underscore

    Returns:
      clean_df, renamed_map (only changed columns)
    """
    clean_df = df.copy()
    original = list(clean_df.columns)

    cols = clean_df.columns.astype(str).str.strip().str.lower()
    cols = cols.str.replace(r"\s+", "_", regex=True)
    cols = cols.str.replace(r"__+", "_", regex=True)
    cols = cols.str.replace(r"[^a-z0-9_]+", "_", regex=True)
    cols = cols.str.replace(r"^_+|_+$", "", regex=True)

    cols = [c if c else f"col_{i}" for i, c in enumerate(cols)]
    clean_df.columns = cols

    renamed = {old: new for old, new in zip(original, clean_df.columns) if str(old) != str(new)}
    return clean_df, renamed