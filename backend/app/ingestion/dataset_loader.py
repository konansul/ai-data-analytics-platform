from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class SheetContext:
    file_name: str
    sheet_name: str
    dataset_id: str

    # Original dataframe (best-effort typed)
    df: pd.DataFrame

    # Parquet-safe dataframe (object columns coerced to string, missing markers normalized)
    df_parquet_safe: pd.DataFrame

    # Diagnostics
    parquet_safe_needed: bool
    parquet_safe_reason: str

    shape: Tuple[int, int]
    dtypes: Dict[str, str]


# --- helpers -----------------------------------------------------------------

_MISSING_MARKERS = {
    "", " ", "  ",
    "-", "—", "--", "---", "…",
    "na", "n/a", "none", "null", "nan",
    "missing", "unknown", "undefined",
    "#n/a", "#na", "#null!",
}


def _excel_engine(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    return "xlrd" if suffix == ".xls" else "openpyxl"


def _cell_to_str(v: Any) -> str:
    # Safe string conversion (handles bytes, ints, floats, etc.)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return bytes(v).decode("utf-8", errors="replace")
        except Exception:
            return str(v)
    s = str(v)
    return s.replace("\r", " ").replace("\n", " ").strip()


def make_df_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dataframe robust for pyarrow parquet conversion:
    - object/string columns -> pandas StringDtype
    - normalize common missing markers to <NA>
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    obj_cols = out.select_dtypes(include=["object", "string"]).columns

    for c in obj_cols:
        s = out[c].map(_cell_to_str).astype("string")
        s_lower = s.str.lower()
        s = s.where(~s_lower.isin(_MISSING_MARKERS), other=pd.NA)
        out[c] = s

    return out


def _detect_parquet_risk(df: pd.DataFrame, *, sample_rows: int = 200) -> Tuple[bool, str]:
    """
    Heuristic: if any object column contains mixed python types (e.g., int + str, bytes + int),
    parquet conversion is likely to fail unless coerced.
    """
    if df is None or df.empty:
        return False, ""

    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return False, ""

    n = min(int(sample_rows), int(len(df)))
    sample = df.iloc[:n]

    for c in obj_cols:
        ser = sample[c]
        ser = ser[ser.notna()]
        if ser.empty:
            continue

        types = set(type(x) for x in ser.tolist())

        # common risky mixes
        if (bytes in types) or (bytearray in types) or (memoryview in types):
            return True, f"Column '{c}' contains bytes-like values."

        # if both numbers and strings exist, parquet often blows up trying to pick a single type
        has_str = any(t is str for t in types)
        has_num = any(t in (int, float) for t in types)
        if has_str and has_num:
            return True, f"Column '{c}' mixes numeric and string values."

        # multiple arbitrary types -> risky
        if len(types) >= 3:
            return True, f"Column '{c}' has mixed python types: {[t.__name__ for t in sorted(types, key=lambda x: x.__name__)]}"

    return False, ""


# --- readers -----------------------------------------------------------------

def _read_excel_all_sheets(source: Union[str, Path, io.BytesIO], file_name: str) -> List[SheetContext]:
    if isinstance(source, io.BytesIO):
        source.seek(0)

    engine = _excel_engine(file_name)
    xls = pd.ExcelFile(source, engine=engine)

    sheet_contexts: List[SheetContext] = []

    for idx, sheet_name in enumerate(xls.sheet_names):
        # NOTE: keep default inference (don’t force dtype=str), to preserve normal datasets
        df = xls.parse(sheet_name=sheet_name)

        dataset_id = f"{file_name}::{sheet_name}::{idx}"

        risky, reason = _detect_parquet_risk(df)
        df_safe = make_df_parquet_safe(df) if risky else df

        sheet_contexts.append(
            SheetContext(
                file_name=file_name,
                sheet_name=str(sheet_name),
                dataset_id=dataset_id,
                df=df,
                df_parquet_safe=df_safe,
                parquet_safe_needed=bool(risky),
                parquet_safe_reason=str(reason),
                shape=(int(df.shape[0]), int(df.shape[1])),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            )
        )

    return sheet_contexts


def _read_csv(source: Union[str, Path, io.BytesIO], file_name: str) -> List[SheetContext]:
    if isinstance(source, io.BytesIO):
        source.seek(0)

    # Keep defaults, but be tolerant to messy CSVs
    df = pd.read_csv(source)

    sheet_name = "CSV"
    dataset_id = f"{file_name}::{sheet_name}::0"

    risky, reason = _detect_parquet_risk(df)
    df_safe = make_df_parquet_safe(df) if risky else df

    return [
        SheetContext(
            file_name=file_name,
            sheet_name=sheet_name,
            dataset_id=dataset_id,
            df=df,
            df_parquet_safe=df_safe,
            parquet_safe_needed=bool(risky),
            parquet_safe_reason=str(reason),
            shape=(int(df.shape[0]), int(df.shape[1])),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        )
    ]


# --- public API ---------------------------------------------------------------

def load_from_path(path: Union[str, Path]) -> List[SheetContext]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return _read_excel_all_sheets(path, file_name=path.name)
    if suffix == ".csv":
        return _read_csv(path, file_name=path.name)

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .xlsx, .xls, .csv")


def load_from_upload(file_bytes: bytes, filename: str) -> List[SheetContext]:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if suffix in [".xlsx", ".xls"]:
        return _read_excel_all_sheets(buffer, file_name=filename)
    if suffix == ".csv":
        return _read_csv(buffer, file_name=filename)

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .xlsx, .xls, .csv")