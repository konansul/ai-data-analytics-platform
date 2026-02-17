from __future__ import annotations

import io
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass
class SheetContext:
    file_name: str
    sheet_name: str
    dataset_id: str
    df: pd.DataFrame
    shape: tuple
    dtypes: dict


def _excel_engine(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".xls":
        return "xlrd"
    return "openpyxl"


def _read_excel_all_sheets(source: Union[str, Path, io.BytesIO], file_name: str) -> List[SheetContext]:

    if isinstance(source, io.BytesIO):
        source.seek(0)

    engine = _excel_engine(file_name)
    xls = pd.ExcelFile(source, engine=engine)

    sheet_contexts: List[SheetContext] = []

    for idx, sheet_name in enumerate(xls.sheet_names):

        df = xls.parse(sheet_name=sheet_name)

        dataset_id = f"{file_name}::{sheet_name}::{idx}"

        sheet_contexts.append(
            SheetContext(
                file_name=file_name,
                sheet_name=sheet_name,
                dataset_id=dataset_id,
                df=df,
                shape=df.shape,
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            )
        )

    return sheet_contexts


def _read_csv(source: Union[str, Path, io.BytesIO], file_name: str) -> List[SheetContext]:
    if isinstance(source, io.BytesIO):
        source.seek(0)

    df = pd.read_csv(source)

    sheet_name = "CSV"
    dataset_id = f"{file_name}::{sheet_name}::0"

    return [
        SheetContext(
            file_name=file_name,
            sheet_name=sheet_name,
            dataset_id=dataset_id,
            df=df,
            shape=df.shape,
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        )
    ]


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