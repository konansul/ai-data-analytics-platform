# backend/test_scripts/messy_xls_reader.py
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import xlrd


@dataclass
class XlsSheetResult:
    sheet_name: str
    df: pd.DataFrame
    n_rows_raw: int
    n_cols_raw: int
    crop_bounds: Tuple[int, int, int, int]  # (r0, r1_exclusive, c0, c1_exclusive)

def _clean_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.replace("\u00a0", " ").strip()  # NBSP -> space
        s = re.sub(r"\s+", " ", s)
        if s in {"", "-", "–", "—", "...", "…"}:
            return None
        return s
    return v


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _crop_empty_edges(matrix: List[List[Any]]) -> Tuple[List[List[Any]], Tuple[int, int, int, int]]:
    """
    Remove fully empty rows/cols on the outer edges only.
    Returns cropped matrix and (r0, r1, c0, c1) bounds in the original matrix.
    """
    if not matrix:
        return matrix, (0, 0, 0, 0)

    n_rows = len(matrix)
    n_cols = max((len(r) for r in matrix), default=0)
    if n_cols == 0:
        return [], (0, 0, 0, 0)

    for r in matrix:
        if len(r) < n_cols:
            r.extend([None] * (n_cols - len(r)))

    def row_empty(i: int) -> bool:
        return all(_is_empty(x) for x in matrix[i])

    def col_empty(j: int) -> bool:
        return all(_is_empty(matrix[i][j]) for i in range(n_rows))

    r0 = 0
    while r0 < n_rows and row_empty(r0):
        r0 += 1

    r1 = n_rows
    while r1 > r0 and row_empty(r1 - 1):
        r1 -= 1

    c0 = 0
    while c0 < n_cols and col_empty(c0):
        c0 += 1

    c1 = n_cols
    while c1 > c0 and col_empty(c1 - 1):
        c1 -= 1

    cropped = [row[c0:c1] for row in matrix[r0:r1]]
    return cropped, (r0, r1, c0, c1)


def _sheet_to_matrix_with_unmerge(sh: xlrd.sheet.Sheet) -> List[List[Any]]:
    """
    Build a 2D matrix of sheet values, then fill merged ranges with the top-left value.
    Requires workbook opened with formatting_info=True to have merged_cells.
    """
    n_rows = sh.nrows
    n_cols = sh.ncols

    matrix: List[List[Any]] = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append(_clean_cell(sh.cell_value(r, c)))
        matrix.append(row)

    for (rlo, rhi, clo, chi) in getattr(sh, "merged_cells", []) or []:
        if rlo >= n_rows or clo >= n_cols:
            continue
        top_left = matrix[rlo][clo]
        if _is_empty(top_left):
            continue
        for rr in range(rlo, min(rhi, n_rows)):
            for cc in range(clo, min(chi, n_cols)):
                if _is_empty(matrix[rr][cc]):
                    matrix[rr][cc] = top_left

    return matrix

def read_xls( source: Union[str, Path, bytes, io.BytesIO], *, sheet_name: Optional[str] = None,
    crop_empty_edges: bool = True,
) -> List[XlsSheetResult]:
    """
    Read .xls from path or bytes and return list of sheets with:
    - merged cells expanded
    - optionally cropped empty outer rows/cols
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        wb = xlrd.open_workbook(str(path), formatting_info=True)
    else:
        if isinstance(source, io.BytesIO):
            data = source.getvalue()
        elif isinstance(source, bytes):
            data = source
        else:
            raise TypeError("source must be path | bytes | BytesIO")
        wb = xlrd.open_workbook(file_contents=data, formatting_info=True)

    results: List[XlsSheetResult] = []
    sheet_names = wb.sheet_names()

    for name in sheet_names:
        if sheet_name and name != sheet_name:
            continue

        sh = wb.sheet_by_name(name)
        matrix = _sheet_to_matrix_with_unmerge(sh)

        bounds = (0, sh.nrows, 0, sh.ncols)
        if crop_empty_edges:
            matrix, bounds = _crop_empty_edges(matrix)

        df = pd.DataFrame(matrix)
        results.append(
            XlsSheetResult(
                sheet_name=name,
                df=df,
                n_rows_raw=sh.nrows,
                n_cols_raw=sh.ncols,
                crop_bounds=bounds,
            )
        )

    return results

def read_xls_from_upload(file_bytes: bytes, filename: str) -> List[XlsSheetResult]:
    suffix = Path(filename).suffix.lower()
    if suffix != ".xls":
        raise ValueError(f"Expected .xls, got {suffix}")
    return read_xls(file_bytes)

def read_xls_from_path(path: Union[str, Path]) -> List[XlsSheetResult]:
    path = Path(path)
    if path.suffix.lower() != ".xls":
        raise ValueError(f"Expected .xls, got {path.suffix}")
    return read_xls(path)