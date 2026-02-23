from __future__ import annotations

import io
import pandas as pd


def df_to_tsv_for_llm(df: pd.DataFrame, max_rows: int = 40, max_cols: int = 25) -> str:
    head = df.iloc[:max_rows, :max_cols].copy()
    head = head.where(pd.notnull(head), "")

    return head.to_csv(sep="\t", index=False)


def csv_from_llm_to_df(csv_text: str) -> pd.DataFrame:
    buf = io.StringIO(csv_text)
    df = pd.read_csv(buf)

    if df.empty:
        raise ValueError("LLM produced empty dataframe")

    required = {"indicator", "year", "value"}
    if not required.issubset(df.columns):
        raise ValueError("CSV missing required columns")

    df["indicator"] = df["indicator"].astype(str).str.strip()

    df["year"] = (
        df["year"]
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.replace(".0", "", regex=False)
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["value"] = (
        df["value"]
        .replace(["-", "…", ""], None)
    )

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["indicator", "year"], how="any")

    if len(df) < 3:
        raise ValueError("Rebuilt dataframe too small")

    return df.reset_index(drop=True)