# backend/test_scripts/test.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.test_scripts.messy_xls_reader import read_xls


def main() -> None:
    parser = argparse.ArgumentParser(description="Test XLS reader (xlrd) with merged-cells expansion.")
    parser.add_argument("path", type=str, help="Path to .xls file")
    parser.add_argument("--sheet", type=str, default=None, help="Exact sheet name (optional)")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping empty outer rows/cols")
    parser.add_argument("--outdir", type=str, default=None, help="Where to save xlsx outputs (optional)")
    parser.add_argument("--maxrows", type=int, default=30, help="Rows to print from the top")
    args = parser.parse_args()

    xls_path = Path(args.path)
    if not xls_path.exists():
        raise FileNotFoundError(f"File not found: {xls_path}")
    if xls_path.suffix.lower() != ".xls":
        raise ValueError(f"Expected .xls, got: {xls_path.suffix}")

    crop = not args.no_crop
    outdir = Path(args.outdir) if args.outdir else xls_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    results = read_xls(str(xls_path), sheet_name=args.sheet, crop_empty_edges=crop)

    if not results:
        print("No sheets found (or sheet filter did not match).")
        return

    print(f"\nLoaded file: {xls_path.name}")
    print(f"Sheets parsed: {len(results)}")
    print(f"Crop empty edges: {crop}")

    for i, r in enumerate(results, start=1):
        df = r.df
        r0, r1, c0, c1 = r.crop_bounds

        print("\n" + "=" * 80)
        print(f"[{i}] Sheet: {r.sheet_name}")
        print(f"Raw size: {r.n_rows_raw} rows x {r.n_cols_raw} cols")
        print(f"Cropped bounds: rows {r0}:{r1}  cols {c0}:{c1}")
        print(f"Result df: {df.shape[0]} rows x {df.shape[1]} cols")

        with pd.option_context("display.max_columns", 50, "display.width", 200):
            print("\nTop preview:")
            print(df.head(args.maxrows))

        safe_sheet = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in r.sheet_name)[:50]
        out_path = outdir / f"{xls_path.stem}__{safe_sheet}__expanded.xlsx"
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="data")

        print(f"\nSaved: {out_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()