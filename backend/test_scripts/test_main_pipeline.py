from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

from backend.app.ingestion.dataset_loader import load_from_path
from backend.app.cleaning.cleaning_steps.main_pipeline import run_cleaning_pipeline


def _pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    test_file = project_root / "backend" / "test_data" / "framingham.csv"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    sheets = load_from_path(test_file)
    sheet = sheets[0]
    df = sheet.df

    print("=" * 80)
    print(f"FILE: {sheet.file_name}")
    print(f"SHEET: {sheet.sheet_name}")
    print(f"DATASET_ID: {sheet.dataset_id}")
    print(f"RAW SHAPE: {df.shape}")
    print("=" * 80)

    clean_df_rb, report_rb = run_cleaning_pipeline(df, use_llm=False)

    print("\n[RUN] Rule-based plan (use_llm=False)")
    print(f"Clean shape: {clean_df_rb.shape}")
    print("Cleaning plan:")
    print(_pretty(report_rb.get("cleaning_plan", {})))

    print("\nSummary (rule-based):")
    print(f"- dropped_total: {report_rb.get('dropped_total')}")
    print(f"- normalized_columns: {len(report_rb.get('normalized_columns', {}))}")
    print(f"- inferred_datetime_columns: {report_rb.get('inferred_datetime_columns')}")
    print(f"- imputation: {report_rb.get('imputation', {})}")

    clean_df_llm, report_llm = run_cleaning_pipeline(
        df,
        use_llm=True,
        llm_model="gemini-2.5-flash",
    )

    print("\n[RUN] LLM plan (use_llm=True)")
    print(f"Clean shape: {clean_df_llm.shape}")
    print("Cleaning plan:")
    print(_pretty(report_llm.get("cleaning_plan", {})))

    print("\nSummary (LLM):")
    print(f"- dropped_total: {report_llm.get('dropped_total')}")
    print(f"- inferred_datetime_columns: {report_llm.get('inferred_datetime_columns')}")
    print(f"- imputation: {report_llm.get('imputation', {})}")

    out_dir = project_root / "backend" / "test_scripts" / "_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "report_rule_based.json").write_text(_pretty(report_rb), encoding="utf-8")
    (out_dir / "report_llm.json").write_text(_pretty(report_llm), encoding="utf-8")

    print("\nSaved:")
    print(f"- {out_dir / 'report_rule_based.json'}")
    print(f"- {out_dir / 'report_llm.json'}")


if __name__ == "__main__":
    main()