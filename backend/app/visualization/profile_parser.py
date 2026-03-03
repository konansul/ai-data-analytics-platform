import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ProfileParser:
    def parse(self, profile: Dict) -> Dict:
        column_keys = ["columns", "column_stats", "fields", "schema", "variables", "features"]
        cols_obj = None
        for key in column_keys:
            val = profile.get(key)
            if val:
                cols_obj = val
                break

        columns_dict: Dict[str, Any] = {}
        if isinstance(cols_obj, dict):
            columns_dict = cols_obj
        elif isinstance(cols_obj, list):
            for item in cols_obj:
                if isinstance(item, dict):
                    name = (
                        item.get("name")
                        or item.get("column")
                        or item.get("col")
                        or item.get("column_name")
                        or item.get("field")
                    )
                    if isinstance(name, str) and name:
                        columns_dict[name] = item

        numeric_cols: List[str] = list(profile.get("numeric_columns") or [])
        cat_cols: List[str] = list(profile.get("categorical_columns") or [])
        date_cols: List[str] = list(profile.get("date_columns") or [])

        if not (numeric_cols or cat_cols or date_cols):
            for name, meta in columns_dict.items():
                if not isinstance(meta, dict):
                    continue
                dtype = (
                    meta.get("dtype")
                    or meta.get("type")
                    or meta.get("data_type")
                    or meta.get("col_type")
                    or ""
                ).lower()
                if any(t in dtype for t in ["int", "float", "double", "number", "numeric", "decimal"]):
                    numeric_cols.append(name)
                elif any(t in dtype for t in ["date", "datetime", "time", "timestamp"]):
                    date_cols.append(name)
                else:
                    cat_cols.append(name)

        cardinality: Dict[str, Any] = {}
        for k, v in columns_dict.items():
            if not isinstance(v, dict):
                continue
            card = (
                v.get("cardinality")
                or v.get("n_unique")
                or v.get("unique_count")
                or v.get("distinct_count")
            )
            cardinality[k] = card

        corr_section = profile.get("correlation") or profile.get("correlations") or {}
        correlations = (
            corr_section.get("top_abs_pairs")
            or corr_section.get("pairs")
            or []
        )

        miss_section = profile.get("missingness") or profile.get("missing") or {}
        missingness = (
            miss_section.get("top_missing_columns")
            or miss_section.get("columns")
            or {}
        )

        return {
            "columns": list(columns_dict.keys()),
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "date_columns": date_cols,
            "cardinality": cardinality,
            "correlations": correlations,
            "missingness": missingness,
            "column_meta": {
                k: {
                    "dtype": (
                        v.get("dtype") or v.get("type") or v.get("data_type")
                    ) if isinstance(v, dict) else None,
                    "cardinality": cardinality.get(k),
                    "is_unique": v.get("is_unique") if isinstance(v, dict) else None,
                    "is_id": v.get("is_id") if isinstance(v, dict) else None,
                }
                for k, v in columns_dict.items()
                if isinstance(v, dict)
            },
        }