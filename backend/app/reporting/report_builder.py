from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import seaborn as sns

from sqlalchemy.orm import Session
from sqlalchemy import or_

from backend.database.models import Artifact
from backend.database.storage import run_key, viz_key, run_root, BLOB_DIR
from backend.api.helpers.artifacts import read_json_from_storage_safe


@dataclass(frozen=True)
class ReportBuilderConfig:
    max_viz_plots: int = 3
    max_forecast_plots: int = 3


class ReportBuilder:
    def __init__(self, *, db: Session, config: Optional[ReportBuilderConfig] = None):
        self.db = db
        self.config = config or ReportBuilderConfig()

    def build(
        self,
        *,
        user_id: str,
        dataset_id: str,
        run_id: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        print(f"[ReportBuilder.build] user_id={user_id} dataset_id={dataset_id} run_id={run_id}", flush=True)

        cleaning_report = self._load_cleaning_report(user_id=user_id, run_id=run_id)

        signals_plots = self._create_signals_plots_from_cleaning(cleaning_report or {})
        print(f"[ReportBuilder.build] signals_plots(in_memory)={len(signals_plots)}", flush=True)

        viz_meta = self._load_viz_metadata(user_id=user_id, run_id=run_id)
        forecast_meta = self._load_forecast_metadata(user_id=user_id, run_id=run_id)

        viz_plots = self._list_plot_artifacts(
            user_id=user_id,
            dataset_id=dataset_id,
            run_id=run_id,
            kind_prefix="viz_plot",
            limit=self.config.max_viz_plots,
            allow_parent_match=False,
        )
        print(f"[ReportBuilder.build] viz_plots(from_db)={len(viz_plots)}", flush=True)

        forecast_plots_db = self._list_plot_artifacts(
            user_id=user_id,
            dataset_id=dataset_id,
            run_id=run_id,
            kind_prefix="forecast_plot",
            limit=self.config.max_forecast_plots,
            allow_parent_match=True,
        )
        print(f"[ReportBuilder.build] forecast_plots(from_db)={len(forecast_plots_db)}", flush=True)

        if forecast_plots_db:
            forecast_plots = forecast_plots_db
        else:
            forecast_plots = self._load_forecast_plots_from_storage(
                user_id=user_id,
                run_id=run_id,
                limit=self.config.max_forecast_plots,
            )
            print(f"[ReportBuilder.build] forecast_plots(from_storage)={len(forecast_plots)}", flush=True)

        return {
            "title": title or "AI Data Analysis Report",
            "user_id": user_id,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "cleaning_report": cleaning_report or {},
            "signals": {},
            "signals_plots": signals_plots,
            "viz_summary": viz_meta or {},
            "forecast_summary": forecast_meta or {},
            "viz_plots": viz_plots,
            "forecast_plots": forecast_plots,
        }

    def _load_cleaning_report(self, *, user_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        key = run_key(user_id, run_id, "report.json")
        data = self._read_json_if_exists(key)
        print(f"[cleaning_report] key={key} found={bool(data)}", flush=True)
        return data

    def _create_signals_plots_from_cleaning(self, cleaning_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        pre = cleaning_report.get("pre_profile") or {}
        cols = (pre.get("columns") or {}).get("numeric") or []
        missing_map = (pre.get("missingness") or {}).get("top_missing_columns") or pre.get("top_missing_columns") or {}
        corr_pairs = ((pre.get("correlation") or {}).get("top_abs_pairs")) or []

        print(f"[signals_plots] numeric_cols={cols}", flush=True)
        print(f"[signals_plots] missing_cols_count={len(missing_map) if isinstance(missing_map, dict) else 0}", flush=True)
        print(f"[signals_plots] corr_pairs_count={len(corr_pairs) if isinstance(corr_pairs, list) else 0}", flush=True)

        corr_png = self._plot_corr_heatmap_sparse(cols, corr_pairs)
        if corr_png:
            out.append(
                {
                    "plot_id": "signals:correlation_heatmap",
                    "artifact_id": None,
                    "kind": "signals_plot_png",
                    "mime_type": "image/png",
                    "storage_key": None,
                    "image_bytes": corr_png,
                    "meta": {"title": "Correlation heatmap"},
                    "data": {
                        "numeric_cols": cols,
                        "top_abs_pairs": corr_pairs,
                    },
                    "created_at": None,
                }
            )
            print("[signals_plots] added correlation heatmap", flush=True)
        else:
            print("[signals_plots] correlation heatmap skipped", flush=True)

        miss_png = self._plot_missingness_heatmap(missing_map)
        if miss_png:
            out.append(
                {
                    "plot_id": "signals:missingness_heatmap",
                    "artifact_id": None,
                    "kind": "signals_plot_png",
                    "mime_type": "image/png",
                    "storage_key": None,
                    "image_bytes": miss_png,
                    "meta": {"title": "Missingness heatmap"},
                    "data": {
                        "missing_map": missing_map,
                    },
                    "created_at": None,
                }
            )
            print("[signals_plots] added missingness heatmap", flush=True)
        else:
            print("[signals_plots] missingness heatmap skipped", flush=True)

        return out

    def _plot_corr_heatmap_sparse(self, numeric_cols: List[str], corr_pairs: List[Dict[str, Any]]) -> Optional[bytes]:
        try:
            cols = [c for c in (numeric_cols or []) if isinstance(c, str) and c.strip()]
            if not cols:
                print("[signals_plots] corr heatmap: no numeric cols", flush=True)
                return None

            m = pd.DataFrame(0.0, index=cols, columns=cols)
            for c in cols:
                m.loc[c, c] = 1.0

            if isinstance(corr_pairs, list):
                for item in corr_pairs:
                    if not isinstance(item, dict):
                        continue
                    x = item.get("col_x")
                    y = item.get("col_y")
                    v = item.get("corr")
                    if x in cols and y in cols:
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        m.loc[x, y] = fv
                        m.loc[y, x] = fv

            fig = Figure(figsize=(7.2, 5.2), dpi=160)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            sns.heatmap(m, ax=ax, annot=True, fmt=".2f", vmin=-1, vmax=1, square=True)
            ax.set_title("Correlation heatmap (numeric columns)")
            fig.tight_layout()

            buf = io.BytesIO()
            canvas.print_png(buf)
            return buf.getvalue()

        except Exception as e:
            print(f"[signals_plots] corr plot error: {repr(e)}", flush=True)
            return None

    def _plot_missingness_heatmap(self, missing_map: Any) -> Optional[bytes]:
        try:
            if not isinstance(missing_map, dict) or not missing_map:
                print("[signals_plots] missingness heatmap: missing_map empty", flush=True)
                return None

            cols = [c for c in missing_map.keys() if isinstance(c, str) and c.strip()]
            if not cols:
                print("[signals_plots] missingness heatmap: no cols", flush=True)
                return None

            vals: List[float] = []
            for c in cols:
                try:
                    vals.append(float(missing_map.get(c, 0.0)))
                except Exception:
                    vals.append(0.0)

            df = pd.DataFrame([vals], columns=cols, index=["missing_%"])

            fig = Figure(figsize=(7.2, 2.4), dpi=160)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            sns.heatmap(df, ax=ax, annot=True, fmt=".2f", cbar=True)
            ax.set_title("Missingness by column (%)")
            fig.tight_layout()

            buf = io.BytesIO()
            canvas.print_png(buf)
            return buf.getvalue()

        except Exception as e:
            print(f"[signals_plots] missingness plot error: {repr(e)}", flush=True)
            return None

    def _find_latest_forecast_run_dir(self, *, user_id: str, run_id: str) -> Optional[Path]:
        try:
            forecast_root = (BLOB_DIR / run_root(user_id, run_id) / "forecast")
            print(f"[forecast_scan] forecast_root={forecast_root} exists={forecast_root.exists()}", flush=True)
            if not forecast_root.exists():
                return None
            result_files = list(forecast_root.glob("frun_*/result.json"))
            print(f"[forecast_scan] found result.json files={len(result_files)}", flush=True)
            if not result_files:
                return None
            newest = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"[forecast_scan] newest_result={newest} mtime={newest.stat().st_mtime}", flush=True)
            return newest.parent
        except Exception as e:
            print(f"[forecast_scan] error: {repr(e)}", flush=True)
            return None

    def _load_forecast_metadata(self, *, user_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            a = (
                self.db.query(Artifact)
                .filter(Artifact.user_id == user_id)
                .filter(or_(Artifact.run_id == run_id, Artifact.parent_run_id == run_id))
                .filter(Artifact.kind.in_(["forecast_result_json", "forecast_result", "forecast_json"]))
                .order_by(Artifact.created_at.desc())
                .first()
            )
            print(f"[forecast_meta] db artifact found={bool(a)}", flush=True)
            if a and a.storage_key:
                data = self._read_json_if_exists(a.storage_key)
                print(f"[forecast_meta] read via artifact storage_key={a.storage_key} found={bool(data)}", flush=True)
                if isinstance(data, dict) and data:
                    return data
        except Exception as e:
            print(f"[forecast_meta] db path failed: {repr(e)}", flush=True)

        frun_dir = self._find_latest_forecast_run_dir(user_id=user_id, run_id=run_id)
        if not frun_dir:
            print("[forecast_meta] no frun_dir found", flush=True)
            return None

        result_path = frun_dir / "result.json"
        print(f"[forecast_meta] result_path={result_path} exists={result_path.exists()}", flush=True)
        if not result_path.exists():
            return None

        storage_key = str(result_path.relative_to(BLOB_DIR)).replace("\\", "/")
        data = self._read_json_if_exists(storage_key)
        print(f"[forecast_meta] read via storage_key={storage_key} found={bool(data)}", flush=True)
        return data if isinstance(data, dict) and data else None

    def _load_viz_metadata(self, *, user_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        candidates = [
            viz_key(user_id, run_id, "viz_plan.json"),
            viz_key(user_id, run_id, "viz_metrics.json"),
        ]
        for k in candidates:
            data = self._read_json_if_exists(k)
            print(f"[viz_meta] try={k} found={bool(data)}", flush=True)
            if isinstance(data, dict) and data:
                return data
        return None

    def _load_forecast_plots_from_storage(self, *, user_id: str, run_id: str, limit: int) -> List[Dict[str, Any]]:
        frun_dir = self._find_latest_forecast_run_dir(user_id=user_id, run_id=run_id)
        if not frun_dir:
            print("[forecast_plots_storage] no frun_dir found", flush=True)
            return []

        plots_dir = frun_dir / "plots"
        print(f"[forecast_plots_storage] plots_dir={plots_dir} exists={plots_dir.exists()}", flush=True)
        if not plots_dir.exists():
            return []

        pngs = list(plots_dir.glob("*.png"))
        print(f"[forecast_plots_storage] found pngs={len(pngs)}", flush=True)
        if not pngs:
            return []

        pngs = sorted(pngs, key=lambda p: p.stat().st_mtime, reverse=True)
        pngs = pngs[: max(0, int(limit))]

        out: List[Dict[str, Any]] = []
        for p in pngs:
            key = str(p.relative_to(BLOB_DIR)).replace("\\", "/")
            out.append(
                {
                    "artifact_id": None,
                    "kind": "forecast_plot_png",
                    "mime_type": "image/png",
                    "storage_key": key,
                    "meta": {"title": p.stem},
                    "created_at": None,
                }
            )
            print(f"[forecast_plots_storage] add plot key={key}", flush=True)

        return out

    def _read_json_if_exists(self, storage_key: str) -> Optional[Dict[str, Any]]:
        try:
            data = read_json_from_storage_safe(storage_key)
            return data if isinstance(data, dict) else None
        except Exception as e:
            print(f"[read_json_if_exists] key={storage_key} failed: {repr(e)}", flush=True)
            return None

    def _list_plot_artifacts(
        self,
        *,
        user_id: str,
        dataset_id: str,
        run_id: str,
        kind_prefix: str,
        limit: int,
        allow_parent_match: bool,
    ) -> List[Dict[str, Any]]:
        q = (
            self.db.query(Artifact)
            .filter(Artifact.user_id == user_id)
            .filter(Artifact.dataset_id == dataset_id)
            .filter(Artifact.kind.like(f"{kind_prefix}%"))
        )

        if allow_parent_match:
            q = q.filter(or_(Artifact.run_id == run_id, Artifact.parent_run_id == run_id))
        else:
            q = q.filter(Artifact.run_id == run_id)

        q = q.order_by(Artifact.created_at.asc()).limit(int(limit))

        rows = q.all()
        print(f"[list_plot_artifacts] kind_prefix={kind_prefix} allow_parent={allow_parent_match} rows={len(rows)}", flush=True)

        out: List[Dict[str, Any]] = []
        for a in rows:
            out.append(
                {
                    "artifact_id": a.artifact_id,
                    "kind": a.kind,
                    "mime_type": a.mime_type,
                    "storage_key": a.storage_key,
                    "meta": a.meta or {},
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
            )
            print(f"[list_plot_artifacts] -> artifact_id={a.artifact_id} kind={a.kind} run_id={a.run_id} parent_run_id={a.parent_run_id} storage_key={a.storage_key}", flush=True)

        return out