from __future__ import annotations

import io
import re

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

from backend.database import storage as blob


@dataclass(frozen=True)
class PDFRendererConfig:
    pagesize: Tuple[int, int] = A4
    title_font_size: int = 18
    section_title_font_size: int = 13
    body_font_size: int = 10
    max_image_width_cm: float = 15.5
    max_image_height_cm: float = 9.5
    caption_font_size: int = 9
    caption_leading: int = 12
    max_text_chars: int = 25_000


class PDFRenderer:
    def __init__(self, config: Optional[PDFRendererConfig] = None):
        self.config = config or PDFRendererConfig()
        self.styles = getSampleStyleSheet()

        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Title"],
                fontSize=self.config.title_font_size,
                leading=self.config.title_font_size + 4,
                spaceAfter=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=self.styles["Heading2"],
                fontSize=self.config.section_title_font_size,
                leading=self.config.section_title_font_size + 3,
                spaceBefore=10,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Body",
                parent=self.styles["BodyText"],
                fontSize=self.config.body_font_size,
                leading=self.config.body_font_size + 3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Caption",
                parent=self.styles["BodyText"],
                fontSize=self.config.caption_font_size,
                leading=self.config.caption_leading,
                textColor=colors.HexColor("#333333"),
                spaceBefore=2,
                spaceAfter=10,
            )
        )

    def render_pdf(
        self,
        *,
        builder_output: Dict[str, Any],
        llm_output: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        buf = io.BytesIO()

        doc = SimpleDocTemplate(
            buf,
            pagesize=self.config.pagesize,
            leftMargin=1.7 * cm,
            rightMargin=1.7 * cm,
            topMargin=1.6 * cm,
            bottomMargin=1.6 * cm,
            title=str(builder_output.get("title") or "AI Data Analysis Report"),
        )

        story: List[Any] = []

        title = str(builder_output.get("title") or "AI Data Analysis Report")
        story.append(Paragraph(self._safe_text(title), self.styles["ReportTitle"]))
        story.append(self._meta_table(builder_output))
        story.append(Spacer(1, 10))

        if isinstance(llm_output, dict):
            summary = llm_output.get("executive_summary")
            if summary:
                story.append(Paragraph("Executive Summary", self.styles["SectionTitle"]))
                story.append(Paragraph(self._safe_text(summary), self.styles["Body"]))
                story.append(Spacer(1, 8))

        story.append(Paragraph("1) Data Cleaning", self.styles["SectionTitle"]))
        story.extend(self._render_cleaning(builder_output.get("cleaning_report") or {}, llm_output))

        story.append(Paragraph("2) Signals & Profiling", self.styles["SectionTitle"]))
        story.extend(self._render_signals(builder_output.get("signals") or {}, builder_output.get("signals_plots") or [], llm_output))

        #story.append(PageBreak())
        story.append(Paragraph("3) Visualization", self.styles["SectionTitle"]))
        story.extend(
            self._render_plots_section(
                title="Visualization Plots",
                plots=builder_output.get("viz_plots") or [],
                llm_text=(llm_output or {}).get("visualization_notes") if isinstance(llm_output, dict) else None,
                llm_output=llm_output,
            )
        )

        #story.append(PageBreak())
        story.append(Paragraph("4) Forecasting", self.styles["SectionTitle"]))
        story.extend(
            self._render_plots_section(
                title="Forecast Plots",
                plots=builder_output.get("forecast_plots") or [],
                llm_text=(llm_output or {}).get("forecasting_notes") if isinstance(llm_output, dict) else None,
                llm_output=llm_output,
            )
        )

        if isinstance(llm_output, dict) and llm_output.get("conclusion"):
            #story.append(PageBreak())
            story.append(Paragraph("5) Conclusion", self.styles["SectionTitle"]))
            story.append(Paragraph(self._safe_text(llm_output["conclusion"]), self.styles["Body"]))

        doc.build(story)
        return buf.getvalue()

    def _meta_table(self, builder_output: Dict[str, Any]) -> Table:
        rows = [
            ["User ID", str(builder_output.get("user_id") or "—")],
            ["Dataset ID", str(builder_output.get("dataset_id") or "—")],
            ["Run ID", str(builder_output.get("run_id") or "—")],
        ]
        t = Table(rows, colWidths=[3.3 * cm, 13.5 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BOX", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    def _render_cleaning(self, cleaning_report: Dict[str, Any], llm_output: Optional[Dict[str, Any]]) -> List[Any]:
        story: List[Any] = []

        if isinstance(llm_output, dict) and llm_output.get("cleaning_notes"):
            story.append(Paragraph(self._safe_text(llm_output["cleaning_notes"]), self.styles["Body"]))
            story.append(Spacer(1, 8))

        pre = cleaning_report.get("pre_profile") or {}
        post = cleaning_report.get("post_profile") or {}
        plan = cleaning_report.get("cleaning_plan") or {}
        drop_rules = cleaning_report.get("drop_rules") or {}
        dedup = cleaning_report.get("deduplicate") or {}
        outliers = cleaning_report.get("outliers") or {}
        imputation = cleaning_report.get("imputation") or {}

        rows = []

        rb = cleaning_report.get("rows_before")
        ra = cleaning_report.get("rows_after")
        cb = cleaning_report.get("cols_before")
        ca = cleaning_report.get("cols_after")

        if rb is not None and ra is not None:
            rows.append(["Rows (before → after)", f"{rb} → {ra}"])
        if cb is not None and ca is not None:
            rows.append(["Columns (before → after)", f"{cb} → {ca}"])

        pre_miss = pre.get("overall_missing_%", (pre.get("missingness") or {}).get("overall_missing_%"))
        post_miss = post.get("overall_missing_%", (post.get("missingness") or {}).get("overall_missing_%"))
        if pre_miss is not None and post_miss is not None:
            try:
                rows.append(["Missing % (before → after)", f"{float(pre_miss):.2f}% → {float(post_miss):.2f}%"])
            except Exception:
                rows.append(["Missing % (before → after)", f"{pre_miss} → {post_miss}"])

        has_time = pre.get("has_time_index")
        time_col = pre.get("time_column")
        if has_time is not None:
            rows.append(["Has time index", "yes" if has_time else "no"])
        if time_col:
            rows.append(["Time column", str(time_col)])

        src = plan.get("source")
        ver = plan.get("version")
        if src:
            rows.append(["Cleaning plan", f"{src} (v{ver})" if ver is not None else str(src)])

        if rows:
            t = Table(rows, colWidths=[6.0 * cm, 10.8 * cm])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("BOX", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 8))

        enabled_steps = plan.get("enabled_steps") or {}
        if enabled_steps:
            enabled_list = [k for k, v in enabled_steps.items() if v]
            disabled_list = [k for k, v in enabled_steps.items() if not v]

            story.append(Paragraph("Enabled steps", self.styles["SectionTitle"]))
            story.append(Paragraph(self._safe_text(", ".join(enabled_list) if enabled_list else "—"), self.styles["Body"]))
            if disabled_list:
                story.append(Paragraph("Disabled steps", self.styles["SectionTitle"]))
                story.append(Paragraph(self._safe_text(", ".join(disabled_list)), self.styles["Body"]))
            story.append(Spacer(1, 8))

        notes = plan.get("notes") or []
        if notes:
            story.append(Paragraph("Cleaning notes", self.styles["SectionTitle"]))
            for n in notes[:8]:
                story.append(Paragraph(f"• {self._safe_text(n)}", self.styles["Body"]))
            story.append(Spacer(1, 8))

        changes: List[str] = []

        dropped_rows = drop_rules.get("dropped_rows_high_missing")
        dropped_cols = drop_rules.get("dropped_total_columns")
        if dropped_rows is not None:
            changes.append(f"Dropped rows (high missing): {dropped_rows}")
        if dropped_cols is not None:
            changes.append(f"Dropped columns (total): {dropped_cols}")

        dropped_dups = dedup.get("dropped_duplicates")
        if dropped_dups is not None:
            changes.append(f"Removed duplicates: {dropped_dups}")

        clipped_cols = outliers.get("columns_clipped")
        if clipped_cols:
            changes.append(f"Outliers clipped in: {', '.join(list(clipped_cols)[:8])}{'…' if len(clipped_cols) > 8 else ''}")

        total_filled = imputation.get("total_filled")
        if total_filled is not None:
            changes.append(f"Imputed cells filled: {total_filled}")

        if changes:
            story.append(Paragraph("Transformations summary", self.styles["SectionTitle"]))
            for c in changes:
                story.append(Paragraph(f"• {self._safe_text(c)}", self.styles["Body"]))
            story.append(Spacer(1, 8))

        post_warnings = post.get("warnings") or []
        if post_warnings:
            story.append(Paragraph("Warnings", self.styles["SectionTitle"]))
            for w in post_warnings[:6]:
                story.append(Paragraph(f"• {self._safe_text(w)}", self.styles["Body"]))
            story.append(Spacer(1, 8))

        return story

    def _render_signals(self, signals: Dict[str, Any], signals_plots: List[Dict[str, Any]], llm_output: Optional[Dict[str, Any]]) -> List[Any]:
        story: List[Any] = []



        rows: List[List[str]] = []
        for k in ["n_rows", "n_cols", "missing_ratio", "num_columns", "cat_columns"]:
            if k in signals:
                v = signals.get(k)
                if isinstance(v, (list, dict)):
                    vs = str(v)
                    v = (vs[:250] + "…") if len(vs) > 250 else vs
                rows.append([k, str(v)])

        if rows:
            t = Table(rows, colWidths=[5 * cm, 11.8 * cm])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("BOX", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 8))
        else:
            None

        if signals_plots:
            story.append(Paragraph("Signals Plots", self.styles["SectionTitle"]))
            story.extend(
                self._render_plots_section(
                    title="",
                    plots=signals_plots,
                    llm_text=None,
                    llm_output=llm_output,
                )
            )

        return story

    def _render_plots_section(
        self,
        *,
        title: str,
        plots: List[Dict[str, Any]],
        llm_text: Optional[str],
        llm_output: Optional[Dict[str, Any]],
    ) -> List[Any]:
        story: List[Any] = []

        if llm_text:
            story.append(Paragraph(self._safe_text(llm_text), self.styles["Body"]))
            story.append(Spacer(1, 8))

        if not plots:
            if title:
                story.append(Paragraph("No plots were saved for this stage.", self.styles["Body"]))
                story.append(Spacer(1, 8))
            return story

        if title:
            story.append(Paragraph(title, self.styles["SectionTitle"]))

        captions: Dict[str, str] = {}
        if isinstance(llm_output, dict):
            c = llm_output.get("plot_captions") or {}
            if isinstance(c, dict):
                captions = c

        for idx, p in enumerate(plots, start=1):
            meta = p.get("meta") or {}
            plot_title = meta.get("title") or meta.get("plot_title") or p.get("kind") or f"Plot {idx}"

            if title:
                story.append(Paragraph(f"{idx}. {self._escape(str(plot_title))}", self.styles["Body"]))
                story.append(Spacer(1, 4))
            else:
                story.append(Paragraph(self._escape(str(plot_title)), self.styles["Body"]))
                story.append(Spacer(1, 4))

            storage_key = p.get("storage_key")
            img_bytes = p.get("image_bytes")

            img_flowable: Optional[Image] = None
            if isinstance(img_bytes, (bytes, bytearray)) and img_bytes:
                img_flowable = self._image_from_bytes(bytes(img_bytes))
            elif isinstance(storage_key, str) and storage_key.strip():
                img_flowable = self._image_from_storage_key(storage_key)

            if img_flowable is None:
                story.append(Paragraph("Could not load image.", self.styles["Body"]))
                story.append(Spacer(1, 10))
                continue

            story.append(img_flowable)
            story.append(Spacer(1, 6))

            plot_id = p.get("plot_id")

            cap_key = None
            if isinstance(plot_id, str) and plot_id.strip():
                cap_key = plot_id.strip()
            elif isinstance(storage_key, str) and storage_key.strip():
                cap_key = storage_key.strip()
            elif isinstance(meta.get("title"), str) and meta.get("title").strip():
                cap_key = meta.get("title").strip()
            elif isinstance(p.get("kind"), str) and p.get("kind").strip():
                cap_key = p.get("kind").strip()

            caption = captions.get(cap_key) if cap_key else None

            caption = captions.get(cap_key) if cap_key else None
            if isinstance(caption, str) and caption.strip():
                story.append(Paragraph(self._safe_text(caption), self.styles["Caption"]))
            else:
                story.append(Spacer(1, 10))

        return story

    def _image_from_storage_key(self, storage_key: str) -> Optional[Image]:
        try:
            img_bytes = blob.get_bytes(storage_key)
            return self._image_from_bytes(img_bytes)
        except Exception:
            return None

    def _image_from_bytes(self, img_bytes: Optional[bytes]) -> Optional[Image]:
        try:
            if not img_bytes:
                return None

            bio = io.BytesIO(img_bytes)

            max_w = self.config.max_image_width_cm * cm
            max_h = self.config.max_image_height_cm * cm

            img = Image(bio)
            iw, ih = float(img.imageWidth), float(img.imageHeight)
            if iw <= 0 or ih <= 0:
                return None

            scale = min(max_w / iw, max_h / ih, 1.0)
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale
            img.hAlign = "LEFT"
            return img
        except Exception:
            return None

    def _safe_text(self, text: Any) -> str:
        s = str(text or "").strip()
        if len(s) > self.config.max_text_chars:
            s = s[: self.config.max_text_chars] + "…"

        s = self._escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = s.replace("\n", "<br/>")

        return s

    def _escape(self, s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def render(self, payload: Dict[str, Any]) -> bytes:
        builder_output: Dict[str, Any] = dict(payload or {})

        llm_output: Dict[str, Any] = {
            "executive_summary": builder_output.get("narrative_text"),
            "cleaning_notes": builder_output.get("cleaning_notes"),
            "signals_notes": builder_output.get("signals_notes"),
            "visualization_notes": builder_output.get("visualization_notes"),
            "forecasting_notes": builder_output.get("forecasting_notes"),
            "conclusion": builder_output.get("conclusion"),
            "plot_captions": builder_output.get("plot_captions") or {},
        }

        return self.render_pdf(builder_output=builder_output, llm_output=llm_output)