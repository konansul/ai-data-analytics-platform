from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StaticRebuildResult:
    version: int
    csv: str
    notes: List[str]
    confidence: float | None = None


def validate_static_result(d: Dict[str, Any]) -> StaticRebuildResult:
    if not isinstance(d, dict):
        raise TypeError("Static rebuild result must be dict")

    csv = d.get("csv")
    if not isinstance(csv, str) or not csv.strip():
        raise ValueError("Missing csv payload from LLM")

    notes = d.get("notes", [])
    if not isinstance(notes, list):
        notes = []

    confidence = d.get("confidence")

    return StaticRebuildResult(
        version=int(d.get("version", 1)),
        csv=csv,
        notes=[str(x) for x in notes],
        confidence=float(confidence) if confidence is not None else None,
    )