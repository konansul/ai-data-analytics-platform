# backend/app/cleaning_agent/llm_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from google import genai


class LLMUnavailableError(RuntimeError):
    pass


JSONType = Union[Dict[str, Any], list]


@dataclass
class LLMClient:
    """
    Gemini client wrapper (Google GenAI SDK).
    - complete(prompt) -> str
    - extract_json(text) -> dict (or list, but we validate higher-level anyway)
    """
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    _client: Any = None

    @staticmethod
    def from_env(model: str = "gemini-2.5-flash") -> "LLMClient":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMUnavailableError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        return LLMClient(model=model, api_key=api_key)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise LLMUnavailableError("Missing api_key.")
        self._client = genai.Client(api_key=self.api_key)

    def complete(self, prompt: str) -> str:
        self._ensure_client()

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = getattr(resp, "text", None)
        if not text:
            text = str(resp)
        return text

    def extract_json(self, text: str) -> JSONType:
        """
        Extract JSON from model output.
        Supports:
        - pure JSON (dict OR list)
        - JSON wrapped in ```json ... ```
        - extra text around JSON
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty LLM response, cannot extract JSON.")

        raw = text.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fence = re.search(
            r"```json\s*([\s\S]*?)\s*```",
            raw,
            flags=re.IGNORECASE,
        )
        if fence:
            candidate = fence.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        brace_obj = re.search(r"(\{[\s\S]*?\})", raw)
        if brace_obj:
            try:
                return json.loads(brace_obj.group(1))
            except json.JSONDecodeError:
                pass

        brace_arr = re.search(r"(\[[\s\S]*?\])", raw)
        if brace_arr:
            try:
                return json.loads(brace_arr.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract JSON from LLM output.")