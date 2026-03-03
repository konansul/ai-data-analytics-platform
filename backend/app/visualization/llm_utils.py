import os
import requests
import logging

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Union

from backend.app.cleaning.cleaning_agent.llm_client import LLMClient


class LLMUtils:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.gemini_client = LLMClient.from_env(model=model)
        self.atllama_url = os.getenv("ATLLAMA_API_URL")

    def safe_llm_call(self, prompt: str) -> Union[Dict, List]:
        try:
            if self.atllama_url:
                resp = requests.post(
                    self.atllama_url,
                    json={"prompt": prompt},
                    timeout=20,
                )
                resp.raise_for_status()
                return self.gemini_client.extract_json(resp.json().get("text", ""))
            raise RuntimeError("No ATLLaMa URL configured")
        except Exception:
            return self.gemini_client.extract_json(
                self.gemini_client.complete(prompt)
            )

    def ensure_list(self, data: Any, key: str) -> List:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get(key, [])
        return []

    def complete(self, prompt: str) -> str:
        return self.gemini_client.complete(prompt)
