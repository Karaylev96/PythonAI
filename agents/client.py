"""Singleton wrapper for the Google GenAI client."""

import os
from threading import RLock
from typing import Any, ClassVar

from dotenv import load_dotenv
from google import genai


_GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


class SingletonMeta(type):
    """Thread-safe singleton metaclass."""

    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[RLock] = RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class GeminiClient(metaclass=SingletonMeta):
    """Google GenAI client wrapper."""

    def __init__(self) -> None:
        load_dotenv()
        self.model_name = os.getenv("GENAI_MODEL_NAME", _GEMINI_DEFAULT_MODEL)
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("GENAI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=api_key)

    def generate_content(self, contents: Any) -> str:
        """Generate content and return the model's text response."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return (response.text or "").strip()
