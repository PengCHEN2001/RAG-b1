"""Shared model-name configuration for application and evaluation LLM calls."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Loading the project .env here keeps CLI, Agent, SQL, and Evaluation defaults
# consistent. Existing process variables still take precedence over .env values.
load_dotenv(ENV_PATH)


def get_gemini_model(override: str | None = None) -> str:
    """Resolve explicit override, then GEMINI_MODEL, then the project default."""
    if override is not None:
        model = override.strip()
        if not model:
            raise ValueError("The model override must be a non-empty string.")
        return model
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL


def get_ragas_model(override: str | None = None) -> str:
    """Resolve the optional judge override, otherwise follow the app model."""
    if override is not None:
        model = override.strip()
        if not model:
            raise ValueError("The RAGAS model override must be a non-empty string.")
        return model
    return os.getenv("RAGAS_MODEL", "").strip() or get_gemini_model()
