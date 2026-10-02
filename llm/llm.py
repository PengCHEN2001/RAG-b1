import os

from dotenv import load_dotenv
from pathlib import Path
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

llm_api_key = os.getenv("LLM_API_KEY")

if not llm_api_key:
    raise RuntimeError("LLM_API_KEY not found in .env")

llm_client = genai.Client(api_key=llm_api_key)

