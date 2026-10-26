import os

from dotenv import load_dotenv
from pathlib import Path
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

llm_client = genai.Client(api_key=google_api_key)

