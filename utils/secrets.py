from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()


class LLM_API_KEY(Enum):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class GEMINI_MODELS(Enum):
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
