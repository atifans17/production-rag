import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
LOG_DIR = "logs"

# bge-small accepts up to 512 tokens. (all-MiniLM-L6-v2 silently cuts off
# everything after 256 tokens, which would break our 500-token chunks.)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_COLLECTION = os.getenv("COLLECTION", "fixed_500_50")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "0"))

# Chunks scoring below this (cross-encoder score) trigger a refusal. Tuned in Step 9.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "-100"))