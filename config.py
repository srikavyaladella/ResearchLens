"""
config.py  –  Central configuration (reads from .env)
------------------------------------------------------
Copy .env.template → .env and fill in your values before running.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ── Embedding model (local, no API key needed) ────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── HuggingFace FREE Inference API ────────────────────────────────────────────
# Get a FREE token at https://huggingface.co/settings/tokens  (read scope only)
# Leave blank → app falls back to local pipeline automatically
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Models used (both are FREE public models, no approval needed)
HF_SUMMARY_MODEL = "facebook/bart-large-cnn"            # summarisation
HF_GAP_MODEL     = "mistralai/Mistral-7B-Instruct-v0.1" # gap analysis

# ── Paths ─────────────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
PAPERS_FOLDER     = os.getenv("PAPERS_FOLDER",     "data/raw_papers")