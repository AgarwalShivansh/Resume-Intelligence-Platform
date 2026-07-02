"""
Centralized configuration for the Resume Intelligence Platform.

WHY THIS FILE EXISTS:
Before this change, the Groq API key, model name, and `load_dotenv()` call
were each duplicated across llm_extractor.py, jd_extractor.py, and
score_engine.py. That meant changing the model name required editing three
files, and there was no single place to see "what can I configure?".

This file is now the ONLY place that reads environment variables or defines
tunable constants. Every other module imports from here instead of calling
os.getenv() or hardcoding values directly.
"""

import os
import logging
from dotenv import load_dotenv

# Load .env file once, at import time, before anything else needs the key.
load_dotenv()


# ── API Configuration ────────────────────────────────────────────────────
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
MODEL_NAME: str = os.getenv("RESUME_AI_MODEL", "llama-3.3-70b-versatile")

# Network behavior for LLM calls — surfaced here so they're easy to tune
# without hunting through extraction code.
LLM_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))


# ── Matching / Scoring Thresholds ────────────────────────────────────────
# How similar two skill embeddings must be (cosine similarity, 0-1) to count
# as the same skill. Raising this makes matching stricter.
SEMANTIC_MATCH_THRESHOLD: float = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.6"))

# Weighting between the TF-IDF baseline and the semantic matcher in the
# final blended score. Must sum to 1.0.
TFIDF_SCORE_WEIGHT: float = float(os.getenv("TFIDF_SCORE_WEIGHT", "0.3"))
SEMANTIC_SCORE_WEIGHT: float = float(os.getenv("SEMANTIC_SCORE_WEIGHT", "0.7"))

# Embedding model used for semantic skill matching.
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


# ── File Upload Constraints ──────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_FILE_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx")

# A resume with fewer than this many characters of extracted text is treated
# as "effectively empty" (e.g., a scanned image PDF with no real text layer).
MIN_RESUME_TEXT_LENGTH: int = int(os.getenv("MIN_RESUME_TEXT_LENGTH", "50"))


# ── Logging Configuration ────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Centralizing this means every module gets consistent formatting and
    log level without each file configuring logging independently.

    Usage in any module:
        from src.config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers if this is called more than once
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)

    return logger


def validate_config() -> list[str]:
    """
    Check that required configuration is present and sane.

    Returns a list of human-readable problem descriptions. An empty list
    means configuration is valid. The Streamlit app calls this at startup
    so missing API keys produce a clear, friendly error instead of a
    confusing traceback deep inside the Groq SDK.
    """
    problems: list[str] = []

    if not GROQ_API_KEY:
        problems.append(
            "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
            "with your free key from https://console.groq.com"
        )

    if abs((TFIDF_SCORE_WEIGHT + SEMANTIC_SCORE_WEIGHT) - 1.0) > 1e-6:
        problems.append(
            f"TFIDF_SCORE_WEIGHT ({TFIDF_SCORE_WEIGHT}) and SEMANTIC_SCORE_WEIGHT "
            f"({SEMANTIC_SCORE_WEIGHT}) must sum to 1.0"
        )

    if not (0.0 <= SEMANTIC_MATCH_THRESHOLD <= 1.0):
        problems.append(f"SEMANTIC_MATCH_THRESHOLD must be between 0 and 1, got {SEMANTIC_MATCH_THRESHOLD}")

    return problems
