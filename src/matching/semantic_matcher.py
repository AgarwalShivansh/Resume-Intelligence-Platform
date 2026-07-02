"""
Semantic skill matcher using sentence embeddings.

Why this matters: TF-IDF only catches exact word matches. If a resume says
"Machine Learning" and the JD says "ML", or the resume says "Postgres" and
the JD says "PostgreSQL", TF-IDF treats those as completely unrelated.
Embeddings capture meaning, so semantically similar skills score as a match
even with different wording — this is the upgrade that makes the matcher
genuinely useful rather than a keyword-search toy.

CHANGE LOG (v1.1):
- Embedding model name and default match threshold now come from src.config
  instead of being hardcoded, so they can be tuned in one place.
- Model loading is wrapped in EmbeddingError so a failure here (e.g., no
  internet access on first run, when the model needs to download) shows a
  clear message instead of a raw library traceback.
- Added logging of model load time and match statistics.
- Public function signatures (semantic_skill_match, semantic_match_score)
  are UNCHANGED apart from the default threshold now reading from config
  rather than being a hardcoded 0.6 literal (same value, just no longer
  duplicated).
"""

import time

from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import get_logger, EMBEDDING_MODEL_NAME, SEMANTIC_MATCH_THRESHOLD
from src.exceptions import EmbeddingError

logger = get_logger(__name__)

# Loaded once and reused — loading this model is the slow part, so we
# don't want to reload it on every function call.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazily load and cache the sentence-embedding model.

    Raises:
        EmbeddingError: if the model fails to load (e.g., first-run download
            fails due to no internet access).
    """
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first call only)...")
            start = time.time()
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Embedding model loaded in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(
                "Couldn't load the skill-matching AI model. This usually means no "
                "internet access on first run (the model needs to download once)."
            ) from e
    return _model


def semantic_skill_match(resume_skills: list[str], jd_skills: list[str], threshold: float = SEMANTIC_MATCH_THRESHOLD):
    """
    For each JD-required skill, find the best-matching resume skill using
    cosine similarity between embeddings.

    Returns:
        matched: list of (jd_skill, matched_resume_skill, similarity_score)
        missing: list of jd_skills with no good match above threshold
    """
    if not jd_skills:
        return [], []
    if not resume_skills:
        return [], [(skill, None, 0.0) for skill in jd_skills]

    model = get_model()

    resume_embeddings = model.encode(resume_skills, normalize_embeddings=True)
    jd_embeddings = model.encode(jd_skills, normalize_embeddings=True)

    similarity_matrix = np.dot(jd_embeddings, resume_embeddings.T)  # cosine sim since normalized

    matched = []
    missing = []

    for i, jd_skill in enumerate(jd_skills):
        best_idx = int(np.argmax(similarity_matrix[i]))
        best_score = float(similarity_matrix[i][best_idx])

        if best_score >= threshold:
            matched.append((jd_skill, resume_skills[best_idx], round(best_score, 3)))
        else:
            missing.append((jd_skill, None, round(best_score, 3)))

    logger.info(f"Semantic matching: {len(matched)}/{len(jd_skills)} JD skills matched (threshold={threshold})")
    return matched, missing


def semantic_match_score(resume_skills: list[str], jd_skills: list[str]) -> float:
    """
    Overall score (0-100) = proportion of JD skills that found a good semantic match.
    """
    if not jd_skills:
        return 100.0

    matched, missing = semantic_skill_match(resume_skills, jd_skills)
    return round((len(matched) / len(jd_skills)) * 100, 2)


if __name__ == "__main__":
    resume_skills = ["Python", "Machine Learning", "Postgres", "Power BI", "Pandas"]
    jd_skills = ["Python", "ML", "PostgreSQL", "Tableau", "Data Visualization"]

    matched, missing = semantic_skill_match(resume_skills, jd_skills)
    print("Matched:", matched)
    print("Missing:", missing)
    print("Score:", semantic_match_score(resume_skills, jd_skills))
