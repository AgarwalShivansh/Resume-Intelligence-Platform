"""
Final scoring and explanation layer.

Combines the TF-IDF baseline score and the semantic skill-match score into
one overall score, then asks the LLM to explain the result in plain language
and suggest concrete improvements — this is what turns a bare number into
something genuinely useful to a candidate or recruiter.

CHANGE LOG (v1.1):
- Score weights (TFIDF_SCORE_WEIGHT / SEMANTIC_SCORE_WEIGHT) now come from
  src.config instead of being hardcoded inline, so they can be tuned in one
  place (or via environment variables) without touching this file.
- Reuses the shared LLM call helper from llm_extractor (no more duplicated
  Groq client setup in this file).
- Added logging and a graceful fallback if explanation generation fails —
  the match score itself is still returned even if the LLM explanation
  step has a transient issue, so one failure doesn't blank out the whole result.
- Public function signatures (compute_final_score, generate_explanation)
  are UNCHANGED.
"""

import json

from src.config import get_logger, TFIDF_SCORE_WEIGHT, SEMANTIC_SCORE_WEIGHT
from src.exceptions import ResumeIntelligenceError
from src.matching.tfidf_baseline import tfidf_similarity_score
from src.matching.semantic_matcher import semantic_skill_match, semantic_match_score
from src.extraction.llm_extractor import _call_llm_with_retries

logger = get_logger(__name__)

EXPLANATION_PROMPT = """You are a career coach helping a candidate understand how well
their resume matches a job description.

Given:
- Matched skills: {matched_skills}
- Missing skills: {missing_skills}
- Overall match score: {score}/100

Write:
1. A short (2-3 sentence) explanation of the score in plain language.
2. Three specific, actionable suggestions to improve the match (e.g., specific skills
   to learn, or how to phrase existing experience better).

Return ONLY valid JSON in this format:
{{
  "explanation": "string",
  "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}
"""

_FALLBACK_EXPLANATION = {
    "explanation": (
        "We couldn't generate a detailed AI explanation right now, but your score "
        "above is based on a blend of keyword overlap and semantic skill matching "
        "between your resume and the job description."
    ),
    "improvement_suggestions": [
        "Review the 'Missing Skills' list above and consider adding relevant experience or projects.",
        "Try again in a moment — this explanation step occasionally has temporary hiccups.",
    ],
}


def compute_final_score(resume_text: str, jd_text: str, resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compute the final weighted match score and generate an explanation.

    Weighting: semantic skill match is weighted higher than TF-IDF by default
    (see TFIDF_SCORE_WEIGHT / SEMANTIC_SCORE_WEIGHT in src/config.py) because
    it's more robust to synonyms and phrasing differences — TF-IDF is kept
    mainly as an interpretable sanity-check baseline.
    """
    logger.info("Computing final match score")

    tfidf_score = tfidf_similarity_score(resume_text, jd_text)
    semantic_score = semantic_match_score(resume_skills, jd_skills)

    overall_score = round(TFIDF_SCORE_WEIGHT * tfidf_score + SEMANTIC_SCORE_WEIGHT * semantic_score, 2)

    matched, missing = semantic_skill_match(resume_skills, jd_skills)
    matched_skill_names = [m[0] for m in matched]
    missing_skill_names = [m[0] for m in missing]

    logger.info(f"Score breakdown — TF-IDF: {tfidf_score}, Semantic: {semantic_score}, Overall: {overall_score}")

    try:
        explanation_data = generate_explanation(matched_skill_names, missing_skill_names, overall_score)
    except ResumeIntelligenceError as e:
        # Don't let a flaky explanation call blank out the whole result — the
        # numeric score and skill lists are still useful on their own.
        logger.warning(f"Explanation generation failed, using fallback text: {e}")
        explanation_data = _FALLBACK_EXPLANATION

    return {
        "overall_score": overall_score,
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "matched_skills": matched_skill_names,
        "missing_skills": missing_skill_names,
        "explanation": explanation_data["explanation"],
        "improvement_suggestions": explanation_data["improvement_suggestions"],
    }


def generate_explanation(matched_skills: list[str], missing_skills: list[str], score: float) -> dict:
    """
    Ask the LLM for a plain-language explanation of the score plus concrete
    improvement suggestions.

    Raises:
        LLMTimeoutError / LLMResponseError: propagated from the shared LLM call helper.
    """
    prompt = EXPLANATION_PROMPT.format(
        matched_skills=matched_skills or "none",
        missing_skills=missing_skills or "none",
        score=score,
    )

    raw_output = _call_llm_with_retries(system_prompt="", user_content=prompt, temperature=0.3)
    return json.loads(raw_output)


if __name__ == "__main__":
    resume_text = "Python, Pandas, SQL, Power BI, Machine Learning, Excel"
    jd_text = "Looking for Python, SQL, Tableau, and cloud data warehouse experience"
    resume_skills = ["Python", "Pandas", "SQL", "Power BI", "Machine Learning", "Excel"]
    jd_skills = ["Python", "SQL", "Tableau", "Snowflake"]

    result = compute_final_score(resume_text, jd_text, resume_skills, jd_skills)
    print(json.dumps(result, indent=2))
