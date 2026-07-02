"""
Extract required skills from a job description using the same LLM backend.
This is a simpler extraction than the full resume parse — we just need
the list of required skills/qualifications.

CHANGE LOG (v1.1):
- Reuses the shared LLM call helper from llm_extractor (no more duplicated
  Groq client setup / model name / load_dotenv() in this file).
- Added EmptyJobDescriptionError for blank/whitespace-only input.
- Added logging.
- Public function signature (extract_jd_requirements) is UNCHANGED.
"""

import json

from src.config import get_logger
from src.exceptions import EmptyJobDescriptionError, LLMResponseError
from src.extraction.llm_extractor import _call_llm_with_retries

logger = get_logger(__name__)

JD_SYSTEM_PROMPT = """You are a precise job-description parser.

Extract the required and preferred skills/technologies from the job description text.
Return ONLY valid JSON in this exact format (no markdown, no commentary):

{
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", "skill2", ...],
  "role_title": "string or null",
  "experience_required": "string describing required experience, or null"
}

Only extract skills explicitly stated or clearly implied by the text. Do not invent skills.
"""


def extract_jd_requirements(jd_text: str) -> dict:
    """
    Extract required/preferred skills and basic metadata from a job
    description using the LLM.

    Raises:
        EmptyJobDescriptionError: if jd_text is blank or whitespace-only.
        LLMTimeoutError / LLMResponseError: propagated from the shared LLM call helper.
    """
    if not jd_text or not jd_text.strip():
        raise EmptyJobDescriptionError("The job description is empty. Please paste the full job description text.")

    logger.info(f"Starting JD extraction ({len(jd_text)} chars of input text)")

    raw_output = _call_llm_with_retries(JD_SYSTEM_PROMPT, jd_text, temperature=0)

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON for JD extraction: {e}")
        raise LLMResponseError(f"The AI returned a response we couldn't understand: {e}") from e

    required_count = len(result.get("required_skills", []))
    preferred_count = len(result.get("preferred_skills", []))
    logger.info(f"JD extraction complete: {required_count} required skills, {preferred_count} preferred skills")

    return result


if __name__ == "__main__":
    sample_jd = """
    We are hiring a Data Analyst with strong SQL and Power BI skills.
    Required: Python, SQL, Power BI, Excel.
    Preferred: experience with cloud data warehouses like Snowflake or BigQuery.
    Minimum 1 year of experience or relevant internship.
    """
    result = extract_jd_requirements(sample_jd)
    print(json.dumps(result, indent=2))
