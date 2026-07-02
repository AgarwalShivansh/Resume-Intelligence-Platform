"""
LLM-based structured extraction.

We send raw resume text to a Groq-hosted LLM (e.g., Llama 3.3 70B) and ask it
to return JSON matching our ResumeData schema. We then validate that JSON
against the Pydantic schema — if the LLM hallucinates a bad shape, this
catches it immediately rather than silently passing broken data downstream.

CHANGE LOG (v1.1):
- API key, model name, and client setup now come from src.config instead of
  being duplicated/hardcoded in this file (and in jd_extractor.py, score_engine.py).
- Added a clear MissingAPIKeyError if the key isn't configured, instead of
  letting the Groq SDK fail with an obscure internal error later.
- Added a retry loop with timeout handling around the LLM call.
- Added logging of each extraction attempt, timing, and outcome.
- Public function signatures (extract_resume_data, validate_extraction_against_source)
  are UNCHANGED — existing callers (streamlit_app.py) keep working with no edits needed.
"""

import json
import time

from groq import Groq, APITimeoutError, APIError

from src.config import get_logger, GROQ_API_KEY, MODEL_NAME, LLM_REQUEST_TIMEOUT_SECONDS, LLM_MAX_RETRIES
from src.exceptions import MissingAPIKeyError, LLMTimeoutError, LLMResponseError
from src.extraction.schema import ResumeData

logger = get_logger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    """
    Lazily create and cache the Groq client.

    Lazy creation (rather than at import time) means importing this module
    never fails just because the API key is missing — the error only surfaces
    when an extraction is actually attempted, with a clear message.
    """
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Create a .env file with your free key "
                "from https://console.groq.com (see .env.example for the format)."
            )
        _client = Groq(api_key=GROQ_API_KEY, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
    return _client


EXTRACTION_SYSTEM_PROMPT = """You are a precise resume-parsing assistant.

Extract structured information from the resume text the user provides.
Return ONLY valid JSON matching this exact structure (no markdown, no commentary, no code fences):

{
  "name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "summary": "string or null (the candidate's professional summary/objective paragraph, if present)",
  "skills": ["skill1", "skill2", ...],
  "education": [{"degree": "...", "institution": "...", "year": "... or null"}],
  "experience": [{"title": "...", "company": "...", "duration": "... or null", "description": "... or null"}],
  "projects": [{"title": "...", "description": "... or null", "technologies": ["..."]}],
  "certifications": [{"name": "...", "issuer": "... or null", "year": "... or null"}],
  "achievements": [{"title": "...", "description": "... or null"}],
  "publications": [{"title": "...", "venue": "... or null", "year": "... or null"}],
  "total_experience_years": number or null
}

CRITICAL RULES:
- Only extract information that is ACTUALLY present in the text. Never invent or assume details.
- If a field is not present, use null (for single values) or an empty list (for lists).
- "skills" should include every technical skill, tool, language, and framework explicitly mentioned anywhere in the resume.
- "experience" should include internships as well as full-time/part-time roles — an internship is just an Experience entry with its own title/company/duration.
- "certifications" covers things like "AWS Certified Cloud Practitioner", "Google Data Analytics Certificate", online course completion certificates, etc.
- "achievements" covers awards, hackathon wins, academic honors (e.g. Dean's List), competition rankings, scholarships — anything presented as a notable accomplishment, NOT regular coursework or projects.
- "publications" covers research papers, articles, or blog posts the candidate authored — only include this if explicitly present; most resumes will have an empty list here, which is correct and expected.
- Return ONLY the JSON object. No explanations, no markdown formatting.
"""


def _call_llm_with_retries(system_prompt: str, user_content: str, temperature: float = 0) -> str:
    """
    Call the Groq chat completion API with retry-on-timeout logic.

    Why retries matter here: free-tier LLM APIs occasionally have transient
    slow responses. Rather than failing the whole resume analysis on one
    blip, we retry a couple of times before giving up.
    """
    client = _get_client()
    last_error: Exception | None = None

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    for attempt in range(1, LLM_MAX_RETRIES + 2):  # +2 = first attempt + retries
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start
            logger.info(f"LLM call succeeded on attempt {attempt} ({elapsed:.2f}s)")
            return response.choices[0].message.content

        except APITimeoutError as e:
            last_error = e
            logger.warning(f"LLM call timed out on attempt {attempt}/{LLM_MAX_RETRIES + 1}")
        except APIError as e:
            last_error = e
            logger.warning(f"LLM API error on attempt {attempt}/{LLM_MAX_RETRIES + 1}: {e}")

    logger.error(f"LLM call failed after {LLM_MAX_RETRIES + 1} attempts: {last_error}")
    if isinstance(last_error, APITimeoutError):
        raise LLMTimeoutError(
            "The AI service took too long to respond. This is usually temporary — please try again."
        ) from last_error
    raise LLMResponseError(f"The AI service returned an error: {last_error}") from last_error


def extract_resume_data(resume_text: str) -> ResumeData:
    """
    Send resume text to the LLM and parse the response into a validated
    ResumeData object.

    Raises:
        MissingAPIKeyError: if GROQ_API_KEY isn't configured.
        LLMTimeoutError: if the LLM call times out even after retries.
        LLMResponseError: if the LLM response isn't valid JSON or doesn't
            match the expected schema.
    """
    logger.info(f"Starting resume extraction ({len(resume_text)} chars of input text)")

    raw_output = _call_llm_with_retries(EXTRACTION_SYSTEM_PROMPT, resume_text, temperature=0)

    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        raise LLMResponseError(f"The AI returned a response we couldn't understand: {e}") from e

    try:
        resume_data = ResumeData(**parsed_json)
    except Exception as e:
        logger.error(f"LLM output failed schema validation: {e}")
        raise LLMResponseError(f"The AI's response didn't match the expected format: {e}") from e

    logger.info(f"Resume extraction complete: {len(resume_data.skills)} skills, "
                f"{len(resume_data.education)} education entries, "
                f"{len(resume_data.experience)} experience entries found")
    return resume_data


def validate_extraction_against_source(resume_data: ResumeData, source_text: str) -> dict:
    """
    Hallucination guard: check that extracted facts actually appear
    (case-insensitively) somewhere in the source text.

    CHANGE LOG (Phase 3): previously this only checked skills. It now also
    checks certification names, achievement titles, and publication titles
    — the fields most likely to contain a fabricated, plausible-sounding
    entry the LLM invented rather than read from the resume. Education and
    experience entries are NOT checked here because they're structured
    multi-field records (degree+institution+year, or title+company+duration)
    where partial verbatim matches are expected and don't indicate
    hallucination the way an invented skill/cert/achievement name would.

    This doesn't catch everything (e.g., a real skill described in different
    wording than the source), but it catches the most common failure mode —
    the LLM inventing an entry that sounds plausible but was never
    mentioned in the resume.

    Returns a dict with a 'verified' and 'unverified' list for each checked
    category, plus an overall 'all_verified' boolean for convenience.
    """
    source_lower = source_text.lower()

    def _check_list(items: list[str]) -> tuple[list[str], list[str]]:
        verified, unverified = [], []
        for item in items:
            if item.lower() in source_lower:
                verified.append(item)
            else:
                unverified.append(item)
        return verified, unverified

    verified_skills, unverified_skills = _check_list(resume_data.skills)
    verified_certs, unverified_certs = _check_list([c.name for c in resume_data.certifications])
    verified_achievements, unverified_achievements = _check_list([a.title for a in resume_data.achievements])
    verified_publications, unverified_publications = _check_list([p.title for p in resume_data.publications])

    total_unverified = len(unverified_skills) + len(unverified_certs) + len(unverified_achievements) + len(unverified_publications)
    if total_unverified > 0:
        logger.warning(
            f"Hallucination check found {total_unverified} unverified item(s) — "
            f"skills={unverified_skills}, certifications={unverified_certs}, "
            f"achievements={unverified_achievements}, publications={unverified_publications}"
        )

    return {
        "verified_skills": verified_skills,
        "unverified_skills": unverified_skills,
        "verified_certifications": verified_certs,
        "unverified_certifications": unverified_certs,
        "verified_achievements": verified_achievements,
        "unverified_achievements": unverified_achievements,
        "verified_publications": verified_publications,
        "unverified_publications": unverified_publications,
        "all_verified": total_unverified == 0,
    }


if __name__ == "__main__":
    sample_text = """
    John Doe
    john.doe@email.com | +91-9876543210

    Education: B.Tech Computer Science, ABC University, 2026

    Skills: Python, Pandas, scikit-learn, SQL, Power BI

    Projects:
    - Churn Prediction Model: Built a classifier using XGBoost and SHAP.
    """
    result = extract_resume_data(sample_text)
    print(result.model_dump_json(indent=2))

    check = validate_extraction_against_source(result, sample_text)
    print("\nHallucination check:", check)
