"""
Analysis orchestration service.

WHY THIS FILE EXISTS (Phase 2):
Previously, the entire analysis pipeline (extract resume text -> LLM-parse
resume -> LLM-parse JD -> compute score) lived inline inside
app/streamlit_app.py, mixed directly with st.spinner(), st.error(), and
other Streamlit UI calls. That meant the "business logic" — the actual
sequence of steps that defines what this product DOES — could not be
tested, reused, or reasoned about without a running Streamlit server.

This module extracts that sequence into one plain Python function,
`run_full_analysis`, with zero Streamlit imports. The UI layer becomes a
thin client of this function: it calls it, gets back a single typed
AnalysisResult, and only handles rendering and error display.

This is also what makes Phase 7 (testing) practical — we can now write a
unit test that calls run_full_analysis() directly with mock inputs, without
needing to spin up a Streamlit server first.
"""

import os
import tempfile
from typing import Callable, Optional

from src.config import get_logger
from src.parsing.extractor import extract_text
from src.extraction.llm_extractor import extract_resume_data, validate_extraction_against_source
from src.extraction.jd_extractor import extract_jd_requirements
from src.scoring.score_engine import compute_final_score
from src.services.result_types import AnalysisResult

logger = get_logger(__name__)

# A progress callback receives a short human-readable stage description.
# The UI layer can pass a function that updates a spinner/status message;
# if no callback is given, progress reporting is simply skipped — the
# service works identically either way.
ProgressCallback = Optional[Callable[[str], None]]


def _report(callback: ProgressCallback, message: str) -> None:
    if callback is not None:
        callback(message)


def run_full_analysis(
    uploaded_file_bytes: bytes,
    uploaded_file_name: str,
    jd_text: str,
    on_progress: ProgressCallback = None,
) -> AnalysisResult:
    """
    Run the complete resume-vs-job-description analysis pipeline.

    This is the single entry point the UI layer should call. It performs,
    in order:
      1. Write the uploaded file to a temp path and extract raw text
      2. LLM-extract structured resume data + run hallucination check
      3. LLM-extract JD requirements (required + preferred skills)
      4. Compute the blended match score + AI explanation

    Args:
        uploaded_file_bytes: raw bytes of the uploaded resume file.
        uploaded_file_name: original filename (used only to determine the
            file extension, e.g. '.pdf' vs '.docx').
        jd_text: the job description text.
        on_progress: optional callback invoked with a short status string
            before each stage starts (e.g. for driving a UI spinner).

    Returns:
        AnalysisResult containing everything the UI needs to render.

    Raises:
        Any ResumeIntelligenceError subclass (EmptyResumeError,
        CorruptedFileError, MissingAPIKeyError, LLMTimeoutError,
        LLMResponseError, EmptyJobDescriptionError, EmbeddingError, etc.)
        propagates unchanged — the caller (UI layer) is responsible for
        catching and displaying these.
    """
    logger.info(f"Starting full analysis pipeline for file '{uploaded_file_name}'")

    # ── Stage 1: Extract raw text from the uploaded file ────────────────
    _report(on_progress, "Extracting resume text...")
    suffix = os.path.splitext(uploaded_file_name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file_bytes)
        tmp_path = tmp.name

    try:
        resume_text = extract_text(tmp_path)
    finally:
        os.remove(tmp_path)

    # ── Stage 2: LLM-extract structured resume data ─────────────────────
    _report(on_progress, "Extracting structured resume data with LLM...")
    resume_data = extract_resume_data(resume_text)
    hallucination_check = validate_extraction_against_source(resume_data, resume_text)

    # ── Stage 3: LLM-extract JD requirements ─────────────────────────────
    _report(on_progress, "Extracting job requirements...")
    jd_requirements = extract_jd_requirements(jd_text)
    jd_skills = jd_requirements.get("required_skills", []) + jd_requirements.get("preferred_skills", [])

    # ── Stage 4: Compute final score + explanation ───────────────────────
    _report(on_progress, "Computing match score and explanation...")
    score_result = compute_final_score(
        resume_text=resume_text,
        jd_text=jd_text,
        resume_skills=resume_data.skills,
        jd_skills=jd_skills,
    )

    logger.info(f"Analysis pipeline complete. Overall score: {score_result['overall_score']}")

    return AnalysisResult(
        resume_text=resume_text,
        resume_data=resume_data,
        hallucination_check=hallucination_check,
        jd_requirements=jd_requirements,
        score_result=score_result,
    )
