"""
Tests for the service layer (src/services/analysis_service.py).

WHY THESE TESTS MATTER:
Before Phase 2, the full analysis pipeline was embedded inside
streamlit_app.py, mixed with st.spinner() and other UI calls — there was
no way to test "does the pipeline correctly raise an error for a bad
file?" without literally running a Streamlit server and clicking buttons.

These tests call run_full_analysis() directly, as plain Python, proving
the orchestration logic is now decoupled from the UI.

Note: tests that require a real LLM call (valid PDF + real API key) are
NOT included here, since they'd require network access and a paid/free
API key to run in CI. These tests focus on the error-handling paths,
which don't need the LLM to be reached at all.

Run with: pytest tests/test_analysis_service.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.services.analysis_service import run_full_analysis
from src.exceptions import UnsupportedFileTypeError, EmptyResumeError


def test_unsupported_file_type_raises_before_any_llm_call():
    """A .txt file should be rejected at the parsing stage, never reaching
    the LLM extraction stage."""
    with pytest.raises(UnsupportedFileTypeError):
        run_full_analysis(
            uploaded_file_bytes=b"irrelevant content",
            uploaded_file_name="resume.txt",
            jd_text="Some job description text",
        )


def test_progress_callback_fires_for_first_stage_only_on_early_failure():
    """If the pipeline fails at stage 1 (file parsing), the progress
    callback should have fired exactly once — for stage 1 — and not for
    any later stage."""
    progress_messages = []

    with pytest.raises(UnsupportedFileTypeError):
        run_full_analysis(
            uploaded_file_bytes=b"irrelevant content",
            uploaded_file_name="resume.txt",
            jd_text="Some job description text",
            on_progress=progress_messages.append,
        )

    assert progress_messages == ["Extracting resume text..."]


def test_empty_pdf_raises_empty_resume_error():
    """A syntactically valid PDF with effectively no text should raise
    EmptyResumeError, not crash with an unrelated error."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "hi", fontsize=10)  # only 2 characters
    pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(EmptyResumeError):
        run_full_analysis(
            uploaded_file_bytes=pdf_bytes,
            uploaded_file_name="empty_resume.pdf",
            jd_text="Some job description text",
        )


if __name__ == "__main__":
    test_unsupported_file_type_raises_before_any_llm_call()
    print("PASS: test_unsupported_file_type_raises_before_any_llm_call")

    test_progress_callback_fires_for_first_stage_only_on_early_failure()
    print("PASS: test_progress_callback_fires_for_first_stage_only_on_early_failure")

    test_empty_pdf_raises_empty_resume_error()
    print("PASS: test_empty_pdf_raises_empty_resume_error")

    print("\nAll service layer tests passed!")
