"""
Resume Intelligence Platform — Streamlit App

Upload a resume (PDF/DOCX), paste a job description, and get:
- Structured resume data (extracted via LLM)
- A match score (TF-IDF + semantic blend)
- Matched/missing skills
- An LLM-generated explanation and improvement suggestions

Run with: streamlit run app/streamlit_app.py

CHANGE LOG (v1.1, Phase 2):
- This file is now PURE UI. All orchestration logic (extract -> parse ->
  match -> score) has moved to src/services/analysis_service.py. This file
  only: renders widgets, calls the service layer, and renders the result.
- Replaced 5 separate function calls + manual variable juggling with a
  single call to run_full_analysis(), which returns one typed AnalysisResult.
- Visible behavior, layout, and error messages are UNCHANGED from the
  previous version — this is an internal refactor, not a feature change.

CHANGE LOG (v1.1, Phase 3):
- Added a new "Resume Profile" expander showing the newly extracted fields
  (summary, certifications, achievements, publications) in readable form,
  instead of only being visible buried in the raw debug JSON.
- Updated the "Hallucination check" expander to show verified/unverified
  results for certifications, achievements, and publications (previously
  skills-only), matching the expanded validate_extraction_against_source().
- Core layout (two-column upload/JD section, button, score section) is
  UNCHANGED — these are additive sections appended after the existing ones.
"""

import sys
import os

# Allow imports from src/ when running via `streamlit run`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.config import get_logger, validate_config
from src.exceptions import ResumeIntelligenceError
from src.services.analysis_service import run_full_analysis
from src.services.result_types import AnalysisResult

logger = get_logger(__name__)

st.set_page_config(page_title="Resume Intelligence Platform", page_icon="📄", layout="wide")

st.title("📄 Resume Intelligence Platform")
st.caption("Upload a resume and a job description to get an AI-powered match score with explainability.")

# Startup check — catches a missing API key immediately with a clear banner,
# instead of letting the user click "Analyze" and hit a confusing error.
config_problems = validate_config()
if config_problems:
    for problem in config_problems:
        st.error(f"⚠️ Configuration issue: {problem}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

with col2:
    st.subheader("2. Paste Job Description")
    jd_text_input = st.text_area("Job description text", height=220, placeholder="Paste the full job description here...")

analyze_clicked = st.button("🔍 Analyze Match", type="primary", use_container_width=False)


def render_result(result: AnalysisResult) -> None:
    """Render an AnalysisResult into the Streamlit UI. Pure presentation —
    no business logic lives here, it only reads fields off the result object."""
    score = result.score_result

    st.divider()
    st.subheader("📊 Match Result")

    score_col, breakdown_col = st.columns([1, 2])

    with score_col:
        st.metric("Overall Match Score", f"{score['overall_score']} / 100")
        st.caption(f"TF-IDF baseline: {score['tfidf_score']}/100")
        st.caption(f"Semantic skill match: {score['semantic_score']}/100")

    with breakdown_col:
        st.markdown("**✅ Matched Skills**")
        st.write(", ".join(score["matched_skills"]) if score["matched_skills"] else "None found")

        st.markdown("**❌ Missing Skills**")
        st.write(", ".join(score["missing_skills"]) if score["missing_skills"] else "None — great match!")

    st.markdown("### 🧠 Explanation")
    st.info(score["explanation"])

    st.markdown("### 💡 Improvement Suggestions")
    for suggestion in score["improvement_suggestions"]:
        st.markdown(f"- {suggestion}")

    with st.expander("📄 Resume Profile (summary, certifications, achievements, publications)"):
        rd = result.resume_data

        if rd.summary:
            st.markdown("**Summary**")
            st.write(rd.summary)

        if rd.certifications:
            st.markdown("**Certifications**")
            for cert in rd.certifications:
                line = cert.name
                if cert.issuer:
                    line += f" — {cert.issuer}"
                if cert.year:
                    line += f" ({cert.year})"
                st.markdown(f"- {line}")

        if rd.achievements:
            st.markdown("**Achievements**")
            for ach in rd.achievements:
                line = ach.title
                if ach.description:
                    line += f" — {ach.description}"
                st.markdown(f"- {line}")

        if rd.publications:
            st.markdown("**Publications**")
            for pub in rd.publications:
                line = pub.title
                if pub.venue:
                    line += f" — {pub.venue}"
                if pub.year:
                    line += f" ({pub.year})"
                st.markdown(f"- {line}")

        if not (rd.summary or rd.certifications or rd.achievements or rd.publications):
            st.caption("No summary, certifications, achievements, or publications were found in this resume.")

    with st.expander("🔬 View extracted resume data (debug view)"):
        st.json(result.resume_data.model_dump())

    with st.expander("🛡️ Hallucination check (extracted facts verified against source text)"):
        hc = result.hallucination_check

        if hc.get("all_verified"):
            st.success("✅ All extracted skills, certifications, achievements, and publications were found in the source text.")

        st.markdown("**Skills**")
        st.write("✅ Verified:", hc["verified_skills"] or "—")
        st.write("⚠️ Unverified (review manually):", hc["unverified_skills"] or "—")

        if result.resume_data.certifications:
            st.markdown("**Certifications**")
            st.write("✅ Verified:", hc["verified_certifications"] or "—")
            st.write("⚠️ Unverified (review manually):", hc["unverified_certifications"] or "—")

        if result.resume_data.achievements:
            st.markdown("**Achievements**")
            st.write("✅ Verified:", hc["verified_achievements"] or "—")
            st.write("⚠️ Unverified (review manually):", hc["unverified_achievements"] or "—")

        if result.resume_data.publications:
            st.markdown("**Publications**")
            st.write("✅ Verified:", hc["verified_publications"] or "—")
            st.write("⚠️ Unverified (review manually):", hc["unverified_publications"] or "—")

    with st.expander("📋 Extracted JD requirements"):
        st.json(result.jd_requirements)


if analyze_clicked:
    if not uploaded_file:
        st.error("Please upload a resume file first.")
    elif not jd_text_input.strip():
        st.error("Please paste a job description.")
    else:
        try:
            with st.spinner("Running analysis..."):
                status_placeholder = st.empty()

                def show_progress(message: str) -> None:
                    status_placeholder.info(f"⏳ {message}")

                result = run_full_analysis(
                    uploaded_file_bytes=uploaded_file.getvalue(),
                    uploaded_file_name=uploaded_file.name,
                    jd_text=jd_text_input,
                    on_progress=show_progress,
                )
                status_placeholder.empty()

            logger.info(f"UI received analysis result. Overall score: {result.score_result['overall_score']}")
            render_result(result)

        except ResumeIntelligenceError as e:
            # Every custom exception we raise has a user-friendly message
            # baked in — show it directly rather than a generic "error occurred".
            logger.error(f"Analysis failed with a known error type: {type(e).__name__}: {e}")
            st.error(f"❌ {e}")

        except Exception as e:
            # Genuinely unexpected errors (bugs, library internals) still get
            # caught here so the app never shows a raw Streamlit traceback to
            # the user — but we log the full detail for debugging.
            logger.exception("Analysis failed with an unexpected error")
            st.error(
                "❌ Something unexpected went wrong while analyzing your resume. "
                "Please try again, and if it persists, check the terminal logs for details."
            )

st.divider()
st.caption("Built with Streamlit, Groq (Llama 3.3), sentence-transformers, and scikit-learn.")
