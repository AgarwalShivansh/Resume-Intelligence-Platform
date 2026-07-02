"""
Result types returned by the service layer.

WHY THIS FILE EXISTS (Phase 2):
Before this change, streamlit_app.py directly called five different backend
functions and manually assembled their outputs into local variables
(resume_text, resume_data, hallucination_check, jd_requirements, result).
The UI file had to know the internal shape of every backend module's output.

AnalysisResult bundles everything the UI needs into ONE typed object. The UI
now only needs to know about this one shape, not five different function
signatures — that's the core of the "separate UI from business logic" goal.
"""

from dataclasses import dataclass, field

from src.extraction.schema import ResumeData


@dataclass
class AnalysisResult:
    """Everything the UI needs to render the result of one resume-vs-JD analysis."""

    # Raw extracted text (kept for transparency / debugging views)
    resume_text: str

    # Structured resume data extracted by the LLM
    resume_data: ResumeData

    # {"verified_skills": [...], "unverified_skills": [...]}
    hallucination_check: dict

    # Raw JD extraction output: {"required_skills": [...], "preferred_skills": [...], ...}
    jd_requirements: dict

    # Scoring output: overall_score, tfidf_score, semantic_score, matched_skills,
    # missing_skills, explanation, improvement_suggestions
    score_result: dict = field(default_factory=dict)
