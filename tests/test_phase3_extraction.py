"""
Tests for Phase 3 additions: extended resume schema (summary, certifications,
achievements, publications) and the expanded hallucination verification that
checks all of those fields, not just skills.

Run with: pytest tests/test_phase3_extraction.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.schema import ResumeData, Certification, Achievement, Publication
from src.extraction.llm_extractor import validate_extraction_against_source


def test_resume_data_new_fields_default_to_empty():
    """An empty ResumeData() should still construct cleanly with the new
    Phase 3 fields defaulting to empty/None — this is what keeps Phase 1/2
    code that constructs ResumeData() working unchanged."""
    data = ResumeData()
    assert data.summary is None
    assert data.certifications == []
    assert data.achievements == []
    assert data.publications == []


def test_resume_data_new_fields_populate_correctly():
    """New fields should store provided values correctly, including nested
    sub-models (Certification, Achievement, Publication)."""
    data = ResumeData(
        summary="A passionate data scientist.",
        certifications=[Certification(name="AWS Certified Cloud Practitioner", issuer="AWS", year="2025")],
        achievements=[Achievement(title="Winner, National Hackathon 2025")],
        publications=[Publication(title="A Study on LLM Hallucinations", venue="arXiv", year="2026")],
    )
    assert data.summary == "A passionate data scientist."
    assert data.certifications[0].name == "AWS Certified Cloud Practitioner"
    assert data.achievements[0].title == "Winner, National Hackathon 2025"
    assert data.publications[0].title == "A Study on LLM Hallucinations"


def test_hallucination_check_flags_fabricated_entries_in_every_category():
    """The core Phase 3 guarantee: a fabricated skill, certification,
    achievement, AND publication should each be correctly flagged as
    unverified, while genuine entries pass as verified."""
    source_text = (
        "John Doe, Data Scientist. Skills: Python, SQL, Machine Learning. "
        "Certified: AWS Certified Cloud Practitioner from Amazon. "
        "Achievements: Winner of National Hackathon 2025."
    )

    resume_data = ResumeData(
        skills=["Python", "Quantum Computing"],  # second one is fabricated
        certifications=[
            Certification(name="AWS Certified Cloud Practitioner"),       # real
            Certification(name="Certified Kubernetes Administrator"),     # fabricated
        ],
        achievements=[
            Achievement(title="Winner of National Hackathon 2025"),       # real
            Achievement(title="Olympic Gold Medalist"),                   # fabricated
        ],
        publications=[
            Publication(title="A totally made up paper title"),          # fabricated
        ],
    )

    result = validate_extraction_against_source(resume_data, source_text)

    assert result["verified_skills"] == ["Python"]
    assert result["unverified_skills"] == ["Quantum Computing"]
    assert result["verified_certifications"] == ["AWS Certified Cloud Practitioner"]
    assert result["unverified_certifications"] == ["Certified Kubernetes Administrator"]
    assert result["verified_achievements"] == ["Winner of National Hackathon 2025"]
    assert result["unverified_achievements"] == ["Olympic Gold Medalist"]
    assert result["unverified_publications"] == ["A totally made up paper title"]
    assert result["all_verified"] is False


def test_hallucination_check_all_verified_true_when_clean():
    """When every extracted item genuinely appears in the source text,
    all_verified should be True — this is the 'everything checks out' path."""
    source_text = "Skills: Python, SQL. Certified: AWS Certified Cloud Practitioner."

    resume_data = ResumeData(
        skills=["Python", "SQL"],
        certifications=[Certification(name="AWS Certified Cloud Practitioner")],
    )

    result = validate_extraction_against_source(resume_data, source_text)
    assert result["all_verified"] is True


def test_hallucination_check_handles_empty_categories_gracefully():
    """A resume with no certifications/achievements/publications at all
    should not error out — empty lists in, empty lists out."""
    result = validate_extraction_against_source(ResumeData(skills=["Python"]), "Python developer.")
    assert result["verified_certifications"] == []
    assert result["unverified_certifications"] == []
    assert result["verified_achievements"] == []
    assert result["verified_publications"] == []
    assert result["all_verified"] is True


if __name__ == "__main__":
    test_resume_data_new_fields_default_to_empty()
    print("PASS: test_resume_data_new_fields_default_to_empty")

    test_resume_data_new_fields_populate_correctly()
    print("PASS: test_resume_data_new_fields_populate_correctly")

    test_hallucination_check_flags_fabricated_entries_in_every_category()
    print("PASS: test_hallucination_check_flags_fabricated_entries_in_every_category")

    test_hallucination_check_all_verified_true_when_clean()
    print("PASS: test_hallucination_check_all_verified_true_when_clean")

    test_hallucination_check_handles_empty_categories_gracefully()
    print("PASS: test_hallucination_check_handles_empty_categories_gracefully")

    print("\nAll Phase 3 tests passed!")
