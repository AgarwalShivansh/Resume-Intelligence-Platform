"""
Basic tests for modules that don't require external API calls
(TF-IDF, semantic matching, schema validation).

Run with: pytest tests/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching.tfidf_baseline import tfidf_similarity_score
from src.extraction.schema import ResumeData, Education


def test_tfidf_similarity_identical_text():
    """Identical text should score very close to 100."""
    text = "Python SQL Machine Learning"
    score = tfidf_similarity_score(text, text)
    assert score > 95


def test_tfidf_similarity_unrelated_text():
    """Completely unrelated text should score low."""
    resume = "Python Pandas SQL"
    jd = "Cooking baking pastry chef kitchen"
    score = tfidf_similarity_score(resume, jd)
    assert score < 20


def test_resume_data_schema_defaults():
    """ResumeData should allow all-optional construction with sensible defaults."""
    data = ResumeData()
    assert data.skills == []
    assert data.education == []
    assert data.name is None


def test_resume_data_schema_with_values():
    """ResumeData should correctly store provided values."""
    data = ResumeData(
        name="Test User",
        skills=["Python", "SQL"],
        education=[Education(degree="B.Tech CS", institution="Test University")],
    )
    assert data.name == "Test User"
    assert len(data.skills) == 2
    assert data.education[0].degree == "B.Tech CS"


if __name__ == "__main__":
    test_tfidf_similarity_identical_text()
    test_tfidf_similarity_unrelated_text()
    test_resume_data_schema_defaults()
    test_resume_data_schema_with_values()
    print("All tests passed!")
