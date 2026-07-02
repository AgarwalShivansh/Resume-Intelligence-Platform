"""
Pydantic schemas define the exact structure we want the LLM to return.

Why this matters: LLMs left unconstrained return inconsistent formats
(sometimes a list, sometimes a paragraph, sometimes missing fields).
By giving the LLM a strict schema (and validating its output against it),
we get reliable, predictable data every time — this is the difference
between a "toy LLM demo" and a production-grade extraction pipeline.

CHANGE LOG (v1.1, Phase 3):
- Added Certification, Achievement, Publication models.
- Added ResumeData fields: summary, certifications, achievements,
  publications. All default to empty/None so existing extraction code
  and existing tests that construct ResumeData() with no arguments still
  work unchanged (backward compatible).
- Experience model unchanged — internships are represented as Experience
  entries (the LLM is instructed to include internships there; we don't
  introduce a separate Internship model since an internship IS a kind of
  experience entry, just with a different title/duration).
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Education(BaseModel):
    degree: str = Field(description="e.g., 'B.Tech Computer Science'")
    institution: str = Field(description="Name of college/university")
    year: Optional[str] = Field(default=None, description="Graduation year or expected year")


class Experience(BaseModel):
    title: str = Field(description="Job title or internship role")
    company: str = Field(description="Company or organization name")
    duration: Optional[str] = Field(default=None, description="e.g., 'Jun 2024 - Aug 2024'")
    description: Optional[str] = Field(default=None, description="Brief summary of responsibilities")


class Project(BaseModel):
    title: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)


class Certification(BaseModel):
    name: str = Field(description="e.g., 'AWS Certified Cloud Practitioner'")
    issuer: Optional[str] = Field(default=None, description="e.g., 'Amazon Web Services'")
    year: Optional[str] = Field(default=None, description="Year obtained, if mentioned")


class Achievement(BaseModel):
    title: str = Field(description="e.g., 'Winner, National Hackathon 2025' or 'Dean's List'")
    description: Optional[str] = Field(default=None, description="Brief context, if mentioned")


class Publication(BaseModel):
    title: str = Field(description="Title of the paper/article")
    venue: Optional[str] = Field(default=None, description="e.g., conference, journal, or platform name")
    year: Optional[str] = Field(default=None, description="Year published, if mentioned")


class ResumeData(BaseModel):
    """The full structured representation of a parsed resume."""

    name: Optional[str] = Field(default=None, description="Candidate's full name")
    email: Optional[str] = None
    phone: Optional[str] = None

    summary: Optional[str] = Field(
        default=None,
        description="The candidate's professional summary/objective statement, if present in the resume"
    )

    skills: List[str] = Field(
        default_factory=list,
        description="Technical and soft skills explicitly mentioned in the resume"
    )

    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    achievements: List[Achievement] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)

    total_experience_years: Optional[float] = Field(
        default=None,
        description="Estimated total years of professional experience (internships count partially)"
    )


class MatchResult(BaseModel):
    """Final output of comparing a resume against a job description."""

    overall_score: float = Field(description="0-100 match score")
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    explanation: str = Field(description="LLM-generated rationale for the score")
    improvement_suggestions: List[str] = Field(default_factory=list)
