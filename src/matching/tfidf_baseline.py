"""
TF-IDF baseline matcher.

This is the "classic ML" comparison point — useful both as a sanity check
and as a talking point in interviews ("I compared a TF-IDF baseline against
a semantic/LLM approach and here's what improved").
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity_score(resume_text: str, job_description: str) -> float:
    """
    Compute cosine similarity between resume and job description using TF-IDF.

    Returns a score between 0 and 100.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


if __name__ == "__main__":
    resume = "Experienced in Python, Pandas, SQL, and machine learning model deployment."
    jd = "Looking for a candidate skilled in Python, SQL, and ML model deployment on cloud."

    score = tfidf_similarity_score(resume, jd)
    print(f"TF-IDF similarity score: {score}/100")
