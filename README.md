# 📄 Resume Intelligence Platform

An AI-powered tool that parses resumes, extracts structured information using an LLM, and scores how well a resume matches a job description — with full explainability (not just a number, but *why*).

Built as a portfolio project demonstrating: NLP, LLM integration, structured extraction, semantic similarity (embeddings), and hallucination-safety checks.

---

### Live-Project link:- https://resume-intelligence-platform-01234.streamlit.app/

---

## 🎯 What it does

1. **Upload a resume** (PDF or DOCX)
2. **Paste a job description**
3. Get back:
   - Structured resume data (skills, education, experience, projects) extracted via LLM
   - A blended match score (TF-IDF baseline + semantic embedding match)
   - Matched and missing skills
   - A plain-language explanation of the score
   - Concrete improvement suggestions
   - A hallucination check confirming extracted skills actually appear in the source text

---

## 🏗️ Architecture

```
Resume (PDF/DOCX) ──► Text Extraction ──► LLM Structured Extraction ──► ResumeData (validated)
                                                                              │
Job Description ──► LLM Requirement Extraction ──► JD Skills                │
                                                          │                  │
                                                          ▼                  ▼
                                              Semantic Matcher (embeddings) + TF-IDF Baseline
                                                          │
                                                          ▼
                                          Weighted Score + LLM Explanation + Suggestions
                                                          │
                                                          ▼
                                                   Streamlit UI
```

**Why two matching methods?** TF-IDF is a fast, fully-interpretable keyword-overlap baseline. The semantic matcher (sentence embeddings) catches synonyms TF-IDF misses (e.g., "ML" ≈ "Machine Learning", "Postgres" ≈ "PostgreSQL"). The final score blends both (70% semantic, 30% TF-IDF), and both are shown separately in the UI so the comparison itself is part of the demo.

**Why a hallucination check?** LLMs can occasionally invent a skill that sounds plausible but isn't actually in the resume. We verify every extracted skill against the raw source text as a lightweight safety net — and show both verified and unverified skills transparently in the UI.

---

## 🚀 Setup & Local Run

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd resume-intelligence-platform
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key

- Sign up at [console.groq.com](https://console.groq.com)
- Create an API key

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=your_actual_key_here
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## ☁️ Deploying to the Cloud (Streamlit Community Cloud — free)

1. Push this project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo, and set the main file path to `app/streamlit_app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_actual_key_here"
   ```
5. Deploy. You'll get a public URL like `https://your-app-name.streamlit.app` — put this link directly on your resume and GitHub README.

---

## 📁 Project Structure

```
resume-intelligence-platform/
├── data/
│   ├── sample_resumes/        # sample resume for testing
│   └── sample_jds/            # sample job descriptions
├── src/
│   ├── config.py              # ALL settings, API keys, thresholds — single source of truth
│   ├── exceptions.py          # custom exception types for specific, friendly error handling
│   ├── parsing/
│   │   └── extractor.py       # PDF/DOCX → raw text
│   ├── extraction/
│   │   ├── schema.py          # Pydantic models (ResumeData, MatchResult)
│   │   ├── llm_extractor.py   # LLM-based resume parsing + hallucination check
│   │   └── jd_extractor.py    # LLM-based JD requirement parsing
│   ├── matching/
│   │   ├── tfidf_baseline.py  # classic TF-IDF cosine similarity
│   │   └── semantic_matcher.py # sentence-embedding skill matching
│   ├── scoring/
│   │   └── score_engine.py    # combines everything into final score + explanation
│   └── services/
│       ├── analysis_service.py # orchestrates the full pipeline (the "business logic" layer)
│       └── result_types.py     # AnalysisResult — the single typed object the UI consumes
├── app/
│   └── streamlit_app.py       # PURE UI — renders widgets, calls the service layer, renders results
├── tests/
│   ├── test_core_modules.py        # unit tests for matching/schema
│   ├── test_analysis_service.py    # tests for the orchestration pipeline (no UI needed)
│   └── test_phase3_extraction.py   # tests for extended schema + hallucination checks
├── requirements.txt
├── .env.example
└── README.md
```

**Why a `services/` layer?** Before this layer existed, `streamlit_app.py` directly called five backend functions and manually wired their outputs together — UI code and business logic were tangled in one file. Now `analysis_service.run_full_analysis()` is the single entry point: it runs the entire extract → parse → match → score pipeline and returns one typed `AnalysisResult`. This means the pipeline can be tested with plain Python (see `test_analysis_service.py`) without ever starting a Streamlit server, and the UI file only needs to know how to render one result object, not five different function signatures.

---

## 🧠 Key Design Decisions (for interview discussion)

- **Why Pydantic schemas for LLM output?** Forces the LLM's JSON response into a strict, validated shape — catching malformed output immediately instead of letting bad data silently flow downstream.
- **Why `temperature=0` for extraction?** Extraction should be deterministic and consistent, not creative — we want the same resume to produce the same extracted data every time.
- **Why blend TF-IDF and semantic scores instead of using only one?** Demonstrates understanding of the tradeoff between interpretable classical NLP and more powerful (but less transparent) embedding-based methods.
- **Why a hallucination check at all?** Production LLM systems must guard against fabricated outputs — this shows awareness of a real failure mode, not just a happy-path demo.
- **Why a centralized `config.py`?** Every tunable value (model name, score weights, thresholds, timeouts) lives in one place instead of being duplicated across files — changing the LLM model or rebalancing score weights is now a one-line edit, not a multi-file search.
- **Why custom exceptions instead of generic `Exception`?** Each failure mode (corrupted file, empty resume, missing API key, LLM timeout) needs a different user-facing message. Specific exception types let the UI layer catch and explain each one precisely, instead of showing "something went wrong" for every failure.
- **Why separate the service layer from the UI?** It makes the core pipeline logic testable and reusable independent of Streamlit — the same `run_full_analysis()` function could power a CLI tool, a FastAPI endpoint, or a batch script with zero changes.
- **Why extend extraction to certifications/achievements/publications instead of just skills/education/experience?** A resume's strength often lives in these "extra" sections — recruiters specifically scan for certifications and notable achievements. Extracting them structurally (rather than leaving them buried in unstructured text) makes them usable for scoring and display.
- **Why check certifications/achievements/publications for hallucination, but not education/experience?** Education and experience are multi-field structured records (degree+institution+year) where the LLM often paraphrases or reformats real content slightly — a strict verbatim check there would produce too many false "unverified" flags on genuinely real data. Skills, certification names, achievement titles, and publication titles are typically copied near-verbatim from the resume, so a missing match is a much stronger hallucination signal for those specific fields.

---

## 🔭 Possible Extensions (not yet built — good "future work" talking points)

- RAG over a skills taxonomy (O*NET/ESCO) to normalize skill synonyms more rigorously
- Batch mode: rank multiple resumes against one JD (recruiter-side view)
- Fairness audit: check if scoring correlates with name-based demographic proxies
- Fine-tuned, smaller model for extraction to reduce latency/cost at scale

---

## 🛠️ Tech Stack

Python · Streamlit · Groq API (Llama 3.3 70B) · Pydantic · PyMuPDF · python-docx · scikit-learn · sentence-transformers

---

## 📝 License

MIT — free to use and adapt for your own portfolio.
