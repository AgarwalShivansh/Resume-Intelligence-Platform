"""
Custom exception types for the Resume Intelligence Platform.

WHY THIS FILE EXISTS:
Without custom exceptions, every failure (corrupted PDF, missing API key,
LLM timeout, empty resume) either raises a generic Exception or a low-level
library error (e.g., from PyMuPDF or the Groq SDK). That makes it impossible
for the Streamlit UI to show a specific, helpful message — it can only catch
"something went wrong" and show a generic error.

By defining specific exception classes, the UI layer can catch each one and
show a tailored, user-friendly message (e.g., "Your PDF appears to be a
scanned image with no selectable text" instead of a raw traceback).
"""


class ResumeIntelligenceError(Exception):
    """Base class for all custom errors in this project. Catch this to
    catch any error raised by our own code (as opposed to third-party
    library internals leaking through)."""


# ── File / Parsing Errors ────────────────────────────────────────────────
class FileTooLargeError(ResumeIntelligenceError):
    """Raised when an uploaded file exceeds the configured size limit."""


class UnsupportedFileTypeError(ResumeIntelligenceError):
    """Raised when a file extension isn't in ALLOWED_FILE_EXTENSIONS."""


class CorruptedFileError(ResumeIntelligenceError):
    """Raised when a PDF/DOCX file can't be opened or parsed at all
    (e.g., truncated download, wrong file renamed to .pdf)."""


class EmptyResumeError(ResumeIntelligenceError):
    """Raised when a file opens fine but contains no extractable text
    (most commonly: a scanned image PDF with no real text layer)."""


class EmptyJobDescriptionError(ResumeIntelligenceError):
    """Raised when the job description field is blank or whitespace-only."""


# ── LLM / AI Errors ──────────────────────────────────────────────────────
class MissingAPIKeyError(ResumeIntelligenceError):
    """Raised when GROQ_API_KEY is not configured."""


class LLMTimeoutError(ResumeIntelligenceError):
    """Raised when the LLM API call exceeds the configured timeout."""


class LLMResponseError(ResumeIntelligenceError):
    """Raised when the LLM returns malformed JSON or output that doesn't
    match the expected schema, even after retries."""


class EmbeddingError(ResumeIntelligenceError):
    """Raised when the sentence-embedding model fails to load or encode text."""
