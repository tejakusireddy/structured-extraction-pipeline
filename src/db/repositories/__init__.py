"""Database repository layer — one repo per aggregate root."""

from src.db.repositories.citation_repo import CitationRepo
from src.db.repositories.extraction_repo import ExtractionRepo
from src.db.repositories.job_repo import JobRepo
from src.db.repositories.opinion_repo import OpinionRepo

__all__ = [
    "CitationRepo",
    "ExtractionRepo",
    "JobRepo",
    "OpinionRepo",
]
