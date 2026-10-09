"""Fixed-first, read-only retrieval from the rules-reference database."""

from .hybrid import retrieve_sql
from .result import RetrievalResult

__all__ = ["retrieve_sql", "RetrievalResult"]
