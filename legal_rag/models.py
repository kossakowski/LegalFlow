"""Data models for the retrieval system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """Search result for a legal provision fragment."""

    id: int
    text: str
    source_file: str
    article_hint: str | None
    score: float
    method: str = "embedding"  # embedding | keyword | both
    embedding_score: Optional[float] = None
    keyword_score: Optional[float] = None

    def __str__(self) -> str:
        """Text representation of the result."""
        article_info = f" [{self.article_hint}]" if self.article_hint else ""
        text_preview = self.text[:300] + "..." if len(self.text) > 300 else self.text
        method_info = f" | Method: {self.method}"
        return (
            f"ID: {self.id} | Score: {self.score:.4f} | "
            f"Source: {self.source_file}{article_info}{method_info}\n"
            f"Text: {text_preview}"
        )
