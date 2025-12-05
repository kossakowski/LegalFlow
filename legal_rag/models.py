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
    display_limit: int = 300  # Maximum characters to display in text preview

    def __str__(self) -> str:
        """Text representation of the result."""
        article_info = f" [{self.article_hint}]" if self.article_hint else ""
        if self.display_limit == 0 or len(self.text) <= self.display_limit:
            text_preview = self.text
        else:
            text_preview = self.text[:self.display_limit] + "..."
        method_info = f" | Method: {self.method}"
        return (
            f"ID: {self.id} | Score: {self.score:.4f} | "
            f"Source: {self.source_file}{article_info}{method_info}\n"
            f"Text: {text_preview}"
        )
