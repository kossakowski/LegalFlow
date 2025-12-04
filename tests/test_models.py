"""Testy dla modeli danych."""

import pytest

from legal_rag.models import SearchResult


class TestSearchResult:
    """Testy dla klasy SearchResult."""
    
    def test_search_result_creation(self):
        """Test tworzenia obiektu SearchResult."""
        result = SearchResult(
            id=1,
            text="Przykładowy tekst przepisu",
            source_file="test.txt",
            article_hint="Art. 1.",
            score=0.85
        )
        
        assert result.id == 1
        assert result.text == "Przykładowy tekst przepisu"
        assert result.source_file == "test.txt"
        assert result.article_hint == "Art. 1."
        assert result.score == 0.85
    
    def test_search_result_without_article_hint(self):
        """Test tworzenia SearchResult bez article_hint."""
        result = SearchResult(
            id=2,
            text="Tekst bez artykułu",
            source_file="test2.txt",
            article_hint=None,
            score=0.75
        )
        
        assert result.article_hint is None
    
    def test_search_result_string_representation(self):
        """Test reprezentacji tekstowej SearchResult."""
        result = SearchResult(
            id=1,
            text="Krótki tekst",
            source_file="test.txt",
            article_hint="Art. 1.",
            score=0.85
        )
        
        str_repr = str(result)
        assert "ID: 1" in str_repr
        assert "Score: 0.8500" in str_repr
        assert "test.txt" in str_repr
        assert "[Art. 1.]" in str_repr
        assert "Krótki tekst" in str_repr
    
    def test_search_result_string_representation_long_text(self):
        """Test reprezentacji tekstowej z długim tekstem."""
        long_text = "A" * 400  # Tekst dłuższy niż 300 znaków
        result = SearchResult(
            id=1,
            text=long_text,
            source_file="test.txt",
            article_hint=None,
            score=0.85
        )
        
        str_repr = str(result)
        assert len(str_repr.split("Text: ")[1]) <= 303  # 300 + "..."
        assert "..." in str_repr
    
    def test_search_result_string_representation_no_article(self):
        """Test reprezentacji tekstowej bez article_hint."""
        result = SearchResult(
            id=1,
            text="Tekst",
            source_file="test.txt",
            article_hint=None,
            score=0.85
        )
        
        str_repr = str(result)
        assert "[Art." not in str_repr
        assert "test.txt" in str_repr

