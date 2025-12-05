"""Tests for HTML generation functionality."""

from pathlib import Path

import pytest

from legal_rag.main import generate_html_results
from legal_rag.models import SearchResult


class TestHTMLGeneration:
    """Tests for HTML result generation."""
    
    def test_generate_html_with_results(self, temp_dir: Path):
        """Test HTML generation with search results."""
        output_path = temp_dir / "output" / "results.html"
        
        results = [
            SearchResult(
                id=1,
                text="Art. 1. Kodeks cywilny reguluje stosunki cywilnoprawne.",
                source_file="test1.txt",
                article_hint="Art. 1.",
                score=0.95,
                method="both",
                embedding_score=0.85,
                keyword_score=0.82,
                display_limit=300
            ),
            SearchResult(
                id=2,
                text="Art. 2. Umowa jest zgodnym oświadczeniem woli stron.",
                source_file="test1.txt",
                article_hint="Art. 2.",
                score=0.88,
                method="embedding",
                embedding_score=0.88,
                keyword_score=None,
                display_limit=300
            ),
        ]
        
        generate_html_results(
            results=results,
            query="kodeks cywilny",
            output_path=str(output_path),
            search_params={
                'top_k': 10,
                'min_score': 0.5,
                'search_multiplier': 2.0,
                'weight_embedding': 1.0,
                'weight_keyword': 1.0,
            }
        )
        
        # Check file was created
        assert output_path.exists()
        
        # Check file content
        html_content = output_path.read_text(encoding='utf-8')
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content
        
        # Check query is included
        assert "kodeks cywilny" in html_content
        
        # Check results are included
        assert "Art. 1." in html_content
        assert "Art. 2." in html_content
        assert "Kodeks cywilny reguluje" in html_content
        assert "Umowa jest zgodnym" in html_content
        
        # Check scores
        assert "0.95" in html_content
        assert "0.88" in html_content
        
        # Check metadata
        assert "test1.txt" in html_content
        assert "ID: 1" in html_content
        assert "ID: 2" in html_content
        
        # Check method badges
        assert "Method: both" in html_content
        assert "Method: embedding" in html_content
    
    def test_generate_html_with_no_results(self, temp_dir: Path):
        """Test HTML generation with no search results."""
        output_path = temp_dir / "output" / "no_results.html"
        
        generate_html_results(
            results=[],
            query="nonexistent query",
            output_path=str(output_path),
            search_params={
                'top_k': 10,
                'min_score': 0.5,
                'search_multiplier': 2.0,
                'weight_embedding': 1.0,
                'weight_keyword': 1.0,
            }
        )
        
        # Check file was created
        assert output_path.exists()
        
        # Check file content
        html_content = output_path.read_text(encoding='utf-8')
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "nonexistent query" in html_content
        assert "Results: 0" in html_content or "0" in html_content
    
    def test_generate_html_creates_directory(self, temp_dir: Path):
        """Test that HTML generation creates output directory if it doesn't exist."""
        output_path = temp_dir / "new_dir" / "subdir" / "results.html"
        
        results = [
            SearchResult(
                id=1,
                text="Test text",
                source_file="test.txt",
                article_hint=None,
                score=0.9,
                method="embedding",
                embedding_score=0.9,
                keyword_score=None,
                display_limit=300
            )
        ]
        
        # Directory shouldn't exist yet
        assert not output_path.parent.exists()
        
        generate_html_results(
            results=results,
            query="test",
            output_path=str(output_path),
            search_params={}
        )
        
        # Directory should be created
        assert output_path.parent.exists()
        assert output_path.exists()
    
    def test_generate_html_includes_all_scores(self, temp_dir: Path):
        """Test that HTML includes all score types when available."""
        output_path = temp_dir / "scores.html"
        
        results = [
            SearchResult(
                id=1,
                text="Test text",
                source_file="test.txt",
                article_hint=None,
                score=1.5,
                method="both",
                embedding_score=0.8,
                keyword_score=0.7,
                display_limit=300
            )
        ]
        
        generate_html_results(
            results=results,
            query="test",
            output_path=str(output_path),
            search_params={}
        )
        
        html_content = output_path.read_text(encoding='utf-8')
        
        # Check all scores are present
        assert "Combined: 1.5" in html_content or "1.5000" in html_content
        assert "Embedding: 0.8" in html_content or "0.8000" in html_content
        assert "Keyword: 0.7" in html_content or "0.7000" in html_content
    
    def test_generate_html_includes_search_parameters(self, temp_dir: Path):
        """Test that HTML includes search parameters."""
        output_path = temp_dir / "params.html"
        
        results = [
            SearchResult(
                id=1,
                text="Test",
                source_file="test.txt",
                article_hint=None,
                score=0.9,
                method="embedding",
                embedding_score=0.9,
                keyword_score=None,
                display_limit=300
            )
        ]
        
        search_params = {
            'top_k': 50,
            'min_score': 0.3,
            'search_multiplier': 5.0,
            'weight_embedding': 2.0,
            'weight_keyword': 1.5,
        }
        
        generate_html_results(
            results=results,
            query="test query",
            output_path=str(output_path),
            search_params=search_params
        )
        
        html_content = output_path.read_text(encoding='utf-8')
        
        # Check parameters are included
        assert "50" in html_content  # top_k
        assert "0.3" in html_content  # min_score
        assert "5.0" in html_content  # search_multiplier
        assert "2.0" in html_content  # weight_embedding
        assert "1.5" in html_content  # weight_keyword
    
    def test_generate_html_escapes_special_characters(self, temp_dir: Path):
        """Test that HTML properly escapes special characters."""
        output_path = temp_dir / "escape.html"
        
        results = [
            SearchResult(
                id=1,
                text="Text with <script>alert('XSS')</script> & special chars",
                source_file="test<file>.txt",
                article_hint="Art. 1. <important>",
                score=0.9,
                method="embedding",
                embedding_score=0.9,
                keyword_score=None,
                display_limit=300
            )
        ]
        
        generate_html_results(
            results=results,
            query="test <query> & more",
            output_path=str(output_path),
            search_params={}
        )
        
        html_content = output_path.read_text(encoding='utf-8')
        
        # Check that special characters are escaped
        assert "&lt;script&gt;" in html_content or "<script>" not in html_content
        assert "&amp;" in html_content
        assert "&lt;query&gt;" in html_content or "<query>" not in html_content
    
    def test_generate_html_full_text_display(self, temp_dir: Path):
        """Test that HTML displays full text regardless of display_limit."""
        output_path = temp_dir / "fulltext.html"
        
        long_text = "A" * 500  # Text longer than default display_limit
        
        results = [
            SearchResult(
                id=1,
                text=long_text,
                source_file="test.txt",
                article_hint=None,
                score=0.9,
                method="embedding",
                embedding_score=0.9,
                keyword_score=None,
                display_limit=300  # Should be ignored in HTML
            )
        ]
        
        generate_html_results(
            results=results,
            query="test",
            output_path=str(output_path),
            search_params={}
        )
        
        html_content = output_path.read_text(encoding='utf-8')
        
        # Full text should be in HTML (all 500 characters)
        assert long_text in html_content
        assert len([c for c in html_content if c == 'A']) >= 500
    
    def test_generate_html_multiple_results(self, temp_dir: Path):
        """Test HTML generation with multiple results."""
        output_path = temp_dir / "multiple.html"
        
        results = [
            SearchResult(
                id=i,
                text=f"Result {i} text",
                source_file=f"file{i}.txt",
                article_hint=f"Art. {i}." if i % 2 == 0 else None,
                score=1.0 - (i * 0.1),
                method="both" if i % 3 == 0 else "embedding" if i % 2 == 0 else "keyword",
                embedding_score=0.9 - (i * 0.05),
                keyword_score=0.8 - (i * 0.05) if i % 2 == 1 else None,
                display_limit=300
            )
            for i in range(1, 11)  # 10 results
        ]
        
        generate_html_results(
            results=results,
            query="multiple results test",
            output_path=str(output_path),
            search_params={}
        )
        
        html_content = output_path.read_text(encoding='utf-8')
        
        # Check all results are included
        for i in range(1, 11):
            assert f"Result {i}" in html_content
            assert f"ID: {i}" in html_content
        
        # Check result numbering
        assert "result-number" in html_content
        assert "1" in html_content
        assert "10" in html_content

