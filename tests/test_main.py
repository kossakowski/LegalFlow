"""Tests for CLI main functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from legal_rag.main import cmd_build_index, cmd_query, generate_html_results
from legal_rag.models import SearchResult


class TestCmdBuildIndex:
    """Tests for cmd_build_index function."""
    
    def test_cmd_build_index_success(self, temp_dir: Path):
        """Test successful index building via CLI."""
        txt_dir = temp_dir / "txt"
        txt_dir.mkdir()
        
        (txt_dir / "test.txt").write_text(
            "Art. 1.\nTest content.\n\nArt. 2.\nMore content.",
            encoding="utf-8"
        )
        
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        from argparse import Namespace
        
        args = Namespace(
            input_dir=str(txt_dir),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        cmd_build_index(args)
        
        # Verify index was created
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "metadata.json").exists()
    
    def test_cmd_build_index_error_handling(self, temp_dir: Path):
        """Test error handling in cmd_build_index."""
        from argparse import Namespace
        
        args = Namespace(
            input_dir="/nonexistent/directory",
            output_dir=str(temp_dir / "index"),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        with pytest.raises(SystemExit):
            cmd_build_index(args)


class TestCmdQuery:
    """Tests for cmd_query function."""
    
    def test_cmd_query_success(self, sample_index: Path):
        """Test successful query via CLI."""
        from argparse import Namespace
        
        args = Namespace(
            index_dir=str(sample_index),
            query="kodeks",
            top_k=5,
            min_score=0.0,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300,
            output_html=None
        )
        
        # Should not raise an exception
        cmd_query(args)
    
    def test_cmd_query_with_html_output(self, sample_index: Path, temp_dir: Path):
        """Test query with HTML output."""
        from argparse import Namespace
        
        output_html = temp_dir / "output" / "test.html"
        
        args = Namespace(
            index_dir=str(sample_index),
            query="umowa",
            top_k=5,
            min_score=0.0,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300,
            output_html=str(output_html)
        )
        
        cmd_query(args)
        
        # HTML should be created
        assert output_html.exists()
    
    def test_cmd_query_no_results(self, sample_index: Path):
        """Test query with no results."""
        from argparse import Namespace
        
        args = Namespace(
            index_dir=str(sample_index),
            query="xyzabc123nonexistent",
            top_k=5,
            min_score=0.0,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300,
            output_html=None
        )
        
        # Should handle no results gracefully
        cmd_query(args)
    
    def test_cmd_query_top_k_zero(self, sample_index: Path):
        """Test query with top_k=0 (all results)."""
        from argparse import Namespace
        
        args = Namespace(
            index_dir=str(sample_index),
            query="kodeks",
            top_k=0,
            min_score=0.0,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300,
            output_html=None
        )
        
        cmd_query(args)
    
    def test_cmd_query_error_handling(self, temp_dir: Path):
        """Test error handling in cmd_query."""
        from argparse import Namespace
        
        args = Namespace(
            index_dir="/nonexistent/index",
            query="test",
            top_k=5,
            min_score=0.0,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300,
            output_html=None
        )
        
        with pytest.raises(SystemExit):
            cmd_query(args)
    
    def test_cmd_query_with_all_parameters(self, sample_index: Path, temp_dir: Path):
        """Test query with all parameters set."""
        from argparse import Namespace
        
        output_html = temp_dir / "output" / "full_test.html"
        
        args = Namespace(
            index_dir=str(sample_index),
            query="test query",
            top_k=10,
            min_score=0.3,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=5.0,
            weight_embedding=1.5,
            weight_keyword=2.0,
            display_limit=500,
            output_html=str(output_html)
        )
        
        cmd_query(args)
        
        # If results found, HTML should be created
        if output_html.exists():
            html = output_html.read_text(encoding='utf-8')
            assert "<!DOCTYPE html>" in html
    
    def test_cmd_query_no_results_with_html(self, sample_index: Path, temp_dir: Path):
        """Test query with no results but HTML output requested."""
        from argparse import Namespace
        
        output_html = temp_dir / "no_results.html"
        
        args = Namespace(
            index_dir=str(sample_index),
            query="xyzabc123nonexistentquery",
            top_k=5,
            min_score=0.0,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300,
            output_html=str(output_html)
        )
        
        cmd_query(args)
        
        # HTML should still be created even with no results
        assert output_html.exists()


class TestGenerateHTMLResults:
    """Tests for generate_html_results function (additional to test_html_generation.py)."""
    
    def test_generate_html_with_various_methods(self, temp_dir: Path):
        """Test HTML generation with different result methods."""
        output_path = temp_dir / "methods.html"
        
        results = [
            SearchResult(
                id=1,
                text="Embedding only",
                source_file="test.txt",
                article_hint=None,
                score=0.8,
                method="embedding",
                embedding_score=0.8,
                keyword_score=None,
                display_limit=300
            ),
            SearchResult(
                id=2,
                text="Keyword only",
                source_file="test.txt",
                article_hint=None,
                score=0.9,
                method="keyword",
                embedding_score=None,
                keyword_score=0.9,
                display_limit=300
            ),
            SearchResult(
                id=3,
                text="Both methods",
                source_file="test.txt",
                article_hint="Art. 3.",
                score=1.5,
                method="both",
                embedding_score=0.8,
                keyword_score=0.7,
                display_limit=300
            ),
        ]
        
        generate_html_results(
            results=results,
            query="test query",
            output_path=str(output_path),
            search_params={'top_k': 10}
        )
        
        assert output_path.exists()
        html = output_path.read_text(encoding='utf-8')
        assert "Method: embedding" in html
        assert "Method: keyword" in html
        assert "Method: both" in html
        assert "Art. 3." in html
