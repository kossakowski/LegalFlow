"""Testy integracyjne dla całego systemu."""

import json
from pathlib import Path

import pytest

from legal_rag.indexing import build_index
from legal_rag.retrieval import LegalRetriever


class TestIntegration:
    """Testy integracyjne całego przepływu."""
    
    def test_full_workflow(self, temp_dir: Path):
        """Test pełnego przepływu: indeksowanie -> wyszukiwanie."""
        # Przygotuj dane testowe
        txt_dir = temp_dir / "txt"
        txt_dir.mkdir()
        
        (txt_dir / "test.txt").write_text(
            "Art. 1.\n"
            "Kodeks cywilny reguluje stosunki cywilnoprawne.\n\n"
            "Art. 2.\n"
            "Umowa jest zgodnym oświadczeniem woli stron.\n\n"
            "Art. 3.\n"
            "Własność może być nabyta na różne sposoby.\n",
            encoding="utf-8"
        )
        
        # Zbuduj indeks
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(txt_dir),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        # Sprawdź czy indeks został utworzony
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "metadata.json").exists()
        
        # Wczytaj retriever
        retriever = LegalRetriever(
            index_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Wyszukaj
        results = retriever.search("umowa", top_k=5)
        
        assert len(results) > 0
        assert any("umowa" in r.text.lower() or "Umowa" in r.text for r in results)
    
    def test_search_finds_relevant_results(self, sample_index: Path):
        """Test czy wyszukiwanie znajduje istotne wyniki."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Wyszukaj "kodeks"
        results = retriever.search("kodeks", top_k=10)
        
        # Sprawdź czy przynajmniej jeden wynik zawiera słowo "kodeks"
        assert len(results) > 0
        assert any(
            "kodeks" in r.text.lower() or "Kodeks" in r.text
            for r in results
        )
    
    def test_multiple_queries(self, sample_index: Path):
        """Test wielu zapytań pod rząd."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        queries = ["kodeks", "umowa", "prawo", "własność"]
        
        for query in queries:
            results = retriever.search(query, top_k=5)
            assert isinstance(results, list)
            assert len(results) <= 5
    
    def test_index_metadata_consistency(self, sample_index: Path):
        """Test spójności metadanych z indeksem."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Liczba metadanych powinna odpowiadać liczbie wektorów w indeksie
        assert len(retriever.metadata) == retriever.index.ntotal
        
        # Wszystkie ID w metadanych powinny być unikalne
        ids = [m["id"] for m in retriever.metadata]
        assert len(ids) == len(set(ids))
    
    def test_search_with_different_parameters(self, sample_index: Path):
        """Test wyszukiwania z różnymi parametrami."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        query = "umowa"
        
        # Różne wartości top_k
        results_10 = retriever.search(query, top_k=10)
        results_20 = retriever.search(query, top_k=20)
        
        assert len(results_10) <= 10
        assert len(results_20) <= 20
        assert len(results_20) >= len(results_10)
        
        # Różne wartości min_score
        results_low = retriever.search(query, top_k=10, min_score=0.0)
        results_high = retriever.search(query, top_k=10, min_score=0.5)
        
        assert len(results_low) >= len(results_high)
        assert all(r.score >= 0.5 for r in results_high)
    
    def test_rebuild_index_with_new_data(self, temp_dir: Path):
        """Test przebudowy indeksu z nowymi danymi."""
        txt_dir = temp_dir / "txt"
        txt_dir.mkdir()
        
        # Pierwsza wersja danych
        (txt_dir / "file1.txt").write_text("Art. 1.\nPierwszy tekst.", encoding="utf-8")
        
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(txt_dir),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        retriever1 = LegalRetriever(index_dir=str(index_dir))
        count1 = len(retriever1.metadata)
        
        # Dodaj nowy plik
        (txt_dir / "file2.txt").write_text("Art. 2.\nDrugi tekst.", encoding="utf-8")
        
        # Przebuduj indeks
        build_index(
            input_dir=str(txt_dir),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        retriever2 = LegalRetriever(index_dir=str(index_dir))
        count2 = len(retriever2.metadata)
        
        # Powinno być więcej wyników
        assert count2 > count1
    
    def test_full_workflow_with_html_generation(self, temp_dir: Path):
        """Test full workflow with HTML output generation."""
        # Prepare test data
        txt_dir = temp_dir / "txt"
        txt_dir.mkdir()
        
        (txt_dir / "test.txt").write_text(
            "Art. 1.\n"
            "Kodeks cywilny reguluje stosunki cywilnoprawne.\n\n"
            "Art. 2.\n"
            "Umowa jest zgodnym oświadczeniem woli stron.\n\n"
            "Art. 3.\n"
            "Własność może być nabyta na różne sposoby.\n",
            encoding="utf-8"
        )
        
        # Build index
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(txt_dir),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        # Search and generate HTML
        from legal_rag.main import cmd_query
        from argparse import Namespace
        
        output_html = temp_dir / "output" / "results.html"
        
        args = Namespace(
            index_dir=str(index_dir),
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
        
        # Check HTML file was created
        assert output_html.exists()
        
        # Check HTML content
        html_content = output_html.read_text(encoding='utf-8')
        assert "<!DOCTYPE html>" in html_content
        assert "umowa" in html_content.lower()
        assert "Results:" in html_content or "results" in html_content.lower()
    
    def test_html_generation_with_no_results(self, sample_index: Path):
        """Test HTML generation when no results are found."""
        from legal_rag.main import cmd_query
        from argparse import Namespace
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_html = Path(tmpdir) / "no_results.html"
            
            args = Namespace(
                index_dir=str(sample_index),
                query="xyzabc123nonexistentquery",
                top_k=10,
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
            
            html_content = output_html.read_text(encoding='utf-8')
            assert "<!DOCTYPE html>" in html_content
            assert "xyzabc123nonexistentquery" in html_content.lower()
    
    def test_full_workflow_all_features(self, temp_dir: Path):
        """Test complete workflow with all features: indexing, searching, HTML generation."""
        # Prepare test data
        txt_dir = temp_dir / "txt"
        txt_dir.mkdir()
        
        (txt_dir / "kodeks.txt").write_text(
            "Art. 1.\n"
            "Kodeks niniejszy reguluje stosunki cywilnoprawne między osobami fizycznymi i prawnymi.\n\n"
            "Art. 2.\n"
            "Każdy ma prawo do własności i innych praw majątkowych.\n\n"
            "Art. 3.\n"
            "Własność może być nabyta na różne sposoby określone w ustawie.\n",
            encoding="utf-8"
        )
        
        (txt_dir / "umowa.txt").write_text(
            "Umowa jest zgodnym oświadczeniem woli dwóch lub więcej stron.\n\n"
            "Umowa powinna być zawarta w formie określonej przez prawo.\n\n"
            "Strony umowy mają obowiązek wykonać zobowiązania wynikające z umowy.\n",
            encoding="utf-8"
        )
        
        # Build index
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(txt_dir),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        # Verify index was created
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "metadata.json").exists()
        
        # Load retriever and search
        retriever = LegalRetriever(
            index_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Test search with different parameters
        results = retriever.search(
            query="umowa",
            top_k=10,
            min_score=0.0,
            search_multiplier=2.0,
            weight_embedding=1.0,
            weight_keyword=1.0,
            display_limit=300
        )
        
        assert len(results) > 0
        
        # Generate HTML
        from legal_rag.main import generate_html_results
        
        output_html = temp_dir / "output" / "complete_results.html"
        
        generate_html_results(
            results=results,
            query="umowa",
            output_path=str(output_html),
            search_params={
                'top_k': 10,
                'min_score': 0.0,
                'search_multiplier': 2.0,
                'weight_embedding': 1.0,
                'weight_keyword': 1.0,
            }
        )
        
        # Verify HTML was created and contains results
        assert output_html.exists()
        html_content = output_html.read_text(encoding='utf-8')
        assert "<!DOCTYPE html>" in html_content
        assert "umowa" in html_content.lower()
        
        # Check that result text is in HTML
        for result in results[:3]:  # Check first 3 results
            assert result.text in html_content or result.text[:50] in html_content



