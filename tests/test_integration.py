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

