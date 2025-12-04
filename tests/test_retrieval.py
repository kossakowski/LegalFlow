"""Testy dla klasy LegalRetriever."""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from legal_rag.retrieval import LegalRetriever


class TestLegalRetriever:
    """Testy dla klasy LegalRetriever."""
    
    def test_retriever_initialization(self, sample_index: Path):
        """Test inicjalizacji retrievera."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        assert retriever.index is not None
        assert len(retriever.metadata) > 0
        assert retriever.embedding_model is not None
    
    def test_retriever_initialization_nonexistent_index(self, temp_dir: Path):
        """Test inicjalizacji z nieistniejącym indeksem."""
        index_dir = temp_dir / "nonexistent"
        index_dir.mkdir()
        
        with pytest.raises(FileNotFoundError, match="Indeks nie istnieje"):
            LegalRetriever(index_dir=str(index_dir))
    
    def test_retriever_initialization_missing_metadata(self, temp_dir: Path):
        """Test inicjalizacji z brakującymi metadanymi."""
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        # Utwórz pusty indeks FAISS
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        
        with pytest.raises(FileNotFoundError, match="Plik metadanych nie istnieje"):
            LegalRetriever(index_dir=str(index_dir))
    
    def test_retriever_search_basic(self, sample_index: Path):
        """Test podstawowego wyszukiwania."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("kodeks", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "score") for r in results)
    
    def test_retriever_search_top_k_limit(self, sample_index: Path):
        """Test ograniczenia liczby wyników przez top_k."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("umowa", top_k=2)
        
        assert len(results) <= 2
    
    def test_retriever_search_min_score_filtering(self, sample_index: Path):
        """Test filtrowania po min_score."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Wyszukaj z wysokim min_score
        results_high = retriever.search("kodeks", top_k=10, min_score=0.9)
        
        # Wszystkie wyniki powinny mieć score >= 0.9
        assert all(r.score >= 0.9 for r in results_high)
    
    def test_retriever_search_results_sorted(self, sample_index: Path):
        """Test czy wyniki są posortowane malejąco po score."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("prawo", top_k=10)
        
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_retriever_search_all_results(self, sample_index: Path):
        """Test wyszukiwania wszystkich wyników (top_k > 100000)."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("umowa", top_k=200000)
        
        # Powinno zwrócić wszystkie dostępne wyniki
        assert len(results) > 0
        assert len(results) <= retriever.index.ntotal
    
    def test_retriever_search_multiplier(self, sample_index: Path):
        """Test parametru search_multiplier."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Z małym mnożnikiem
        results_small = retriever.search("kodeks", top_k=5, search_multiplier=1.0)
        
        # Z dużym mnożnikiem
        results_large = retriever.search("kodeks", top_k=5, search_multiplier=5.0)
        
        # Oba powinny zwrócić wyniki (może być różna liczba po filtrowaniu)
        assert len(results_small) <= 5
        assert len(results_large) <= 5
    
    def test_retriever_search_empty_query(self, sample_index: Path):
        """Test wyszukiwania z pustym zapytaniem."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("", top_k=5)
        
        # Powinno zwrócić listę (może być pusta)
        assert isinstance(results, list)
    
    def test_retriever_search_polish_query(self, sample_index: Path):
        """Test wyszukiwania z polskim zapytaniem."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("Jakie są prawa rzeczowe?", top_k=10)
        
        assert isinstance(results, list)
        assert all(isinstance(r.score, float) for r in results)
    
    def test_retriever_search_result_structure(self, sample_index: Path):
        """Test struktury wyników wyszukiwania."""
        retriever = LegalRetriever(
            index_dir=str(sample_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        results = retriever.search("kodeks", top_k=1)
        
        if results:
            result = results[0]
            assert hasattr(result, "id")
            assert hasattr(result, "text")
            assert hasattr(result, "source_file")
            assert hasattr(result, "article_hint")
            assert hasattr(result, "score")
            assert isinstance(result.id, int)
            assert isinstance(result.text, str)
            assert isinstance(result.source_file, str)
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0  # Cosine similarity

