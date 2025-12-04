"""Testy dla modelu embeddingowego."""

import numpy as np
import pytest

from legal_rag.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Testy dla klasy EmbeddingModel."""
    
    def test_embedding_model_initialization(self):
        """Test inicjalizacji modelu embeddingowego."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        assert model.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        assert model.model is not None
    
    def test_embedding_model_encode_single_text(self):
        """Test kodowania pojedynczego tekstu."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        embeddings = model.encode(["Przykładowy tekst"])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0  # Wymiar embeddingu
    
    def test_embedding_model_encode_multiple_texts(self):
        """Test kodowania wielu tekstów."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        texts = [
            "Pierwszy tekst",
            "Drugi tekst",
            "Trzeci tekst"
        ]
        embeddings = model.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0
    
    def test_embedding_model_normalization(self):
        """Test czy embeddingi są znormalizowane (L2 norm = 1)."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        embeddings = model.encode(["Test normalizacji"])
        
        # Sprawdź normalizację L2 (norma każdego wektora powinna być ~1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_embedding_model_polish_text(self):
        """Test kodowania polskiego tekstu."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        polish_text = "Kodeks cywilny reguluje stosunki cywilnoprawne między osobami."
        embeddings = model.encode([polish_text])
        
        assert embeddings.shape[0] == 1
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
    
    def test_embedding_model_similarity(self):
        """Test czy podobne teksty mają podobne embeddingi."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        text1 = "Umowa sprzedaży"
        text2 = "Kontrakt kupna-sprzedaży"
        text3 = "Zupełnie inny temat"
        
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        emb3 = model.encode([text3])[0]
        
        # Cosine similarity (iloczyn skalarny znormalizowanych wektorów)
        similarity_12 = np.dot(emb1, emb2)
        similarity_13 = np.dot(emb1, emb3)
        
        # Teksty 1 i 2 powinny być bardziej podobne niż 1 i 3
        assert similarity_12 > similarity_13
    
    def test_embedding_model_empty_list(self):
        """Test kodowania pustej listy."""
        model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        embeddings = model.encode([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0


