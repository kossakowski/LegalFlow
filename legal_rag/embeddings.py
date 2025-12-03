"""Model embeddingowy do obliczania wektorów reprezentacji tekstu."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingModel:
    """Klasa opakowująca SentenceTransformer z normalizacją embeddingów."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicjalizuje model embeddingowy.
        
        Args:
            model_name: Nazwa modelu z sentence-transformers obsługującego język polski.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Oblicza embeddingi dla listy tekstów z normalizacją L2.
        
        Args:
            texts: Lista tekstów do zakodowania.
            
        Returns:
            Tablica numpy o kształcie (len(texts), embedding_dim) z znormalizowanymi embeddingami.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizacja L2 dla cosine similarity
        )
        return embeddings

