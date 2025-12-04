"""Embedding model for computing text representation vectors."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingModel:
    """Wrapper class for SentenceTransformer with embedding normalization."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model supporting Polish language.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts with L2 normalization.
        
        Args:
            texts: List of texts to encode.
            
        Returns:
            Numpy array of shape (len(texts), embedding_dim) with normalized embeddings.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        return embeddings
