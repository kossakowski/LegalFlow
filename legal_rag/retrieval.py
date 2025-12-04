"""Class for searching legal provision fragments in FAISS index."""

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .embeddings import EmbeddingModel
from .models import SearchResult


class LegalRetriever:
    """Class for searching legal provisions in FAISS index."""
    
    def __init__(
        self,
        index_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize retriever by loading FAISS index and metadata.
        
        Args:
            index_dir: Directory containing index.faiss and metadata.json.
            model_name: Name of embedding model (must be same as used for building).
        """
        index_path = Path(index_dir) / "index.faiss"
        metadata_path = Path(index_dir) / "metadata.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index does not exist: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
        
        print(f"Loading index from: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.embedding_model = EmbeddingModel(model_name)
        
        print(f"Initialized retriever with {len(self.metadata)} fragments.")
    
    def search(
        self,
        query: str,
        top_k: int = 50,
        min_score: float = 0.0,
        search_multiplier: float = 2.0
    ) -> List[SearchResult]:
        """
        Search for legal provision fragments similar to query.
        
        Args:
            query: Query text.
            top_k: Maximum number of results to return. If very large (>100000), returns all results.
            min_score: Minimum score (cosine similarity) to include.
            search_multiplier: Multiplier determining how many more candidates to search than top_k (default: 2.0).
        
        Returns:
            List of SearchResult objects sorted descending by score.
        """
        # Compute query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # If top_k is very large, search all available results
        if top_k > 100000:
            search_k = self.index.ntotal
        else:
            # Search in index (we use larger top_k to be able to filter by min_score)
            search_k = min(int(top_k * search_multiplier), self.index.ntotal)
        
        scores, indices = self.index.search(query_embedding.astype("float32"), search_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty results
                continue
            
            if score < min_score:
                continue
            
            meta = self.metadata[idx]
            results.append(SearchResult(
                id=meta["id"],
                text=meta["text"],
                source_file=meta["source_file"],
                article_hint=meta.get("article_hint"),
                score=float(score)
            ))
        
        # Sort descending by score and limit to top_k (if not very large)
        results.sort(key=lambda x: x.score, reverse=True)
        if top_k > 100000:
            return results  # Return all results
        else:
            return results[:top_k]
