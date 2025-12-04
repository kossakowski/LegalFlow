"""Klasa do wyszukiwania fragmentów przepisów w indeksie FAISS."""

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .embeddings import EmbeddingModel
from .models import SearchResult


class LegalRetriever:
    """Klasa do wyszukiwania przepisów prawnych w indeksie FAISS."""
    
    def __init__(
        self,
        index_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Inicjalizuje retriever ładując indeks FAISS i metadane.
        
        Args:
            index_dir: Katalog zawierający index.faiss i metadata.json.
            model_name: Nazwa modelu embeddingowego (musi być taki sam jak przy budowie).
        """
        index_path = Path(index_dir) / "index.faiss"
        metadata_path = Path(index_dir) / "metadata.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Indeks nie istnieje: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Plik metadanych nie istnieje: {metadata_path}")
        
        print(f"Ładowanie indeksu z: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        print(f"Ładowanie metadanych z: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.embedding_model = EmbeddingModel(model_name)
        
        print(f"Zainicjalizowano retriever z {len(self.metadata)} fragmentami.")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Wyszukuje fragmenty przepisów podobne do zapytania.
        
        Args:
            query: Tekst zapytania.
            top_k: Maksymalna liczba wyników do zwrócenia.
            min_score: Minimalny score (cosine similarity) do uwzględnienia.
        
        Returns:
            Lista obiektów SearchResult posortowana malejąco po score.
        """
        # Oblicz embedding zapytania
        query_embedding = self.embedding_model.encode([query])
        
        # Wyszukaj w indeksie (używamy top_k większego, żeby móc filtrować po min_score)
        search_k = min(top_k * 2, self.index.ntotal)  # Pobierz więcej, żeby mieć zapas po filtrowaniu
        scores, indices = self.index.search(query_embedding.astype("float32"), search_k)
        
        # Przygotuj wyniki
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS zwraca -1 dla pustych wyników
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
        
        # Sortuj malejąco po score i ogranicz do top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


