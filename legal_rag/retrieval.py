"""Class for searching legal provision fragments in FAISS index."""

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

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

        # BM25 initialization
        self._bm25 = None
        self._bm25_corpus: List[List[str]] = []
        self._init_bm25()

        print(f"Initialized retriever with {len(self.metadata)} fragments.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on whitespace, strip basic punctuation."""
        # Very light punctuation stripping to keep it fast/simple
        stripped = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
        return stripped.split()

    def _init_bm25(self) -> None:
        """Initialize BM25 corpus and index. Gracefully degrade if it fails."""
        try:
            self._bm25_corpus = [self._tokenize(meta["text"]) for meta in self.metadata]
            if not self._bm25_corpus:
                print("Warning: BM25 corpus is empty; falling back to embeddings only.")
                return
            self._bm25 = BM25Okapi(self._bm25_corpus)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Failed to initialize BM25 ({exc}). Falling back to embeddings only.")
            self._bm25 = None
            self._bm25_corpus = []

    def search(
        self,
        query: str,
        top_k: int = 50,
        min_score: float = 0.0,
        search_multiplier: float = 2.0,
        weight_embedding: float = 1.0,
        weight_keyword: float = 1.0
    ) -> List[SearchResult]:
        """
        Search for legal provision fragments similar to query using hybrid embedding + BM25.

        Args:
            query: Query text.
            top_k: Maximum number of results to return. If very large (>100000), returns all results.
            min_score: Minimum combined score to include.
            search_multiplier: Multiplier determining how many more candidates to search than top_k (default: 2.0).
            weight_embedding: Weight for embedding-based score (default: 1.0).
            weight_keyword: Weight for keyword/BM25 score (default: 1.0).

        Returns:
            List of SearchResult objects sorted descending by combined score.
        """
        # Embedding search
        query_embedding = self.embedding_model.encode([query])

        if top_k > 100000:
            emb_search_k = self.index.ntotal
        else:
            emb_search_k = min(int(top_k * search_multiplier), self.index.ntotal)

        emb_scores, emb_indices = self.index.search(query_embedding.astype("float32"), emb_search_k)
        emb_scores_by_idx: Dict[int, float] = {
            int(idx): float(score)
            for score, idx in zip(emb_scores[0], emb_indices[0])
            if idx >= 0
        }

        # Keyword/BM25 search
        kw_scores_by_idx: Dict[int, float] = {}
        normalized_kw_scores_by_idx: Dict[int, float] = {}

        if self._bm25 is not None and self._bm25_corpus:
            query_tokens = self._tokenize(query)
            raw_kw_scores = self._bm25.get_scores(query_tokens)

            # pick top-k BM25 candidates
            kw_search_k = self.index.ntotal if top_k > 100000 else min(int(top_k * search_multiplier), len(raw_kw_scores))
            sorted_kw_indices = np.argsort(raw_kw_scores)[::-1][:kw_search_k]

            for idx in sorted_kw_indices:
                kw_scores_by_idx[int(idx)] = float(raw_kw_scores[idx])

            if kw_scores_by_idx:
                max_kw_score = max(kw_scores_by_idx.values())
                for idx, score in kw_scores_by_idx.items():
                    normalized_kw_scores_by_idx[idx] = score / max_kw_score if max_kw_score > 0 else 0.0
        else:
            # BM25 unavailable
            normalized_kw_scores_by_idx = {}

        # Merge results
        candidate_indices = set(emb_scores_by_idx.keys()) | set(normalized_kw_scores_by_idx.keys())
        results: List[SearchResult] = []

        for idx in candidate_indices:
            emb_score = emb_scores_by_idx.get(idx, 0.0)
            kw_score = normalized_kw_scores_by_idx.get(idx, 0.0)

            combined_score = emb_score * weight_embedding + kw_score * weight_keyword
            if combined_score < min_score:
                continue

            if emb_score > 0 and kw_score > 0:
                method = "both"
            elif emb_score > 0:
                method = "embedding"
            else:
                method = "keyword"

            meta = self.metadata[idx]
            results.append(
                SearchResult(
                    id=meta["id"],
                    text=meta["text"],
                    source_file=meta["source_file"],
                    article_hint=meta.get("article_hint"),
                    score=combined_score,
                    method=method,
                    embedding_score=emb_score or None,
                    keyword_score=kw_score or None,
                )
            )

        # Sort and truncate
        results.sort(key=lambda x: x.score, reverse=True)
        if top_k > 100000:
            return results
        return results[:top_k]
