"""Konfiguracja i fixtures dla testów."""

import json
import tempfile
from pathlib import Path
from typing import Generator, Tuple

import faiss
import numpy as np
import pytest

from legal_rag.embeddings import EmbeddingModel


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Tworzy tymczasowy katalog dla testów."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_files(temp_dir: Path) -> Path:
    """Tworzy przykładowe pliki tekstowe z przepisami."""
    txt_dir = temp_dir / "txt"
    txt_dir.mkdir()
    
    # Plik 1: Z artykułami
    file1 = txt_dir / "kodeks_cywilny.txt"
    file1.write_text(
        "Art. 1.\n"
        "Kodeks niniejszy reguluje stosunki cywilnoprawne między osobami fizycznymi i prawnymi.\n\n"
        "Art. 2.\n"
        "Każdy ma prawo do własności i innych praw majątkowych.\n\n"
        "Art. 3.\n"
        "Własność może być nabyta na różne sposoby określone w ustawie.\n",
        encoding="utf-8"
    )
    
    # Plik 2: Bez artykułów
    file2 = txt_dir / "umowa.txt"
    file2.write_text(
        "Umowa jest zgodnym oświadczeniem woli dwóch lub więcej stron.\n\n"
        "Umowa powinna być zawarta w formie określonej przez prawo.\n\n"
        "Strony umowy mają obowiązek wykonać zobowiązania wynikające z umowy.\n",
        encoding="utf-8"
    )
    
    return txt_dir


@pytest.fixture
def sample_index(temp_dir: Path, sample_text_files: Path) -> Path:
    """Tworzy przykładowy indeks FAISS dla testów."""
    from legal_rag.indexing import build_index
    
    index_dir = temp_dir / "index"
    index_dir.mkdir()
    
    # Buduj indeks z przykładowych plików
    build_index(
        input_dir=str(sample_text_files),
        output_dir=str(index_dir),
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_chunk_size=200
    )
    
    return index_dir


@pytest.fixture
def embedding_model() -> EmbeddingModel:
    """Tworzy model embeddingowy dla testów."""
    return EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@pytest.fixture
def sample_metadata() -> list:
    """Przykładowe metadane dla testów."""
    return [
        {
            "id": 0,
            "text": "Art. 1. Kodeks niniejszy reguluje stosunki cywilnoprawne.",
            "source_file": "test1.txt",
            "article_hint": "Art. 1."
        },
        {
            "id": 1,
            "text": "Art. 2. Każdy ma prawo do własności.",
            "source_file": "test1.txt",
            "article_hint": "Art. 2."
        },
        {
            "id": 2,
            "text": "Umowa jest zgodnym oświadczeniem woli.",
            "source_file": "test2.txt",
            "article_hint": None
        }
    ]


@pytest.fixture
def sample_faiss_index(embedding_model: EmbeddingModel, sample_metadata: list) -> Tuple[faiss.Index, list]:
    """Tworzy przykładowy indeks FAISS w pamięci."""
    # Oblicz embeddingi
    texts = [meta["text"] for meta in sample_metadata]
    embeddings = embedding_model.encode(texts)
    
    # Utwórz indeks
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    
    return index, sample_metadata

