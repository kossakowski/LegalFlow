"""Testy dla funkcji indeksowania."""

import json
from pathlib import Path

import pytest

from legal_rag.indexing import build_index, load_text_files, split_text_into_chunks


class TestLoadTextFiles:
    """Testy dla funkcji load_text_files."""
    
    def test_load_text_files_success(self, sample_text_files: Path):
        """Test wczytywania plików tekstowych."""
        files = load_text_files(str(sample_text_files))
        
        assert len(files) == 2
        assert any("kodeks_cywilny.txt" in path for path in files.keys())
        assert any("umowa.txt" in path for path in files.keys())
    
    def test_load_text_files_nonexistent_dir(self):
        """Test wczytywania z nieistniejącego katalogu."""
        with pytest.raises(FileNotFoundError):
            load_text_files("/nonexistent/directory")
    
    def test_load_text_files_empty_dir(self, temp_dir: Path):
        """Test loading from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No .txt files found"):
            load_text_files(str(empty_dir))
    
    def test_load_text_files_recursive(self, temp_dir: Path):
        """Test rekurencyjnego wczytywania plików."""
        root = temp_dir / "root"
        root.mkdir()
        subdir = root / "subdir"
        subdir.mkdir()
        
        (root / "file1.txt").write_text("Content 1", encoding="utf-8")
        (subdir / "file2.txt").write_text("Content 2", encoding="utf-8")
        
        files = load_text_files(str(root))
        
        assert len(files) == 2
        assert any("file1.txt" in path for path in files.keys())
        assert any("file2.txt" in path for path in files.keys())


class TestSplitTextIntoChunks:
    """Testy dla funkcji split_text_into_chunks."""
    
    def test_split_text_with_articles(self):
        """Test dzielenia tekstu z artykułami."""
        text = (
            "Art. 1.\n"
            "Pierwszy artykuł z treścią.\n\n"
            "Art. 2.\n"
            "Drugi artykuł z treścią.\n\n"
            "Art. 3.\n"
            "Trzeci artykuł z treścią."
        )
        
        chunks = split_text_into_chunks(text, "test.txt")
        
        assert len(chunks) == 3
        assert chunks[0]["article_hint"] == "Art. 1."
        assert chunks[1]["article_hint"] == "Art. 2."
        assert chunks[2]["article_hint"] == "Art. 3."
        assert all(chunk["source_file"] == "test.txt" for chunk in chunks)
    
    def test_split_text_without_articles(self):
        """Test dzielenia tekstu bez artykułów."""
        text = (
            "Pierwszy akapit.\n\n"
            "Drugi akapit.\n\n"
            "Trzeci akapit."
        )
        
        chunks = split_text_into_chunks(text, "test.txt", max_chunk_size=50)
        
        assert len(chunks) >= 1
        assert all(chunk["article_hint"] is None for chunk in chunks)
        assert all(chunk["source_file"] == "test.txt" for chunk in chunks)
    
    def test_split_text_single_article(self):
        """Test dzielenia tekstu z jednym artykułem."""
        text = "Art. 1.\nTreść jedynego artykułu."
        
        chunks = split_text_into_chunks(text, "test.txt")
        
        assert len(chunks) == 1
        assert chunks[0]["article_hint"] == "Art. 1."
    
    def test_split_text_max_chunk_size(self):
        """Test dzielenia tekstu z ograniczeniem rozmiaru chunka."""
        text = "A" * 5000  # Długi tekst bez artykułów
        
        chunks = split_text_into_chunks(text, "test.txt", max_chunk_size=1000)
        
        assert len(chunks) >= 5  # Powinno być co najmniej 5 chunków
        assert all(len(chunk["text"]) <= 1000 for chunk in chunks)
    
    def test_split_text_empty(self):
        """Test dzielenia pustego tekstu."""
        chunks = split_text_into_chunks("", "test.txt")
        
        assert len(chunks) == 0
    
    def test_split_text_article_with_letter(self):
        """Test dzielenia tekstu z artykułami zawierającymi litery (np. Art. 123a.)."""
        text = (
            "Art. 1.\nTreść.\n\n"
            "Art. 2a.\nTreść z literą.\n\n"
            "Art. 3.\nTreść."
        )
        
        chunks = split_text_into_chunks(text, "test.txt")
        
        assert len(chunks) == 3
        assert chunks[1]["article_hint"] == "Art. 2a."


class TestBuildIndex:
    """Testy dla funkcji build_index."""
    
    def test_build_index_success(self, sample_text_files: Path, temp_dir: Path):
        """Test budowy indeksu."""
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(sample_text_files),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        # Sprawdź czy pliki zostały utworzone
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "metadata.json").exists()
        
        # Sprawdź metadane
        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        assert len(metadata) > 0
        assert all("id" in m for m in metadata)
        assert all("text" in m for m in metadata)
        assert all("source_file" in m for m in metadata)
    
    def test_build_index_creates_valid_faiss(self, sample_text_files: Path, temp_dir: Path):
        """Test czy utworzony indeks FAISS jest poprawny."""
        import faiss
        
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(sample_text_files),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        # Wczytaj indeks
        index = faiss.read_index(str(index_dir / "index.faiss"))
        
        assert index.ntotal > 0
        assert index.d > 0  # Wymiar embeddingu
    
    def test_build_index_metadata_structure(self, sample_text_files: Path, temp_dir: Path):
        """Test struktury metadanych."""
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        build_index(
            input_dir=str(sample_text_files),
            output_dir=str(index_dir),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=200
        )
        
        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Sprawdź strukturę pierwszego elementu
        first = metadata[0]
        assert "id" in first
        assert "text" in first
        assert "source_file" in first
        assert "article_hint" in first
        assert isinstance(first["id"], int)
        assert isinstance(first["text"], str)
        assert isinstance(first["source_file"], str)
    
    def test_build_index_empty_input(self, temp_dir: Path):
        """Test building index from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        index_dir = temp_dir / "index"
        index_dir.mkdir()
        
        with pytest.raises(ValueError, match="No .txt files found"):
            build_index(
                input_dir=str(empty_dir),
                output_dir=str(index_dir)
            )

