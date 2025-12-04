"""Logika budowy indeksu FAISS z przepisów prawnych."""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np

from .embeddings import EmbeddingModel


def load_text_files(input_dir: str) -> Dict[str, str]:
    """
    Wczytuje wszystkie pliki .txt z katalogu (rekurencyjnie).
    
    Args:
        input_dir: Ścieżka do katalogu z plikami .txt.
        
    Returns:
        Słownik: {ścieżka_pliku: zawartość_tekstowa}
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Katalog wejściowy nie istnieje: {input_dir}")
    
    files_content = {}
    for txt_file in input_path.rglob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    files_content[str(txt_file)] = content
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się wczytać pliku {txt_file}: {e}")
    
    if not files_content:
        raise ValueError(f"Nie znaleziono żadnych plików .txt w katalogu {input_dir}")
    
    return files_content


def split_text_into_chunks(
    text: str,
    source_file: str,
    max_chunk_size: int = 1200
) -> List[Dict[str, Any]]:
    """
    Dzieli tekst na fragmenty (chunki) na poziomie artykułów.
    
    Próbuje wykrywać nagłówki w stylu "Art. 118." i grupować tekst od nagłówka
    do następnego nagłówka jako jeden fragment. Jeśli nie ma takich nagłówków,
    dzieli po pustych liniach lub co max_chunk_size znaków.
    
    Args:
        text: Tekst do podziału.
        source_file: Ścieżka do pliku źródłowego.
        max_chunk_size: Maksymalna długość chunka w znakach (gdy brak nagłówków).
        
    Returns:
        Lista słowników z polami: text, source_file, article_hint.
    """
    chunks = []
    
    # Wzorzec do wykrywania nagłówków artykułów (np. "Art. 118.", "Art. 1.", "Art. 123a.")
    article_pattern = re.compile(r'^Art\.\s*\d+[a-z]?\.', re.MULTILINE | re.IGNORECASE)
    
    # Znajdź wszystkie pozycje nagłówków
    matches = list(article_pattern.finditer(text))
    
    if len(matches) > 1:
        # Mamy nagłówki - dziel po artykułach
        for i in range(len(matches)):
            start_pos = matches[i].start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            chunk_text = text[start_pos:end_pos].strip()
            if chunk_text:
                article_hint = matches[i].group().strip()
                chunks.append({
                    "text": chunk_text,
                    "source_file": source_file,
                    "article_hint": article_hint
                })
    elif len(matches) == 1:
        # Jeden artykuł - cały tekst jako jeden chunk
        article_hint = matches[0].group().strip()
        chunks.append({
            "text": text.strip(),
            "source_file": source_file,
            "article_hint": article_hint
        })
    else:
        # Brak nagłówków - dziel po pustych liniach lub co max_chunk_size znaków
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 1 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "source_file": source_file,
                        "article_hint": None
                    })
                current_chunk = para
        
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "source_file": source_file,
                "article_hint": None
            })
    
    return chunks


def build_index(
    input_dir: str,
    output_dir: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_chunk_size: int = 1200
) -> None:
    """
    Buduje indeks FAISS z przepisów prawnych.
    
    Wczytuje pliki .txt z katalogu wejściowego, dzieli na fragmenty,
    oblicza embeddingi i buduje indeks FAISS. Zapisuje:
    - index.faiss - indeks FAISS
    - metadata.json - metadane fragmentów
    
    Args:
        input_dir: Katalog z plikami .txt.
        output_dir: Katalog wyjściowy dla indeksu i metadanych.
        model_name: Nazwa modelu embeddingowego.
        max_chunk_size: Maksymalna długość chunka w znakach.
    """
    print(f"Wczytywanie plików z katalogu: {input_dir}")
    files_content = load_text_files(input_dir)
    print(f"Wczytano {len(files_content)} plików.")
    
    # Dzielenie na fragmenty
    print("Dzielenie tekstów na fragmenty...")
    all_chunks = []
    for file_path, content in files_content.items():
        chunks = split_text_into_chunks(content, file_path, max_chunk_size)
        all_chunks.extend(chunks)
    
    print(f"Utworzono {len(all_chunks)} fragmentów.")
    
    if not all_chunks:
        raise ValueError("Nie udało się utworzyć żadnych fragmentów z plików wejściowych.")
    
    # Obliczanie embeddingów
    print(f"Obliczanie embeddingów za pomocą modelu: {model_name}")
    embedding_model = EmbeddingModel(model_name)
    
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts)
    
    print(f"Obliczono embeddingi o wymiarze: {embeddings.shape[1]}")
    
    # Budowa indeksu FAISS (IndexFlatIP z znormalizowanymi embeddingami)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product dla cosine similarity z normalizacją
    index.add(embeddings.astype("float32"))
    
    print(f"Zbudowano indeks FAISS z {index.ntotal} wektorami.")
    
    # Przygotowanie metadanych
    metadata = []
    for i, chunk in enumerate(all_chunks):
        metadata.append({
            "id": i,
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "article_hint": chunk["article_hint"]
        })
    
    # Zapis na dysk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.json"
    
    print(f"Zapisywanie indeksu do: {index_path}")
    faiss.write_index(index, str(index_path))
    
    print(f"Zapisywanie metadanych do: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("Indeks został pomyślnie zbudowany i zapisany.")


