"""FAISS index building logic from legal provisions."""

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
    Load all .txt files from directory (recursively).
    
    Args:
        input_dir: Path to directory with .txt files.
        
    Returns:
        Dictionary: {file_path: text_content}
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    files_content = {}
    for txt_file in input_path.rglob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    files_content[str(txt_file)] = content
        except Exception as e:
            print(f"Warning: Failed to load file {txt_file}: {e}")
    
    if not files_content:
        raise ValueError(f"No .txt files found in directory {input_dir}")
    
    return files_content


def split_text_into_chunks(
    text: str,
    source_file: str,
    max_chunk_size: int = 1200
) -> List[Dict[str, Any]]:
    """
    Split text into fragments (chunks) at article level.
    
    Attempts to detect headers in style "Art. 118." and group text from header
    to next header as one fragment. If no such headers exist,
    splits by empty lines or every max_chunk_size characters.
    
    Args:
        text: Text to split.
        source_file: Path to source file.
        max_chunk_size: Maximum chunk length in characters (when no headers).
        
    Returns:
        List of dictionaries with fields: text, source_file, article_hint.
    """
    chunks = []
    
    # Pattern to detect article headers (e.g., "Art. 118.", "Art. 1.", "Art. 123a.")
    article_pattern = re.compile(r'^Art\.\s*\d+[a-z]?\.', re.MULTILINE | re.IGNORECASE)
    
    # Find all header positions
    matches = list(article_pattern.finditer(text))
    
    if len(matches) > 1:
        # We have headers - split by articles
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
        # Single article - entire text as one chunk
        article_hint = matches[0].group().strip()
        chunks.append({
            "text": text.strip(),
            "source_file": source_file,
            "article_hint": article_hint
        })
    else:
        # No headers - split by empty lines or every max_chunk_size characters
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If a single paragraph is longer than max_chunk_size, split it
            if len(para) > max_chunk_size:
                # First save current chunk if it exists
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "source_file": source_file,
                        "article_hint": None
                    })
                    current_chunk = ""
                
                # Split long paragraph into smaller fragments
                start = 0
                while start < len(para):
                    end = start + max_chunk_size
                    chunk_text = para[start:end]
                    chunks.append({
                        "text": chunk_text,
                        "source_file": source_file,
                        "article_hint": None
                    })
                    start = end
            elif len(current_chunk) + len(para) + 1 <= max_chunk_size:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk and start new one
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
    Build FAISS index from legal provisions.
    
    Loads .txt files from input directory, splits into fragments,
    computes embeddings and builds FAISS index. Saves:
    - index.faiss - FAISS index
    - metadata.json - fragment metadata
    
    Args:
        input_dir: Directory with .txt files.
        output_dir: Output directory for index and metadata.
        model_name: Name of embedding model.
        max_chunk_size: Maximum chunk length in characters.
    """
    print(f"Loading files from directory: {input_dir}")
    files_content = load_text_files(input_dir)
    print(f"Loaded {len(files_content)} files.")
    
    # Splitting into fragments
    print("Splitting texts into fragments...")
    all_chunks = []
    for file_path, content in files_content.items():
        chunks = split_text_into_chunks(content, file_path, max_chunk_size)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} fragments.")
    
    if not all_chunks:
        raise ValueError("Failed to create any fragments from input files.")
    
    # Computing embeddings
    print(f"Computing embeddings using model: {model_name}")
    embedding_model = EmbeddingModel(model_name)
    
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts)
    
    print(f"Computed embeddings with dimension: {embeddings.shape[1]}")
    
    # Building FAISS index (IndexFlatIP with normalized embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity with normalization
    index.add(embeddings.astype("float32"))
    
    print(f"Built FAISS index with {index.ntotal} vectors.")
    
    # Preparing metadata
    metadata = []
    for i, chunk in enumerate(all_chunks):
        metadata.append({
            "id": i,
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "article_hint": chunk["article_hint"]
        })
    
    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.json"
    
    print(f"Saving index to: {index_path}")
    faiss.write_index(index, str(index_path))
    
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("Index successfully built and saved.")
