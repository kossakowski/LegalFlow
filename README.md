# LegalFlow

Simple legal provision retrieval system based on local embeddings and FAISS index.

## Requirements

- Python 3.10 or newer
- WSL environment (Linux)
- pip

## Installation

1. Create and activate virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place `.txt` files with legal provisions in the `./data/txt` directory. Files should be encoded in UTF-8.

Example structure:

```
data/
  txt/
    kodeks_cywilny.txt
    kodeks_pracy.txt
    ...
```

## Building Index

To build FAISS index from source files:

```bash
python -m legal_rag.main build-index --input-dir ./data/txt --output-dir ./data/index
```

Optional parameters:

- `--model-name` - name of embedding model (default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- `--max-chunk-size` - maximum chunk length in characters (default: 1200)

## Searching (hybrid: embeddings + BM25)

To search for legal provision fragments (hybrid embeddings + keyword/BM25):

```bash
python -m legal_rag.main query \
  --index-dir ./data/index \
  --query "when does a claim for payment from a sales contract become time-barred?" \
  --top-k 50 \
  --min-score 0.0 \
  --weight-embedding 1.0 \
  --weight-keyword 1.0
```

Parameters:

- `--index-dir` - directory containing `index.faiss` and `metadata.json`
- `--query` - query text
- `--top-k` - maximum number of results (default: 50). Use `0` to display all results
- `--min-score` - minimum combined score to include (default: 0.0)
- `--model-name` - name of embedding model (must be same as used for building)
- `--search-multiplier` - multiplier determining how many more candidates to search than top_k (default: 2.0)
- `--weight-embedding` - weight for embedding-based score (default: 1.0)
- `--weight-keyword` - weight for keyword/BM25-based score (default: 1.0)

### Usage Examples

```bash
# Basic hybrid search (50 results)
python -m legal_rag.main query --index-dir ./data/index --query "sales contract"

# More results
python -m legal_rag.main query --index-dir ./data/index --query "liability" --top-k 100

# All results
python -m legal_rag.main query --index-dir ./data/index --query "statute of limitations" --top-k 0

# With minimum combined similarity filtering
python -m legal_rag.main query --index-dir ./data/index --query "property" --min-score 0.5

# With larger search multiplier (more candidates)
python -m legal_rag.main query --index-dir ./data/index --query "contract" --search-multiplier 3.0

# Tuning weights (favor embeddings)
python -m legal_rag.main query --index-dir ./data/index --query "umowa sprzedaży" --weight-embedding 1.5 --weight-keyword 0.5

# Tuning weights (favor keywords/BM25)
python -m legal_rag.main query --index-dir ./data/index --query "umowa sprzedaży" --weight-embedding 0.5 --weight-keyword 1.5
```

## Architecture

The project is designed for easy extension with RAG and LLM:

- `LegalRetriever` - class for searching fragments
- `SearchResult` - data structure containing text, metadata and score
- Results can be easily passed to LLM as context

## Project Structure

```
LegalFlow/
├── legal_rag/
│   ├── __init__.py
│   ├── models.py          # Data models (SearchResult)
│   ├── embeddings.py      # EmbeddingModel class
│   ├── indexing.py        # Index building logic
│   ├── retrieval.py       # LegalRetriever class
│   └── main.py            # CLI interface
├── data/
│   ├── txt/               # Source .txt files
│   └── index/             # FAISS index and metadata
├── tests/                  # Unit and integration tests
├── requirements.txt
└── README.md
```

## Tests

To run tests:

```bash
pytest
```

With code coverage report:

```bash
pytest --cov=legal_rag --cov-report=html
```

More information in [tests/README.md](tests/README.md).
