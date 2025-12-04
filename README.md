# LegalFlow

Prosty system retrieval przepisów prawnych oparty na lokalnych embeddingach i indeksie FAISS.

## Wymagania

- Python 3.10 lub nowszy
- Środowisko WSL (Linux)
- pip

## Instalacja

1. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

## Przygotowanie danych

Umieść pliki `.txt` z przepisami prawnymi w katalogu `./data/txt`. Pliki powinny być zakodowane w UTF-8.

Przykładowa struktura:

```
data/
  txt/
    kodeks_cywilny.txt
    kodeks_pracy.txt
    ...
```

## Budowa indeksu

Aby zbudować indeks FAISS z plików źródłowych:

```bash
python -m legal_rag.main build-index --input-dir ./data/txt --output-dir ./data/index
```

Opcjonalne parametry:

- `--model-name` - nazwa modelu embeddingowego (domyślnie: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- `--max-chunk-size` - maksymalna długość chunka w znakach (domyślnie: 1200)

## Wyszukiwanie

Aby wyszukać fragmenty przepisów:

```bash
python -m legal_rag.main query --index-dir ./data/index --query "ile przedawnia się roszczenie o wynagrodzenie z umowy sprzedaży?" --top-k 20 --min-score 0.0
```

Parametry:

- `--index-dir` - katalog zawierający `index.faiss` i `metadata.json`
- `--query` - tekst zapytania
- `--top-k` - maksymalna liczba wyników (domyślnie: 20)
- `--min-score` - minimalny score (cosine similarity) do uwzględnienia (domyślnie: 0.0)
- `--model-name` - nazwa modelu embeddingowego (musi być taki sam jak przy budowie)

## Architektura

Projekt został zaprojektowany z myślą o łatwej rozbudowie o RAG z LLM:

- `LegalRetriever` - klasa do wyszukiwania fragmentów
- `SearchResult` - struktura danych zawierająca tekst, metadane i score
- Wyniki można łatwo przekazać do LLM jako kontekst

## Struktura projektu

```
LegalFlow/
├── legal_rag/
│   ├── __init__.py
│   ├── models.py          # Modele danych (SearchResult)
│   ├── embeddings.py      # Klasa EmbeddingModel
│   ├── indexing.py        # Logika budowy indeksu
│   ├── retrieval.py       # Klasa LegalRetriever
│   └── main.py            # Interfejs CLI
├── data/
│   ├── txt/               # Pliki źródłowe .txt
│   └── index/             # Indeks FAISS i metadane
├── requirements.txt
└── README.md
```


