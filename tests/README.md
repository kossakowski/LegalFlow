# Testy dla LegalFlow

Ten katalog zawiera testy jednostkowe i integracyjne dla systemu LegalFlow.

## Struktura testów

- `test_models.py` - Testy dla modeli danych (SearchResult)
- `test_embeddings.py` - Testy dla modelu embeddingowego
- `test_indexing.py` - Testy dla funkcji indeksowania
- `test_retrieval.py` - Testy dla klasy LegalRetriever
- `test_integration.py` - Testy integracyjne całego systemu
- `conftest.py` - Konfiguracja i fixtures dla testów

## Uruchamianie testów

### Wszystkie testy

```bash
pytest
```

### Z raportem pokrycia kodu

```bash
pytest --cov=legal_rag --cov-report=html
```

Raport HTML będzie dostępny w `htmlcov/index.html`.

### Pojedynczy plik testowy

```bash
pytest tests/test_models.py
```

### Pojedynczy test

```bash
pytest tests/test_models.py::TestSearchResult::test_search_result_creation
```

### Verbose output

```bash
pytest -v
```

## Wymagania

Testy wymagają zainstalowanych zależności z `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Uwagi

- Testy używają rzeczywistego modelu embeddingowego, więc mogą być wolniejsze
- Niektóre testy tworzą tymczasowe pliki i katalogi
- Testy integracyjne mogą wymagać więcej czasu na wykonanie

