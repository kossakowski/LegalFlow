# Tests for LegalFlow

This directory contains unit and integration tests for the LegalFlow system.

## Test Structure

- `test_models.py` - Tests for data models (SearchResult)
- `test_embeddings.py` - Tests for embedding model
- `test_indexing.py` - Tests for indexing functions
- `test_retrieval.py` - Tests for LegalRetriever class
- `test_integration.py` - Integration tests for the entire system
- `conftest.py` - Configuration and fixtures for tests

## Running Tests

### All Tests

```bash
pytest
```

### With Code Coverage Report

```bash
pytest --cov=legal_rag --cov-report=html
```

HTML report will be available in `htmlcov/index.html`.

### Single Test File

```bash
pytest tests/test_models.py
```

### Single Test

```bash
pytest tests/test_models.py::TestSearchResult::test_search_result_creation
```

### Verbose Output

```bash
pytest -v
```

## Requirements

Tests require dependencies from `requirements.txt` to be installed:

```bash
pip install -r requirements.txt
```

## Notes

- Tests use real embedding model, so they may be slower
- Some tests create temporary files and directories
- Integration tests may require more time to execute
