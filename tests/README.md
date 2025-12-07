# Tests for LegalFlow

This directory contains unit and integration tests for the LegalFlow system.

## Test Structure

- `test_models.py` - Tests for data models (SearchResult)
- `test_embeddings.py` - Tests for embedding model
- `test_indexing.py` - Tests for indexing functions
- `test_retrieval.py` - Tests for LegalRetriever class
- `test_integration.py` - Integration tests for the entire system
- `test_search_accuracy.py` - Tests for search accuracy and result quality validation
- `test_search_accuracy_cases.json` - Test cases (queries and expected articles) for accuracy tests
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

## Test Types

### Unit Tests
- `test_models.py`, `test_embeddings.py`, `test_indexing.py`, `test_retrieval.py` - Test individual components

### Integration Tests
- `test_integration.py` - Test the full workflow from indexing to searching

### Accuracy/Quality Tests
- `test_search_accuracy.py` - Validates search accuracy against the production index in `data/index`
  - Uses production data (bankruptcy/restructuring laws), not synthetic samples
  - Test cases defined in `test_search_accuracy_cases.json` (queries + expected articles)
  - Validates relevance, ranking, and presence of expected articles in top results

## Notes

- Tests use real embedding model, so they may be slower
- Some tests create temporary files and directories
- Integration tests may require more time to execute
- Accuracy tests validate search quality, not just functionality
