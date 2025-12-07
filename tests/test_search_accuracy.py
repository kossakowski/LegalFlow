"""Tests for validating search accuracy and result quality.

These tests perform actual searches and validate that expected articles
appear in the search results, ensuring the search functionality is accurate
and returns relevant legal provisions.
"""

import json
import os
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pytest

from legal_rag.retrieval import LegalRetriever


def load_test_cases() -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    test_file = Path(__file__).parent / "test_search_accuracy_cases.json"
    if not test_file.exists():
        pytest.skip(f"Test cases file not found: {test_file}")
    
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("test_cases", [])


PRODUCTION_INDEX = Path(__file__).resolve().parent.parent / "data" / "index"
DEFAULT_TOP_K = 100
DEFAULT_SEARCH_MULTIPLIER = 100.0
RESULTS_LOG: List[Dict[str, Any]] = []


def _window_size(top_k: int, results_len: int) -> int:
    """Return window size for top-25% window (at least 1, max results_len)."""
    w = max(1, int(top_k * 0.25))
    return min(w, results_len)


def _report(
    name: str,
    query: str,
    expected: List[str],
    found: List[str],
    top_found: List[str],
    all_articles: List[str],
    missing: List[str],
    rank_info: str,
    rendered_all: str,
    capsys=None,
) -> None:
    """Print and record a concise per-query report (always logs, even on failures)."""
    in_any = any(e in found for e in expected) if expected else False
    in_top = any(e in top_found for e in expected) if expected else False
    msg = (
        f"[{name}] query='{query}' expected={expected} "
        f"in_results={in_any} in_top25%={in_top} top_found={top_found} missing={missing} ranks={rank_info}"
    )
    if capsys is not None:
        with capsys.disabled():
            print(msg, flush=True)
    else:
        print(msg, flush=True)
    
    RESULTS_LOG.append({
        "name": name,
        "query": query,
        "expected": expected,
        "in_results": in_any,
        "in_top": in_top,
        "top_found": top_found,
        "found_all": all(e in found for e in expected) if expected else True,
        "all_articles": rendered_all,
        "missing": missing,
        "rank_info": rank_info,
    })


def _write_html_report(path: Path, log: List[Dict[str, Any]]) -> None:
    """Write an HTML report summarizing per-query results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in log:
        status = "PASS" if entry["in_results"] and entry["in_top"] else "WARN" if entry["in_results"] else "FAIL"
        status_class = "pass" if status == "PASS" else "warn" if status == "WARN" else "fail"
        in_results_class = "yes" if entry["in_results"] else "no"
        in_top_class = "yes" if entry["in_top"] else "no"
        rows.append(f"""
        <tr>
            <td>{escape(entry['name'])}</td>
            <td>{escape(entry['query'])}</td>
            <td>{escape(', '.join(entry['expected']) if entry['expected'] else '—')}</td>
            <td><span class="pill {status_class}">{status}</span></td>
            <td><span class="pill {in_results_class}">{'YES' if entry['in_results'] else 'NO'}</span></td>
            <td><span class="pill {in_top_class}">{'YES' if entry['in_top'] else 'NO'}</span></td>
            <td>{escape(entry['rank_info'])}</td>
            <td class="small-text">{entry['all_articles'] or '—'}</td>
        </tr>
        """)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Search Accuracy Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: #f5f5f5;
      color: #2c3e50;
      padding: 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: #fff;
      padding: 24px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    h1 {{
      border-bottom: 3px solid #3498db;
      padding-bottom: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 20px;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid #e0e0e0;
      text-align: left;
    }}
    th {{
      background: #34495e;
      color: #fff;
    }}
    tr:nth-child(even) {{ background: #f9fafb; }}
    .pass {{ color: #27ae60; background: #e8f9ef; }}
    .fail {{ color: #e74c3c; background: #fdecea; }}
    .warn {{ color: #f39c12; background: #fff4e5; }}
    .yes {{ color: #1e8449; background: #e8f9ef; }}
    .no {{ color: #b03a2e; background: #fdecea; }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.9em;
    }}
    .small-text {{
      font-size: 0.85em;
      color: #555;
      line-height: 1.4;
      word-break: break-word;
    }}
    .match {{
      background: #e8f9ef;
      color: #1e8449;
      font-weight: 700;
      padding: 1px 4px;
      border-radius: 6px;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Search Accuracy Report</h1>
    <p>Generated: {escape(now)}</p>
    <table>
      <thead>
        <tr>
          <th>Test Case</th>
          <th>Query</th>
          <th>Expected Articles</th>
          <th>Status</th>
          <th>In Results</th>
          <th>In Top 25%</th>
          <th>Expected Rank</th>
          <th>Top K Articles</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


class TestSearchAccuracy:
    """Tests for validating search accuracy and result quality."""
    
    @pytest.fixture(scope="session")
    def production_index(self) -> Path:
        """Use the real production index from data/index."""
        index_dir = PRODUCTION_INDEX
        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "metadata.json"
        if not index_file.exists() or not metadata_file.exists():
            pytest.skip(f"Production index not found at {index_dir}")
        return index_dir

    @pytest.fixture
    def test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from JSON file."""
        return load_test_cases()
    
    @pytest.fixture(scope="session", autouse=True)
    def html_reporter(self, request):
        """Generate HTML report. Defaults to quality_test_results/result.html unless overridden by TEST_SEARCH_ACCURACY_HTML."""
        yield
        output = os.getenv("TEST_SEARCH_ACCURACY_HTML")
        if not output:
            output = "quality_test_results/result.html"
        _write_html_report(Path(output), RESULTS_LOG)

    def test_search_returns_expected_articles_via_retriever(
        self, 
        production_index: Path, 
        test_cases: List[Dict[str, Any]],
        capsys
    ):
        """Test that searches return expected articles using LegalRetriever directly."""
        retriever = LegalRetriever(
            index_dir=str(production_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        for test_case in test_cases:
            query = test_case["query"]
            expected_articles = test_case.get("expected_articles", [])
            top_k = test_case.get("top_k", DEFAULT_TOP_K)
            search_multiplier = test_case.get("search_multiplier", DEFAULT_SEARCH_MULTIPLIER)
            
            results = retriever.search(
                query=query,
                top_k=top_k,
                min_score=0.0,
                search_multiplier=search_multiplier,
                weight_embedding=1.0,
                weight_keyword=1.0
            )
            
            # Collect article hints from results
            found_articles = [
                r.article_hint 
                for r in results 
                if r.article_hint is not None
            ]
            all_articles = found_articles  # already filtered None and limited by top_k
            missing_articles = [
                art for art in expected_articles 
                if art not in found_articles
            ] if expected_articles else []
            
            # At least some results should be returned
            assert len(results) > 0, (
                f"Test case '{test_case['name']}': "
                f"Query '{query}' returned no results."
            )

            # Report presence overall and in top 25%
            window = _window_size(top_k, len(results))
            top_articles = [
                r.article_hint for r in results[:window] if r.article_hint is not None
            ]
            # Rank info for expected articles
            rank_map = {art: (found_articles.index(art) + 1) for art in expected_articles if art in found_articles}
            rank_info = ", ".join(f"{art}: {rank_map.get(art, '—')}" for art in expected_articles) if expected_articles else "—"
            # Render all articles with matches highlighted
            rendered_all = []
            for art in all_articles:
                if art in expected_articles:
                    rendered_all.append(f"<span class=\"match\">{escape(art)}</span>")
                else:
                    rendered_all.append(escape(art))
            rendered_all_html = ", ".join(rendered_all) if rendered_all else "—"
            _report(
                test_case["name"],
                query,
                expected_articles,
                found_articles,
                top_articles,
                all_articles,
                missing_articles,
                rank_info,
                rendered_all_html,
                capsys=capsys
            )

            # If expected articles are specified, check that they appear
            if expected_articles:
                assert len(missing_articles) == 0, (
                    f"Test case '{test_case['name']}': "
                    f"Query '{query}' did not return expected articles: {missing_articles}. "
                    f"Found articles: {found_articles}. "
                    f"Expected: {expected_articles}. "
                    f"Description: {test_case.get('description', 'N/A')}"
                )
    
    def test_search_results_contain_query_terms(
        self, 
        production_index: Path, 
        test_cases: List[Dict[str, Any]]
    ):
        """Test that search results are relevant (contain query terms or are semantically related)."""
        retriever = LegalRetriever(
            index_dir=str(production_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        for test_case in test_cases:
            query = test_case["query"]
            top_k = test_case.get("top_k", DEFAULT_TOP_K)
            search_multiplier = test_case.get("search_multiplier", DEFAULT_SEARCH_MULTIPLIER)
            
            results = retriever.search(
                query=query,
                top_k=top_k,
                min_score=0.0,
                search_multiplier=search_multiplier,
                weight_embedding=1.0,
                weight_keyword=1.0
            )
            
            # At least the top result should be relevant
            # Check if query terms appear in top results or if scores are reasonable
            if results:
                top_result = results[0]
                
                # Query should have some relevance - either terms match or score is reasonable
                query_lower = query.lower()
                result_text_lower = top_result.text.lower()
                
                # Check if any significant word from query appears in result
                query_words = [w for w in query_lower.split() if len(w) > 3]
                has_relevance = (
                    any(word in result_text_lower for word in query_words) or
                    top_result.score > 0.3  # Semantic relevance threshold
                )
                
                assert has_relevance, (
                    f"Test case '{test_case['name']}': "
                    f"Top result for query '{query}' seems irrelevant. "
                    f"Score: {top_result.score}, "
                    f"Result text (first 100 chars): {top_result.text[:100]}"
                )
    
    def test_search_with_expected_articles_at_top(
        self, 
        production_index: Path, 
        test_cases: List[Dict[str, Any]]
    ):
        """Test that expected articles appear near the top of the ranked results."""
        retriever = LegalRetriever(
            index_dir=str(production_index),
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        for test_case in test_cases:
            query = test_case["query"]
            expected_articles = test_case.get("expected_articles", [])
            top_k = test_case.get("top_k", DEFAULT_TOP_K)
            search_multiplier = test_case.get("search_multiplier", DEFAULT_SEARCH_MULTIPLIER)
            
            # Skip if no expected articles
            if not expected_articles:
                continue
            
            results = retriever.search(
                query=query,
                top_k=top_k,
                min_score=0.0,
                search_multiplier=search_multiplier,
                weight_embedding=1.0,
                weight_keyword=1.0
            )
            
            # Look at a reasonable window near the top (up to 10 or half of top_k)
            window_size = min(max(5, top_k // 2), 10, len(results))
            top_results = results[:window_size]
            top_articles = [
                r.article_hint 
                for r in top_results 
                if r.article_hint is not None
            ]
            
            # At least one expected article should be in this window
            found_in_top = any(art in top_articles for art in expected_articles)
            
            assert found_in_top, (
                f"Test case '{test_case['name']}': "
                f"Expected articles {expected_articles} not found in top {window_size} results "
                f"for query '{query}'. Top articles: {top_articles}. "
                f"Description: {test_case.get('description', 'N/A')}"
            )

