"""Unit-level tests for helper utilities used in search accuracy reporting.

These tests avoid running heavy searches and instead validate the helper
functions (window sizing, logging, HTML rendering) introduced for the
quality test reporting.
"""

import os
from pathlib import Path

from tests.test_search_accuracy import (
    _window_size,
    _report,
    _write_html_report,
    RESULTS_LOG,
)


def test_window_size_logic():
    # Basic 25% calculation with cap at results length
    assert _window_size(100, 200) == 25
    assert _window_size(10, 4) == 2  # 25% of 10 is 2.5 -> 2, but capped to results_len
    assert _window_size(4, 3) == 1   # max(1, 1) and capped
    assert _window_size(1, 1) == 1   # minimum window is 1


def test_report_logs_success_and_failure(capsys):
    # Reset global log
    RESULTS_LOG.clear()

    # Success case
    _report(
        name="ok-case",
        query="good query",
        expected=["Art. 1."],
        found=["Art. 1.", "Art. 2."],
        top_found=["Art. 1."],
        all_articles=["Art. 1.", "Art. 2."],
        missing=[],
        rank_info="Art. 1.: 1",
        rendered_all="<span class=\"match\">Art. 1.</span>, Art. 2.",
        capsys=capsys,
    )

    # Failure case (missing expected)
    _report(
        name="missing-case",
        query="bad query",
        expected=["Art. 5."],
        found=["Art. 1."],
        top_found=["Art. 1."],
        all_articles=["Art. 1."],
        missing=["Art. 5."],
        rank_info="Art. 5.: —",
        rendered_all="Art. 1.",
        capsys=capsys,
    )

    assert len(RESULTS_LOG) == 2
    ok, missing = RESULTS_LOG
    assert ok["in_results"] is True and ok["in_top"] is True and ok["missing"] == []
    assert missing["in_results"] is False or missing["missing"] == ["Art. 5."]
    # Clean up
    RESULTS_LOG.clear()


def test_write_html_report_renders_columns(tmp_path: Path):
    log = [
        {
            "name": "case-1",
            "query": "q1",
            "expected": ["Art. 1."],
            "in_results": True,
            "in_top": True,
            "top_found": ["Art. 1."],
            "found_all": True,
            "all_articles": '<span class="match">Art. 1.</span>, Art. 2.',
            "missing": [],
            "rank_info": "Art. 1.: 1",
        },
        {
            "name": "case-2",
            "query": "q2",
            "expected": ["Art. 9."],
            "in_results": False,
            "in_top": False,
            "top_found": ["Art. 3."],
            "found_all": False,
            "all_articles": "Art. 3., Art. 4.",
            "missing": ["Art. 9."],
            "rank_info": "Art. 9.: —",
        },
    ]
    out = tmp_path / "report.html"
    _write_html_report(out, log)
    html = out.read_text(encoding="utf-8")

    # Check key headers/columns exist
    assert "Expected Rank" in html
    assert "Top K Articles" in html
    # Check highlighting and pill classes are present
    assert "class=\"match\"" in html
    assert "class=\"pill pass\"" in html or "class=\"pill fail\"" in html
    # Check ranks rendered
    assert "Art. 1.: 1" in html
    assert "Art. 9.: —" in html


