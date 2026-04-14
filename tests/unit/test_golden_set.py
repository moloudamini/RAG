"""Unit tests for the SQL golden set evaluator."""

import pytest
from src.evaluation.golden_set import _score_entry, _aggregate_by_tag, _load_golden_set, _normalize_result


# --- _score_entry ---

def make_entry(**kwargs):
    base = {
        "id": "test_entry",
        "query": "How many companies are there?",
        "expected_sql_keywords": ["SELECT", "COUNT", "companies"],
        "expected_columns": ["count"],
        "expected_row_count": 1,
        "tags": ["aggregation"],
    }
    base.update(kwargs)
    return base


def test_score_entry_perfect_match():
    entry = make_entry()
    sql = "SELECT COUNT(*) AS count FROM companies;"
    sql_result = {"columns": ["count"], "data": [{"count": 5}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result)

    assert result["passed"] is True
    assert result["overall"] == pytest.approx(1.0)
    assert result["scores"]["execution_success"] == 1.0
    assert result["scores"]["safety"] == 1.0
    assert result["scores"]["keyword_coverage"] == 1.0


def test_score_entry_execution_error():
    entry = make_entry()
    sql = "SELECT COUNT(*) FROM companies;"
    sql_result = {"error": "relation does not exist", "data": [], "row_count": 0}
    result = _score_entry(entry, sql, sql_result)

    assert result["scores"]["execution_success"] == 0.0
    assert result["scores"]["column_match"] == 0.0
    assert result["passed"] is False


def test_score_entry_dangerous_sql():
    entry = make_entry(expected_sql_keywords=["SELECT"])
    sql = "DROP TABLE companies;"
    sql_result = {"columns": [], "data": [], "row_count": 0}
    result = _score_entry(entry, sql, sql_result)

    assert result["scores"]["safety"] == 0.0
    assert result["passed"] is False


def test_score_entry_wrong_row_count():
    entry = make_entry(expected_row_count=1)
    sql = "SELECT COUNT(*) FROM companies;"
    sql_result = {"columns": ["count"], "data": [{"count": 3}, {"count": 2}], "row_count": 2}
    result = _score_entry(entry, sql, sql_result)

    assert result["scores"]["row_count_match"] == 0.0


def test_score_entry_row_count_minimum():
    entry = make_entry(expected_row_count_min=1)
    del entry["expected_row_count"]
    sql = "SELECT name FROM companies;"
    sql_result = {"columns": ["name"], "data": [{"name": "Acme"}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result)

    assert result["scores"]["row_count_match"] == 1.0


def test_score_entry_partial_keyword_match():
    entry = make_entry(expected_sql_keywords=["SELECT", "COUNT", "companies", "WHERE"])
    sql = "SELECT COUNT(*) FROM companies;"  # missing WHERE
    sql_result = {"columns": ["count"], "data": [{"count": 5}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result)

    assert result["scores"]["keyword_coverage"] == pytest.approx(3 / 4)


def test_score_entry_no_expected_columns():
    entry = make_entry(expected_columns=[])
    sql = "SELECT name FROM companies;"
    sql_result = {"columns": ["name"], "data": [{"name": "Acme"}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result)

    assert result["scores"]["column_match"] == 1.0


# --- result_match via reference_result ---

def test_score_entry_result_match_exact():
    entry = make_entry()
    sql = "SELECT COUNT(*) AS count FROM companies;"
    data = [{"count": 5}]
    sql_result = {"columns": ["count"], "data": data, "row_count": 1}
    reference_result = {"columns": ["count"], "data": data, "row_count": 1}
    result = _score_entry(entry, sql, sql_result, reference_result)

    assert result["scores"]["result_match"] == 1.0
    assert result["passed"] is True


def test_score_entry_result_match_mismatch():
    entry = make_entry()
    sql = "SELECT COUNT(*) AS count FROM companies;"
    sql_result = {"columns": ["count"], "data": [{"count": 3}], "row_count": 1}
    reference_result = {"columns": ["count"], "data": [{"count": 5}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result, reference_result)

    assert result["scores"]["result_match"] == 0.0


def test_score_entry_result_match_ordering_ignored():
    """Rows in different order should still match."""
    entry = make_entry(expected_row_count_min=0)
    del entry["expected_row_count"]
    sql = "SELECT name FROM companies;"
    sql_result = {
        "columns": ["name"],
        "data": [{"name": "Acme"}, {"name": "Beta"}],
        "row_count": 2,
    }
    reference_result = {
        "columns": ["name"],
        "data": [{"name": "Beta"}, {"name": "Acme"}],
        "row_count": 2,
    }
    result = _score_entry(entry, sql, sql_result, reference_result)

    assert result["scores"]["result_match"] == 1.0


def test_score_entry_result_match_column_case_ignored():
    """Column name casing differences should not cause mismatch."""
    entry = make_entry()
    sql = "SELECT COUNT(*) AS Count FROM companies;"
    sql_result = {"columns": ["Count"], "data": [{"Count": 5}], "row_count": 1}
    reference_result = {"columns": ["count"], "data": [{"count": 5}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result, reference_result)

    assert result["scores"]["result_match"] == 1.0


def test_score_entry_no_result_match_when_reference_missing():
    """result_match dimension is skipped when no reference_result provided."""
    entry = make_entry()
    sql = "SELECT COUNT(*) AS count FROM companies;"
    sql_result = {"columns": ["count"], "data": [{"count": 5}], "row_count": 1}
    result = _score_entry(entry, sql, sql_result, reference_result=None)

    assert "result_match" not in result["scores"]


def test_score_entry_no_result_match_when_reference_errored():
    """result_match is skipped when the reference SQL itself failed."""
    entry = make_entry()
    sql = "SELECT COUNT(*) AS count FROM companies;"
    sql_result = {"columns": ["count"], "data": [{"count": 5}], "row_count": 1}
    reference_result = {"error": "syntax error", "data": [], "row_count": 0}
    result = _score_entry(entry, sql, sql_result, reference_result)

    assert "result_match" not in result["scores"]


# --- _normalize_result ---

def test_normalize_result_lowercases_keys():
    data = [{"Name": "Acme", "COUNT": 5}]
    assert _normalize_result(data) == [{"name": "Acme", "count": 5}]


def test_normalize_result_sorts_rows():
    data = [{"name": "Beta"}, {"name": "Acme"}]
    normalized = _normalize_result(data)
    assert normalized[0] == {"name": "Acme"}
    assert normalized[1] == {"name": "Beta"}


# --- fixture: reference_sql ---

def test_golden_set_all_entries_have_reference_sql():
    entries = _load_golden_set()
    for entry in entries:
        assert "reference_sql" in entry, f"{entry['id']} missing reference_sql"
        assert entry["reference_sql"].strip().upper().startswith("SELECT"), \
            f"{entry['id']} reference_sql must be a SELECT statement"


# --- _aggregate_by_tag ---

def test_aggregate_by_tag():
    results = [
        {"tags": ["aggregation", "products"], "overall": 1.0},
        {"tags": ["aggregation"], "overall": 0.5},
        {"tags": ["products"], "overall": 0.8},
    ]
    agg = _aggregate_by_tag(results)

    assert agg["aggregation"]["count"] == 2
    assert agg["aggregation"]["avg_score"] == pytest.approx(0.75)
    assert agg["aggregation"]["pass_rate"] == pytest.approx(0.5)
    assert agg["products"]["count"] == 2
    assert agg["products"]["pass_rate"] == pytest.approx(1.0)


def test_aggregate_by_tag_empty():
    assert _aggregate_by_tag([]) == {}


# --- fixture file ---

def test_golden_set_fixture_loads():
    entries = _load_golden_set()
    assert len(entries) > 0
    for entry in entries:
        assert "id" in entry
        assert "query" in entry
        assert "tags" in entry


def test_golden_set_all_entries_have_sql_keywords():
    entries = _load_golden_set()
    for entry in entries:
        assert "expected_sql_keywords" in entry, f"{entry['id']} missing expected_sql_keywords"
        assert len(entry["expected_sql_keywords"]) > 0
