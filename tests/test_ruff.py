import pytest
import json
import subprocess
from unittest.mock import patch

from review_agent.ruff_adapter import diagnostic_key, parse_ruff_json, run_ruff


def payload(code="F821", row=1, message="Undefined name `x`"):
    return json.dumps([{"code": code, "message": message, "location": {"row": row, "column": 1},
                        "end_location": {"row": row, "column": 2}, "fix": None}])


def test_parse_ruff_preserves_rule_symbol_position():
    finding = parse_ruff_json(payload(), "x\n")[0]
    assert diagnostic_key(finding)[:3] == ("F821", "x", 1)


def test_same_line_distinct_symbols_are_not_same_key():
    data = json.dumps([
        {"code":"F821","message":"Undefined name `a`","location":{"row":1,"column":1},"end_location":{"row":1,"column":2}},
        {"code":"F821","message":"Undefined name `b`","location":{"row":1,"column":4},"end_location":{"row":1,"column":5}},
    ])
    findings = parse_ruff_json(data, "a, b\n")
    assert len({diagnostic_key(x) for x in findings}) == 2


@patch("review_agent.ruff_adapter.subprocess.run")
def test_ruff_completed(mock_run):
    mock_run.return_value = subprocess.CompletedProcess([], 1, payload(), "")
    findings, status = run_ruff("x\n")
    assert status.status == "completed" and len(findings) == 1
    assert mock_run.call_args.kwargs["shell"] is False
    assert mock_run.call_args.kwargs["encoding"] == "utf-8"


@patch("review_agent.ruff_adapter.subprocess.run", side_effect=FileNotFoundError)
def test_ruff_unavailable(_):
    assert run_ruff("x=1\n")[1].status == "unavailable"


@patch("review_agent.ruff_adapter.subprocess.run", side_effect=subprocess.TimeoutExpired("ruff", 1))
def test_ruff_timeout(_):
    assert run_ruff("x=1\n")[1].status == "timeout"


@patch("review_agent.ruff_adapter.subprocess.run")
def test_ruff_bad_json(mock_run):
    mock_run.return_value = subprocess.CompletedProcess([], 1, "not-json", "")
    assert run_ruff("x=1\n")[1].status == "error"


@patch("review_agent.ruff_adapter.subprocess.run")
def test_ruff_abnormal_exit(mock_run):
    mock_run.return_value = subprocess.CompletedProcess([], 2, "[]", "failure")
    assert run_ruff("x=1\n")[1].status == "error"


@patch("review_agent.ruff_adapter.subprocess.run")
def test_ruff_syntax_status(mock_run):
    mock_run.return_value = subprocess.CompletedProcess([], 1, payload("E999", message="Syntax error"), "")
    assert run_ruff("x=\n")[1].status == "syntax_error"


@pytest.mark.parametrize("rule,severity,category", [
    ("F401", "low", "maintainability"),
    ("F541", "low", "maintainability"),
    ("F841", "low", "maintainability"),
    ("F821", "high", "bug"),
    ("F822", "high", "bug"),
    ("F823", "high", "bug"),
    ("E999", "high", "bug"),
    ("F601", "medium", "bug"),
])
def test_ruff_impact_levels(rule, severity, category):
    item = parse_ruff_json(payload(rule), "x\n")[0]
    assert (item.severity, item.category) == (severity, category)


def test_long_ruff_line_keeps_diagnostic_in_bounded_source_excerpt():
    code = "x = '" + "a" * 9000 + "'; print(missing)\n"
    data = json.loads(payload())
    data[0]["location"]["column"] = code.index("missing") + 1
    item = parse_ruff_json(json.dumps(data), code)[0]
    assert item.evidence.startswith("missing") and item.evidence in code
    assert len(item.evidence) <= 8000
    assert "证据较长" in item.scope
