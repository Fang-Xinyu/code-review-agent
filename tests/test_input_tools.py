import hashlib

import pytest

from review_agent.ast_rules import analyze_ast
from review_agent.config import MAX_SOURCE_BYTES, MAX_SOURCE_LINES
from review_agent.tools import EvidenceTools
from review_agent.verification import normalize_and_validate


def test_normalize_and_hash():
    code, digest = normalize_and_validate("x=1\r\ny=2\r", "x.py")
    assert code == "x=1\ny=2\n"
    assert digest == hashlib.sha256(code.encode()).hexdigest()


@pytest.mark.parametrize(("code", "filename", "message"), [
    ("", "x.py", "不能为空"),
    (" \n", "x.py", "不能为空"),
    ("x\x00y", "x.py", "空字节"),
    ("x=1", "x.txt", ".py"),
    ("\n".join("x=1" for _ in range(MAX_SOURCE_LINES + 1)), "x.py", str(MAX_SOURCE_LINES)),
])
def test_input_rejections(code, filename, message):
    with pytest.raises(ValueError, match=message):
        normalize_and_validate(code, filename)


def test_maximum_line_count_is_accepted():
    code = "\n".join("x" for _ in range(MAX_SOURCE_LINES))
    normalized, _ = normalize_and_validate(code, "x.py")
    assert len(normalized.splitlines()) == MAX_SOURCE_LINES


def test_maximum_utf8_size_is_accepted():
    code = "#" + "a" * (MAX_SOURCE_BYTES - 1)
    normalized, _ = normalize_and_validate(code, "x.py")
    assert len(normalized.encode("utf-8")) == MAX_SOURCE_BYTES


def test_utf8_size_over_two_mib_is_rejected():
    code = "#" + "中" * (MAX_SOURCE_BYTES // 3 + 1)
    with pytest.raises(ValueError, match="2 MiB"):
        normalize_and_validate(code, "x.py")


def make_tools():
    code = "def one(x):\n    return x\n\ndef two():\n    return one(2)\n"
    findings, _ = analyze_ast(code)
    return code, EvidenceTools(code, findings)


def test_read_range():
    _, tools = make_tools()
    result = tools.execute("read_source_range", {"start_line": 1, "end_line": 2}, "c1")
    assert result["ok"] and "1 | def one" in result["content"]


def test_read_range_out_of_bounds():
    _, tools = make_tools()
    assert tools.execute("read_source_range", {"start_line": 1, "end_line": 99}, "c1")["error"]["code"] == "invalid_range"


def test_invalid_json_and_unknown_tool():
    _, tools = make_tools()
    assert tools.execute("read_source_range", "{", "c1")["error"]["code"] == "invalid_json"
    assert tools.execute("delete_file", {}, "c2")["error"]["code"] == "unknown_tool"


def test_duplicate_call():
    _, tools = make_tools()
    tools.execute("read_source_range", {"start_line": 1, "end_line": 1}, "c1")
    assert tools.execute("read_source_range", {"start_line": 1, "end_line": 1}, "c2")["error"]["code"] == "duplicate_call"


def test_symbol_context_and_not_found():
    _, tools = make_tools()
    result = tools.execute("get_symbol_context", {"symbol_name": "one"}, "c1")
    assert result["found"] and result["call_lines"] == [5]
    assert tools.execute("get_symbol_context", {"symbol_name": "missing"}, "c2")["error"] == "not_found"


def test_ambiguous_symbol():
    code = "def x(): pass\ndef x(a): pass\n"
    tools = EvidenceTools(code, [])
    assert tools.execute("get_symbol_context", {"symbol_name": "x"}, "c1")["error"] == "ambiguous"


def test_static_query_does_not_rerun():
    code = "x=1/0\n"
    findings, _ = analyze_ast(code)
    tools = EvidenceTools(code, findings)
    result = tools.execute("get_static_findings", {"rule_id": "AST104"}, "c1")
    assert result["count"] == 1
