import ast
import json
from pathlib import Path

import pytest
from conftest import ScriptedModel, no_ruff

from review_agent.ast_rules import analyze_ast
from review_agent.graph import ReviewAgent
from review_agent.models import FindingDraft, PatchEdit
from review_agent.patches import apply_and_validate_patches
from review_agent.reporting import report_json, report_markdown
from review_agent.ruff_adapter import run_ruff
from review_agent.tools import EvidenceTools
from review_agent.verification import ground_model_findings, reconcile


@pytest.mark.parametrize("code", ["print(a, b)\n", "print(a, a)\n"])
def test_same_line_ruff_findings_have_distinct_ids_and_complete_reflection(code):
    report = ReviewAgent(ScriptedModel(), "fake").review(code)
    findings = report.findings
    assert len(findings) == len({item.finding_id for item in findings}) == 2
    decisions = [record for record in report.validation_records if record.action == "reflection"]
    assert {record.finding_ref for record in decisions} == {item.finding_id for item in findings}
    assert not report.limitations
    assert report.usage.model_calls == 2
    assert len(json.loads(report_json(report))["findings"]) == 2
    assert all(item.finding_id in report_markdown(report) for item in findings)


def test_web_same_line_findings_leave_export_and_followup_available(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("review_agent.llm.DeepSeekModel", lambda config: ScriptedModel())
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)
    next(field for field in app.text_input if field.label == "API Key").set_value("offline-only").run()
    next(field for field in app.text_area if field.label == "Python 代码").set_value("print(a, b)\n").run()
    next(button for button in app.button if button.label == "开始审查").click().run(timeout=20)
    assert not app.exception
    fixes = [button for button in app.button if button.label == "生成修复"]
    assert len(fixes) == len({button.key for button in fixes}) == 2
    assert len(app.get("download_button")) == 2
    assert not app.chat_input[0].disabled


@pytest.mark.parametrize("code", ["x=1/0; y=2/0\n", "x=1/0; y=1/0\n"])
def test_same_line_ast_expressions_survive_merge(code):
    static, _ = analyze_ast(code)
    findings, _ = reconcile(code, [], static)
    assert len(static) == len(findings) == 2
    assert len({item.finding_id for item in findings}) == 2
    # Repeated observations of one diagnostic are still deduplicated.
    assert len(reconcile(code, [], [*static, static[0]])[0]) == 2


@pytest.mark.parametrize("code", ['s="value=%d" % 0\n', 's=f"value=%d" % 0\n', "def f(template):\n    return template % 0\n"])
def test_string_or_unknown_modulo_is_not_confirmed_division_by_zero(code):
    assert not any(item.rule_id == "AST104" for item in analyze_ast(code)[0])


@pytest.mark.parametrize("expression", ["1 % 0", "-1 % 0", "+1.5 % 0", "value / 0", "value // 0"])
def test_numeric_zero_divisor_is_still_detected(expression):
    assert any(item.rule_id == "AST104" for item in analyze_ast(f"result = {expression}\n")[0])


@pytest.mark.parametrize("code", ["def f():\n\n", "def f():\n    ", "(\n", "if True:\n\n"])
def test_syntax_errors_with_empty_or_eof_locations_still_produce_report(code):
    report = ReviewAgent(ScriptedModel(), "fake").review(code)
    finding = next(item for item in report.findings if item.rule_id == "AST001")
    assert finding.evidence.strip() and finding.evidence in code
    assert report.tool_statuses[0].status == "syntax_error"


def candidate(evidence, line=1):
    return FindingDraft(title="未定义名称", severity="high", category="bug",
        line_start=line, line_end=line, evidence=evidence, trigger="执行到此处",
        explanation="名称未定义", suggestion="定义名称")


def test_executable_evidence_is_not_rejected_due_to_matching_inline_comment():
    code = "x = missing # x = missing\n"
    findings, records = ground_model_findings(code, [candidate("x = missing")], [])
    assert len(findings) == 1 and not any(record.action == "rejected" for record in records)
    # The full comment is still rejected, including surrounding indentation.
    assert not ground_model_findings(code, [candidate(" # x = missing")], [])[0]


def test_docstring_filter_uses_utf8_columns_and_keeps_executable_evidence():
    code = 'def f():\n    """中文 missing()"""\n    missing()\n'
    assert not ground_model_findings(code, [candidate("missing()", 2)], [])[0]
    assert len(ground_model_findings(code, [candidate("missing()", 3)], [])[0]) == 1


def test_missing_findings_requires_repair_but_explicit_empty_list_does_not():
    model = ScriptedModel([{}, {"findings": []}])
    report = ReviewAgent(model, "fake", no_ruff).review("x=1\n")
    assert [call.stage for call in report.usage.calls] == ["agent_reasoning", "draft_repair"]
    assert not report.findings and not report.pending_findings
    assert "findings" in json.loads(model.calls[1][1])["schema"]["required"]
    explicit = ReviewAgent(ScriptedModel([{"findings": []}]), "fake", no_ruff).review("x=1\n")
    assert explicit.usage.model_calls == 1


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85", "\v", "\f"])
def test_source_tools_and_patches_preserve_string_characters(separator):
    code = f'text = """a{separator}b"""\ny=1\n'
    tools = EvidenceTools(code, [])
    line = tools.execute("read_source_range", {"start_line": 2, "end_line": 2}, "read")
    assert line["content"] == "   2 | y=1"
    edit = PatchEdit(line_start=2, line_end=2, original="y=1", replacement="y=2", rationale="更新 y")
    patched = apply_and_validate_patches(code, [edit])
    assert patched.valid and patched.patched_code == code.replace("y=1", "y=2")
    assert ast.parse(patched.patched_code).body[0].value.value == f"a{separator}b"
    # Characters introduced by the requested replacement are preserved too.
    replacement = f'text = """c{separator}d"""'
    edit = PatchEdit(line_start=1, line_end=1, original=code.split("\n")[0],
                     replacement=replacement, rationale="更新 text")
    assert apply_and_validate_patches(code, [edit]).patched_code == replacement + "\ny=1\n"


def test_model_and_ruff_evidence_share_python_line_numbers():
    code = 'text = "a\u2028b"\nprint(missing)\n'
    model = ScriptedModel()
    report = ReviewAgent(model, "fake").review(code)
    numbered = json.loads(model.calls[0][1])["source"]
    assert numbered == '   1 | text = "a\u2028b"\n   2 | print(missing)'
    assert report.findings[0].line_start == 2
    assert report.findings[0].evidence == "print(missing)"


def test_installed_ruff_syntax_diagnostic_is_high_and_marked_as_syntax_error():
    findings, status = run_ruff("x =\n")
    assert status.status == "syntax_error"
    assert findings and all(item.severity == "high" for item in findings)


