from review_agent.ast_rules import analyze_ast
from review_agent.models import FindingDraft, StaticFinding
from review_agent.verification import ground_model_findings, reconcile, static_to_reviewed


def draft(**updates):
    data = dict(title="problem", severity="medium", category="bug", line_start=1, line_end=1,
                evidence="x = risky()", trigger="when called", explanation="may fail", suggestion="guard it", rule_id=None)
    data.update(updates)
    return FindingDraft(**data)


def test_exact_evidence():
    items, records = ground_model_findings("x = risky()\n", [draft()], [])
    assert len(items) == 1 and items[0].line_start == 1


def test_unique_relocation():
    item = draft(line_start=9, line_end=9)
    items, records = ground_model_findings("a=1\nx = risky()\n", [item], [])
    assert items[0].line_start == 2
    assert any(x.action == "relocated" for x in records)


def test_ambiguous_evidence_rejected():
    items, records = ground_model_findings("x = risky()\nx = risky()\n", [draft(line_start=9, line_end=9)], [])
    assert not items and records[0].action == "rejected"


def test_missing_and_discontinuous_evidence_rejected():
    items, _ = ground_model_findings("a=1\nb=2\n", [draft(evidence="a=1\nb=3")], [])
    assert not items


def test_comment_only_rejected():
    items, _ = ground_model_findings("# x = risky()\n", [draft(evidence="x = risky()")], [])
    assert not items


def test_docstring_only_rejected():
    code = '"""x = risky()"""\n'
    items, _ = ground_model_findings(code, [draft(evidence="x = risky()")], [])
    assert not items


def test_comment_and_implementation_evidence_is_accepted():
    code = "# 返回包含右端点的整数列表\ndef numbers(start, end):\n    return list(range(start, end))\n"
    item = draft(title="注释与实现不一致", category="maintainability", severity="low",
                 line_start=1, line_end=3, evidence=code.rstrip("\n"))
    items, _ = ground_model_findings(code, [item], [])
    assert len(items) == 1
    assert items[0].origin == "model" and items[0].status == "needs_confirmation"


def test_docstring_and_implementation_evidence_is_accepted():
    code = 'def numbers(start, end):\n    """包含右端点。"""\n    return list(range(start, end))\n'
    item = draft(category="maintainability", line_start=1, line_end=3, evidence=code.rstrip("\n"))
    items, _ = ground_model_findings(code, [item], [])
    assert len(items) == 1


def test_forged_rule_id_is_unlinked():
    items, records = ground_model_findings("x = risky()\n", [draft(rule_id="AST104")], [])
    assert items[0].rule_id is None and items[0].origin == "model"
    assert any(x.action == "rule_unlinked" for x in records)


def test_matching_rule_gets_static_origin():
    code = "x = 1 / 0\n"
    static, _ = analyze_ast(code)
    item = draft(title="zero", severity="high", evidence="1 / 0", rule_id="AST104")
    items, _ = ground_model_findings(code, [item], static)
    assert items[0].origin == "ast"


def test_static_backfill():
    code = "x = 1 / 0\n"
    static, _ = analyze_ast(code)
    items, records = reconcile(code, [], static)
    assert items[0].status == "static_confirmed"
    assert any(x.action == "backfilled" for x in records)


def test_context_static_becomes_pending_without_changing_severity():
    code = "eval(text)\n"
    static, _ = analyze_ast(code)
    item = static_to_reviewed(static[0].model_copy(update={"severity": "high"}))
    assert item.status == "needs_confirmation" and item.severity == "high"


def test_model_high_stays_high_while_pending():
    items, _ = ground_model_findings("x = risky()\n", [draft(severity="high")], [])
    assert items[0].status == "needs_confirmation" and items[0].severity == "high"


def test_limit_30_records_truncation():
    code = "\n".join(f"x{i} = risky()" for i in range(35))
    drafts = [draft(line_start=i+1, line_end=i+1, evidence=f"x{i} = risky()", title=f"p{i}") for i in range(35)]
    items, records = reconcile(code, drafts, [])
    assert len(items) == 30
    assert any(x.action == "truncated" for x in records)


def test_reconcile_preserves_same_line_distinct_ruff_diagnostics():
    base = dict(title="undefined", severity="medium", category="bug", line_start=1, line_end=1,
                evidence="print(a, b)", trigger="t", suggestion="s", rule_id="F821", origin="ruff", scope="ruff")
    first = StaticFinding(**base, explanation="Undefined name `a`", symbol="a")
    second = StaticFinding(**base, explanation="Undefined name `b`", symbol="b")
    items, _ = reconcile("print(a, b)\n", [], [first, second])
    assert len(items) == 2
