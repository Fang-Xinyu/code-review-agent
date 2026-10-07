import json

import pytest
from conftest import ScriptedModel, no_ruff
from pydantic import ValidationError
from test_graph_reflection import finding

from review_agent.graph import ReviewAgent
from review_agent.llm import ModelResult
from review_agent.models import ReflectionResponse, ReviewedFinding
from review_agent.prompts import INITIAL_SYSTEM, PATCH_SYSTEM, REFLECTION_SYSTEM
from review_agent.reflection import reflect


def decision(item, verdict="supported", **extra):
    return {"finding_id": item.finding_id, "verdict": verdict, "note": "核对具体执行路径与保护分支", **extra}


def test_no_changes_preserves_all_candidate_content_and_schema_is_real():
    item = finding()
    model = ScriptedModel([{"decisions": [decision(item)]}])
    items, *_ = reflect(model, [item], "x()\n")
    expected = {**item.model_dump(), "status": "model_reviewed", "review_note": "核对具体执行路径与保护分支"}
    assert items[0].model_dump() == expected
    payload = json.loads(model.calls[0][1])
    assert payload["source"] == "x()\n"
    assert payload["reflection_schema"] == ReflectionResponse.model_json_schema()
    assert payload["reflection_schema"]["$defs"]["ReflectionDecision"]["required"] == ["finding_id", "verdict", "note"]


def test_changes_only_updates_explicit_fields_and_revalidates(monkeypatch):
    item = finding()
    model = ScriptedModel([{"decisions": [decision(item, changes={"severity": "low", "suggestion": "检查调用约定"})]}])
    original = ReviewedFinding.model_validate
    validated = []

    def validate(value):
        validated.append(value)
        return original(value)

    monkeypatch.setattr(ReviewedFinding, "model_validate", validate)
    items, *_ = reflect(model, [item])
    expected = {**item.model_dump(), "severity": "low", "suggestion": "检查调用约定",
                "status": "model_reviewed", "review_note": "核对具体执行路径与保护分支"}
    assert items[0].model_dump() == expected
    assert validated == [expected]


@pytest.mark.parametrize("extra", [
    {"title": "旧协议"}, {"changes": None}, {"changes": {"evidence": "伪造"}},
    {"changes": {"line_start": 2}}, {"changes": {"origin": "ast"}},
    {"changes": {"severity": "critical"}}, {"changes": {"severity": None}},
    {"changes": {"title": ""}}, {"changes": {"trigger": None}},
    {"changes": {"explanation": " "}}, {"changes": {"suggestion": 12}},
    {"changes": {"title": "x" * 161}}, {"note": ""}, {"verdict": "yes"},
])
def test_invalid_decisions_rejected(extra):
    with pytest.raises(ValidationError):
        ReflectionResponse.model_validate({"decisions": [decision(finding(), **extra)]})


@pytest.mark.parametrize("verdict,origin,status,expected_status,kept", [
    ("uncertain", "model", "needs_confirmation", "needs_confirmation", True),
    ("rejected", "model", "needs_confirmation", None, False),
    ("rejected", "ast", "static_confirmed", "needs_confirmation", True),
    ("rejected", "ruff", "static_confirmed", "needs_confirmation", True),
    ("uncertain", "ast", "static_confirmed", "static_confirmed", True),
    ("supported", "ast", "needs_confirmation", "model_reviewed", True),
])
def test_confirmation_verdict_does_not_cap_severity(verdict, origin, status, expected_status, kept):
    item = finding(origin=origin, status=status).model_copy(update={"severity": "high"})
    items, *_ = reflect(ScriptedModel([{"decisions": [decision(item, verdict, changes={"severity": "high"})]}]), [item])
    assert bool(items) == kept
    if kept:
        assert items[0].status == expected_status
        assert items[0].severity == "high"


def test_no_patch_instructions_in_review_and_followup_patch_preserved():
    for prompt in (INITIAL_SYSTEM, REFLECTION_SYSTEM):
        assert "original" not in prompt and "replacement" not in prompt and "补丁" not in prompt
    assert "original 必须与对应的完整源码行段完全一致" in PATCH_SYSTEM


def test_reflection_retry_tokens_accumulate():
    item = finding()
    model = ScriptedModel([ModelResult('{"decisions":[]}', [], 10, 3),
                           ModelResult(json.dumps({"decisions": [decision(item)]}), [], 12, 4)])
    _, _, calls, incoming, outgoing = reflect(model, [item])
    assert (calls, incoming, outgoing) == (2, 22, 7)


def test_failed_reflection_calls_tokens_and_per_call_metadata_survive():
    model = ScriptedModel([ModelResult('{"findings":[]}', [], 10, 2, "stop", 1),
                           ModelResult('{"decisions":[]}', [], 20, 3, "length", 2),
                           RuntimeError("service failure")])
    agent = ReviewAgent(model, "fake", no_ruff)
    report = agent.review("x=1/0\n", "x.py")
    assert (report.usage.model_calls, report.usage.http_attempts) == (3, 3)
    assert (report.usage.input_tokens, report.usage.output_tokens) == (30, 5)
    calls = report.usage.calls
    assert len(calls) == 3
    assert calls[0].answer_chars == len('{"findings":[]}')
    assert calls[1].finish_reason == "length" and calls[1].reasoning_tokens == 2
    assert calls[2].failed and calls[2].input_tokens is None
    assert all(x.elapsed_ms >= 0 for x in calls)
    # A second run must not include earlier requests in its HTTP counters.
    second = agent.review("x=1\n", "x.py")
    assert second.usage.model_calls == second.usage.http_attempts == 1


def test_quality_scenarios_keep_grounding_and_verdict_boundaries():
    # Real defect, guarded counterexample, unknown contract: verify pipeline semantics,
    # not the fake model's ability to discover defects.
    cases = [
        ("def f(xs):\n    return xs[0]\n", "supported", "空列表输入会在索引处抛出异常"),
        ("def f(xs):\n    if not xs:\n        return None\n    return xs[0]\n", "rejected", "空列表被已有保护分支阻断"),
        ("def f(xs):\n    return xs[0]\n", "uncertain", "缺少调用约定，无法确定是否允许空列表"),
    ]
    for code, verdict, note in cases:
        line = next(i for i, text in enumerate(code.splitlines(), 1) if "return xs[0]" in text)
        item = finding().model_copy(update={"line_start": line, "line_end": line, "evidence": "    return xs[0]"})
        items, records, *_ = reflect(ScriptedModel([{"decisions": [decision(item, verdict, note=note)]}]), [item], code)
        assert records[0].detail.endswith(note)
        if items:
            assert items[0].evidence in code
            assert items[0].status == ("model_reviewed" if verdict == "supported" else "needs_confirmation")
        else:
            assert verdict == "rejected"


def test_high_model_finding_remains_high_after_supported_review():
    from review_agent.models import FindingDraft
    from review_agent.verification import ground_model_findings

    draft = FindingDraft(title="除零", severity="high", category="bug", line_start=1, line_end=1,
                         evidence="x=1/0", trigger="执行此行", explanation="除数为零", suggestion="检查除数")
    items, _ = ground_model_findings("x=1/0\n", [draft], [])
    assert items[0].severity == "high"
    result, *_ = reflect(ScriptedModel([{"decisions": [decision(items[0], changes={"severity": "high"})]}]), items)
    assert result[0].status == "model_reviewed"
    assert result[0].severity == "high"


def test_json_repair_and_format_failure_keep_usage():
    responses = [ModelResult('bad', [], 3, 1), ModelResult('{"findings":[]}', [], 5, 2)]
    report = ReviewAgent(ScriptedModel(responses), "fake", no_ruff).review("x=1\n")
    assert (report.usage.input_tokens, report.usage.output_tokens, report.usage.model_calls) == (8, 3, 2)
    agent = ReviewAgent(ScriptedModel([ModelResult('bad', [], 3, 1), ModelResult('bad', [], 5, 2)]), "fake", no_ruff)
    with pytest.raises(ValueError):
        agent.review("x=1\n")
    assert agent.model.totals()["input_tokens"] == 8
    assert len(agent.model.calls) == 2


def test_per_call_stats_export_and_old_report_compatibility():
    from review_agent.models import ReviewReport
    from review_agent.reporting import report_json, report_markdown

    report = ReviewAgent(ScriptedModel([ModelResult('{"findings":[]}', [], 9, 2, "stop", 1)]), "fake", no_ruff).review("x=1\n")
    exported = json.loads(report_json(report))
    assert exported["usage"]["calls"][0]["reasoning_tokens"] == 1
    markdown = report_markdown(report)
    assert "逐次模型调用" in markdown and "finish_reason" in markdown
    del exported["usage"]["calls"]
    assert ReviewReport.model_validate(exported).usage.calls == []
