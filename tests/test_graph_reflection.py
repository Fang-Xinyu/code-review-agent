import json

import pytest
from conftest import ScriptedModel, no_ruff

from review_agent.graph import ReviewAgent
from review_agent.llm import ModelResult
from review_agent.models import ReviewedFinding
from review_agent.reflection import reflect


def test_graph_direct_end_and_static_backfill():
    model = ScriptedModel()
    report = ReviewAgent(model, "fake", no_ruff).review("x = 1 / 0\n", "x.py")
    assert any(x.rule_id == "AST104" for x in report.findings)
    assert report.usage.model_calls == 2  # initial + reflection
    assert report.tool_statuses[0].name == "ast"
    reflection_payload = json.loads(model.calls[1][1])
    assert reflection_payload["source"] == "x = 1 / 0\n"
    assert reflection_payload["candidates"]


def test_legal_long_default_evidence_does_not_abort_review():
    from review_agent.verification import normalize_and_validate

    code = "def f(items=['" + "x" * 9000 + "']):\n    return items\n"
    normalize_and_validate(code, "long.py")
    report = ReviewAgent(ScriptedModel(), "fake", no_ruff).review(code, "long.py")
    item = next(x for x in report.findings if x.rule_id == "AST101")
    assert len(item.evidence) == 8000 and item.evidence in code
    assert item.line_start == item.line_end == 1
    assert "证据较长" in item.validation_hint


def test_graph_normal_tool_loop():
    model = ScriptedModel([
        ModelResult("", [{"id":"t1","name":"read_source_range","args":{"start_line":1,"end_line":1}}]),
        {"findings": []},
    ])
    report = ReviewAgent(model, "fake", no_ruff).review("x = 1\n", "x.py")
    assert report.usage.tool_calls == 1
    assert any(x.stage == "tool_execution" for x in report.events)


def test_progress_reports_stage_before_model_and_tracks_tool_rounds():
    progress = []

    class ProgressModel(ScriptedModel):
        def invoke(self, system, user, tools=None):
            assert progress[-1] == ("agent_reasoning", len(self.calls) + 1)
            return super().invoke(system, user, tools)

    model = ProgressModel([
        ModelResult("", [{"id": "t1", "name": "read_source_range", "args": {"start_line": 1, "end_line": 1}}]),
        {"findings": []},
    ])
    report = ReviewAgent(model, "fake", no_ruff, progress_callback=lambda s, r: progress.append((s, r))).review("x=1\n", "x.py")
    assert progress[0][0] == "validate_input"
    assert ("tool_execution", 1) in progress
    assert ("agent_reasoning", 2) in progress
    assert progress[-1][0] == "finalize_report"
    assert report.usage.model_calls == 2


def test_graph_illegal_tool_observation_counts():
    model = ScriptedModel([
        ModelResult("", [{"id":"t1","name":"delete_file","args":{}}]),
        {"findings": []},
    ])
    report = ReviewAgent(model, "fake", no_ruff).review("x = 1\n", "x.py")
    assert report.usage.tool_calls == 1


@pytest.mark.parametrize("requested", [4, 5, 10])
def test_graph_tool_limit_per_round_and_second_round_can_use_tools(requested):
    calls = [{"id":f"t{i}","name":"read_source_range","args":{"start_line":i + 1,"end_line":i + 1}} for i in range(requested)]
    model = ScriptedModel([ModelResult("", calls), ModelResult("", [{"id":"next","name":"read_source_range","args":{"start_line":11,"end_line":11}}]), {"findings": []}])
    report = ReviewAgent(model, "fake", no_ruff).review("x = 1\n" * 11, "x.py")
    assert report.usage.tool_calls == min(requested, 5) + 1
    assert model.calls[1][2] is not None
    assert model.calls[2][2] is None
    observations = json.loads(model.calls[1][1])["tool_observations"]
    executed = [x for x in observations if x["ok"]]
    skipped = [x for x in observations if not x["ok"]]
    assert len(executed) == min(requested, 5)
    assert len(skipped) == max(0, requested - 5)
    assert all(x["error"]["code"] == "tool_call_limit" for x in skipped)
    assert [x["tool_call_id"] for x in executed] == [f"t{i}" for i in range(min(requested, 5))]
    assert bool(report.limitations) == (requested > 5)


def test_graph_round_limit_forces_final():
    responses = [ModelResult("", [{"id":f"t{i}","name":"read_source_range","args":{"start_line":1,"end_line":1}}]) for i in range(2)]
    responses.append(ModelResult("", [{"id":"late","name":"read_source_range","args":{"start_line":1,"end_line":1}}]))
    report = ReviewAgent(ScriptedModel(responses), "fake", no_ruff).review("x=1\n", "x.py")
    assert report.usage.model_calls == 3
    assert any("轮次上限" in x for x in report.limitations)


def test_bad_json_repaired_once():
    model = ScriptedModel([ModelResult("not json", []), {"findings": []}])
    report = ReviewAgent(model, "fake", no_ruff).review("x=1\n", "x.py")
    assert report.usage.model_calls == 2


def test_bad_json_twice_fails_review():
    model = ScriptedModel([ModelResult("bad", []), ModelResult("still bad", [])])
    with pytest.raises(ValueError, match="两次失败"):
        ReviewAgent(model, "fake", no_ruff).review("x=1\n", "x.py")


def finding(fid="F-123456789abc", origin="model", status="needs_confirmation"):
    return ReviewedFinding(title="x", severity="medium", category="bug", line_start=1, line_end=1, evidence="x()",
        trigger="call", explanation="bad", suggestion="fix", rule_id=None, finding_id=fid, origin=origin,
        status=status, review_note="", validation_hint="")


@pytest.mark.parametrize("bad", [
    {"decisions": []},
    {"decisions": [{"finding_id":"F-000000000000","verdict":"supported","note":"具体依据"}]},
])
def test_reflection_missing_or_new_id_fails_after_repair(bad):
    model = ScriptedModel([bad, bad])
    with pytest.raises(ValueError, match="ID"):
        reflect(model, [finding()])


def test_reflection_duplicate_id_fails():
    decision = {"finding_id":"F-123456789abc","verdict":"supported","note":"具体依据"}
    model = ScriptedModel([{"decisions":[decision, decision]}, {"decisions":[decision, decision]}])
    with pytest.raises(ValueError, match="ID"):
        reflect(model, [finding()])


def test_reflection_supported_model_becomes_reviewed():
    model = ScriptedModel()
    items, _, _, _, _ = reflect(model, [finding()])
    assert items[0].status == "model_reviewed"


def test_reflection_rejected_static_is_preserved():
    item = finding(origin="ast", status="static_confirmed")
    d = {"finding_id":item.finding_id,"verdict":"rejected","note":"具体依据"}
    items, *_ = reflect(ScriptedModel([{"decisions":[d]}]), [item])
    assert len(items) == 1
    assert items[0].status == "needs_confirmation"


def test_reflection_supported_contextual_static_becomes_reviewed():
    item = finding(origin="ast", status="needs_confirmation")
    d = {"finding_id":item.finding_id,"verdict":"supported","note":"源码足以支持"}
    items, *_ = reflect(ScriptedModel([{"decisions":[d]}]), [item])
    assert items[0].status == "model_reviewed"


def test_graph_reflection_failure_changes_confirmation_but_preserves_severity():
    candidate = {"title":"other","severity":"high","category":"bug","line_start":1,"line_end":1,"evidence":"x=1",
                 "trigger":"call","explanation":"maybe","suggestion":"check","rule_id":None}
    model = ScriptedModel([{"findings":[candidate]}, {"decisions":[]}, {"decisions":[]}])
    report = ReviewAgent(model, "fake", no_ruff).review("x=1\ny=1/0\n", "x.py")
    assert report.pending_findings
    assert report.pending_findings[0].severity == "high"
    assert any(x.rule_id == "AST104" for x in report.findings)
