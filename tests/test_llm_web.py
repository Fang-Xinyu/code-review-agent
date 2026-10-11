from pathlib import Path

import pytest

from review_agent.config import ModelConfig
from review_agent.llm import DeepSeekModel, ModelRequestError


class HttpError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class FailingClient:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0
    def invoke(self, messages):
        self.calls += 1
        error = self.errors.pop(0)
        raise error


def test_model_defaults_and_client_limits(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("review_agent.llm.ChatOpenAI", fake_client)
    config = ModelConfig(api_key="placeholder")
    DeepSeekModel(config)
    assert config.timeout == 40.0
    assert config.max_tokens == 4096
    assert config.max_retries == 2
    assert captured["timeout"] == 40.0
    assert captured["max_tokens"] == 4096
    assert captured["max_retries"] == 0  # retries remain managed by DeepSeekModel
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in captured


def test_thinking_disabled_in_http_requests_for_tools_review_and_reflection(monkeypatch):
    import json

    import httpx
    from conftest import no_ruff
    from langchain_openai import ChatOpenAI

    from review_agent.graph import ReviewAgent

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) <= 2:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call-{len(requests)}", "type": "function", "function": {
                    "name": "read_source_range",
                    "arguments": json.dumps({"start_line": 1, "end_line": 1}),
                },
            }]}
            finish_reason = "tool_calls"
        else:
            payload = json.loads(body["messages"][-1]["content"])
            content = {"findings": []}
            if "candidates" in payload:
                content = {"decisions": [{"finding_id": item["finding_id"],
                    "verdict": "supported", "note": "Division by literal zero."}
                    for item in payload["candidates"]]}
            message = {"role": "assistant", "content": json.dumps(content)}
            finish_reason = "stop"
        return httpx.Response(200, json={"id": "offline-response", "object": "chat.completion",
            "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr("review_agent.llm.ChatOpenAI",
            lambda **kwargs: ChatOpenAI(**kwargs, http_client=http_client))
        model = DeepSeekModel(ModelConfig(api_key="offline-test-key"))
        report = ReviewAgent(model, model.config.model, no_ruff).review("x = 1 / 0\n", "x.py")

    assert len(requests) == 4  # two tool decisions, final draft, independent reflection
    assert all(body["thinking"] == {"type": "disabled"} for body in requests)
    assert all("reasoning_effort" not in body for body in requests)
    assert all("extra_body" not in body for body in requests)  # merged into the HTTP JSON body
    assert all(body.get("tools") for body in requests[:2])
    assert requests[2].get("tools")  # early final response may still have tools available
    assert "tools" not in requests[3]  # independent reflection has no tools
    assert [call.stage for call in report.usage.calls] == ["agent_reasoning"] * 3 + ["reflection"]
    assert report.usage.tool_calls == 2
    assert any(record.action == "reflection" for record in report.validation_records)
    assert any(item.rule_id == "AST104" for item in report.findings)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_non_retryable_http_status(status):
    model = DeepSeekModel(ModelConfig(api_key="placeholder", max_retries=2))
    model.client = FailingClient([HttpError(status)])
    with pytest.raises(ModelRequestError, match="配置"):
        model.invoke("s", "u")
    assert model.attempts == 1


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_http_status_retries(monkeypatch, status):
    monkeypatch.setattr("review_agent.llm.time.sleep", lambda _: None)
    model = DeepSeekModel(ModelConfig(api_key="placeholder", max_retries=2))
    model.client = FailingClient([HttpError(status), HttpError(status), HttpError(status)])
    with pytest.raises(ModelRequestError, match="不可用"):
        model.invoke("s", "u")
    assert model.attempts == 3


def test_streamlit_initial_screen_has_disabled_review_button():
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=20)
    assert not at.exception
    buttons = {button.label: button for button in at.button}
    assert buttons["开始审查"].disabled is True
    assert buttons["测试连接"].disabled is True
    assert at.title[0].value == "TraceReview"
    assert any("100 KB（102400 字节） UTF-8" in caption.value for caption in at.caption)


def test_streamlit_review_displays_chinese_filters_and_seconds(monkeypatch):
    from conftest import ScriptedModel
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("review_agent.llm.DeepSeekModel", lambda config: ScriptedModel())
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=20)
    next(field for field in at.text_input if field.label == "API Key").set_value("offline-test-key").run()
    next(button for button in at.button if button.label == "开始审查").click().run(timeout=20)
    assert not at.exception
    assert not at.multiselect
    metric_labels = {metric.label for metric in at.metric}
    assert metric_labels == {"已确认问题", "需要人工确认", "总用时（s）"}
    assert any(metric.label == "总用时（s）" for metric in at.metric)
    assert any("已确认问题" in heading.value for heading in at.markdown)
    assert any(expander.label.startswith(":red-badge[高严重度]") for expander in at.expander)
    assert any(expander.label == "查看审查过程" for expander in at.expander)
    assert any("审查完成" in message.value and " s" in message.value for message in at.success)


def test_retry_success_metadata_and_request_attempts(monkeypatch):
    from types import SimpleNamespace

    from review_agent.llm import RecordingModel

    monkeypatch.setattr("review_agent.llm.time.sleep", lambda _: None)
    model = DeepSeekModel(ModelConfig(api_key="secret-not-in-stats", max_retries=2))

    class RetryClient:
        count = 0

        def invoke(self, messages):
            self.count += 1
            if self.count == 1:
                raise HttpError(429)
            return SimpleNamespace(content='{"ok":true}', tool_calls=[],
                usage_metadata={"input_tokens": 25, "output_tokens": 8, "output_token_details": {"reasoning": 5}},
                response_metadata={"finish_reason": "stop"}, additional_kwargs={"reasoning_content": "private-reasoning"})

    model.client = RetryClient()
    recorded = RecordingModel(model)
    recorded.invoke("s", "u")
    assert recorded.totals() == {"model_calls": 1, "http_attempts": 2, "input_tokens": 25, "output_tokens": 8}
    call = recorded.calls[0]
    assert len(call.attempts) == 2
    assert call.attempts[0].failed and call.attempts[0].input_tokens is None
    assert call.attempts[1].reasoning_tokens == 5
    assert call.answer_chars == 11 and call.finish_reason == "stop"
    assert "secret-not-in-stats" not in call.model_dump_json()
    assert "private-reasoning" not in call.model_dump_json()


def test_exhausted_http_retries_record_failure(monkeypatch):
    from review_agent.llm import RecordingModel

    monkeypatch.setattr("review_agent.llm.time.sleep", lambda _: None)
    model = DeepSeekModel(ModelConfig(api_key="placeholder", max_retries=2))
    model.client = FailingClient([HttpError(503)] * 3)
    recorded = RecordingModel(model)
    with pytest.raises(ModelRequestError):
        recorded.invoke("s", "u")
    assert recorded.totals() == {"model_calls": 1, "http_attempts": 3, "input_tokens": None, "output_tokens": None}
    assert len(recorded.calls[0].attempts) == 3
    assert recorded.calls[0].failed
