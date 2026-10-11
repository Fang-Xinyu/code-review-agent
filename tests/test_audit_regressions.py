import json
from pathlib import Path

import httpx
import pytest
from conftest import ScriptedModel, no_ruff
from pydantic import ValidationError

from review_agent.cli import main
from review_agent.config import MAX_SOURCE_BYTES, ModelConfig
from review_agent.graph import ReviewAgent
from review_agent.llm import DeepSeekModel, ModelResult, RecordingModel
from review_agent.models import FollowupResponse, ReviewReport
from review_agent.patches import generate_patch
from review_agent.reporting import report_json, report_markdown, snapshot_is_stale
from review_agent.tools import EvidenceTools


EDIT = {'patches': [{'line_start': 1, 'line_end': 1, 'original': 'x=1',
                    'replacement': 'x=2', 'rationale': 'change requested'}]}


def test_full_eight_call_review_with_twenty_four_real_http_attempts(monkeypatch):
    from langchain_openai import ChatOpenAI

    requests = []
    monkeypatch.setattr('review_agent.llm.time.sleep', lambda _: None)

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) % 3:
            return httpx.Response(503, json={'error': {'message': 'test retry'}})
        call = len(requests) // 3
        if call <= 4:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f't{call}', 'type': 'function', 'function': {'name': 'read_source_range',
                    'arguments': json.dumps({'start_line': call, 'end_line': call})}}]}
            finish = 'tool_calls'
        else:
            payload = json.loads(body['messages'][-1]['content'])
            content = 'invalid JSON' if call in (5, 7) else json.dumps({'findings': []})
            if call == 8:
                content = json.dumps({'decisions': [{'finding_id': item['finding_id'],
                    'verdict': 'supported', 'note': 'Literal division by zero.'}
                    for item in payload['candidates']]})
            message = {'role': 'assistant', 'content': content}
            finish = 'stop'
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr('review_agent.llm.ChatOpenAI', lambda **kwargs: ChatOpenAI(**kwargs, http_client=client))
        report = ReviewAgent(DeepSeekModel(ModelConfig(api_key='offline-only')), 'fake').review(
            'x=1/0\n' + 'y=1\n' * 3)
    usage = report.usage
    assert usage.model_calls == usage.model_call_limit == 8
    assert usage.http_attempts == len(requests) == 24
    assert usage.plan_rounds == 5 and usage.initial_stop_reason == 'round_limit'
    assert usage.tool_calls == 4 and usage.tool_call_limit == 20
    assert [item.stage for item in usage.calls] == ['agent_reasoning'] * 5 + ['draft_repair', 'reflection', 'reflection_repair']
    assert all(item.http_attempts == len(item.attempts) == 3 for item in usage.calls)
    assert usage.input_tokens == 80 and usage.output_tokens == 24
    assert all(request.get('tools') for request in requests[:12])
    assert all('tools' not in request for request in requests[12:])
    md = report_markdown(report)
    data = json.loads(report_json(report))
    assert '模型调用：8/8' in md and '初审轮数：5/5' in md
    assert 'HTTP 请求：24' in md and '修正初审格式' in md and '修正复核格式' in md
    assert '## 执行轨迹' in md and '确认情况：规则检查已确认' in md
    assert data['usage']['model_calls'] == len(data['usage']['calls']) == 8
    assert data['schema_version'] == '1.1'
    assert 'finalize_report' in usage.stage_ms
    assert all(event.elapsed_ms is not None for event in report.events)


def test_cli_rejects_raw_crlf_size_before_model_call(monkeypatch, tmp_path, capsys):
    path = tmp_path / 'large.py'
    path.write_bytes(b'#' + b'a' * (MAX_SOURCE_BYTES - 2) + b'\r\n')
    model = ScriptedModel()
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-only')
    monkeypatch.setattr('review_agent.cli.DeepSeekModel', lambda config: model)
    assert main(['review', str(path)]) == 1
    assert '100 KB' in capsys.readouterr().err
    assert not model.calls


@pytest.mark.parametrize('review', [ModelResult('invalid', []), {'verdict': 'supported', 'note': None},
                                    {'verdict': 'supported'}, {'verdict': 'unsupported', 'note': 'bad'}])
def test_invalid_patch_review_never_regenerates_or_exposes_code(review):
    base = ScriptedModel([EDIT, review, EDIT, {'verdict': 'supported', 'note': 'unused'}])
    recorded = RecordingModel(base)
    result = generate_patch(recorded, 'x=1\n', 'change', ruff_runner=no_ruff)
    assert not result.valid and result.diff is None and result.patched_code is None
    assert [call.stage for call in recorded.calls] == ['patch_generation', 'patch_review']
    assert len(base.calls) == 2


def test_patch_generation_repair_has_three_calls_with_correct_stages():
    model = RecordingModel(ScriptedModel([ModelResult('bad', []), EDIT,
        {'verdict': 'supported', 'note': 'Valid local change.'}]))
    result = generate_patch(model, 'x=1\n', 'change', ruff_runner=no_ruff)
    assert result.valid
    assert [call.stage for call in model.calls] == ['patch_generation', 'patch_generation', 'patch_review']


@pytest.mark.parametrize('answer', [None, 3, {}, [], '', '  \n '])
def test_invalid_followup_answer_is_not_coerced_to_success(answer):
    with pytest.raises(ValidationError):
        FollowupResponse.model_validate({'answer': answer})


def test_filename_change_invalidates_report():
    report = ReviewAgent(ScriptedModel(), 'fake', no_ruff).review('x=1\n', 'x.py')
    assert not snapshot_is_stale('x=1\n', report, 'x.py')
    assert snapshot_is_stale('x=1\n', report, 'renamed.py')
    assert any('无需独立复核' in event.message for event in report.events)
    assert report.usage.model_calls == 1


def test_markdown_evidence_containing_backticks_keeps_valid_fence():
    evidence = 'x = "````"'
    candidate = {'title': 'example', 'severity': 'medium', 'category': 'bug', 'line_start': 1,
        'line_end': 1, 'evidence': evidence, 'trigger': 'example', 'explanation': 'example',
        'suggestion': 'example', 'rule_id': None}
    report = ReviewAgent(ScriptedModel([{'findings': [candidate]}]), 'fake', no_ruff).review(evidence)
    md = report_markdown(report)
    assert '`````python\n' + evidence + '\n`````' in md
    assert '确认情况：AI 复核通过' in md


def test_symbol_call_list_omission_is_explicit():
    tools = EvidenceTools('def f():\n    pass\n' + 'f()\n' * 25, [])
    result = tools.execute('get_symbol_context', {'symbol_name': 'f'}, 'a')
    assert result['call_count'] == 25 and len(result['call_lines']) == 20
    assert result['call_lines_truncated'] and result['truncated']


def test_old_json_report_still_loads_with_new_usage_defaults():
    report = ReviewAgent(ScriptedModel(), 'fake', no_ruff).review('x=1\n')
    data = report.model_dump()
    data['schema_version'] = '1.0'
    for key in ('model_call_limit', 'plan_rounds', 'plan_round_limit', 'tool_call_limit', 'initial_stop_reason'):
        data['usage'].pop(key)
    assert ReviewReport.model_validate(data).schema_version == '1.0'


def start_web(monkeypatch, responses=None):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr('review_agent.llm.DeepSeekModel', lambda config: ScriptedModel(responses))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    next(field for field in at.text_input if field.label == 'API Key').set_value('offline-only').run()
    next(button for button in at.button if button.label == '开始审查').click().run(timeout=20)
    assert not at.exception
    return at


def test_failed_second_web_review_clears_previous_report(monkeypatch):
    at = start_web(monkeypatch)
    assert at.session_state['report'] is not None
    monkeypatch.setattr('review_agent.llm.DeepSeekModel', lambda config: ScriptedModel([RuntimeError('offline failure')]))
    next(button for button in at.button if button.label == '开始审查').click().run(timeout=20)
    assert not at.exception
    assert at.session_state['report'] is None
    assert any('审查失败' in item.value for item in at.error)
    assert not at.metric


def test_web_patch_request_failure_is_handled_and_counted(monkeypatch):
    at = start_web(monkeypatch)
    monkeypatch.setattr('review_agent.llm.DeepSeekModel', lambda config: ScriptedModel([RuntimeError('offline failure')]))
    next(button for button in at.button if button.label == '生成修复').click().run(timeout=20)
    assert not at.exception
    assert at.session_state['patch'] is None
    assert at.session_state['interaction_usage']['model_calls'] == 1
    assert any('生成修复失败' in item.value for item in at.error)


def test_web_followup_usage_separate_from_review(monkeypatch):
    at = start_web(monkeypatch)
    before = at.session_state['report'].usage.model_calls
    monkeypatch.setattr('review_agent.llm.DeepSeekModel', lambda config: ScriptedModel([{'answer': '说明'}]))
    at.chat_input[0].set_value('为什么？').run(timeout=20)
    assert not at.exception
    assert at.session_state['chat'][-1]['content'] == '说明'
    assert at.session_state['interaction_usage']['model_calls'] == 1
    assert at.session_state['report'].usage.model_calls == before


def test_web_null_answer_is_rejected(monkeypatch):
    at = start_web(monkeypatch)
    monkeypatch.setattr('review_agent.llm.DeepSeekModel', lambda config: ScriptedModel([{'answer': None}]))
    at.chat_input[0].set_value('为什么？').run(timeout=20)
    assert not at.exception
    assert not at.session_state['chat']
    assert at.session_state['interaction_usage']['model_calls'] == 1
    assert any('追问失败' in item.value for item in at.error)


@pytest.mark.parametrize('example', ['buggy_basic.py', 'buggy_context.py', 'clean_example.py'])
def test_actual_cli_exports_with_real_static_tools(monkeypatch, tmp_path, example):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-only')
    monkeypatch.setattr('review_agent.cli.DeepSeekModel', lambda config: ScriptedModel())
    outputs = {}
    for fmt in ('json', 'markdown'):
        path = tmp_path / f'{example}.{fmt}'
        assert main(['review', str(root / 'examples' / example), '--format', fmt, '--output', str(path)]) == 0
        outputs[fmt] = path.read_text()
    report = ReviewReport.model_validate_json(outputs['json'])
    assert report.source_hash in outputs['markdown']
    assert report.summary.total_formal == len(report.findings)
    assert report.summary.total_pending == len(report.pending_findings)
    assert report.summary.high + report.summary.medium + report.summary.low == len(report.findings) + len(report.pending_findings)
    assert report.usage.model_calls == len(report.usage.calls) <= 8
    assert report.usage.http_attempts == sum(call.http_attempts for call in report.usage.calls)
    assert all(tool.status == 'completed' for tool in report.tool_statuses)
    assert all(event.elapsed_ms is not None for event in report.events)
    assert report_markdown(report)  # Export remains valid after JSON round-trip.


def test_repeated_reviews_reset_budget_and_duplicate_tool_detection():
    tool = ModelResult('', [{'id': 'a', 'name': 'read_source_range', 'args': {'start_line': 1, 'end_line': 1}}])
    base = ScriptedModel([tool, {'findings': []}, tool, {'findings': []}])
    agent = ReviewAgent(base, 'fake', no_ruff)
    for _ in range(2):
        report = agent.review('x=1\n')
        assert report.usage.model_calls == 2 and report.usage.tool_calls == 1
    assert json.loads(base.calls[3][1])['tool_observations'][0]['ok'] is True


def test_web_filename_change_disables_followup_and_patch(monkeypatch):
    at = start_web(monkeypatch)
    next(field for field in at.text_input if field.label == '文件名').set_value('renamed.py').run(timeout=20)
    assert not at.exception
    assert at.chat_input[0].disabled
    assert all(button.disabled for button in at.button if button.label == '生成修复')
    assert any('文件名或源码' in item.value for item in at.warning)


def test_web_removing_key_disables_patch_buttons(monkeypatch):
    at = start_web(monkeypatch)
    next(field for field in at.text_input if field.label == 'API Key').set_value('').run(timeout=20)
    assert not at.exception
    assert at.chat_input[0].disabled
    assert all(button.disabled for button in at.button if button.label == '生成修复')


def test_real_adapter_returns_malformed_tool_arguments_to_executor(monkeypatch):
    from langchain_openai import ChatOpenAI

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'bad', 'type': 'function',
                'function': {'name': 'read_source_range', 'arguments': '{'}}]}
            finish = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': '{"findings":[]}'}
            finish = 'stop'
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr('review_agent.llm.ChatOpenAI', lambda **kwargs: ChatOpenAI(**kwargs, http_client=client))
        report = ReviewAgent(DeepSeekModel(ModelConfig(api_key='offline-only')), 'fake', no_ruff).review('x=1\n')
    assert report.usage.model_calls == 2 and report.usage.tool_calls == 1
    payload = json.loads(requests[1]['messages'][-1]['content'])
    assert payload['tool_observations'][0]['error']['code'] == 'invalid_json'
    assert report.usage.calls[1].stage == 'agent_reasoning'


def test_web_migrates_report_instance_kept_before_code_reload(monkeypatch):
    from types import SimpleNamespace

    at = start_web(monkeypatch)
    data = at.session_state['report'].model_dump()
    data['schema_version'] = '1.0'
    for key in ('model_call_limit', 'plan_rounds', 'plan_round_limit', 'tool_call_limit', 'initial_stop_reason'):
        data['usage'].pop(key)
    at.session_state['report'] = SimpleNamespace(model_dump=lambda: data)
    at.run(timeout=20)
    assert not at.exception
    assert at.session_state['report'].schema_version == '1.0'
    assert any('旧版本报告' in item.value for item in at.info)
