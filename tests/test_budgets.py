import json

import pytest
from conftest import ScriptedModel, no_ruff

from review_agent.ast_rules import analyze_ast
from review_agent.config import MAX_MODEL_CALLS, MAX_SOURCE_BYTES
from review_agent.graph import ReviewAgent
from review_agent.llm import ModelBudgetExceeded, ModelResult, RecordingModel
from review_agent.tools import EvidenceTools
from review_agent.verification import normalize_and_validate


def test_full_review_uses_eight_calls_including_both_format_repairs():
    tools = [ModelResult('', [{'id': str(i), 'name': 'read_source_range',
        'args': {'start_line': i, 'end_line': i}}]) for i in range(1, 5)]
    model = ScriptedModel([*tools, ModelResult('bad draft', []), {'findings': []},
                           ModelResult('bad reflection', [])])
    report = ReviewAgent(model, 'fake', no_ruff).review('x=1/0\n' + 'y=1\n' * 3)
    assert report.usage.model_calls == MAX_MODEL_CALLS == 8
    assert report.usage.tool_calls == 4
    assert all(call[2] for call in model.calls[:4])
    assert all(call[2] is None for call in model.calls[4:])
    assert any(item.rule_id == 'AST104' for item in report.findings)


def test_guard_counts_failures_and_blocks_ninth_call_before_network():
    base = ScriptedModel([RuntimeError('failed')])
    model = RecordingModel(base)
    with pytest.raises(RuntimeError, match='failed'):
        model.invoke('s', 'u')
    for _ in range(7):
        model.invoke('s', 'u')
    with pytest.raises(ModelBudgetExceeded):
        model.invoke('s', 'u')
    assert len(base.calls) == len(model.calls) == 8
    assert model.calls[0].failed
    assert model.remaining == 0


def test_four_rounds_allow_twenty_tools_then_force_final():
    responses = [ModelResult('', [{'id': str(i), 'name': 'read_source_range',
        'args': {'start_line': i, 'end_line': i}} for i in range(lo, lo + 5)])
        for lo in (1, 6, 11, 16)]
    base = ScriptedModel([*responses, {'findings': []}])
    report = ReviewAgent(base, 'fake', no_ruff).review('x=1\n' * 20)
    assert report.usage.model_calls == 5
    assert report.usage.tool_calls == 20
    assert base.calls[4][2] is None


def test_utf8_size_checked_before_newline_normalization():
    code = '# ' + '中' * ((MAX_SOURCE_BYTES - 2) // 3)
    normalize_and_validate(code, 'x.py')
    with pytest.raises(ValueError, match='100 KB'):
        normalize_and_validate(code + '中', 'x.py')
    with pytest.raises(ValueError, match='100 KB'):
        normalize_and_validate('#' + 'a' * (MAX_SOURCE_BYTES - 2) + '\r\n', 'x.py')


def test_source_truncation_preserves_lines_and_continuation():
    code = '\n'.join('# ' + '\\"中' * 15 for _ in range(20))
    tools = EvidenceTools(code, [], max_chars=700)
    first = tools.execute('read_source_range', {'start_line': 1, 'end_line': 20}, 'a')
    assert len(json.dumps(first, ensure_ascii=False)) <= 700
    assert first['truncated'] and not first.get('partial_item')
    assert first['next_start_line'] == first['returned_end_line'] + 1
    assert len(first['content'].splitlines()) == first['returned_end_line']
    second = tools.execute('read_source_range', {'start_line': first['next_start_line'], 'end_line': 20}, 'b')
    assert second['returned_start_line'] == first['next_start_line']


def test_error_and_single_long_line_are_bounded():
    tools = EvidenceTools('# ' + '中' * 10000, [], max_chars=512)
    for result in [tools.execute('x' * 10000, {}, 'id'),
                   tools.execute('read_source_range', {'start_line': 1, 'end_line': 1}, 'id')]:
        assert len(json.dumps(result, ensure_ascii=False)) <= 512
        assert result['truncated']
    assert result['partial_item']


def test_static_and_symbol_pagination():
    code = 'x=1/0\n' * 40
    findings, _ = analyze_ast(code)
    tools = EvidenceTools(code, findings)
    first = tools.execute('get_static_findings', {}, 'a')
    assert first['truncated'] and first['next_offset'] > 0
    second = tools.execute('get_static_findings', {'offset': first['next_offset']}, 'b')
    assert first['findings'][-1]['line_start'] < second['findings'][0]['line_start']
    symbol_tools = EvidenceTools('def f():\n' + '    pass\n' * 200, [])
    first = symbol_tools.execute('get_symbol_context', {'symbol_name': 'f'}, 'a')
    assert first['truncated'] and first['next_start_line'] == 121
    second = symbol_tools.execute('get_symbol_context', {'symbol_name': 'f',
        'context_start_line': first['next_start_line']}, 'b')
    assert second['returned_start_line'] == 121
    assert second['next_start_line'] is None


def test_truncated_tool_result_is_recorded_in_report():
    base = ScriptedModel([ModelResult('', [{'id': 'a', 'name': 'read_source_range',
        'args': {'start_line': 1, 'end_line': 1}}]), {'findings': []}])
    report = ReviewAgent(base, 'fake', no_ruff).review('#' + 'a' * 10000)
    assert any('输出已截断' in detail for detail in report.limitations)
