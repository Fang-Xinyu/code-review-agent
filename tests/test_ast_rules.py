import pytest

from review_agent.ast_rules import analyze_ast


@pytest.mark.parametrize(("code", "rule"), [
    ("def f(x=[]):\n return x\n", "AST101"),
    ("def f(x={}):\n return x\n", "AST101"),
    ("def f(x=set()):\n return x\n", "AST101"),
    ("def f(x=tuple()):\n return x\n", None),
    ("try:\n x()\nexcept:\n y()\n", "AST102"),
    ("try:\n x()\nexcept ValueError:\n y()\n", None),
    ("try:\n x()\nexcept ValueError:\n pass\n", "AST103"),
    ("try:\n x()\nexcept ValueError:\n y()\n", None),
    ("x = 1 / 0\n", "AST104"),
    ("x = 1 // 0\n", "AST104"),
    ("x = 1 % 0\n", "AST104"),
    ("x = 1 / divisor\n", None),
    ("def f():\n try:\n  x()\n finally:\n  return 1\n", "AST105"),
    ("def f():\n try:\n  return 1\n finally:\n  cleanup()\n", None),
    ("eval(text)\n", "AST106"),
    ("exec(text)\n", "AST106"),
    ("parser.eval(text)\n", None),
    ("def f(p):\n h=open(p)\n return h.read()\n", "AST107"),
    ("def f(p):\n h=open(p)\n h.close()\n", None),
    ("def f(p):\n with open(p) as h:\n  return h.read()\n", None),
    ("import subprocess\nsubprocess.run(cmd, shell=True)\n", "AST108"),
    ("import subprocess\nsubprocess.run(cmd, shell=False)\n", None),
    ("try:\n x()\nexcept Exception:\n y()\n", "AST109"),
    ("try:\n x()\nexcept RuntimeError:\n y()\n", None),
])
def test_ast_positive_and_correct_negative(code, rule):
    findings, status = analyze_ast(code)
    ids = {x.rule_id for x in findings}
    assert status == "completed"
    if rule:
        assert rule in ids
    else:
        assert not ids


def test_syntax_error():
    findings, status = analyze_ast("def broken(:\n")
    assert status == "syntax_error"
    assert findings[0].rule_id == "AST001"
    assert findings[0].evidence == "def broken(:"


def test_resource_rule_is_context_dependent():
    findings, _ = analyze_ast("h = open('x')\n")
    assert findings[0].needs_context is True


def test_broad_exception_is_low():
    findings, _ = analyze_ast("try:\n x()\nexcept Exception:\n y()\n")
    assert findings[0].severity == "low"


def test_return_inside_nested_function_in_finally_is_not_flagged():
    code = "def outer():\n try:\n  work()\n finally:\n  def inner():\n   return 1\n"
    findings, _ = analyze_ast(code)
    assert "AST105" not in {x.rule_id for x in findings}
