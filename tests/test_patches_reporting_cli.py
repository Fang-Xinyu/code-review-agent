import json

from conftest import ScriptedModel, no_ruff
from review_agent.cli import main
from review_agent.graph import ReviewAgent
from review_agent.models import PatchEdit
from review_agent.patches import apply_and_validate_patches, generate_patch
from review_agent.reporting import report_json, report_markdown, snapshot_is_stale
from review_agent.ast_rules import analyze_ast
from review_agent.verification import static_to_reviewed


def patch(start=1, end=1, original="x=1", replacement="x=2"):
    return PatchEdit(line_start=start, line_end=end, original=original, replacement=replacement, rationale="fix")


def test_patch_out_of_range():
    assert not apply_and_validate_patches("x=1\n", [patch(2, 2)], ruff_runner=no_ruff).valid


def test_patch_overlap():
    patches = [patch(1, 1), patch(1, 1)]
    assert "重叠" in ";".join(apply_and_validate_patches("x=1\n", patches, ruff_runner=no_ruff).errors)


def test_patch_original_mismatch():
    assert "不匹配" in ";".join(apply_and_validate_patches("x=1\n", [patch(original="x=0")], ruff_runner=no_ruff).errors)


def test_patch_syntax_error_hidden():
    result = apply_and_validate_patches("x=1\n", [patch(replacement="x=")], ruff_runner=no_ruff)
    assert not result.valid and result.patched_code is None


def test_patch_success_diff():
    result = apply_and_validate_patches("x=1\n", [patch()], ruff_runner=no_ruff)
    assert result.valid and "-x=1" in result.diff and "+x=2" in result.diff


def test_patch_requires_target_ast_diagnostic_to_disappear():
    code = "x=1/0\n"
    target = static_to_reviewed(analyze_ast(code)[0][0])
    unchanged = apply_and_validate_patches(code, [patch(original="x=1/0", replacement="x = 1 / 0")], target=target, ruff_runner=no_ruff)
    fixed = apply_and_validate_patches(code, [patch(original="x=1/0", replacement="x = 1 / 2")], target=target, ruff_runner=no_ruff)
    assert not unchanged.valid
    assert fixed.valid


def test_shifted_unfixed_ast_target_is_rejected():
    code = "x=1/0\n"
    target = static_to_reviewed(analyze_ast(code)[0][0])
    result = apply_and_validate_patches(code, [patch(original="x=1/0", replacement="# shifted\nx=1/0")],
                                        target=target, ruff_runner=no_ruff)
    assert not result.valid and result.patched_code is None


def test_ast_target_fixed_while_same_rule_elsewhere_shifts():
    code = "x=1/0\ny=2/0\n"
    target = static_to_reviewed(analyze_ast(code)[0][0])
    result = apply_and_validate_patches(code, [patch(original="x=1/0", replacement="# fixed\nx=1/2")],
                                        target=target, ruff_runner=no_ruff)
    assert result.valid


def test_existing_ruff_warning_moving_after_insert_or_delete_is_not_new():
    from review_agent.ruff_adapter import run_ruff

    for code, edit in [
        ("x=1\nimport json\n", patch(replacement="# shifted\nx=2")),
        ("x=1\nimport json\n", patch(replacement="")),
    ]:
        result = apply_and_validate_patches(code, [edit], ruff_runner=run_ruff)
        assert result.valid and not result.new_diagnostics and not result.resolved_diagnostics


def test_fix_one_ruff_target_with_another_same_rule_remaining():
    from review_agent.ruff_adapter import run_ruff

    code = "import json\nimport csv\n"
    target = static_to_reviewed(run_ruff(code)[0][0])
    result = apply_and_validate_patches(code, [patch(original="import json", replacement="")],
                                        target=target, ruff_runner=run_ruff)
    assert result.valid and len(result.resolved_diagnostics) == 1 and not result.new_diagnostics


def test_duplicate_existing_warning_in_replacement_is_new():
    from review_agent.ruff_adapter import run_ruff

    code = "print(missing)\n"
    result = apply_and_validate_patches(code, [patch(original="print(missing)", replacement="print(missing)\nprint(missing)")],
                                        ruff_runner=run_ruff)
    assert not result.valid and len(result.new_diagnostics) == 1


def test_patch_new_ruff_diagnostic_fails():
    from review_agent.models import StaticFinding, ToolStatus
    def runner(code, filename):
        findings = []
        if "bad" in code:
            findings = [StaticFinding(title="F821", severity="medium", category="bug", line_start=1, line_end=1,
                evidence="bad", trigger="t", explanation="Undefined name `bad`", suggestion="s", rule_id="F821",
                origin="ruff", scope="test", symbol="bad")]
        return findings, ToolStatus(name="ruff", status="completed", finding_count=len(findings))
    result = apply_and_validate_patches("x=1\n", [patch(replacement="x=bad")], ruff_runner=runner)
    assert not result.valid and result.new_diagnostics


def test_generate_patch_reviewed_success():
    model = ScriptedModel([
        {"patches":[{"line_start":1,"line_end":1,"original":"x=1","replacement":"x=2","rationale":"fix"}]},
        {"verdict":"supported","note":"looks local"},
    ])
    result = generate_patch(model, "x=1\n", "change", ruff_runner=no_ruff)
    assert result.valid and result.model_note == "looks local"


def test_generate_patch_corrects_once_then_hides():
    bad = {"patches":[{"line_start":2,"line_end":2,"original":"z=0","replacement":"z=1","rationale":"fix"}]}
    result = generate_patch(ScriptedModel([bad, bad]), "x=1\n", "change", ruff_runner=no_ruff)
    assert not result.valid and result.patched_code is None


def test_markdown_json_same_report_and_no_secret():
    report = ReviewAgent(ScriptedModel(), "fake", no_ruff).review("x=1\n", "x.py")
    js = report_json(report)
    md = report_markdown(report)
    assert json.loads(js)["source_hash"] in md
    assert "api_key" not in js.lower() and "api_key" not in md.lower()


def test_markdown_uses_chinese_labels_and_preserves_json_schema():
    report = ReviewAgent(ScriptedModel(), "fake", no_ruff).review("x=1/0\n", "x.py")
    md = report_markdown(report)
    assert "【高严重度】" in md
    assert "AST104：字面量除零" in md
    assert "AI 复核结论：" in md
    assert "Python 语法结构检查（AST）" in md
    assert "基础检查：" in md and " 秒" in md
    assert "static_confirmed" not in md and "stage_ms" not in md
    assert json.loads(report_json(report))["findings"][0]["status"] == "static_confirmed"


def test_snapshot_staleness():
    report = ReviewAgent(ScriptedModel(), "fake", no_ruff).review("x=1\n", "x.py")
    assert not snapshot_is_stale("x=1\n", report)
    assert snapshot_is_stale("x=2\n", report)


def test_cli_missing_key_exit(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["review", "missing.py"]) == 2
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err


def test_cli_success(monkeypatch, tmp_path, capsys):
    source = tmp_path / "x.py"
    source.write_text("x=1\n")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.setattr("review_agent.cli.DeepSeekModel", lambda config: ScriptedModel())
    assert main(["review", str(source), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["filename"] == "x.py"


def test_package_excludes_credentials(tmp_path):
    from scripts.package_submission import allowed
    (tmp_path / ".env").write_text("SECRET=x")
    assert not allowed(tmp_path / ".env", tmp_path)
    assert allowed(tmp_path / ".env.example", tmp_path)
