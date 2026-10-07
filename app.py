from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import streamlit as st

from review_agent.config import MAX_SOURCE_BYTES, MAX_SOURCE_LINES, ModelConfig
from review_agent.graph import ReviewAgent
from review_agent.llm import DeepSeekModel, parse_json_object, test_connection
from review_agent.patches import generate_patch
from review_agent.prompts import FOLLOWUP_SYSTEM
from review_agent.reporting import report_json, report_markdown, snapshot_is_stale
from review_agent.ui_text import ACTIONS, ORIGINS, STAGES, TOOL_STATUSES, escape_markdown_tildes, finding_basis
from review_agent.verification import normalize_and_validate


ROOT = Path(__file__).parent
EXAMPLES = {
    "buggy_basic.py": ROOT / "examples" / "buggy_basic.py",
    "buggy_context.py": ROOT / "examples" / "buggy_context.py",
    "clean_example.py": ROOT / "examples" / "clean_example.py",
}
SEVERITY_BADGES = {"high": ":red-badge[高严重度]", "medium": ":orange-badge[中严重度]", "low": ":gray-badge[低严重度]"}


def review_with_progress(model, model_name, code, name, focus):
    updates = Queue()
    started = time.perf_counter()
    stage, round_number = "validate_input", 0
    phase_names = ["输入检查", "基础检查", "AI 分析", "核对问题", "完成"]
    phases = {"validate_input": 0, "static_analysis": 1, "agent_reasoning": 2, "tool_execution": 2,
              "parse_draft": 3, "reconcile_and_ground": 3, "reflection": 3, "finalize_report": 3}
    display = st.empty()
    with ThreadPoolExecutor(max_workers=1) as executor:
        agent = ReviewAgent(model, model_name, progress_callback=lambda s, r: updates.put((s, r)))
        future = executor.submit(agent.review, code, name, focus)
        while True:
            try:
                while True:
                    stage, round_number = updates.get_nowait()
            except Empty:
                pass
            elapsed = time.perf_counter() - started
            current = STAGES[stage]
            if stage == "agent_reasoning":
                current += f"（第 {round_number} 轮）"
            phase = phases[stage]
            steps = " → ".join(f"{label} {'✓' if i < phase else '进行中' if i == phase else ''}".strip()
                               for i, label in enumerate(phase_names))
            display.info(f"正在审查代码 · 当前步骤：{current} · 已用时：{elapsed:.1f} s\n\n{steps}")
            if future.done():
                try:
                    result = future.result()
                except Exception:
                    display.error(f"审查中断 · 已用时：{elapsed:.1f} s · 中断步骤：{current}")
                    raise
                display.success(f"审查完成 · 总用时：{result.usage.total_ms / 1000:.1f} s")
                return result
            time.sleep(0.2)

st.set_page_config(page_title="TraceReview", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "选择文件";
        font-size: 0.875rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("TraceReview")
st.caption("Python 代码审查 Agent")


def init_state() -> None:
    defaults = {"source": EXAMPLES["buggy_basic.py"].read_text(encoding="utf-8"), "filename": "buggy_basic.py",
                "report": None, "chat": [], "patch": None, "patch_hash": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def config_from_sidebar() -> ModelConfig:
    st.sidebar.header("模型配置")
    key = st.sidebar.text_input("API Key", type="password", help="仅保留在当前 Streamlit 会话与模型客户端中")
    base = st.sidebar.text_input("Base URL", "https://api.deepseek.com")
    model = st.sidebar.text_input("模型", "deepseek-flash")
    if st.sidebar.button("测试连接", disabled=not key):
        try:
            st.sidebar.success(test_connection(DeepSeekModel(ModelConfig(api_key=key, base_url=base, model=model))))
        except Exception as exc:
            st.sidebar.error(str(exc))
    if not key:
        st.sidebar.info("请输入 API Key 后开始在线审查。")
    return ModelConfig(api_key=key, base_url=base, model=model)


init_state()
config = config_from_sidebar()

st.subheader("1. 输入代码")
st.caption(f"请选择一种输入方式，最多审查 {MAX_SOURCE_LINES} 行、2 MiB UTF-8 Python 代码。")
input_method = st.radio(
    "代码来源",
    ["内置示例", "上传代码", "直接复制粘贴"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_method == "内置示例":
    selected = st.selectbox("选择示例", list(EXAMPLES))
    example_source = EXAMPLES[selected].read_text(encoding="utf-8")
    example_lines = len(example_source.splitlines())
    st.caption(f"{selected} · {example_lines} 行")
    with st.expander("查看示例代码"):
        st.code(example_source, language="python", line_numbers=True, height=320)
    if st.button("使用此示例", type="primary"):
        st.session_state.source = example_source
        st.session_state.filename = selected
        st.rerun()

elif input_method == "上传代码":
    st.caption("请选择或拖放一个不超过 2 MiB、UTF-8 编码的 .py 文件。")
    upload = st.file_uploader("Python 代码文件", type=["py"], label_visibility="collapsed")
    if upload is not None:
        uploaded_source = None
        uploaded_normalized = None
        upload_error = None
        try:
            uploaded_bytes = upload.getvalue()
            uploaded_lines = 0
            if len(uploaded_bytes) > MAX_SOURCE_BYTES:
                upload_error = f"该文件的 UTF-8 大小超过 2 MiB 上限，无法载入审查。"
            else:
                uploaded_source = uploaded_bytes.decode("utf-8")
                uploaded_lines = len(uploaded_source.replace("\r\n", "\n").replace("\r", "\n").splitlines())
            if upload_error is None and uploaded_lines > MAX_SOURCE_LINES:
                upload_error = f"该文件共 {uploaded_lines} 行，超过 {MAX_SOURCE_LINES} 行上限，无法载入审查。"
            elif upload_error is None:
                uploaded_normalized, _ = normalize_and_validate(uploaded_source, upload.name)
        except UnicodeDecodeError:
            uploaded_lines = 0
            upload_error = "上传文件必须使用 UTF-8 编码。"
        except ValueError:
            upload_error = "代码内容过长或格式不符合要求，无法载入审查。"

        if upload_error:
            st.error(upload_error)
        else:
            st.success(f"已读取 {upload.name} · {uploaded_lines} 行")
        if uploaded_source is not None:
            preview = uploaded_source if uploaded_lines <= MAX_SOURCE_LINES else "\n".join(uploaded_source.splitlines()[:200])
            preview_title = "查看上传代码" if uploaded_lines <= MAX_SOURCE_LINES else "查看前 200 行"
            with st.expander(preview_title):
                st.code(preview, language="python", line_numbers=True, height=320)
        if st.button("载入编辑区", type="primary", disabled=uploaded_normalized is None):
            st.session_state.source = uploaded_normalized
            st.session_state.filename = upload.name
            st.rerun()

else:
    st.info("请在下方编辑区直接粘贴 Python 代码。")

st.markdown("#### 代码预览与编辑")
filename = st.text_input("文件名", key="filename", help="文件名必须以 .py 结尾")
source = st.text_area("Python 代码", key="source", height=420)
source_lines = len(source.replace("\r\n", "\n").replace("\r", "\n").splitlines())
source_bytes = len(source.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
st.caption(f"当前 {source_lines} / {MAX_SOURCE_LINES} 行 · UTF-8 {source_bytes / 1024 / 1024:.2f} / 2.00 MiB")
try:
    _, st.session_state.current_hash = normalize_and_validate(source, filename)
    input_error = None
except ValueError as exc:
    st.session_state.current_hash = None
    input_error = str(exc)
if input_error:
    st.warning(input_error)

st.subheader("2. 审查设置")
focus = st.text_input("审查重点", "全面检查正确性、异常处理、资源和安全")
review_disabled = not bool(config.api_key) or st.session_state.current_hash is None
if st.button("开始审查", type="primary", disabled=review_disabled):
    try:
        with st.spinner("正在审查代码…"):
            st.session_state.report = review_with_progress(DeepSeekModel(config), config.model, source, filename, focus)
            st.session_state.chat = []
            st.session_state.patch = None
            st.session_state.patch_hash = None
    except Exception as exc:
        st.error(f"审查失败：{exc}")

report = st.session_state.report
if report:
    stale = snapshot_is_stale(source, report)
    st.subheader("3. 审查报告")
    if stale:
        st.warning("当前编辑区与报告源码快照不同。报告仍可查看，但追问和旧补丁已禁用；请重新审查。")
    summary = report.summary
    metric_cols = st.columns(3)
    for col, label, value in zip(metric_cols, ["已确认问题", "需要人工确认", "总用时（s）"],
                                  [summary.total_formal, summary.total_pending, f"{report.usage.total_ms / 1000:.1f}"]):
        col.metric(label, value)
    reflection_details = {record.finding_ref: record.detail for record in report.validation_records if record.action == "reflection"}

    def render_finding_group(title, findings, empty_text):
        st.markdown(f"### {title}")
        if not findings:
            st.caption(empty_text)
            return
        for origin in ORIGINS:
            source_findings = [finding for finding in findings if finding.origin == origin]
            if not source_findings:
                continue
            st.markdown(f"#### {ORIGINS[origin]}")
            for finding in source_findings:
                safe_title = escape_markdown_tildes(finding.title)
                label = f"{SEVERITY_BADGES[finding.severity]} {safe_title} · 第 {finding.line_start}-{finding.line_end} 行"
                with st.expander(label, expanded=finding.severity == "high"):
                    if finding.rule_id:
                        st.write(f"**检查依据：** {finding_basis(finding)}")
                    if finding.finding_id in reflection_details:
                        st.write(f"**AI 复核结论：** {escape_markdown_tildes(reflection_details[finding.finding_id])}")
                    if finding.validation_hint:
                        st.write(f"**程序核验依据：** {escape_markdown_tildes(finding.validation_hint)}")
                    st.code(finding.evidence, language="python")
                    st.write(f"**触发：** {escape_markdown_tildes(finding.trigger)}")
                    st.write(f"**解释：** {escape_markdown_tildes(finding.explanation)}")
                    st.write(f"**建议：** {escape_markdown_tildes(finding.suggestion)}")
                    if st.button("生成修复", key=f"fix-{finding.finding_id}", disabled=stale):
                        with st.spinner("正在生成并验证局部补丁…"):
                            result = generate_patch(DeepSeekModel(config), source, f"修复 {finding.title}", filename, finding)
                            st.session_state.patch = result
                            st.session_state.patch_hash = report.source_hash

    render_finding_group("已确认问题", report.findings, "本次没有已确认问题。")
    render_finding_group("需要人工确认", report.pending_findings, "本次没有需要人工确认的问题。")
    for tool in report.tool_statuses:
        if tool.status in {"unavailable", "timeout", "error"}:
            st.warning(escape_markdown_tildes(f"{ORIGINS.get(tool.name, tool.name)}：{TOOL_STATUSES[tool.status]}。{tool.detail}"))
    for limitation in report.limitations:
        st.warning(escape_markdown_tildes(limitation))
    with st.expander("查看审查过程"):
        st.caption("查看各步骤是否完成，以及问题如何通过源码核对与复核。AI 复核通过不代表已运行行为测试。")
        st.markdown("**审查步骤**")
        for event in report.events:
            st.write(escape_markdown_tildes(f"{STAGES.get(event.stage, event.stage)}：{event.message}"))
        st.markdown("**基础检查结果**")
        for tool in report.tool_statuses:
            st.write(escape_markdown_tildes(f"{ORIGINS.get(tool.name, tool.name)}：{TOOL_STATUSES[tool.status]} · 发现 {tool.finding_count} 条问题。{tool.detail}"))
        st.markdown("**问题核对记录**")
        for record in report.validation_records:
            st.write(escape_markdown_tildes(f"{ACTIONS[record.action]}：{record.detail}"))
        if not report.validation_records:
            st.caption("本次没有额外的问题核对记录。")
        st.markdown("**阶段用时**")
        st.table([{"步骤": STAGES.get(stage, stage), "用时（s）": f"{milliseconds / 1000:.2f}"}
                  for stage, milliseconds in report.usage.stage_ms.items()])
        st.caption(f"模型调用 {report.usage.model_calls} 次 · 服务请求 {report.usage.http_attempts} 次 · 补充取证 {report.usage.tool_calls} 次")
    st.download_button("导出 Markdown", report_markdown(report), file_name="trace-review.md", mime="text/markdown")
    st.download_button("导出 JSON", report_json(report), file_name="trace-review.json", mime="application/json")

    st.subheader("4. 追问与修复")
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.write(escape_markdown_tildes(message["content"]))
    question = st.chat_input("围绕当前源码和报告追问", disabled=stale or not bool(config.api_key))
    if question:
        history = st.session_state.chat[-8:]
        try:
            if any(word in question.lower() for word in ("修复", "补丁", "fix", "patch")):
                patch_result = generate_patch(DeepSeekModel(config), source, question, filename)
                st.session_state.patch = patch_result
                st.session_state.patch_hash = report.source_hash
                answer = "已生成并验证补丁，请查看下方结果。" if patch_result.valid else "补丁未通过验证：" + "; ".join(patch_result.errors)
            else:
                payload = {"source": source, "report": report.model_dump(), "history": history, "question": question}
                raw = DeepSeekModel(config).invoke(FOLLOWUP_SYSTEM, json.dumps(payload, ensure_ascii=False)).content
                answer = str(parse_json_object(raw).get("answer", ""))
                if not answer:
                    raise ValueError("追问响应缺少 answer")
            st.session_state.chat = [*history, {"role": "user", "content": question}, {"role": "assistant", "content": answer}][-8:]
            st.rerun()
        except Exception as exc:
            st.error(escape_markdown_tildes(f"追问失败：{exc}"))
    patch = st.session_state.patch
    if patch:
        if stale or st.session_state.patch_hash != report.source_hash:
            st.warning("补丁属于旧源码快照，已禁用预览。")
        elif patch.valid:
            st.success("补丁已通过原文、语法、Ruff 前后比较与模型复核；未运行行为测试。")
            st.code(patch.diff, language="diff")
            st.text_area("可复制的修复后完整代码", patch.patched_code, height=300)
            st.write({"已消失 Ruff 诊断": patch.resolved_diagnostics, "新 Ruff 诊断": patch.new_diagnostics, "复核": patch.model_note})
        else:
            st.error(escape_markdown_tildes("补丁未通过验证，正文已隐藏：" + "; ".join(patch.errors)))
