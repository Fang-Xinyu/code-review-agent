# TraceReview

TraceReview 是一个面向课程项目的 **Python 代码审查 Agent**。它审查单个 Python 文件，用每次必跑的 AST/Ruff 事实作为基线，让 LangGraph Agent 在有界循环中自主追加取证，再由程序验证源码引用和工具来源。高/中风险或证据质量不足的候选会选择性复核。

该实现对应课程要求的可观察状态图、确定性静态检查、证据落地、三级确认状态、追问、局部补丁、导出与离线测试。

## 安装

需要 Python 3.9+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 网页启动

```bash
streamlit run app.py
```

在侧边栏的密码框填写 DeepSeek API Key，根据需要修改 Base URL 和模型名，可先点击“测试连接”。未填 Key 时审查按钮会禁用。Key 只存在当前 Streamlit 会话和短命模型客户端中，不写入文件、图状态、报告、日志或异常正文。

## 命令行使用

```bash
export DEEPSEEK_API_KEY='...'
python -m review_agent review examples/buggy_basic.py --focus "重点检查异常处理"
python -m review_agent review examples/buggy_basic.py --format markdown --output report.md
python -m review_agent review examples/buggy_basic.py --format json --output report.json
```

命令行与网页复用同一个 `ReviewAgent`、LangGraph 图、工具、验证和报告模块。

## 演示流程

1. 加载 `buggy_basic.py`，观察可变默认参数、字面量除零、裸异常与静默忽略。
2. 加载 `buggy_context.py`，观察依赖输入来源的资源／安全启发式如何进入“需要人工确认”。
3. 加载 `clean_example.py`，检查正确保护不被误报。
4. 在“已确认问题”和“需要人工确认”中按检查来源查看结果；严重度标签位于具体问题缩略行最前面，展开后可查看“规则编号：具体含义”及复核结论。
5. 展开执行轨迹和校验记录，导出 Markdown/JSON。
6. 针对某问题生成修复，检查统一 diff 和 Ruff 前后变化。修改编辑区后，旧追问与补丁会禁用。

## 测试与打包

```bash
pytest
python scripts/package_submission.py --output dist/trace-review-agent.zip
```

测试使用脚本化假模型，不访问网络，用于验证程序流程和边界处理，不代表在线 AI 审查效果。

网页与 Markdown 报告使用一致的中文严重度、来源、校验记录和步骤名称，用时以秒显示。JSON 导出保留英文字段、枚举及毫秒数值，供程序读取；源码、规则编号及工具原始诊断保留原文。当前报告在全部审查完成后展示，基础检查结果尚未提前展示。

## 边界与安全

- 只处理一个 `.py` 文件；最多 1000 行、2 MiB UTF-8。两个上限同时生效。
- 仅在线模型审查；无 Key 不伪造成功报告。
- 绝不导入、执行或运行被审查源码/用户测试，不联网分析依赖。
- Ruff 仅以 stdin、`shell=False`、`--isolated --no-cache --select F,E9` 执行。
- 补丁只在内存中组合并供查看/复制，不回写上传文件。语法和静态对比不等于行为正确性证明。
- AST 资源／安全规则是有界启发式；依赖缺失的外部输入、第三方行为、调用约定或性能实验的结论保持“需要人工确认”。
- 模型可用性、限流和 JSON 遵循程度会影响审查；报告会明示失败和覆盖限制。
