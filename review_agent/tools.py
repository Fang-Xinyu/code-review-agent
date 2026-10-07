from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import StaticFinding


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadRangeArgs(ToolArgs):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class SymbolArgs(ToolArgs):
    symbol_name: str = Field(min_length=1, max_length=200)


class StaticQueryArgs(ToolArgs):
    severity: Optional[Literal["high", "medium", "low"]] = None
    rule_id: Optional[str] = None


TOOL_SCHEMAS = [
    {"name": "read_source_range", "description": "读取当前内存源码的有界行范围", "parameters": ReadRangeArgs.model_json_schema()},
    {"name": "get_symbol_context", "description": "查找唯一函数或类的定义与调用上下文", "parameters": SymbolArgs.model_json_schema()},
    {"name": "get_static_findings", "description": "查询本次已执行的 AST/Ruff 结果", "parameters": StaticQueryArgs.model_json_schema()},
]


class EvidenceTools:
    def __init__(self, code: str, findings: list[StaticFinding], max_chars: int = 6000):
        self.code = code
        self.lines = code.splitlines()
        self.findings = findings
        self.max_chars = max_chars
        self.seen: Counter[str] = Counter()

    def execute(self, name: str, arguments: str | dict[str, Any], call_id: str) -> dict[str, Any]:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                raise ValueError("参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error(call_id, "invalid_json", str(exc))
        signature = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
        self.seen[signature] += 1
        if self.seen[signature] > 1:
            return self._error(call_id, "duplicate_call", "相同工具和参数已调用，不重复返回内容")
        mapping: dict[str, tuple[type[ToolArgs], Callable[[ToolArgs], dict[str, Any]]]] = {
            "read_source_range": (ReadRangeArgs, self._range),
            "get_symbol_context": (SymbolArgs, self._symbol),
            "get_static_findings": (StaticQueryArgs, self._static),
        }
        if name not in mapping:
            return self._error(call_id, "unknown_tool", f"未知工具: {name}")
        schema, function = mapping[name]
        try:
            parsed = schema.model_validate(args)
            result = function(parsed)
        except ValidationError as exc:
            return self._error(call_id, "invalid_arguments", exc.errors(include_url=False)[0]["msg"])
        except ValueError as exc:
            return self._error(call_id, "invalid_range", str(exc))
        result.update({"ok": True, "tool_call_id": call_id, "tool": name})
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded) > self.max_chars:
            result = {"ok": True, "tool_call_id": call_id, "tool": name, "truncated": True, "content": encoded[: self.max_chars]}
        return result

    @staticmethod
    def _error(call_id: str, code: str, message: str) -> dict[str, Any]:
        return {"ok": False, "tool_call_id": call_id, "error": {"code": code, "message": message}}

    def _range(self, args: ReadRangeArgs) -> dict[str, Any]:
        if args.end_line < args.start_line or args.end_line > len(self.lines) or args.end_line - args.start_line + 1 > 120:
            raise ValueError("行号必须在源码内，顺序正确且一次不超过 120 行")
        content = "\n".join(f"{i:4d} | {self.lines[i - 1]}" for i in range(args.start_line, args.end_line + 1))
        return {"start_line": args.start_line, "end_line": args.end_line, "content": content}

    def _symbol(self, args: SymbolArgs) -> dict[str, Any]:
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return {"found": False, "error": "syntax_error"}
        definitions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == args.symbol_name]
        if len(definitions) != 1:
            return {"found": False, "error": "not_found" if not definitions else "ambiguous", "matches": len(definitions)}
        node = definitions[0]
        calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == args.symbol_name) or (isinstance(n.func, ast.Attribute) and n.func.attr == args.symbol_name))][:20]
        start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
        lo, hi = max(1, start - 3), min(len(self.lines), end + 3)
        return {"found": True, "kind": type(node).__name__, "start_line": start, "end_line": end,
                "signature": self.lines[start - 1][:500], "call_lines": calls,
                "context": "\n".join(f"{i:4d} | {self.lines[i - 1]}" for i in range(lo, hi + 1))}

    def _static(self, args: StaticQueryArgs) -> dict[str, Any]:
        selected = [f for f in self.findings if (not args.severity or f.severity == args.severity) and (not args.rule_id or f.rule_id == args.rule_id)]
        return {"count": len(selected), "findings": [f.model_dump() for f in selected[:30]], "truncated": len(selected) > 30}
