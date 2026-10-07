from __future__ import annotations

import json
from typing import Any

from review_agent.llm import ModelResult


class ScriptedModel:
    def __init__(self, responses: list[Any] | None = None):
        self.responses = list(responses or [])
        self.calls = []
        self.attempts = 0

    def invoke(self, system: str, user: str, tools=None) -> ModelResult:
        self.calls.append((system, user, tools))
        self.attempts += 1
        if self.responses:
            value = self.responses.pop(0)
            if isinstance(value, Exception):
                raise value
            if isinstance(value, ModelResult):
                return value
            return ModelResult(content=json.dumps(value, ensure_ascii=False), tool_calls=[])
        if "independent-reflection" in system:
            payload = json.loads(user)
            decisions = []
            for item in payload["candidates"]:
                decisions.append({"finding_id": item["finding_id"], "verdict": "supported",
                                  "note": "supported by scripted reflection"})
            return ModelResult(content=json.dumps({"decisions": decisions}), tool_calls=[])
        return ModelResult(content='{"findings":[]}', tool_calls=[])


def no_ruff(code: str, filename: str = "input.py"):
    from review_agent.models import ToolStatus
    return [], ToolStatus(name="ruff", status="completed", detail="test", finding_count=0)
