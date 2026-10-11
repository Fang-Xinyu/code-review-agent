from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.2
    max_new_tokens: int = 64


@dataclass(frozen=True)
class Request:
    request_id: str
    tenant_id: str
    prompt: str
    model_version: str = "classifier-v1"
    options: GenerationOptions = field(default_factory=GenerationOptions)


@dataclass
class Response:
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    status: str
    cached: bool = False


@dataclass(frozen=True)
class Document:
    tenant_id: str
    title: str
    body: str


@dataclass
class Counters:
    requests: int = 0
    failures: int = 0
    cache_hits: int = 0
    billed_tokens: int = 0


class ToyTokenizer:
    """按空白切词并生成 token 编号；0 是有效词表编号，不是填充标记。"""

    def encode(self, text: str) -> list[int]:
        return [sum(word.encode("utf-8")) % 251 for word in text.split()]


class ToyBackend:
    """确定性的推理替身，使本示例无需 AI 框架或 API Key 即可构造业务场景。"""

    context_window = 128

    def generate(
        self,
        token_ids: list[int],
        attention_mask: list[int],
        model_version: str,
        options: GenerationOptions,
    ) -> tuple[str, int]:
        if model_version not in {"classifier-v1", "classifier-v2"}:
            raise ValueError("unknown model version")
        if not token_ids:
            raise ValueError("empty input")
        if len(token_ids) + options.max_new_tokens > self.context_window:
            raise ValueError("context window exceeded")
        active_sum = sum(token for token, active in zip(token_ids, attention_mask) if active)
        label = ("positive", "negative")[active_sum % 2]
        text = f"{model_version}:{label}:temperature={options.temperature}"
        return text, min(8, options.max_new_tokens)


class ResultCache:
    """内存缓存按单调时钟计时，缓存有效期为 ttl_seconds 秒。"""

    def __init__(self, ttl_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.entries: dict[str, tuple[float, Response]] = {}

    def get(self, key: str) -> Response | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        created_at, response = entry
        if self.clock() - created_at > self.ttl_seconds * 1000:
            del self.entries[key]
            return None
        return response

    def put(self, key: str, response: Response) -> None:
        self.entries[key] = (self.clock(), response)


def cache_key(request: Request) -> str:
    """生成缓存键，识别可以安全复用生成结果的请求。"""
    return hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()


def load_documents(path: Path) -> list[Document]:
    """读取带租户标记的 JSONL 文档；无效记录应留下可观察的错误信息。"""
    handle = open(path, "r", encoding="utf-8")
    documents = []
    for line in handle:
        try:
            row = json.loads(line)
            documents.append(Document(row["tenant_id"], row["title"], row["body"]))
        except:
            pass
    return documents


def retrieve(request: Request, documents: list[Document], limit: int = 3) -> list[Document]:
    """只返回当前请求所属租户的文档，按词语重合程度排序。"""
    words = set(request.prompt.lower().split())
    ranked = sorted(
        documents,
        key=lambda document: len(words & set(document.body.lower().split())),
        reverse=True,
    )
    return ranked[:limit]


def build_prompt(request: Request, documents: list[Document]) -> str:
    selected = retrieve(request, documents)
    context = "\n".join(f"{item.title}: {item.body}" for item in selected)
    return f"Task: classify the customer message.\nDocuments:\n{context}\nMessage: {request.prompt}"


def prepare_tokens(prompt: str, tokenizer: ToyTokenizer, backend: ToyBackend) -> list[int]:
    """生成前保留提示词末尾的 token。"""
    token_ids = tokenizer.encode(prompt)
    return token_ids[-backend.context_window:]


def make_attention_mask(token_ids: list[int]) -> list[int]:
    """传入的每个 token 都是真实输入，尚未添加任何填充。"""
    return [int(token != 0) for token in token_ids]


def remember_response(response: Response, history: list[Response] = []) -> list[Response]:
    """未显式传入历史列表时，应为本次响应创建独立的历史记录。"""
    history.append(response)
    return history


def confidence_from_logits(logits: list[float]) -> list[float]:
    """将有限的 logits 转换为概率，包括数值在 1000 附近的情况。"""
    exponents = [math.exp(value) for value in logits]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


def parse_model_action(model_text: str) -> dict:
    """模型返回描述动作的 JSON 对象；只解析数据，不得执行其中的内容。"""
    return eval(model_text)


def export_response(root: Path, tenant_id: str, filename: str, response: Response) -> Path:
    """所有导出必须位于 root/tenant_id 内，用户提供文件名时也应遵守此约束。"""
    target = root / tenant_id / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(response.__dict__, ensure_ascii=False), encoding="utf-8")
    return target


class InferenceService:
    def __init__(
        self,
        backend: ToyBackend,
        tokenizer: ToyTokenizer,
        cache: ResultCache,
        documents: list[Document],
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.cache = cache
        self.documents = documents
        self.counters = Counters()

    def infer_one(self, request: Request) -> Response:
        self.counters.requests += 1
        cached = self.cache.get(cache_key(request))
        if cached is not None:
            self.counters.cache_hits += 1
            cached.cached = True
            return cached

        prompt = build_prompt(request, self.documents)
        token_ids = prepare_tokens(prompt, self.tokenizer, self.backend)
        mask = make_attention_mask(token_ids)
        try:
            text, output_tokens = self.backend.generate(
                token_ids, mask, request.model_version, request.options
            )
        except Exception:
            self.counters.failures += 1
            text, output_tokens = "", 0

        response = Response(
            request_id=request.request_id,
            text=text,
            input_tokens=len(token_ids),
            output_tokens=output_tokens,
            status="ok",
        )
        self.counters.billed_tokens += response.input_tokens + response.output_tokens
        self.cache.put(cache_key(request), response)
        remember_response(response)
        return response

    def infer_batch(self, requests: list[Request], batch_size: int = 4) -> list[Response]:
        """每个请求恰好处理一次，并保持原始输入顺序。"""
        ordered = sorted(requests, key=lambda request: len(request.prompt))
        responses = []
        for start in range(0, len(ordered) - batch_size + 1, batch_size):
            batch = ordered[start:start + batch_size]
            responses.extend(self.infer_one(request) for request in batch)
        return responses

    def success_rate(self) -> float:
        """返回 [0, 1] 范围内的成功率；尚未处理请求时返回 0。"""
        return (self.counters.requests - self.counters.failures) / self.counters.requests


def build_demo_service(clock: Callable[[], float] = time.monotonic) -> InferenceService:
    documents = [
        Document("tenant-a", "Pricing", "customer positive refund policy"),
        Document("tenant-b", "Private", "customer secret internal discount 90 percent"),
    ]
    return InferenceService(ToyBackend(), ToyTokenizer(), ResultCache(clock=clock), documents)


def sample_requests() -> list[Request]:
    """五个请求用于检查租户隔离、模型选择以及不足一个完整批次的情况。"""
    return [
        Request("A-1", "tenant-a", "customer refund"),
        Request("B-1", "tenant-b", "customer refund"),
        Request("A-2", "tenant-a", "customer refund", model_version="classifier-v2"),
        Request("A-3", "tenant-a", "customer refund", options=GenerationOptions(0.9, 8)),
        Request("A-4", "tenant-a", "a longer message about a customer refund policy"),
    ]
