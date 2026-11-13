"""公共 API ↔ 运行器适配器（文档 §5.1、§6）。

适配器只负责把公共 API 的输入输出映射到运行器；新运行器或全新接口须先作「待适配」，
适配器进入项目并经过测试后再试。首版登记 llama.cpp server 的 OpenAI 兼容接口。
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol


class Adapter(Protocol):
    name: str

    def chat_payload(self, messages: list[dict[str, Any]], *, stream: bool = False,
                     temperature: float | None = None, max_tokens: int | None = None,
                     model: str = "local-model") -> dict[str, Any]: ...

    def parse_response(self, data: dict[str, Any], public_model: str,
                       backend: str) -> dict[str, Any]: ...

    def translate_stream(self, line: str, public_model: str,
                         meta: dict[str, Any] | None = None) -> str | None: ...


class LlamaCppOpenAIAdapter:
    """llama.cpp llama-server 的 /v1/chat/completions 与 /health 接口。"""

    name = "llamacpp-openai"
    health_path = "/health"
    chat_path = "/v1/chat/completions"
    models_path = "/v1/models"

    def chat_payload(self, messages, *, stream=False, temperature=None,
                     max_tokens=None, model="local-model") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": stream,
            "cache_prompt": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["n_predict"] = max_tokens
            payload["max_tokens"] = max_tokens
        return payload

    def parse_response(self, data: dict[str, Any], public_model: str,
                       backend: str) -> dict[str, Any]:
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or first.get("text") or {}
        content = message.get("content") if isinstance(message, dict) else str(message)
        if content is None:
            content = first.get("text", "")
        usage = data.get("usage") or {}
        return {
            "id": data.get("id") or f"chatcmpl-local-{int(time.time() * 1000)}",
            "object": "chat.completion", "created": data.get("created") or int(time.time()),
            "model": public_model,
            "choices": [{
                "index": 0, "message": {"role": "assistant", "content": content},
                "finish_reason": first.get("finish_reason") or "stop",
            }],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "doomsday": {"backend": backend, "cloud_fallback": False},
        }

    def translate_stream(self, line: str, public_model: str,
                         meta: dict[str, Any] | None = None) -> str | None:
        if not line:
            return None
        text = line.strip()
        if text.startswith("data:"):
            text = text[5:].strip()
        if text == "[DONE]":
            return "data: [DONE]\n\n"
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if "error" in data:
            return "data: " + json.dumps({"error": data["error"]}, ensure_ascii=False) + "\n\n"
        choices = data.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        content = delta.get("content") if isinstance(delta, dict) else None
        if content is None and choices:
            content = choices[0].get("text")
        chunk = {
            "id": data.get("id") or "chatcmpl-local",
            "object": "chat.completion.chunk", "created": data.get("created") or int(time.time()),
            "model": public_model,
            "choices": [{"index": 0, "delta": {"content": content or ""},
                         "finish_reason": choices[0].get("finish_reason") if choices else None}],
            "doomsday": {"cloud_fallback": False, **(meta or {})},
        }
        return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


ADAPTERS: dict[str, Adapter] = {LlamaCppOpenAIAdapter.name: LlamaCppOpenAIAdapter()}


def get_adapter(name: str) -> Adapter:
    if name not in ADAPTERS:
        raise KeyError(f"适配器 {name} 未登记：作业应标为「待适配」")
    return ADAPTERS[name]


def registered_adapters() -> list[dict[str, str]]:
    return [{"id": name, "chat_path": a.chat_path, "health_path": a.health_path}
            for name, a in ADAPTERS.items()]
