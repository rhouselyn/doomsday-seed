"""测试用最小运行器：模拟 llama.cpp server 的 OpenAI 兼容接口。

只使用标准库、只监听回环、不访问网络。它的应答是确定性的，
对应 resources/eval/stage1.json 与 debug_samples.json 的固定样例。
生产环境请使用锁定的 llama-server 构建，见 data/runtimes/registry.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("llama-server (test fixture) b-test")
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8177)
parser.add_argument("-m", "--model", default="")
parser.add_argument("-c", "--ctx", default="2048")
parser.add_argument("-np", default="1")
parser.add_argument("-ngl", default="0")
parser.add_argument("--mmproj", default="")
parser.add_argument("--chat-template", default="")
ARGS, _UNKNOWN = parser.parse_known_args()


def answer_for(messages: list[dict]) -> str:
    text = ""
    for message in reversed(messages or []):
        if message.get("role") == "user":
            text = str(message.get("content", ""))
            break
    if "OFFLINE READY" in text:
        return "OFFLINE READY"
    if "首都" in text:
        return "北京"
    if "一加一" in text:
        return "2"
    if "工号" in text:
        return "A-17"
    if "JSON" in text or "json" in text:
        return '{"ok": true, "lang": "zh"}'
    if "03:00" in text or "同步时间" in text:
        return "03:00"
    if "翻译成英文" in text:
        return "seed vault"
    if "列表" in text:
        return "苹果、香蕉、橙子"
    if "调试就绪" in text:
        return "调试就绪"
    if "离线" in text:
        return "我可以离线工作"
    return "离线可用"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "doomsday-test-runner/1.0"

    def log_message(self, *_args):
        return

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json(200, {"status": "ok"})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": "local-test-model", "object": "model", "owned_by": "local"}]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "invalid json"})
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        messages = payload.get("messages", [])
        answer = answer_for(messages)
        created = int(time.time())
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunks = []
            for index in range(0, len(answer), 3):
                chunks.append({"id": "chatcmpl-test", "object": "chat.completion.chunk",
                               "created": created, "model": "local-test-model",
                               "choices": [{"index": 0,
                                            "delta": {"content": answer[index:index + 3]},
                                            "finish_reason": None}]})
            chunks.append({"id": "chatcmpl-test", "object": "chat.completion.chunk",
                           "created": created, "model": "local-test-model",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            for chunk in chunks:
                data = ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        return self._json(200, {
            "id": "chatcmpl-test", "object": "chat.completion", "created": created,
            "model": "local-test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": max(1, len(answer)),
                      "total_tokens": 12 + max(1, len(answer))},
        })


if __name__ == "__main__":
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    server.daemon_threads = True
    server.serve_forever()
