"""本地服务安全边界（文档 §6）。

服务和推理进程默认只监听回环地址；浏览器同源访问，限制 Host / Origin，
变更类请求做本地会话校验，其他客户端用本地 API 密钥。上传限制文件类型与尺寸，
推理过程不外发用户内容，默认不保留试用输入。联网同步与本地推理由不同进程负责。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from .config import ALLOWED_TEXT_SUFFIXES, MAX_UPLOAD_BYTES, Paths
from .db import Database

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def hostname_of(host_header: str | None) -> str:
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.split(":", 1)[0]


class LocalSecurity:
    def __init__(self, paths: Paths, db: Database):
        self.paths = paths
        self.db = db
        paths.ensure()
        self.session_token = self._load_or_create(paths.session_token, "session-")
        self.api_key = self._load_or_create(paths.api_key, "dsk-")
        self.db.add_api_key(hashlib.sha256(self.api_key.encode()).hexdigest(), "local")

    @staticmethod
    def _load_or_create(path: Path, prefix: str) -> str:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        value = prefix + secrets.token_urlsafe(32)
        path.write_text(value, encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return value

    def verify_session(self, token: str | None) -> bool:
        return bool(token) and hmac.compare_digest(token, self.session_token)

    def verify_api_key(self, key: str | None) -> bool:
        if not key:
            return False
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.db.has_api_key(digest)

    def check_host(self, host_header: str | None) -> bool:
        return hostname_of(host_header) in ALLOWED_HOSTS

    def check_origin(self, origin: str | None) -> bool:
        if not origin:
            return True  # 非浏览器客户端不发送 Origin
        cleaned = origin.strip().rstrip("/").lower()
        for scheme in ("http://", "https://"):
            if cleaned.startswith(scheme):
                host = hostname_of(cleaned[len(scheme):])
                if host in ALLOWED_HOSTS:
                    return True
                return False
        return False


def install_middleware(app, security: LocalSecurity) -> None:
    """只读取头部，不触碰请求体，避免影响流式响应。"""

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if not security.check_host(request.headers.get("host")):
            return _deny("Host 不在回环白名单内")
        if not security.check_origin(request.headers.get("origin")):
            return _deny("Origin 不在本地同源白名单内")
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    def _deny(reason: str):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={"detail": f"本地访问被拒绝：{reason}"})


def require_local(request: Request, security: LocalSecurity) -> None:
    """全局依赖：仅变更类请求需要会话或 API 密钥。"""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    token = request.headers.get("X-DS-Session")
    auth = request.headers.get("Authorization", "")
    key = auth[7:].strip() if auth.startswith("Bearer ") else None
    if security.verify_session(token) or security.verify_api_key(key):
        return None
    raise HTTPException(
        status_code=403,
        detail="变更类请求需要本地会话校验或本地 API 密钥（X-DS-Session / Authorization: Bearer）")


def validate_upload(filename: str, size: int, content_type: str = "") -> dict[str, Any]:
    """§6 上传限制文件类型与尺寸；推理过程不外发用户内容。"""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_TEXT_SUFFIXES:
        raise HTTPException(status_code=415,
                            detail=f"只接受本地文本文件：{', '.join(ALLOWED_TEXT_SUFFIXES)}")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB 限制")
    return {"filename": filename, "size": size, "suffix": suffix,
            "local_only": True, "retained": False}
