"""Store the provider credential encrypted for the current Windows account."""
from __future__ import annotations

import ctypes
import os
import tempfile
from ctypes import wintypes
from pathlib import Path


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(value: bytes, encrypt: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("本机安全凭据存储目前仅支持 Windows")
    source_data = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source = _Blob(len(value), source_data)
    output = _Blob()
    crypt = ctypes.windll.crypt32
    kernel = ctypes.windll.kernel32
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    function = crypt.CryptProtectData if encrypt else crypt.CryptUnprotectData
    function.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise RuntimeError("无法使用当前 Windows 账户保护凭据")
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))


class CredentialStore:
    def __init__(self, directory: Path):
        self.path = directory / "deepseek.credential"

    def get(self) -> str:
        if not self.path.exists():
            return os.environ.get("DEEPSEEK_API_KEY", "").strip()
        try:
            return _crypt(self.path.read_bytes(), False).decode("utf-8")
        except (OSError, UnicodeError, RuntimeError):
            return ""

    def set(self, key: str) -> None:
        key = key.strip()
        if not key or len(key) > 1024 or any(c.isspace() for c in key):
            raise ValueError("请输入有效的 API Key")
        encrypted = _crypt(key.encode("utf-8"), True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encrypted)
        try:
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
