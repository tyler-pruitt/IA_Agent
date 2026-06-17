"""
MCP 客户端封装 — 统一通过 MCP 工具访问数据层

默认使用 stdio 传输在本地拉起 `python -m akshare_mcp_server`，
也支持通过 SSE 连接外部 MCP Server。
"""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import shlex
import sys
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from config.settings import (
    MCP_CONNECT_TIMEOUT,
    MCP_SERVER_URL,
    MCP_SSE_READ_TIMEOUT,
    MCP_STDIO_ARGS,
    MCP_STDIO_COMMAND,
    MCP_TOOL_TIMEOUT,
    MCP_TRANSPORT,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_stdio_args(args: str) -> tuple[str, ...]:
    """解析环境变量中的 stdio 参数字符串。"""
    if not args.strip():
        return ("-m", "akshare_mcp_server")
    return tuple(shlex.split(args))


@dataclass(frozen=True)
class MCPClientConfig:
    """MCP 客户端连接配置。"""

    transport: str = MCP_TRANSPORT
    server_url: str = MCP_SERVER_URL
    stdio_command: str = MCP_STDIO_COMMAND or sys.executable
    stdio_args: tuple[str, ...] = _parse_stdio_args(MCP_STDIO_ARGS)
    tool_timeout: int = MCP_TOOL_TIMEOUT
    connect_timeout: int = MCP_CONNECT_TIMEOUT
    sse_read_timeout: int = MCP_SSE_READ_TIMEOUT
    cwd: str = str(PROJECT_ROOT)
    cache_dir: str | None = None


class MCPToolClient:
    """
    后台持久化 MCP 会话。

    对外暴露同步的 `call_tool()`，内部通过后台事件循环管理异步 MCP Session。
    """

    def __init__(self, config: MCPClientConfig | None = None):
        self._config = config or MCPClientConfig()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: Exception | None = None
        self._transport_cm: Any = None
        self._session: ClientSession | None = None
        self._call_count = 0
        self._error_count = 0
        self._stats_lock = threading.Lock()
        atexit.register(self.close)

    def ensure_started(self):
        """惰性启动后台 MCP 会话。"""
        if self._closed:
            raise RuntimeError("MCP client 已关闭，无法再次启动")

        if self._thread and self._thread.is_alive() and self._ready.is_set():
            if self._startup_error is not None:
                raise RuntimeError("MCP client 启动失败") from self._startup_error
            return

        with self._start_lock:
            if self._thread and self._thread.is_alive() and self._ready.is_set():
                if self._startup_error is not None:
                    raise RuntimeError("MCP client 启动失败") from self._startup_error
                return

            self._ready.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._run_loop,
                name="investment-advisor-mcp-client",
                daemon=True,
            )
            self._thread.start()
            self._ready.wait(timeout=self._config.connect_timeout + 5)

            if not self._ready.is_set():
                raise TimeoutError("MCP client 启动超时")
            if self._startup_error is not None:
                raise RuntimeError("MCP client 启动失败") from self._startup_error

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """同步调用 MCP 工具并返回解析后的 JSON 结果。"""
        self.ensure_started()
        if self._loop is None:
            raise RuntimeError("MCP client 事件循环未初始化")

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments or {}),
            self._loop,
        )
        payload = future.result(timeout=self._config.tool_timeout + 5)
        with self._stats_lock:
            self._call_count += 1
            if isinstance(payload, dict) and payload.get("error"):
                self._error_count += 1
        return payload

    def close(self):
        """关闭后台会话和事件循环。"""
        if self._closed:
            return
        self._closed = True

        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return

        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    @property
    def stats(self) -> dict[str, Any]:
        """客户端调用统计。"""
        with self._stats_lock:
            return {
                "transport": self._config.transport,
                "calls": self._call_count,
                "errors": self._error_count,
                "hits": 0,
                "misses": 0,
                "hit_rate": "0.0%",
            }

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._startup_async())
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return

        self._ready.set()

        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self._shutdown_async())
            finally:
                loop.close()

    async def _startup_async(self):
        if self._config.transport == "sse":
            self._transport_cm = sse_client(
                self._config.server_url,
                timeout=self._config.connect_timeout,
                sse_read_timeout=self._config.sse_read_timeout,
            )
        elif self._config.transport == "stdio":
            args = list(self._config.stdio_args)
            if self._config.cache_dir:
                args.extend(["--cache-dir", self._config.cache_dir])
            server = StdioServerParameters(
                command=self._config.stdio_command,
                args=args,
                cwd=self._config.cwd,
            )
            self._transport_cm = stdio_client(server)
        else:
            raise ValueError(f"不支持的 MCP transport: {self._config.transport}")

        read_stream, write_stream = await self._transport_cm.__aenter__()
        self._session = ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=self._config.tool_timeout),
        )
        await self._session.__aenter__()
        await self._session.initialize()
        logger.info("[MCP] connected via %s", self._config.transport)

    async def _shutdown_async(self):
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                logger.debug("[MCP] session close failed", exc_info=True)
            finally:
                self._session = None

        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except Exception:
                logger.debug("[MCP] transport close failed", exc_info=True)
            finally:
                self._transport_cm = None

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MCP session 未初始化")

        result = await self._session.call_tool(
            name,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=self._config.tool_timeout),
        )
        return self._parse_tool_result(name, result)

    @staticmethod
    def _parse_tool_result(name: str, result) -> dict[str, Any]:
        """解析 MCP tool 返回值。"""
        if result.structuredContent is not None:
            return MCPToolClient._normalize_payload(result.structuredContent)

        text_parts = []
        for item in result.content or []:
            if getattr(item, "type", None) == "text":
                text_parts.append(item.text)

        raw = "\n".join(part for part in text_parts if part).strip()
        if not raw:
            return {"error": f"MCP 工具 {name} 未返回可解析内容"}

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"text": raw}

        if result.isError and "error" not in payload:
            return {"error": f"MCP 工具 {name} 调用失败", "detail": payload}
        return MCPToolClient._normalize_payload(payload)

    @staticmethod
    def _normalize_payload(payload: Any) -> dict[str, Any]:
        """兼容 FastMCP 对字符串返回值的 result 包装。"""
        if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
            wrapped = payload["result"]
            if isinstance(wrapped, dict):
                return wrapped
            if isinstance(wrapped, str):
                try:
                    parsed = json.loads(wrapped)
                except json.JSONDecodeError:
                    return {"result": wrapped}
                return parsed if isinstance(parsed, dict) else {"data": parsed}

        if isinstance(payload, dict):
            return payload
        return {"data": payload}


_SHARED_CLIENTS: dict[MCPClientConfig, MCPToolClient] = {}
_SHARED_CLIENTS_LOCK = threading.Lock()


def get_shared_client(config: MCPClientConfig | None = None) -> MCPToolClient:
    """按配置复用 MCP 客户端，避免重复拉起 stdio server。"""
    client_config = config or MCPClientConfig()
    with _SHARED_CLIENTS_LOCK:
        client = _SHARED_CLIENTS.get(client_config)
        if client is None:
            client = MCPToolClient(client_config)
            _SHARED_CLIENTS[client_config] = client
        return client


def close_shared_clients():
    """关闭所有共享 MCP 客户端。"""
    with _SHARED_CLIENTS_LOCK:
        clients = list(_SHARED_CLIENTS.values())
        _SHARED_CLIENTS.clear()

    for client in clients:
        client.close()
