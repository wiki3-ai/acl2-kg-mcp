"""End-to-end MCP protocol test over stdio.

Spawns the server as a subprocess, exchanges JSON-RPC messages via
stdin/stdout, and verifies the responses.

Run:
    pytest tests/test_mcp_protocol.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import pytest

# The installed entry-point is "acl2-kg-mcp"; fallback to python -m.
SERVER_CMD = [sys.executable, "-m", "acl2_kg_mcp.server"]


def _rpc(method: str, params: dict | None = None, id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": id}
    if params is not None:
        msg["params"] = params
    return msg


async def _send_recv(proc: asyncio.subprocess.Process,
                     request: dict) -> dict:
    """Send a JSON-RPC request and read the response line."""
    assert proc.stdin is not None and proc.stdout is not None
    payload = json.dumps(request) + "\n"
    proc.stdin.write(payload.encode())
    await proc.stdin.drain()

    line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
    return json.loads(line)


async def _start_server() -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        *SERVER_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return proc


async def _initialize(proc: asyncio.subprocess.Process) -> dict:
    """Send the MCP initialize handshake."""
    resp = await _send_recv(proc, _rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1"},
    }, id=1))
    # Send initialized notification (no response expected)
    notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(notif) + "\n").encode())
    await proc.stdin.drain()
    return resp


@pytest.mark.asyncio
async def test_initialize() -> None:
    proc = await _start_server()
    try:
        resp = await _initialize(proc)
        assert "result" in resp
        assert resp["result"]["serverInfo"]["name"] == "acl2-kg-mcp"
    finally:
        proc.terminate()
        await proc.wait()


@pytest.mark.asyncio
async def test_tools_list() -> None:
    proc = await _start_server()
    try:
        await _initialize(proc)

        resp = await _send_recv(proc, _rpc("tools/list", {}, id=2))
        assert "result" in resp
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert len(tools) == 8
        assert "kg_stats" in names
        assert "kg_search" in names
        assert "kg_get_symbol" in names
    finally:
        proc.terminate()
        await proc.wait()


@pytest.mark.asyncio
async def test_call_kg_stats() -> None:
    proc = await _start_server()
    try:
        await _initialize(proc)

        resp = await _send_recv(proc, _rpc("tools/call", {
            "name": "kg_stats",
            "arguments": {},
        }, id=3))
        assert "result" in resp
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"

        data = json.loads(content[0]["text"])
        assert "collections" in data
        assert data["collections"]["ACL2Symbol"] > 400_000
    finally:
        proc.terminate()
        await proc.wait()


@pytest.mark.asyncio
async def test_call_kg_search_symbol() -> None:
    proc = await _start_server()
    try:
        await _initialize(proc)

        resp = await _send_recv(proc, _rpc("tools/call", {
            "name": "kg_search",
            "arguments": {
                "query": "merge sort",
                "target": "symbol",
                "mode": "semantic",
                "limit": 3,
            },
        }, id=4))
        assert "result" in resp
        data = json.loads(resp["result"]["content"][0]["text"])
        assert len(data["results"]) > 0
    finally:
        proc.terminate()
        await proc.wait()


@pytest.mark.asyncio
async def test_call_kg_get_symbol() -> None:
    proc = await _start_server()
    try:
        await _initialize(proc)

        resp = await _send_recv(proc, _rpc("tools/call", {
            "name": "kg_get_symbol",
            "arguments": {
                "qualified_name": "ACL2::BRR@",
                "include": ["definition"],
            },
        }, id=5))
        assert "result" in resp
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["qualified_name"] == "ACL2::BRR@"
        assert data["kind"] == "macro"
        assert "definition" in data
    finally:
        proc.terminate()
        await proc.wait()


@pytest.mark.asyncio
async def test_call_kg_list_notebooks() -> None:
    proc = await _start_server()
    try:
        await _initialize(proc)

        resp = await _send_recv(proc, _rpc("tools/call", {
            "name": "kg_list_notebooks",
            "arguments": {"filter": "defsort", "limit": 10},
        }, id=6))
        assert "result" in resp
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["total"] == 6
    finally:
        proc.terminate()
        await proc.wait()
