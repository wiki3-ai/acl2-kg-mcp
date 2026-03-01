"""Integration tests for the MCP server layer.

These tests exercise the MCP tool dispatch by calling the server's
`call_tool` directly (no subprocess/stdio). They require a running
Weaviate instance.

Run:
    pytest tests/test_server.py -v
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from acl2_kg_mcp.server import call_tool, list_tools


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def _ensure_weaviate() -> None:
    from acl2_kg_mcp import weaviate_client as wc
    try:
        client = wc.get_client()
        assert client.is_ready()
    except Exception:
        pytest.skip("Weaviate is not reachable")


def _parse(result: Any) -> dict:
    """Extract the JSON payload from a call_tool result."""
    assert len(result) == 1
    assert result[0].type == "text"
    return json.loads(result[0].text)


# ── list_tools ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_tools_returns_ten() -> None:
    tools = await list_tools()
    assert len(tools) == 10
    names = {t.name for t in tools}
    assert names == {
        "kg_stats", "kg_search", "kg_get_symbol", "kg_get_notebook",
        "kg_get_cell", "kg_get_summary", "kg_list_notebooks",
        "kg_search_docs", "kg_search_acl2_docs", "kg_get_include_book",
    }

@pytest.mark.asyncio
async def test_tools_all_readonly() -> None:
    tools = await list_tools()
    for t in tools:
        assert t.annotations.readOnlyHint is True, f"{t.name} not readonly"


# ── kg_stats ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_stats() -> None:
    data = _parse(await call_tool("kg_stats", {}))
    assert "collections" in data
    assert data["collections"]["ACL2Symbol"] > 400_000
    assert data["collections"]["ACL2Cell"] > 500_000


# ── kg_search ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_search_symbol_semantic() -> None:
    data = _parse(await call_tool("kg_search", {
        "query": "merge sort", "target": "symbol",
        "mode": "semantic", "limit": 5,
    }))
    assert len(data["results"]) > 0
    assert data["limit"] == 5

@pytest.mark.asyncio
async def test_kg_search_code_keyword() -> None:
    data = _parse(await call_tool("kg_search", {
        "query": "defun", "target": "code",
        "mode": "keyword", "limit": 3,
    }))
    assert len(data["results"]) == 3

@pytest.mark.asyncio
async def test_kg_search_docs_semantic() -> None:
    data = _parse(await call_tool("kg_search", {
        "query": "theorem proving", "target": "docs",
        "mode": "semantic", "limit": 3,
    }))
    assert len(data["results"]) > 0

@pytest.mark.asyncio
async def test_kg_search_acl2_docs_via_unified() -> None:
    data = _parse(await call_tool("kg_search", {
        "query": "guard verification", "target": "acl2_docs",
        "mode": "semantic", "limit": 3,
    }))
    assert len(data["results"]) > 0
    for item in data["results"]:
        assert "title" in item
        assert "doc_type" in item

@pytest.mark.asyncio
async def test_kg_search_unknown_target() -> None:
    data = _parse(await call_tool("kg_search", {
        "query": "x", "target": "bogus", "mode": "semantic",
    }))
    assert "error" in data

@pytest.mark.asyncio
async def test_kg_search_summary() -> None:
    data = _parse(await call_tool("kg_search", {
        "query": "sanity check", "target": "summary",
        "mode": "semantic", "limit": 5,
    }))
    assert len(data["results"]) > 0


# ── kg_get_symbol ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_get_symbol_basic() -> None:
    data = _parse(await call_tool("kg_get_symbol", {
        "qualified_name": "ACL2::BRR@",
    }))
    assert data["qualified_name"] == "ACL2::BRR@"
    assert data["kind"] == "macro"
    assert "definition" in data
    assert "dependencies" in data

@pytest.mark.asyncio
async def test_kg_get_symbol_selective_include() -> None:
    data = _parse(await call_tool("kg_get_symbol", {
        "qualified_name": "ACL2::BRR@",
        "include": ["definition"],
    }))
    assert "definition" in data
    assert "dependencies" not in data

@pytest.mark.asyncio
async def test_kg_get_symbol_not_found() -> None:
    data = _parse(await call_tool("kg_get_symbol", {
        "qualified_name": "NONEXISTENT::ZZZZZ",
    }))
    assert "error" in data

@pytest.mark.asyncio
async def test_kg_get_symbol_deps_paging() -> None:
    p1 = _parse(await call_tool("kg_get_symbol", {
        "qualified_name": "COMMON-LISP::APPEND",
        "include": ["dependents"],
        "deps_limit": 5, "deps_offset": 0,
    }))
    p2 = _parse(await call_tool("kg_get_symbol", {
        "qualified_name": "COMMON-LISP::APPEND",
        "include": ["dependents"],
        "deps_limit": 5, "deps_offset": 5,
    }))
    names1 = {d["qualified_name"] for d in p1["dependents"]["results"]}
    names2 = {d["qualified_name"] for d in p2["dependents"]["results"]}
    assert names1.isdisjoint(names2), "Pages overlap"


# ── kg_get_notebook ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_get_notebook() -> None:
    data = _parse(await call_tool("kg_get_notebook", {
        "source_file": "books/defsort/defsort.lisp",
    }))
    assert data["source_file"] == "books/defsort/defsort.lisp"
    assert data["cell_count"] > 0
    assert "cells" in data

@pytest.mark.asyncio
async def test_kg_get_notebook_not_found() -> None:
    data = _parse(await call_tool("kg_get_notebook", {
        "source_file": "nonexistent/file.lisp",
    }))
    assert "error" in data

@pytest.mark.asyncio
async def test_kg_get_notebook_no_cells() -> None:
    data = _parse(await call_tool("kg_get_notebook", {
        "source_file": "books/defsort/defsort.lisp",
        "include_cells": False,
    }))
    assert "cells" not in data

@pytest.mark.asyncio
async def test_kg_get_notebook_cell_paging() -> None:
    data = _parse(await call_tool("kg_get_notebook", {
        "source_file": "books/defsort/defsort.lisp",
        "cell_limit": 3, "cell_offset": 0,
    }))
    assert len(data["cells"]["results"]) == 3


# ── kg_get_cell ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_get_cell() -> None:
    data = _parse(await call_tool("kg_get_cell", {
        "source_file": "books/defsort/defsort.lisp",
        "cell_index": 2,
    }))
    assert data["cell_index"] == 2
    assert data["notebook_source"] == "books/defsort/defsort.lisp"

@pytest.mark.asyncio
async def test_kg_get_cell_not_found() -> None:
    data = _parse(await call_tool("kg_get_cell", {
        "source_file": "books/defsort/defsort.lisp",
        "cell_index": 99999,
    }))
    assert "error" in data


# ── kg_get_summary ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_get_summary() -> None:
    data = _parse(await call_tool("kg_get_summary", {
        "ref_key": "acl2-check.lisp",
    }))
    assert data["scope"] == "notebook"
    assert len(data["what"]) > 0

@pytest.mark.asyncio
async def test_kg_get_summary_not_found() -> None:
    data = _parse(await call_tool("kg_get_summary", {
        "ref_key": "nonexistent/zzz",
    }))
    assert "error" in data


# ── kg_list_notebooks ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_list_notebooks_filter() -> None:
    data = _parse(await call_tool("kg_list_notebooks", {
        "filter": "defsort", "limit": 10,
    }))
    assert data["total"] == 6
    for nb in data["results"]:
        assert "defsort" in nb["source_file"]

@pytest.mark.asyncio
async def test_kg_list_notebooks_paging() -> None:
    p1 = _parse(await call_tool("kg_list_notebooks", {
        "filter": "defsort", "limit": 3, "offset": 0,
    }))
    p2 = _parse(await call_tool("kg_list_notebooks", {
        "filter": "defsort", "limit": 3, "offset": 3,
    }))
    assert p1["has_more"] is True
    assert p2["has_more"] is False
    names1 = {nb["source_file"] for nb in p1["results"]}
    names2 = {nb["source_file"] for nb in p2["results"]}
    assert names1.isdisjoint(names2)


# ── kg_search_docs ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_search_docs() -> None:
    data = _parse(await call_tool("kg_search_docs", {
        "query": "theorem proving", "mode": "semantic", "limit": 3,
    }))
    assert len(data["results"]) > 0
    for item in data["results"]:
        assert "paper_title" in item

@pytest.mark.asyncio
async def test_kg_search_docs_keyword() -> None:
    data = _parse(await call_tool("kg_search_docs", {
        "query": "ACL2", "mode": "keyword", "limit": 3,
    }))
    assert len(data["results"]) > 0


# ── kg_search_acl2_docs ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_search_acl2_docs() -> None:
    data = _parse(await call_tool("kg_search_acl2_docs", {
        "query": "guard verification", "mode": "semantic", "limit": 5,
    }))
    assert len(data["results"]) > 0
    for item in data["results"]:
        assert "title" in item
        assert "source_path" in item
        assert "doc_type" in item

@pytest.mark.asyncio
async def test_kg_search_acl2_docs_keyword() -> None:
    data = _parse(await call_tool("kg_search_acl2_docs", {
        "query": "defun", "mode": "keyword", "limit": 3,
    }))
    assert len(data["results"]) > 0

@pytest.mark.asyncio
async def test_kg_search_acl2_docs_doc_type_filter() -> None:
    data = _parse(await call_tool("kg_search_acl2_docs", {
        "query": "ACL2", "doc_type": "readme", "limit": 5,
    }))
    for item in data["results"]:
        assert item["doc_type"] == "readme"


# ── Error handling ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tool() -> None:
    data = _parse(await call_tool("nonexistent_tool", {}))
    assert "error" in data

@pytest.mark.asyncio
async def test_missing_required_arg() -> None:
    data = _parse(await call_tool("kg_search", {}))  # missing "query"
    assert "error" in data


# ── kg_get_include_book ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_get_include_book() -> None:
    data = _parse(await call_tool("kg_get_include_book", {
        "source_file": "books/defsort/defsort.lisp",
    }))
    assert data["include_book"] == '(include-book "defsort/defsort" :dir :system)'

@pytest.mark.asyncio
async def test_kg_get_include_book_not_found() -> None:
    data = _parse(await call_tool("kg_get_include_book", {
        "source_file": "nonexistent/file.lisp",
    }))
    assert "error" in data


# ── Dependents total ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kg_get_symbol_dependents_total() -> None:
    data = _parse(await call_tool("kg_get_symbol", {
        "qualified_name": "COMMON-LISP::APPEND",
        "include": ["dependents"],
        "deps_limit": 5,
    }))
    assert data["dependents"]["total"] is not None
    assert data["dependents"]["total"] > 100
