"""ACL2 Knowledge Graph MCP Server.

Exposes the Weaviate-hosted ACL2 knowledge graph and DoclingPapers collection
as MCP tools for coding, deep-research, and Q&A agents.

Usage:
    acl2-kg-mcp                          # stdio transport (default)
    acl2-kg-mcp --tcp 9800               # TCP socket transport
    acl2-kg-mcp --weaviate-host localhost # override Weaviate host

Environment variables:
    WEAVIATE_HOST      (default: host.docker.internal)
    WEAVIATE_HTTP_PORT (default: 8080)
    WEAVIATE_GRPC_PORT (default: 50051)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from . import weaviate_client as wc

logger = logging.getLogger(__name__)

app: Server = Server("acl2-kg-mcp")


# ── Tool definitions ─────────────────────────────────────────────────


@app.list_tools()  # type: ignore[misc]
async def list_tools() -> list[Tool]:
    """Return the catalogue of KG query tools."""
    return [
        Tool(
            name="kg_stats",
            description=(
                "Get overview statistics for the ACL2 Knowledge Graph: "
                "collection counts (notebooks, cells, symbols, summaries, docs), "
                "symbol kind distribution, and summary scope breakdown."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_search",
            description=(
                "Unified ranked search across the ACL2 Knowledge Graph. "
                "Search targets: symbol (ACL2 definitions), code (cell source code), "
                "comment (markdown/comment cells), summary (LLM-generated summaries), "
                "docs (academic papers from DoclingPapers), "
                "acl2_docs (READMEs, HTML, PDFs from the ACL2 source tree). "
                "Results are ranked by semantic distance (semantic mode) or relevance (keyword mode). "
                "Supports pagination via offset/limit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["symbol", "code", "comment", "summary", "docs", "acl2_docs"],
                        "default": "symbol",
                        "description": "What to search: symbol, code, comment, summary, docs, or acl2_docs",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "keyword"],
                        "default": "semantic",
                        "description": "Search mode: semantic (vector similarity) or keyword (substring match)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum results to return per page",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Number of results to skip (for pagination)",
                    },
                    "paper_filter": {
                        "type": "string",
                        "description": "(docs target only) Filter by paper title substring",
                    },
                    "version": {
                        "type": "string",
                        "description": "(summary target only) Filter summaries by version label, e.g. 'v1-qwen3-coder'",
                    },
                },
                "required": ["query"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_get_symbol",
            description=(
                "Get full details for an ACL2 symbol by its qualified name "
                "(e.g. ACL2::APPEND, COMMON-LISP::CAR). Returns the symbol's kind, "
                "package, defining code, dependencies (what it calls), "
                "dependents (what calls it), and LLM-generated summaries "
                "(a list of what/why/how dicts, one per distinct idea). "
                "Dependencies and dependents are paginated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "qualified_name": {
                        "type": "string",
                        "description": "Fully qualified symbol name, e.g. ACL2::APPEND",
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["definition", "dependencies", "dependents", "summary"],
                        },
                        "default": ["definition", "dependencies", "dependents", "summary"],
                        "description": "Which sections to include in the response",
                    },
                    "deps_limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Max dependencies/dependents per page",
                    },
                    "deps_offset": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Offset for paging through dependencies/dependents",
                    },
                    "version": {
                        "type": "string",
                        "description": "Filter summaries by version label, e.g. 'v1-qwen3-coder'",
                    },
                },
                "required": ["qualified_name"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_get_notebook",
            description=(
                "Get an ACL2 notebook (source file) with metadata and cell listing. "
                "Returns cell count, code cell count, source type, ACL2 version, "
                "notebook-level summary, and a paginated list of cells with their "
                "code/comments and defined symbols."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_file": {
                        "type": "string",
                        "description": "Notebook path, e.g. 'books/defsort/defsort.lisp' or 'acl2-check.lisp'",
                    },
                    "include_cells": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include the cell listing",
                    },
                    "cell_limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Max cells per page",
                    },
                    "cell_offset": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Offset for paging through cells",
                    },
                    "version": {
                        "type": "string",
                        "description": "Filter summaries by version label, e.g. 'v1-qwen3-coder'",
                    },
                },
                "required": ["source_file"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_get_cell",
            description=(
                "Get a single cell's full content from an ACL2 notebook. "
                "Returns the complete code or comment text, cell type, package, "
                "execution count, defined symbols, stdout, execute_result, "
                "and LLM-generated summaries (a list of what/why/how dicts, "
                "one per distinct idea) if available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_file": {
                        "type": "string",
                        "description": "Notebook path containing the cell",
                    },
                    "cell_index": {
                        "type": "integer",
                        "description": "0-based cell index within the notebook",
                    },
                    "version": {
                        "type": "string",
                        "description": "Filter summaries by version label, e.g. 'v1-qwen3-coder'",
                    },
                },
                "required": ["source_file", "cell_index"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_get_summary",
            description=(
                "Get an LLM-generated summary by its reference key. "
                "Returns the what/why/how summaries, scope (cell/notebook/directory), "
                "source file, cell index, directory, and symbol names."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ref_key": {
                        "type": "string",
                        "description": "Summary reference key, e.g. 'books/defsort/defsort.lisp' (notebook) or 'books/defsort/defsort.lisp:5:0' (cell, summary_index=0)",
                    },
                },
                "required": ["ref_key"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_list_notebooks",
            description=(
                "List ACL2 notebooks (source files) in the knowledge graph. "
                "Optionally filter by path substring. Returns source file path, "
                "cell counts, and notebook-level summary when available. "
                "Supports pagination via offset/limit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional path substring filter, e.g. 'defsort' or 'kestrel/bv'",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Max notebooks per page",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Offset for pagination",
                    },
                },
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_search_docs",
            description=(
                "Search academic papers in the DoclingPapers collection. "
                "These are RAG-chunked papers on ACL2, theorem proving, "
                "program verification, and related topics. "
                "Semantic search embeds the query via Ollama nomic-embed-text "
                "and matches against document chunk vectors. "
                "Supports pagination and optional paper title filtering."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "keyword"],
                        "default": "semantic",
                        "description": "Search mode",
                    },
                    "paper_filter": {
                        "type": "string",
                        "description": "Filter by paper title substring",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max results per page",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Offset for pagination",
                    },
                },
                "required": ["query"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_search_acl2_docs",
            description=(
                "Search ACL2 source tree documentation in the ACL2Docs collection. "
                "Contains ~211K chunks from READMEs, HTML reference docs, and PDFs "
                "(design notes, lecture slides, talks) from the ACL2 books directory. "
                "Semantic search embeds the query via Ollama nomic-embed-text. "
                "Supports filtering by doc_type (readme, html, pdf) and title. "
                "Supports pagination."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "keyword"],
                        "default": "semantic",
                        "description": "Search mode",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["readme", "html", "pdf"],
                        "description": "Filter by document type",
                    },
                    "title_filter": {
                        "type": "string",
                        "description": "Filter by title substring",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max results per page",
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "Offset for pagination",
                    },
                },
                "required": ["query"],
            },
            annotations={"readOnlyHint": True},
        ),
        Tool(
            name="kg_get_include_book",
            description=(
                "Map an ACL2 notebook source file to its (include-book ...) form. "
                "Given a source_file path from the KG (e.g. 'books/defsort/defsort.lisp'), "
                "returns the correct ACL2 include-book command with :dir :system "
                "for community books. Use this to bridge from KG exploration to "
                "loading code in a live ACL2 session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_file": {
                        "type": "string",
                        "description": "Notebook source_file path from the KG",
                    },
                },
                "required": ["source_file"],
            },
            annotations={"readOnlyHint": True},
        ),
    ]


# ── Tool dispatch ────────────────────────────────────────────────────

def _json_response(data: Any) -> list[TextContent]:
    """Wrap a Python object as a JSON TextContent response."""
    return [TextContent(type="text", text=json.dumps(data, indent=2, default=str))]


def _error_response(message: str) -> list[TextContent]:
    """Return an error JSON response."""
    return [TextContent(type="text", text=json.dumps({"error": message}))]


@app.call_tool()  # type: ignore[misc]
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Dispatch tool calls to weaviate_client query functions."""
    try:
        if name == "kg_stats":
            return _json_response(wc.get_stats())

        elif name == "kg_search":
            query = arguments["query"]
            target = arguments.get("target", "symbol")
            mode = arguments.get("mode", "semantic")
            limit = min(int(arguments.get("limit", 20)), 200)
            offset = max(int(arguments.get("offset", 0)), 0)

            if target == "symbol":
                return _json_response(
                    wc.search_symbols(query, mode=mode,
                                      offset=offset, limit=limit))
            elif target in ("code", "comment"):
                return _json_response(
                    wc.search_cells(query, target=target, mode=mode,
                                    offset=offset, limit=limit))
            elif target == "summary":
                version = arguments.get("version")
                return _json_response(
                    wc.search_summaries(query, mode=mode,
                                        offset=offset, limit=limit,
                                        version=version))
            elif target == "docs":
                paper_filter = arguments.get("paper_filter")
                return _json_response(
                    wc.search_docs(query, mode=mode,
                                   paper_filter=paper_filter,
                                   offset=offset, limit=limit))
            elif target == "acl2_docs":
                doc_type = arguments.get("doc_type")
                title_filter = arguments.get("title_filter")
                return _json_response(
                    wc.search_acl2_docs(query, mode=mode,
                                        doc_type=doc_type,
                                        title_filter=title_filter,
                                        offset=offset, limit=limit))
            else:
                return _error_response(
                    f"Unknown search target: {target}. "
                    "Use: symbol, code, comment, summary, docs, or acl2_docs")

        elif name == "kg_get_symbol":
            qn = arguments["qualified_name"]
            include = arguments.get("include",
                                    ["definition", "dependencies",
                                     "dependents", "summary"])
            deps_limit = min(int(arguments.get("deps_limit", 50)), 500)
            deps_offset = max(int(arguments.get("deps_offset", 0)), 0)
            version = arguments.get("version")

            result = wc.get_symbol(
                qn, include=include,
                deps_offset=deps_offset, deps_limit=deps_limit,
                version=version,
            )
            if result is None:
                return _error_response(f"Symbol not found: {qn}")
            return _json_response(result)

        elif name == "kg_get_notebook":
            source_file = arguments["source_file"]
            include_cells = arguments.get("include_cells", True)
            cell_limit = min(int(arguments.get("cell_limit", 50)), 500)
            cell_offset = max(int(arguments.get("cell_offset", 0)), 0)
            version = arguments.get("version")

            result = wc.get_notebook(
                source_file,
                include_cells=include_cells,
                cell_offset=cell_offset, cell_limit=cell_limit,
                version=version,
            )
            if result is None:
                return _error_response(f"Notebook not found: {source_file}")
            return _json_response(result)

        elif name == "kg_get_cell":
            source_file = arguments["source_file"]
            cell_index = int(arguments["cell_index"])
            version = arguments.get("version")

            result = wc.get_cell(source_file, cell_index, version=version)
            if result is None:
                return _error_response(
                    f"Cell not found: {source_file} cell {cell_index}")
            return _json_response(result)

        elif name == "kg_get_summary":
            ref_key = arguments["ref_key"]
            result = wc.get_summary(ref_key)
            if result is None:
                return _error_response(f"Summary not found: {ref_key}")
            return _json_response(result)

        elif name == "kg_list_notebooks":
            filter_path = arguments.get("filter")
            limit = min(int(arguments.get("limit", 50)), 200)
            offset = max(int(arguments.get("offset", 0)), 0)

            return _json_response(
                wc.list_notebooks(filter_path, offset=offset, limit=limit))

        elif name == "kg_search_docs":
            query = arguments["query"]
            mode = arguments.get("mode", "semantic")
            paper_filter = arguments.get("paper_filter")
            limit = min(int(arguments.get("limit", 10)), 100)
            offset = max(int(arguments.get("offset", 0)), 0)

            return _json_response(
                wc.search_docs(query, mode=mode, paper_filter=paper_filter,
                               offset=offset, limit=limit))

        elif name == "kg_search_acl2_docs":
            query = arguments["query"]
            mode = arguments.get("mode", "semantic")
            doc_type = arguments.get("doc_type")
            title_filter = arguments.get("title_filter")
            limit = min(int(arguments.get("limit", 10)), 100)
            offset = max(int(arguments.get("offset", 0)), 0)

            return _json_response(
                wc.search_acl2_docs(query, mode=mode, doc_type=doc_type,
                                    title_filter=title_filter,
                                    offset=offset, limit=limit))

        elif name == "kg_get_include_book":
            source_file = arguments["source_file"]
            result = wc.get_include_book(source_file)
            if result is None:
                return _error_response(f"Notebook not found: {source_file}")
            return _json_response(result)

        else:
            return _error_response(f"Unknown tool: {name}")

    except KeyError as e:
        return _error_response(f"Missing required argument: {e}")
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return _error_response(f"Internal error: {e}")


# ── Transport ────────────────────────────────────────────────────────

async def _run_stdio() -> None:
    """Run over stdio (default MCP transport)."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


async def _run_tcp(port: int) -> None:
    """Run over TCP socket transport.

    Each connection gets its own JSON-RPC session, but they share the
    same Weaviate client via the singleton in weaviate_client.
    """
    from mcp.server.stdio import stdio_server as _stdio  # noqa: F811
    import anyio
    from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream

    async def handle_client(reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.info("TCP client connected: %s", peer)

        # Adapt asyncio streams to anyio memory streams that mcp expects.
        # We run a simple read-loop and write-loop to bridge them.
        read_send: MemoryObjectSendStream
        read_recv: MemoryObjectReceiveStream
        write_send: MemoryObjectSendStream
        write_recv: MemoryObjectReceiveStream

        read_send, read_recv = anyio.create_memory_object_stream(0)
        write_send, write_recv = anyio.create_memory_object_stream(0)

        async def tcp_reader() -> None:
            """Read JSON-RPC messages from the TCP socket."""
            buf = b""
            try:
                while True:
                    chunk = await asyncio.to_thread(
                        lambda: asyncio.get_event_loop().run_until_complete(
                            reader.read(4096)
                        )
                    )
                    if not chunk:
                        break
                    buf += chunk
                    # Simple newline-delimited JSON
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if line:
                            import json as _json
                            msg = _json.loads(line)
                            await read_send.send(msg)
            except Exception:
                pass
            finally:
                await read_send.aclose()

        async def tcp_writer() -> None:
            """Write JSON-RPC messages to the TCP socket."""
            try:
                async for msg in write_recv:
                    import json as _json
                    data = _json.dumps(msg) + "\n"
                    writer.write(data.encode())
                    await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()

        async with anyio.create_task_group() as tg:
            tg.start_soon(tcp_reader)
            tg.start_soon(tcp_writer)
            await app.run(
                read_recv,  # type: ignore[arg-type]
                write_send,  # type: ignore[arg-type]
                app.create_initialization_options(),
            )

    server = await asyncio.start_server(handle_client, "0.0.0.0", port)
    logger.info("TCP MCP server listening on port %d", port)
    async with server:
        await server.serve_forever()


async def _run_sse(port: int) -> None:
    """Run over HTTP + SSE transport (for LM Studio, Claude Desktop, etc.).

    Exposes two endpoints:
      GET  /sse        – SSE stream (client connects here)
      POST /messages/  – client posts JSON-RPC messages here
    """
    from mcp.server.sse import SseServerTransport
    import uvicorn

    sse_transport = SseServerTransport("/messages/")

    async def asgi_app(scope: dict, receive: Any, send: Any) -> None:
        """Plain ASGI handler — avoids Starlette expecting a Response return value."""
        if scope["type"] != "http":
            return
        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET")
        if method == "GET" and path == "/sse":
            async with sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await app.run(
                    read_stream,
                    write_stream,
                    app.create_initialization_options(),
                )
        elif path.startswith("/messages/"):
            await sse_transport.handle_post_message(scope, receive, send)
        else:
            # 404 for anything else
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Not found"})

    config = uvicorn.Config(asgi_app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    logger.info("SSE MCP server listening on http://0.0.0.0:%d/sse", port)
    await server.serve()


def main() -> None:
    """Entry point: parse args and run the server."""
    parser = argparse.ArgumentParser(
        description="ACL2 Knowledge Graph MCP Server")
    parser.add_argument(
        "--tcp", type=int, default=None, metavar="PORT",
        help="Run over raw TCP on the given port instead of stdio")
    parser.add_argument(
        "--sse", type=int, default=None, metavar="PORT",
        help="Run over HTTP+SSE on the given port (for LM Studio, Claude Desktop, etc.)")
    parser.add_argument(
        "--weaviate-host", default=None,
        help="Override WEAVIATE_HOST env var")
    parser.add_argument(
        "--weaviate-http-port", type=int, default=None,
        help="Override WEAVIATE_HTTP_PORT env var")
    parser.add_argument(
        "--weaviate-grpc-port", type=int, default=None,
        help="Override WEAVIATE_GRPC_PORT env var")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Apply overrides
    wc.configure(
        host=args.weaviate_host,
        http_port=args.weaviate_http_port,
        grpc_port=args.weaviate_grpc_port,
    )

    # Ignore SIGPIPE for graceful client disconnects
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    if args.sse is not None:
        asyncio.run(_run_sse(args.sse))
    elif args.tcp is not None:
        asyncio.run(_run_tcp(args.tcp))
    else:
        asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
