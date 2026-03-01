# ACL2 Knowledge Graph MCP Server — Design Plan

## Overview

MCP server that exposes the Weaviate-hosted ACL2 Knowledge Graph and
DoclingPapers collection as tools for coding, deep-research, and Q&A agents.

Built with the Python `mcp` SDK (v1.26.0), supports **stdio** (default) and
**TCP** (`--tcp PORT`) transport.  All list/query results use offset-based
pagination with a standard JSON envelope.

## Collections Served

| Collection | Objects | Description |
|------------|---------|-------------|
| ACL2Notebook | ~14,500 | One per ingested .lisp file |
| ACL2Cell | ~507,000 | One per Jupyter cell (code/markdown/raw) |
| ACL2Symbol | ~403,000 | One per ACL2 symbol with dependency graph (~6.75M edges) |
| ACL2Summary | ~15,000+ | LLM-generated what/why/how summaries (cell/notebook/directory) |
| DoclingPapers | ~14,000 | RAG chunks from academic papers |

## Tools (8)

### `kg_stats`
Collection counts, symbol kind distribution, summary scope breakdown.
No arguments.

### `kg_search`
Unified ranked search.  Targets: `symbol`, `code`, `comment`, `summary`, `docs`.
Modes: `semantic` (vector similarity) or `keyword` (substring match).
Paginated via `offset`/`limit`.

### `kg_get_symbol`
Full symbol detail by qualified name (e.g. `ACL2::APPEND`).
Uses deterministic UUID5 for exact lookup (avoids tokenization issues).
Includes: definition (source code), dependencies, dependents, summary.
Dependencies and dependents are independently paginated.

### `kg_get_notebook`
Notebook metadata + paginated cell listing with defined symbols and summaries.
Cells sorted by index; post-filtered for exact source_file match.

### `kg_get_cell`
Single cell by source_file + cell_index.  Returns full code/comment text,
defined symbols, stdout, execute_result, and cell summary.

### `kg_get_summary`
Fetch a specific summary by its ref_key.

### `kg_list_notebooks`
Browse/filter notebooks by path substring.  Paginated, sorted alphabetically.
Includes notebook summary `what` when available.

### `kg_search_docs`
Search DoclingPapers (academic papers).  Semantic mode uses client-side
Ollama nomic-embed-text embeddings + `near_vector`.  Supports paper title
filtering.

## Response Envelope

All paginated results return:
```json
{
  "results": [...],
  "total": 403000,
  "offset": 0,
  "limit": 50,
  "has_more": true
}
```

Single-object lookups return the object directly (no envelope).

## Architecture

```
acl2_kg_mcp/
├── __init__.py          # version
├── server.py            # MCP tool definitions + dispatch + transport
└── weaviate_client.py   # Weaviate singleton + query functions
```

- **weaviate_client.py**: Lazy singleton client via `get_client()`.
  All query functions accept `offset`/`limit` and return typed dicts.
  Uses `generate_uuid5()` for deterministic UUID lookups.
  Post-filters source_file matches due to Weaviate TEXT tokenization.

- **server.py**: `@app.list_tools()` returns 8 Tool definitions with JSON Schema.
  `@app.call_tool()` dispatches to weaviate_client functions.
  All responses are JSON-serialized TextContent.
  Supports stdio (default) and TCP (`--tcp PORT`) transport.

## Key Design Decisions

1. **Offset-based pagination**: Simple, idempotent, Weaviate supports offset
   on both `fetch_objects` and `near_text`.

2. **JSON in TextContent**: Agents parse JSON easily; `structuredContent` is
   not universally supported.

3. **Separate `kg_search_docs`** alongside `kg_search target=docs`: Allows
   paper-specific filtering without bloating the main search tool.

4. **Client-side embeddings for DoclingPapers**: That collection has no
   Weaviate vectorizer (created by LangChain with `Vectorizers.NONE`).
   The server embeds queries via `OllamaEmbeddings` and uses `near_vector`.

5. **All tools `readOnlyHint: true`**: No mutations, safe to call freely.

6. **Post-filtering for exact path match**: Weaviate TEXT tokenization splits
   on `:`, `@`, `-`, `/`, etc., so source_file filters may return extras.

## Installation

```bash
pip install -e external/acl2-kg-mcp
```

## DevContainer Registration

In `.devcontainer/devcontainer.json`:
```json
"acl2-kg-mcp": {
    "command": "${containerWorkspaceFolder}/.venv/bin/acl2-kg-mcp",
    "args": [],
    "env": {
        "WEAVIATE_HOST": "host.docker.internal"
    }
}
```
