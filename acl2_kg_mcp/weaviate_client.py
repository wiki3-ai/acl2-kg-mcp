"""Weaviate client and query helpers for the ACL2 Knowledge Graph.

Provides a lazy-init singleton client and typed query functions for each
collection, all with offset/limit pagination.

Collections
-----------
- ACL2Notebook  — one per ingested .lisp file
- ACL2Cell      — one per Jupyter cell (code, markdown, raw)
- ACL2Symbol    — one per ACL2 symbol (function, macro, theorem, …)
- ACL2Summary   — LLM-generated summaries at cell/notebook/directory scope
- DoclingPapers — RAG chunks from academic papers (no Weaviate vectorizer;
                  requires client-side OllamaEmbeddings for semantic search)
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any

import weaviate
from weaviate.classes.query import Filter, MetadataQuery, QueryReference
from weaviate.util import generate_uuid5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: weaviate.WeaviateClient | None = None

_cfg: dict[str, Any] = {
    "host": os.environ.get("WEAVIATE_HOST", "host.docker.internal"),
    "http_port": int(os.environ.get("WEAVIATE_HTTP_PORT", "8080")),
    "grpc_port": int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
}


def configure(*, host: str | None = None,
              http_port: int | None = None,
              grpc_port: int | None = None) -> None:
    """Override connection settings before first use."""
    if host is not None:
        _cfg["host"] = host
    if http_port is not None:
        _cfg["http_port"] = http_port
    if grpc_port is not None:
        _cfg["grpc_port"] = grpc_port


def get_client() -> weaviate.WeaviateClient:
    """Return the singleton Weaviate client, connecting on first call."""
    global _client
    if _client is None:
        _client = weaviate.connect_to_custom(
            http_host=_cfg["host"],
            http_port=_cfg["http_port"],
            http_secure=False,
            grpc_host=_cfg["host"],
            grpc_port=_cfg["grpc_port"],
            grpc_secure=False,
        )
        atexit.register(_client.close)
        logger.info("Connected to Weaviate at %s:%s", _cfg["host"], _cfg["http_port"])
    return _client


# ---------------------------------------------------------------------------
# Pagination envelope
# ---------------------------------------------------------------------------

def _envelope(results: list[dict[str, Any]], *,
              total: int | None,
              offset: int,
              limit: int) -> dict[str, Any]:
    """Wrap a list of results in a standard paging envelope."""
    has_more = len(results) == limit if total is None else (offset + limit) < total
    return {
        "results": results,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
    }


# ---------------------------------------------------------------------------
# kg_stats
# ---------------------------------------------------------------------------

def get_stats() -> dict[str, Any]:
    """Aggregate counts for all collections."""
    client = get_client()
    stats: dict[str, Any] = {"collections": {}}

    for name in ("ACL2Notebook", "ACL2Cell", "ACL2Symbol",
                 "ACL2Summary", "DoclingPapers"):
        try:
            col = client.collections.get(name)
            agg = col.aggregate.over_all(total_count=True)
            count = agg.total_count
        except Exception:
            count = None
        stats["collections"][name] = count

    # Symbol kind distribution
    try:
        sym = client.collections.get("ACL2Symbol")
        kind_agg = sym.aggregate.over_all(
            group_by="kind",
            total_count=True,
        )
        stats["symbol_kinds"] = {
            g.grouped_by.value: g.total_count
            for g in kind_agg.groups
        }
    except Exception:
        stats["symbol_kinds"] = {}

    # Summary scope distribution
    try:
        summ = client.collections.get("ACL2Summary")
        scope_agg = summ.aggregate.over_all(
            group_by="scope",
            total_count=True,
        )
        stats["summary_scopes"] = {
            g.grouped_by.value: g.total_count
            for g in scope_agg.groups
        }
    except Exception:
        stats["summary_scopes"] = {}

    return stats


# ---------------------------------------------------------------------------
# kg_search  — unified search across collections
# ---------------------------------------------------------------------------

def search_symbols(query: str, mode: str = "semantic",
                   offset: int = 0, limit: int = 20,
                   ) -> dict[str, Any]:
    """Search ACL2Symbol by semantic or keyword."""
    client = get_client()
    sym = client.collections.get("ACL2Symbol")

    if mode == "semantic":
        resp = sym.query.near_text(
            query=query, limit=limit, offset=offset,
            target_vector="symbol_vector",
            return_metadata=MetadataQuery(distance=True),
        )
    else:
        resp = sym.query.fetch_objects(
            filters=Filter.by_property("qualified_name").like(f"*{query}*"),
            limit=limit, offset=offset,
        )

    results = []
    for obj in resp.objects:
        dist = getattr(obj.metadata, "distance", None) if obj.metadata else None
        results.append({
            "qualified_name": obj.properties["qualified_name"],
            "kind": obj.properties.get("kind", ""),
            "package": obj.properties.get("package", ""),
            "distance": dist,
        })
    return _envelope(results, total=None, offset=offset, limit=limit)


def search_cells(query: str, target: str = "code",
                 mode: str = "semantic",
                 offset: int = 0, limit: int = 20,
                 ) -> dict[str, Any]:
    """Search ACL2Cell code or comments."""
    client = get_client()
    cell = client.collections.get("ACL2Cell")
    vec = "code_vector" if target == "code" else "comment_vector"
    prop = "code_text" if target == "code" else "comment_text"

    if mode == "semantic":
        resp = cell.query.near_text(
            query=query, limit=limit, offset=offset,
            target_vector=vec,
            return_metadata=MetadataQuery(distance=True),
        )
    else:
        resp = cell.query.fetch_objects(
            filters=Filter.by_property(prop).like(f"*{query}*"),
            limit=limit, offset=offset,
        )

    results = []
    for obj in resp.objects:
        dist = getattr(obj.metadata, "distance", None) if obj.metadata else None
        text = obj.properties.get(prop) or ""
        results.append({
            "notebook_source": obj.properties.get("notebook_source", ""),
            "cell_index": obj.properties.get("cell_index", 0),
            "cell_type": obj.properties.get("cell_type", ""),
            "preview": text[:300],
            "distance": dist,
        })
    return _envelope(results, total=None, offset=offset, limit=limit)


def search_summaries(query: str, mode: str = "semantic",
                     vector: str = "what_vector",
                     offset: int = 0, limit: int = 20,
                     ) -> dict[str, Any]:
    """Search ACL2Summary."""
    client = get_client()
    col = client.collections.get("ACL2Summary")

    if mode == "semantic":
        resp = col.query.near_text(
            query=query, limit=limit, offset=offset,
            target_vector=vector,
            return_metadata=MetadataQuery(distance=True),
        )
    else:
        resp = col.query.fetch_objects(
            filters=Filter.by_property("what_summary").like(f"*{query}*"),
            limit=limit, offset=offset,
        )

    results = []
    for obj in resp.objects:
        p = obj.properties
        dist = getattr(obj.metadata, "distance", None) if obj.metadata else None
        results.append({
            "scope": p.get("scope", ""),
            "ref_key": p.get("ref_key", ""),
            "what": p.get("what_summary", ""),
            "why": p.get("why_summary", ""),
            "how": p.get("how_summary", ""),
            "source_file": p.get("source_file", ""),
            "cell_index": p.get("cell_index", -1),
            "symbol_names": p.get("symbol_names", []),
            "distance": dist,
        })
    return _envelope(results, total=None, offset=offset, limit=limit)


def search_docs(query: str, mode: str = "semantic",
                paper_filter: str | None = None,
                offset: int = 0, limit: int = 10,
                ) -> dict[str, Any]:
    """Search DoclingPapers.

    This collection uses client-side Ollama embeddings (no Weaviate vectorizer),
    so semantic search requires embedding the query and using near_vector.
    Falls back to keyword search when embeddings are unavailable.
    """
    client = get_client()
    col = client.collections.get("DoclingPapers")

    filt = None
    if paper_filter:
        filt = Filter.by_property("paper_title").like(f"*{paper_filter}*")

    if mode == "semantic":
        # Embed query client-side using Ollama
        try:
            from langchain_ollama import OllamaEmbeddings  # type: ignore[import-untyped]
            ollama_url = f"http://{_cfg['host']}:11434"
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text:latest",
                base_url=ollama_url,
            )
            qvec = embeddings.embed_query(query)
        except Exception as e:
            logger.warning("Ollama embedding failed, falling back to keyword: %s", e)
            return search_docs(query, mode="keyword",
                               paper_filter=paper_filter,
                               offset=offset, limit=limit)

        kwargs: dict[str, Any] = {
            "near_vector": qvec,
            "limit": limit,
            "offset": offset,
            "return_metadata": MetadataQuery(distance=True),
        }
        if filt:
            kwargs["filters"] = filt
        resp = col.query.near_vector(**kwargs)
    else:
        kw_filt = Filter.by_property("text").like(f"*{query}*")
        if filt:
            kw_filt = kw_filt & filt
        resp = col.query.fetch_objects(filters=kw_filt,
                                       limit=limit, offset=offset)

    results = []
    for obj in resp.objects:
        p = obj.properties
        dist = getattr(obj.metadata, "distance", None) if obj.metadata else None
        results.append({
            "paper_title": p.get("paper_title", ""),
            "source": p.get("source", ""),
            "text": p.get("text", ""),
            "distance": dist,
        })
    return _envelope(results, total=None, offset=offset, limit=limit)


# ---------------------------------------------------------------------------
# kg_get_symbol
# ---------------------------------------------------------------------------

def get_symbol(qualified_name: str, *,
               include: list[str] | None = None,
               deps_offset: int = 0,
               deps_limit: int = 50,
               ) -> dict[str, Any] | None:
    """Get full symbol detail with paged deps/dependents.

    ``include`` may contain: definition, dependencies, dependents, summary.
    Default: all of them.
    """
    if include is None:
        include = ["definition", "dependencies", "dependents", "summary"]

    client = get_client()
    sym_col = client.collections.get("ACL2Symbol")

    symbol_uuid = generate_uuid5(f"symbol:{qualified_name}")

    # Build reference list depending on what's requested
    refs = []
    if "dependencies" in include:
        refs.append(QueryReference(
            link_on="dependsOn",
            return_properties=["qualified_name", "kind", "package"],
        ))
    if "definition" in include:
        refs.append(QueryReference(
            link_on="definedInCell",
            return_properties=["cell_index", "notebook_source",
                               "code_text", "comment_text"],
        ))

    obj = sym_col.query.fetch_object_by_id(
        symbol_uuid,
        return_references=refs if refs else None,
    )
    if obj is None:
        return None

    result: dict[str, Any] = {
        "qualified_name": obj.properties["qualified_name"],
        "name": obj.properties.get("name", ""),
        "package": obj.properties.get("package", ""),
        "kind": obj.properties.get("kind", ""),
    }

    # Defining cell
    if "definition" in include:
        cell_ref = obj.references.get("definedInCell")
        if cell_ref and cell_ref.objects:
            c = cell_ref.objects[0].properties
            result["definition"] = {
                "cell_index": c["cell_index"],
                "notebook_source": c["notebook_source"],
                "code": c.get("code_text") or c.get("comment_text") or "",
            }

    # Forward dependencies (paged)
    if "dependencies" in include:
        deps_ref = obj.references.get("dependsOn")
        all_deps: list[dict[str, Any]] = []
        if deps_ref and deps_ref.objects:
            all_deps = sorted(
                [{"qualified_name": d.properties["qualified_name"],
                  "kind": d.properties.get("kind", ""),
                  "package": d.properties.get("package", "")}
                 for d in deps_ref.objects],
                key=lambda x: x["qualified_name"],
            )
        total_deps = len(all_deps)
        paged_deps = all_deps[deps_offset:deps_offset + deps_limit]
        result["dependencies"] = _envelope(
            paged_deps, total=total_deps,
            offset=deps_offset, limit=deps_limit,
        )

    # Reverse dependencies (paged)
    if "dependents" in include:
        try:
            rev_resp = sym_col.query.fetch_objects(
                filters=Filter.by_ref("dependsOn").by_id().equal(symbol_uuid),
                limit=deps_limit,
                offset=deps_offset,
            )
            rev_deps = sorted(
                [{"qualified_name": r.properties["qualified_name"],
                  "kind": r.properties.get("kind", "")}
                 for r in rev_resp.objects],
                key=lambda x: x["qualified_name"],
            )
            result["dependents"] = _envelope(
                rev_deps, total=None,
                offset=deps_offset, limit=deps_limit,
            )
        except Exception:
            result["dependents"] = _envelope(
                [], total=0, offset=deps_offset, limit=deps_limit,
            )

    # Cell summary
    if "summary" in include and "definition" in result:
        defn = result["definition"]
        sums = _get_cell_summaries(client, defn["notebook_source"])
        s = sums.get(defn["cell_index"])
        if s:
            result["summary"] = s

    return result


# ---------------------------------------------------------------------------
# kg_get_notebook
# ---------------------------------------------------------------------------

def get_notebook(source_file: str, *,
                 include_cells: bool = True,
                 cell_offset: int = 0,
                 cell_limit: int = 50,
                 ) -> dict[str, Any] | None:
    """Get notebook metadata + paged cell listing."""
    client = get_client()

    # Notebook metadata
    nb_col = client.collections.get("ACL2Notebook")
    nb_resp = nb_col.query.fetch_objects(
        filters=Filter.by_property("source_file").equal(source_file),
        limit=5,
    )
    # Post-filter for exact match (word tokenization may return extras)
    notebook = None
    for obj in nb_resp.objects:
        if obj.properties.get("source_file") == source_file:
            notebook = dict(obj.properties)
            break
    if notebook is None:
        return None

    result: dict[str, Any] = {
        "source_file": notebook["source_file"],
        "cell_count": notebook.get("cell_count", 0),
        "code_cell_count": notebook.get("code_cell_count", 0),
        "is_bootstrap": notebook.get("is_bootstrap", False),
        "source_type": notebook.get("source_type", ""),
        "acl2_version": notebook.get("acl2_version", ""),
    }

    # Notebook summary
    nb_summary = _get_notebook_summary(client, source_file)
    if nb_summary:
        result["summary"] = nb_summary

    # Cells (paged)
    if include_cells:
        cell_col = client.collections.get("ACL2Cell")
        resp = cell_col.query.fetch_objects(
            filters=Filter.by_property("notebook_source").equal(source_file),
            limit=10000,  # fetch all, then page in-memory (sorted by index)
            return_references=[
                QueryReference(link_on="definesSymbols",
                               return_properties=["qualified_name", "kind"]),
            ],
        )

        all_cells: list[dict[str, Any]] = []
        seen: set[int] = set()
        for obj in resp.objects:
            # Post-filter for exact source match
            if obj.properties.get("notebook_source") != source_file:
                continue
            idx = obj.properties["cell_index"]
            if idx in seen:
                continue
            seen.add(idx)

            syms_ref = obj.references.get("definesSymbols")
            defined_symbols = []
            if syms_ref and syms_ref.objects:
                defined_symbols = sorted(
                    [s.properties["qualified_name"] for s in syms_ref.objects]
                )

            all_cells.append({
                "cell_index": idx,
                "cell_type": obj.properties["cell_type"],
                "code_text": obj.properties.get("code_text") or "",
                "comment_text": obj.properties.get("comment_text") or "",
                "package": obj.properties.get("package") or "",
                "execution_count": obj.properties.get("execution_count"),
                "is_portcullis": obj.properties.get("is_portcullis", False),
                "defined_symbols": defined_symbols,
            })

        all_cells.sort(key=lambda c: c["cell_index"])
        total_cells = len(all_cells)
        paged_cells = all_cells[cell_offset:cell_offset + cell_limit]

        # Attach cell summaries
        cell_sums = _get_cell_summaries(client, source_file)
        for cell in paged_cells:
            s = cell_sums.get(cell["cell_index"])
            if s:
                cell["summary"] = s

        result["cells"] = _envelope(
            paged_cells, total=total_cells,
            offset=cell_offset, limit=cell_limit,
        )

    return result


# ---------------------------------------------------------------------------
# kg_get_cell
# ---------------------------------------------------------------------------

def get_cell(source_file: str, cell_index: int) -> dict[str, Any] | None:
    """Get a single cell's full content + summary."""
    client = get_client()
    cell_col = client.collections.get("ACL2Cell")

    cell_uuid = generate_uuid5(f"cell:{source_file}:{cell_index}")
    obj = cell_col.query.fetch_object_by_id(
        cell_uuid,
        return_references=[
            QueryReference(link_on="definesSymbols",
                           return_properties=["qualified_name", "kind"]),
        ],
    )
    if obj is None:
        return None

    p = obj.properties
    defined_symbols = []
    syms_ref = obj.references.get("definesSymbols")
    if syms_ref and syms_ref.objects:
        defined_symbols = sorted(
            [{"qualified_name": s.properties["qualified_name"],
              "kind": s.properties.get("kind", "")}
             for s in syms_ref.objects],
            key=lambda x: x["qualified_name"],
        )

    result: dict[str, Any] = {
        "notebook_source": p.get("notebook_source", ""),
        "cell_index": p.get("cell_index", cell_index),
        "cell_type": p.get("cell_type", ""),
        "code_text": p.get("code_text") or "",
        "comment_text": p.get("comment_text") or "",
        "package": p.get("package") or "",
        "execution_count": p.get("execution_count"),
        "is_portcullis": p.get("is_portcullis", False),
        "defined_symbols": defined_symbols,
        "stdout": p.get("stdout") or "",
        "execute_result": p.get("execute_result") or "",
    }

    # Cell summary
    sums = _get_cell_summaries(client, source_file)
    s = sums.get(cell_index)
    if s:
        result["summary"] = s

    return result


# ---------------------------------------------------------------------------
# kg_get_summary
# ---------------------------------------------------------------------------

def get_summary(ref_key: str) -> dict[str, Any] | None:
    """Fetch a single summary by its ref_key."""
    client = get_client()
    try:
        col = client.collections.get("ACL2Summary")
    except Exception:
        return None

    # TEXT tokenization splits on special chars, so "equal" is token-based.
    # Fetch candidates and post-filter for an exact ref_key match.
    resp = col.query.fetch_objects(
        filters=Filter.by_property("ref_key").equal(ref_key),
        limit=50,
    )
    match = None
    for obj in resp.objects:
        if obj.properties.get("ref_key") == ref_key:
            match = obj
            break
    if match is None:
        return None

    p = match.properties
    return {
        "scope": p.get("scope", ""),
        "ref_key": p.get("ref_key", ""),
        "what": p.get("what_summary", ""),
        "why": p.get("why_summary", ""),
        "how": p.get("how_summary", ""),
        "source_file": p.get("source_file", ""),
        "cell_index": p.get("cell_index", -1),
        "directory": p.get("directory", ""),
        "symbol_names": p.get("symbol_names", []),
    }


# ---------------------------------------------------------------------------
# kg_list_notebooks
# ---------------------------------------------------------------------------

def list_notebooks(filter_path: str | None = None, *,
                   offset: int = 0, limit: int = 50,
                   ) -> dict[str, Any]:
    """List/filter notebooks with optional summaries."""
    client = get_client()
    nb_col = client.collections.get("ACL2Notebook")

    if filter_path:
        resp = nb_col.query.fetch_objects(
            filters=Filter.by_property("source_file").like(f"*{filter_path}*"),
            limit=10000,
        )
    else:
        resp = nb_col.query.fetch_objects(limit=10000)

    # Sort all results by source_file, then page
    all_nbs = sorted(
        [obj.properties for obj in resp.objects],
        key=lambda x: x["source_file"],
    )
    total = len(all_nbs)
    paged = all_nbs[offset:offset + limit]

    # Attach notebook summaries
    results = []
    for nb in paged:
        sf = nb["source_file"]
        entry: dict[str, Any] = {
            "source_file": sf,
            "cell_count": nb.get("cell_count", 0),
            "code_cell_count": nb.get("code_cell_count", 0),
            "is_bootstrap": nb.get("is_bootstrap", False),
        }
        s = _get_notebook_summary(client, sf)
        if s and s.get("what"):
            entry["summary_what"] = s["what"]
        results.append(entry)

    return _envelope(results, total=total, offset=offset, limit=limit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cell_summaries(client: weaviate.WeaviateClient,
                        source_file: str) -> dict[int, dict[str, str]]:
    """Fetch all cell-level summaries for a notebook."""
    try:
        col = client.collections.get("ACL2Summary")
    except Exception:
        return {}

    resp = col.query.fetch_objects(
        filters=(
            Filter.by_property("scope").equal("cell")
            & Filter.by_property("source_file").equal(source_file)
        ),
        limit=10000,
    )

    sums: dict[int, dict[str, str]] = {}
    for obj in resp.objects:
        p = obj.properties
        if p.get("source_file") != source_file:
            continue
        idx = p.get("cell_index", -1)
        if idx >= 0:
            sums[idx] = {
                "what": p.get("what_summary", ""),
                "why": p.get("why_summary", ""),
                "how": p.get("how_summary", ""),
            }
    return sums


def _get_notebook_summary(client: weaviate.WeaviateClient,
                          source_file: str) -> dict[str, str] | None:
    """Fetch the notebook-level summary."""
    try:
        col = client.collections.get("ACL2Summary")
    except Exception:
        return None

    resp = col.query.fetch_objects(
        filters=(
            Filter.by_property("scope").equal("notebook")
            & Filter.by_property("source_file").equal(source_file)
        ),
        limit=1,
    )
    for obj in resp.objects:
        p = obj.properties
        if p.get("source_file") == source_file:
            return {
                "what": p.get("what_summary", ""),
                "why": p.get("why_summary", ""),
                "how": p.get("how_summary", ""),
            }
    return None
