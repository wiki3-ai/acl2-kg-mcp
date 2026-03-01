"""Integration tests for acl2_kg_mcp.weaviate_client.

These tests require a running Weaviate instance with the ACL2 KG
collections populated.  Set WEAVIATE_HOST if not the default
(host.docker.internal).

Run:
    pytest tests/test_weaviate_client.py -v
"""

import pytest

from acl2_kg_mcp import weaviate_client as wc


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def _ensure_weaviate() -> None:
    """Skip the entire module if Weaviate is unreachable."""
    try:
        client = wc.get_client()
        assert client.is_ready()
    except Exception:
        pytest.skip("Weaviate is not reachable")


# ── kg_stats ─────────────────────────────────────────────────────────

class TestGetStats:
    def test_returns_all_collections(self) -> None:
        stats = wc.get_stats()
        for name in ("ACL2Notebook", "ACL2Cell", "ACL2Symbol",
                      "ACL2Summary", "DoclingPapers"):
            assert name in stats["collections"], f"Missing collection {name}"

    def test_counts_are_positive(self) -> None:
        stats = wc.get_stats()
        for name, count in stats["collections"].items():
            assert count is None or count > 0, f"{name} has count {count}"

    def test_symbol_kinds_present(self) -> None:
        stats = wc.get_stats()
        kinds = stats["symbol_kinds"]
        assert "function" in kinds
        assert "theorem" in kinds
        assert "macro" in kinds

    def test_summary_scopes_present(self) -> None:
        stats = wc.get_stats()
        scopes = stats["summary_scopes"]
        assert "cell" in scopes
        assert "notebook" in scopes


# ── kg_search: symbols ───────────────────────────────────────────────

class TestSearchSymbols:
    def test_semantic_returns_results(self) -> None:
        r = wc.search_symbols("merge sort", mode="semantic", limit=5)
        assert len(r["results"]) > 0
        assert r["limit"] == 5
        assert r["offset"] == 0

    def test_keyword_returns_results(self) -> None:
        r = wc.search_symbols("APPEND", mode="keyword", limit=5)
        assert len(r["results"]) > 0
        # Every result should contain APPEND in the qualified name
        for item in r["results"]:
            assert "APPEND" in item["qualified_name"].upper()

    def test_semantic_has_distance(self) -> None:
        r = wc.search_symbols("binary tree", mode="semantic", limit=3)
        for item in r["results"]:
            assert item["distance"] is not None
            assert isinstance(item["distance"], float)

    def test_pagination_offset(self) -> None:
        page1 = wc.search_symbols("list", mode="semantic", limit=3, offset=0)
        page2 = wc.search_symbols("list", mode="semantic", limit=3, offset=3)
        names1 = {r["qualified_name"] for r in page1["results"]}
        names2 = {r["qualified_name"] for r in page2["results"]}
        # Pages should not overlap
        assert names1.isdisjoint(names2), "Pages overlap"

    def test_result_fields(self) -> None:
        r = wc.search_symbols("car", mode="semantic", limit=1)
        assert len(r["results"]) == 1
        item = r["results"][0]
        assert "qualified_name" in item
        assert "kind" in item
        assert "package" in item


# ── kg_search: cells ─────────────────────────────────────────────────

class TestSearchCells:
    def test_code_semantic(self) -> None:
        r = wc.search_cells("defun factorial", target="code",
                            mode="semantic", limit=5)
        assert len(r["results"]) > 0
        for item in r["results"]:
            assert "notebook_source" in item
            assert "cell_index" in item
            assert "preview" in item

    def test_comment_keyword(self) -> None:
        r = wc.search_cells("theorem", target="comment",
                            mode="keyword", limit=5)
        assert len(r["results"]) > 0

    def test_code_keyword(self) -> None:
        r = wc.search_cells("defun", target="code",
                            mode="keyword", limit=5)
        assert len(r["results"]) == 5
        for item in r["results"]:
            assert "defun" in item["preview"].lower()


# ── kg_search: summaries ─────────────────────────────────────────────

class TestSearchSummaries:
    def test_semantic(self) -> None:
        r = wc.search_summaries("sanity check", mode="semantic", limit=5)
        assert len(r["results"]) > 0
        for item in r["results"]:
            assert "scope" in item
            assert "ref_key" in item
            assert "what" in item

    def test_keyword(self) -> None:
        r = wc.search_summaries("sanity", mode="keyword", limit=5)
        assert len(r["results"]) > 0


# ── kg_search: docs ──────────────────────────────────────────────────

class TestSearchDocs:
    def test_semantic(self) -> None:
        r = wc.search_docs("theorem proving", mode="semantic", limit=3)
        assert len(r["results"]) > 0
        for item in r["results"]:
            assert "paper_title" in item
            assert "text" in item
            assert item["distance"] is not None

    def test_keyword(self) -> None:
        r = wc.search_docs("ACL2", mode="keyword", limit=3)
        assert len(r["results"]) > 0

    def test_paper_filter(self) -> None:
        r = wc.search_docs("verification", mode="semantic",
                           paper_filter="HOList", limit=5)
        for item in r["results"]:
            assert "HOList" in item["paper_title"]


# ── kg_get_symbol ─────────────────────────────────────────────────────

class TestGetSymbol:
    def test_exact_lookup_brr_at(self) -> None:
        """ACL2::BRR@ uses @ which trips up word tokenization;
        UUID lookup must return the exact match."""
        s = wc.get_symbol("ACL2::BRR@")
        assert s is not None
        assert s["qualified_name"] == "ACL2::BRR@"
        assert s["kind"] == "macro"

    def test_includes_definition(self) -> None:
        s = wc.get_symbol("ACL2::BRR@", include=["definition"])
        assert s is not None
        assert "definition" in s
        assert "code" in s["definition"]
        assert "defmacro brr@" in s["definition"]["code"].lower()

    def test_includes_dependencies(self) -> None:
        s = wc.get_symbol("ACL2::BRR@",
                          include=["dependencies"], deps_limit=10)
        assert s is not None
        deps = s["dependencies"]
        assert deps["total"] > 0
        assert len(deps["results"]) <= 10

    def test_dependencies_paging(self) -> None:
        s = wc.get_symbol("ACL2::BRR@",
                          include=["dependencies"],
                          deps_limit=5, deps_offset=0)
        assert s is not None
        page1_names = [d["qualified_name"] for d in s["dependencies"]["results"]]

        s2 = wc.get_symbol("ACL2::BRR@",
                           include=["dependencies"],
                           deps_limit=5, deps_offset=5)
        assert s2 is not None
        page2_names = [d["qualified_name"] for d in s2["dependencies"]["results"]]

        # Pages should not overlap
        assert set(page1_names).isdisjoint(set(page2_names))

    def test_includes_dependents(self) -> None:
        s = wc.get_symbol("COMMON-LISP::APPEND",
                          include=["dependents"], deps_limit=5)
        assert s is not None
        dependents = s["dependents"]
        assert len(dependents["results"]) > 0
        assert dependents["has_more"] is True

    def test_dependents_has_total(self) -> None:
        """Dependents should now report a total count via aggregate."""
        s = wc.get_symbol("COMMON-LISP::APPEND",
                          include=["dependents"], deps_limit=5)
        assert s is not None
        assert s["dependents"]["total"] is not None
        assert s["dependents"]["total"] > 100  # APPEND has many dependents

    def test_dependents_paging(self) -> None:
        s1 = wc.get_symbol("COMMON-LISP::APPEND",
                           include=["dependents"],
                           deps_limit=5, deps_offset=0)
        s2 = wc.get_symbol("COMMON-LISP::APPEND",
                           include=["dependents"],
                           deps_limit=5, deps_offset=5)
        assert s1 is not None and s2 is not None
        names1 = {d["qualified_name"] for d in s1["dependents"]["results"]}
        names2 = {d["qualified_name"] for d in s2["dependents"]["results"]}
        assert names1.isdisjoint(names2), "Dependent pages overlap"

    def test_includes_summary(self) -> None:
        s = wc.get_symbol("ACL2::BRR@",
                          include=["definition", "summary"])
        assert s is not None
        if "summary" in s:
            assert "what" in s["summary"]
            assert "why" in s["summary"]

    def test_not_found(self) -> None:
        s = wc.get_symbol("NONEXISTENT::ZZZZZ")
        assert s is None

    def test_selective_include(self) -> None:
        s = wc.get_symbol("ACL2::BRR@", include=["definition"])
        assert s is not None
        assert "definition" in s
        assert "dependencies" not in s
        assert "dependents" not in s


# ── kg_get_notebook ───────────────────────────────────────────────────

class TestGetNotebook:
    def test_basic(self) -> None:
        nb = wc.get_notebook("books/defsort/defsort.lisp")
        assert nb is not None
        assert nb["source_file"] == "books/defsort/defsort.lisp"
        assert nb["cell_count"] > 0
        assert nb["code_cell_count"] > 0

    def test_cells_included(self) -> None:
        nb = wc.get_notebook("books/defsort/defsort.lisp",
                             cell_limit=5)
        assert nb is not None
        cells = nb["cells"]
        assert len(cells["results"]) <= 5
        assert cells["total"] > 0
        # Cells should be sorted by index
        indices = [c["cell_index"] for c in cells["results"]]
        assert indices == sorted(indices)

    def test_cells_paging(self) -> None:
        p1 = wc.get_notebook("books/defsort/defsort.lisp",
                             cell_limit=3, cell_offset=0)
        p2 = wc.get_notebook("books/defsort/defsort.lisp",
                             cell_limit=3, cell_offset=3)
        assert p1 is not None and p2 is not None
        idx1 = {c["cell_index"] for c in p1["cells"]["results"]}
        idx2 = {c["cell_index"] for c in p2["cells"]["results"]}
        assert idx1.isdisjoint(idx2)

    def test_no_cells(self) -> None:
        nb = wc.get_notebook("books/defsort/defsort.lisp",
                             include_cells=False)
        assert nb is not None
        assert "cells" not in nb

    def test_not_found(self) -> None:
        nb = wc.get_notebook("nonexistent/file.lisp")
        assert nb is None

    def test_cell_defined_symbols(self) -> None:
        nb = wc.get_notebook("books/defsort/defsort.lisp",
                             cell_limit=50)
        assert nb is not None
        # At least some cells should define symbols
        cells_with_syms = [
            c for c in nb["cells"]["results"]
            if c.get("defined_symbols")
        ]
        assert len(cells_with_syms) > 0


# ── kg_get_cell ───────────────────────────────────────────────────────

class TestGetCell:
    def test_basic(self) -> None:
        c = wc.get_cell("books/defsort/defsort.lisp", 2)
        assert c is not None
        assert c["notebook_source"] == "books/defsort/defsort.lisp"
        assert c["cell_index"] == 2
        assert c["cell_type"] == "code"

    def test_has_code(self) -> None:
        c = wc.get_cell("books/defsort/defsort.lisp", 2)
        assert c is not None
        assert len(c["code_text"]) > 0

    def test_not_found(self) -> None:
        c = wc.get_cell("books/defsort/defsort.lisp", 99999)
        assert c is None


# ── kg_get_summary ────────────────────────────────────────────────────

class TestGetSummary:
    def test_notebook_summary(self) -> None:
        """Fetch a known notebook summary by ref_key."""
        s = wc.get_summary("acl2-check.lisp")
        assert s is not None
        assert s["scope"] == "notebook"
        assert len(s["what"]) > 0
        assert len(s["why"]) > 0

    def test_not_found(self) -> None:
        s = wc.get_summary("nonexistent/ref/key/zzz")
        assert s is None


# ── kg_list_notebooks ─────────────────────────────────────────────────

class TestListNotebooks:
    def test_filter(self) -> None:
        r = wc.list_notebooks("defsort", limit=10)
        assert r["total"] == 6
        assert len(r["results"]) == 6
        for nb in r["results"]:
            assert "defsort" in nb["source_file"]

    def test_pagination(self) -> None:
        p1 = wc.list_notebooks("defsort", limit=3, offset=0)
        p2 = wc.list_notebooks("defsort", limit=3, offset=3)
        names1 = {nb["source_file"] for nb in p1["results"]}
        names2 = {nb["source_file"] for nb in p2["results"]}
        assert names1.isdisjoint(names2)
        assert p1["has_more"] is True
        assert p2["has_more"] is False

    def test_sorted_alphabetically(self) -> None:
        r = wc.list_notebooks("defsort", limit=10)
        paths = [nb["source_file"] for nb in r["results"]]
        assert paths == sorted(paths)

    def test_no_filter(self) -> None:
        r = wc.list_notebooks(None, limit=5)
        assert len(r["results"]) == 5
        assert r["has_more"] is True


# ── Envelope structure ────────────────────────────────────────────────

class TestEnvelope:
    def test_envelope_fields(self) -> None:
        r = wc.search_symbols("car", mode="semantic", limit=3)
        assert "results" in r
        assert "total" in r
        assert "offset" in r
        assert "limit" in r
        assert "has_more" in r
        assert isinstance(r["results"], list)
        assert r["offset"] == 0
        assert r["limit"] == 3

    def test_has_more_true(self) -> None:
        r = wc.search_symbols("function", mode="semantic", limit=1)
        assert r["has_more"] is True

    def test_has_more_false(self) -> None:
        r = wc.list_notebooks("defsort/defsort.lisp", limit=50)
        assert r["has_more"] is False


# ── Enriched symbol search ────────────────────────────────────────────

class TestEnrichedSymbolSearch:
    def test_vague_query_gets_enriched(self) -> None:
        """A conceptual query that would get poor symbol-vector hits
        should be enriched via code-cell search."""
        r = wc.search_symbols("binary search tree insertion",
                              mode="semantic", limit=5)
        assert len(r["results"]) > 0
        # If enrichment kicked in, some results should have source="code_cell"
        # or at minimum, the distance should be reasonable
        best_dist = r["results"][0].get("distance")
        if best_dist is not None:
            # Either enrichment improved it or symbol search was adequate
            assert best_dist < 0.55

    def test_precise_query_not_enriched(self) -> None:
        """Good keyword queries should not be enriched (mode=keyword)."""
        r = wc.search_symbols("APPEND", mode="keyword", limit=5)
        for item in r["results"]:
            assert "source" not in item  # no enrichment marker


# ── kg_get_include_book ───────────────────────────────────────────────

class TestGetIncludeBook:
    def test_community_book(self) -> None:
        r = wc.get_include_book("books/defsort/defsort.lisp")
        assert r is not None
        assert r["include_book"] == '(include-book "defsort/defsort" :dir :system)'
        assert r["book_path"] == "defsort/defsort"
        assert r["dir"] == ":system"

    def test_nested_community_book(self) -> None:
        r = wc.get_include_book("books/kestrel/utilities/defmergesort.lisp")
        assert r is not None
        assert '"kestrel/utilities/defmergesort"' in r["include_book"]
        assert ":dir :system" in r["include_book"]

    def test_non_books_path(self) -> None:
        r = wc.get_include_book("acl2-check.lisp")
        assert r is not None
        assert ":dir" not in r["include_book"]
        assert r["include_book"] == '(include-book "acl2-check")'

    def test_not_found(self) -> None:
        r = wc.get_include_book("nonexistent/file.lisp")
        assert r is None
