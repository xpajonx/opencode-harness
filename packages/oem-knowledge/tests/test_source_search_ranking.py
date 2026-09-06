from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest

from oem_knowledge.services.source_corpus import (
    _classify_source_result,
    _detect_source_query_intent,
    _extract_source_identifiers,
    _has_boundary_identifier_match,
    _matched_source_identifiers,
    SourceCorpusService,
)


class TestExtractSourceIdentifiers:
    def test_extracts_dotted_identifiers(self):
        ids = _extract_source_identifiers("chat.ask getNotebook timeout")
        assert "chat.ask" in ids
        assert "getNotebook" in ids

    def test_preserves_camelcase_and_snakecase(self):
        ids = _extract_source_identifiers("getNotebook my_function UPPERCASE")
        assert "getNotebook" in ids
        assert "my_function" in ids
        assert "UPPERCASE" in ids

    def test_stopwords_excluded(self):
        ids = _extract_source_identifiers("the for how to fix a bug")
        for stopword in ("the", "for", "how", "to", "a"):
            assert stopword not in ids

    def test_splits_longer_dotted_identifiers(self):
        ids = _extract_source_identifiers("chat.ask.getNotebook")
        assert "chat.ask.getNotebook" in ids

    def test_empty_query_returns_empty_list(self):
        assert _extract_source_identifiers("") == []


class TestBoundaryIdentifierMatch:
    def test_exact_match_found(self):
        assert _has_boundary_identifier_match(
            "getNotebook", "def getNotebook(): pass"
        )

    def test_substring_not_matched(self):
        assert not _has_boundary_identifier_match(
            "ask", "def chat_ask_helper(): pass"
        )

    def test_dotted_identifier_as_prefix_not_matched(self):
        assert not _has_boundary_identifier_match(
            "chat", "def chat_ask(): pass"
        )

    def test_dotted_identifier_found(self):
        assert _has_boundary_identifier_match(
            "chat.ask", "some code about chat.ask and stuff"
        )

    def test_empty_document_returns_false(self):
        assert not _has_boundary_identifier_match("foo", "")

    def test_empty_string_returns_false(self):
        assert not _has_boundary_identifier_match("foo", "")

    def test_underscore_prefix_identifier_matches(self):
        assert _has_boundary_identifier_match(
            "SourceIndexStore", "class _SourceIndexStore: pass"
        )

    def test_dot_prefixed_identifer_not_matched(self):
        assert not _has_boundary_identifier_match(
            "ask", "def chat.ask(): pass"
        )


class TestCountMatchedIdentifiers:
    def test_zero_when_no_identifiers_match(self):
        assert len(_matched_source_identifiers(
            ["foo", "bar"], "nothing matches here"
        )) == 0

    def test_counts_only_distinct_matching_identifiers(self):
        assert len(_matched_source_identifiers(
            ["foo", "bar", "baz"],
            "foo is here and bar is also here foo again",
        )) == 2

    def test_returns_minimum_of_identifiers_length_and_matches(self):
        assert len(_matched_source_identifiers(
            ["getNotebook", "source_ids"],
            "getNotebook uses source_ids and source_ids again",
        )) == 2

    def test_empty_identifiers_returns_zero(self):
        assert len(_matched_source_identifiers([], "any doc")) == 0


class TestDetectSourceQueryIntent:
    def test_source_intent_detected(self):
        result = _detect_source_query_intent("find chat.ask implementation")
        assert result["source_intent"] is True

    def test_debug_intent_detected(self):
        result = _detect_source_query_intent("debug getNotebook timeout error")
        assert result["debug_intent"] is True

    def test_test_intent_detected(self):
        result = _detect_source_query_intent("test source_ids regression")
        assert result["test_intent"] is True

    def test_doc_intent_detected(self):
        result = _detect_source_query_intent("documentation for guidelines")
        assert result["doc_intent"] is True

    def test_config_intent_detected(self):
        result = _detect_source_query_intent(
            "configure pyproject.toml settings"
        )
        assert result["config_intent"] is True

    def test_mixed_intent_allows_multiple(self):
        result = _detect_source_query_intent(
            "fix getNotebook test regression"
        )
        assert result["debug_intent"] is True
        assert result["test_intent"] is True

    def test_domain_terms_detected(self):
        result = _detect_source_query_intent(
            "chat.ask source_ids notebook"
        )
        assert "chat.ask" in result["identifiers"]
        assert "source_ids" in result["identifiers"]
        assert "notebook" in result.get("domain_terms", [])

    def test_empty_query_returns_no_intent(self):
        result = _detect_source_query_intent("")
        assert result["source_intent"] is False
        assert result["debug_intent"] is False
        assert result["test_intent"] is False
        assert result["doc_intent"] is False


class TestClassifySourceResult:
    def test_agents_md_classified_as_agent_instruction(self):
        assert (
            _classify_source_result(
                ".agents/foo.md", "## This is agent instruction", []
            )
            == "agent_instruction"
        )

    def test_python_implementation_classified(self):
        assert (
            _classify_source_result(
                "src/module.py", "def my_function(): pass", []
            )
            == "implementation_code"
        )

    def test_adapter_classified_by_path(self):
        result = _classify_source_result(
            "adapters/slack/adapter.py",
            "class SlackAdapter: pass",
            [],
        )
        assert result == "adapter_code"

    def test_service_classified_by_path(self):
        result = _classify_source_result(
            "services/llm/service.py",
            "class LLMService: pass",
            [],
        )
        assert result == "service_code"

    def test_client_classified_by_path(self):
        result = _classify_source_result(
            "clients/github/client.py",
            "class GitHubClient: pass",
            [],
        )
        assert result == "client_code"

    def test_relevant_test_with_identifier(self):
        result = _classify_source_result(
            "tests/test_module.py",
            "# my_function tests\ndef test_my_function(): pass",
            ["my_function"],
        )
        assert result == "relevant_test"

    def test_unrelated_test_without_identifier(self):
        result = _classify_source_result(
            "tests/test_other.py",
            "def test_other(): pass",
            [],
        )
        assert result == "unrelated_test"

    def test_generated_or_cache_detected(self):
        assert (
            _classify_source_result(
                "generated/output.txt", "some generated content", []
            )
            == "generated_or_cache"
        )

    def test_readme_classified_as_readme_doc(self):
        assert (
            _classify_source_result(
                "README.md", "# Project", []
            )
            == "readme_doc"
        )

    def test_project_doc_classified(self):
        assert (
            _classify_source_result(
                "docs/guide.md", "# Guide", []
            )
            == "project_doc"
        )

    def test_config_file_classified(self):
        assert (
            _classify_source_result(
                "pyproject.toml", "[tool.poetry]", []
            )
            == "config_file"
        )

    def test_unknown_extension_classified_unknown(self):
        assert (
            _classify_source_result(
                "some/file.xyz", "binary content", []
            )
            == "unknown"
        )

    def test_agents_md_override_by_classifier_rules(self):
        """AGENTS.md must always classify as agent_instruction regardless of content."""
        result = _classify_source_result(
            "AGENTS.md",
            "def some_function(): pass",
            [],
        )
        assert result == "agent_instruction"

    def test_toml_config_does_not_match_short_rel_path_py(self):
        """.toml files should not be misclassified as python code."""
        result = _classify_source_result(
            "config/settings.toml", "[settings]", []
        )
        assert result == "config_file"


class TestSearchRankingIntegration:
    """Integration-style tests that verify the search() method produces correct
    rankings when given controlled mock store data.

    These use direct patching of _SourceIndexStore and _bm25_scores to
    avoid needing a real SQLite index.
    """

    @pytest.fixture
    def engine(self, tmp_path):
        from oem_knowledge.engine import KnowledgeEngine

        eng = KnowledgeEngine(project_path=tmp_path)
        eng.init_project(str(tmp_path))
        oem_dir = tmp_path / ".oem"
        oem_dir.mkdir(parents=True, exist_ok=True)
        (oem_dir / "source_manifest.json").write_text("{}")
        (oem_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (oem_dir / "indexes" / "source_index.sqlite").touch()
        return eng

    def _make_mock_rows(self, entries):
        """Build rows matching iter_chunks() schema from simple (rel_path, document) pairs."""
        rows = []
        for i, (rel_path, document, snippet) in enumerate(entries):
            metadata = {
                "rel_path": rel_path,
                "start_line": 1,
                "end_line": len(document.splitlines()),
            }
            rows.append({
                "id": i + 1,
                "rel_path": rel_path,
                "document": document,
                "snippet": snippet or document,
                "metadata": metadata,
                "path_text": rel_path,
                "symbols_text": "",
                "start_line": 1,
                "end_line": len(document.splitlines()),
            })
        return rows

    @pytest.mark.parametrize(
        ("query", "document", "bm25_score", "dense_score", "expected_status", "expected_evidence", "expected_threshold", "expected_component_match"),
        [
            ("source", "class SourceManifest: pass", 0.0, 0.30, "success", True, 0.25, True),
            ("quantum entanglement laboratory measurements", "def actual_handler(): return 1", 0.0, 0.50, "no_relevant_source_results", False, 0.55, False),
            ("quantum entanglement laboratory measurements", "def actual_handler(): return 1", 0.0, 0.60, "success", True, 0.55, False),
            ("quantum entanglement laboratory measurements", "def actual_handler(): return 1", 0.10, 0.30, "success", True, 0.25, False),
        ],
    )
    def test_hybrid_dense_evidence_uses_conditional_threshold(
        self,
        engine,
        query,
        document,
        bm25_score,
        dense_score,
        expected_status,
        expected_evidence,
        expected_threshold,
        expected_component_match,
    ):
        class DenseFakeStore:
            def __init__(self, rows):
                self.rows = rows

            def iter_chunks(self):
                return self.rows

            def load_embeddings(self, model_name, dimension):
                assert dimension == 2
                return {
                    (row["id"], row["content_hash"]): [
                        dense_score,
                        (1.0 - dense_score**2) ** 0.5,
                    ]
                    for row in self.rows
                }

            def close(self):
                pass

        rows = self._make_mock_rows(
            [("src/implementation.py", document, None)]
        )
        rows[0]["content_hash"] = "hash-1"
        model_name = engine.resolve_embedding_model()
        manifest = {
            "embedding": {"status": "ready", "model": model_name, "dimension": 2}
        }
        with patch(
            "oem_knowledge.services.source_corpus._SourceIndexStore",
            return_value=DenseFakeStore(rows),
        ), patch.object(engine.source, "_load_manifest", return_value=manifest), patch.object(
            engine.source, "_bm25_scores", return_value=[bm25_score]
        ), patch.object(engine.search, "set_retrieval_mode", wraps=engine.search.set_retrieval_mode), patch.object(
            engine.search, "embed", return_value=[[1.0, 0.0]]
        ):
            engine.search.set_retrieval_mode("hybrid")
            result = engine.source.search(query, k=1)

        assert result["status"] == expected_status
        if result["results"]:
            diagnostics = result["results"][0]["metadata"]["source_diagnostics"]
            assert diagnostics["dense_available"] is True
            assert diagnostics["dense_evidence"] is expected_evidence
            assert diagnostics["dense_evidence_threshold"] == expected_threshold
            assert diagnostics["lexical_component_match"] is expected_component_match
        else:
            assert expected_status == "no_relevant_source_results"

    def test_agents_md_penalized_over_implementation(self, engine):
        """AGENTS.md should rank below implementation code for a code query."""
        rows = self._make_mock_rows([
            (
                "AGENTS.md",
                "## Agent instructions for the project.",
                None,
            ),
            (
                "src/chat.py",
                "def ask(): return chat.ask(query)",
                None,
            ),
        ])
        with patch(
            "oem_knowledge.services.source_corpus._SourceIndexStore",
            return_value=_MockStore(rows),
        ), patch.object(
            engine.source, "_bm25_scores", return_value=[0.8, 0.6]
        ):
            result = engine.source.search(
                "chat.ask implementation", k=2
            )
        assert result["status"] == "success"
        results = result["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/chat.py" in rel_paths
        assert "AGENTS.md" not in rel_paths

    def test_relevant_test_boosted_for_test_query(self, engine):
        """Test query should boost relevant_test results."""
        rows = self._make_mock_rows([
            (
                "src/chat.py",
                "def ask(): return chat.ask(query)",
                None,
            ),
            (
                "tests/test_chat.py",
                "def test_ask(): chat.ask returns correct result",
                None,
            ),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.5, 0.4]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search(
                    "test chat.ask", k=2
                )
        assert result["status"] == "success"
        results = result["results"]
        diagnostics = {}
        for r in results:
            rel = r["metadata"]["rel_path"]
            diagnostics[rel] = r["metadata"].get("source_diagnostics", {})
        test_diag = diagnostics.get("tests/test_chat.py", {})
        assert test_diag.get("source_type") == "relevant_test", (
            f"test_chat.py should be relevant_test, got {test_diag.get('source_type')}"
        )
        assert "relevant_test" in test_diag.get("ranking_boosts", {}), (
            f"relevant_test boost missing in {test_diag.get('ranking_boosts')}"
        )

    def test_unrelated_test_penalized_for_source_query(self, engine):
        """Unrelated test should be penalized for non-test source query."""
        rows = self._make_mock_rows([
            (
                "src/module.py",
                "def my_function(): pass",
                None,
            ),
            (
                "tests/test_other.py",
                "def test_unrelated(): pass",
                None,
            ),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.4, 0.5]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search(
                    "find my_function", k=2
                )
        assert result["status"] == "success"
        results = result["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/module.py" in rel_paths
        assert "tests/test_other.py" not in rel_paths

    def test_generated_cache_heavily_penalized(self, engine):
        """Generated/cache files should be heavily penalized for any query."""
        rows = self._make_mock_rows([
            (
                "src/module.py",
                "def real_code(): pass",
                None,
            ),
            (
                "generated/output.txt",
                "## This file is auto-generated.",
                None,
            ),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.3, 0.7]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search(
                    "find code", k=2
                )
        assert result["status"] == "success"
        results = result["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/module.py" in rel_paths
        assert "generated/output.txt" not in rel_paths

    def test_diagnostics_shape_is_present(self, engine):
        """Every search result should include source_diagnostics."""
        rows = self._make_mock_rows([
            ("src/module.py", "def foo(): pass", None),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.5]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search("foo", k=1)
        assert result["status"] == "success"
        diag = result["results"][0]["metadata"].get("source_diagnostics", {})
        assert "source_type" in diag
        assert "base_score" in diag
        assert "final_score" in diag
        assert "ranking_reason" in diag
        assert "ranking_boosts" in diag
        assert "ranking_penalties" in diag

    def test_exact_identifier_boost_dominates_bm25(self, engine):
        """Exact identifier match should overcome lower BM25."""
        rows = self._make_mock_rows([
            (
                "src/chat.py",
                "def ask(): return chat.ask(query)",
                None,
            ),
            (
                "docs/general.md",
                "# General documentation about everything",
                None,
            ),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.3, 0.9]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search(
                    "chat.ask", k=2
                )
        assert result["status"] == "success"
        results = result["results"]
        src_rank = next(i for i, r in enumerate(results) if "src/chat.py" in r["metadata"]["rel_path"])
        doc_matches = [i for i, r in enumerate(results) if "docs/general.md" in r["metadata"]["rel_path"]]
        if doc_matches:
            assert src_rank < doc_matches[0], (
                f"src/chat.py at rank {src_rank} should be above docs at rank {doc_matches[0]}"
            )

    def test_boundary_matching_prevents_false_positive(self, engine):
        """Identifier 'ask' should NOT match document containing only 'chat.ask'."""
        rows = self._make_mock_rows([
            (
                "src/asking.py",
                "def helper_ask(): pass  # the only 'ask' reference",
                None,
            ),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.5]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search("ask", k=1)
        assert result["status"] == "success"
        diag = result["results"][0]["metadata"].get("source_diagnostics", {})
        if diag.get("ranking_boosts", {}).get("exact_identifier"):
            pytest.fail(
                f"query 'ask' should not produce exact_identifier boost for "
                f"doc containing only 'helper_ask' in a function name: {diag}"
            )

    def test_source_search_symbol_definition_boosts_exact_function_definition(self, engine):
        """Definition file ranks above prose-only mention; diagnostics show symbol_definition boost and reason."""
        rows = self._make_mock_rows([
            (
                "docs/notebook.md",
                "# Notebook\nget_notebook is a helper mentioned here in prose only.",
                None,
            ),
            (
                "src/notebook.py",
                "def get_notebook(source_ids):\n    return chat.ask(source_ids)",
                None,
            ),
        ])
        # BM25 slightly favors prose doc; definition should still win via symbol boost
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.6, 0.5]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search("get_notebook", k=2)
        assert result["status"] == "success"
        results = result["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        # definition file ranks first
        assert rel_paths[0] == "src/notebook.py"
        diag = results[0]["metadata"]["source_diagnostics"]
        assert any("symbol definition: get_notebook" in reason for reason in diag["ranking_reason"])
        assert "symbol_definition" in diag.get("ranking_boosts", {})

    def test_empty_query_returns_no_results(self, engine):
        """Empty query should return success but empty results."""
        rows = self._make_mock_rows([
            ("src/module.py", "def foo(): pass", None),
        ])
        with patch.object(
            engine.source, "_bm25_scores", return_value=[0.0]
        ):
            with patch(
                "oem_knowledge.services.source_corpus._SourceIndexStore",
                return_value=_MockStore(rows),
            ):
                result = engine.source.search("", k=1)
        assert result["status"] == "success"


class _MockStore:
    """Stand-in for _SourceIndexStore that returns pre-canned rows."""

    def __init__(self, rows):
        self._rows = rows

    def iter_chunks(self):
        return self._rows

    def close(self):
        pass


class TestIndexedSearchRanking:
    @pytest.fixture
    def indexed_engine(self, tmp_path):
        from oem_knowledge.engine import KnowledgeEngine

        # Create files
        (tmp_path / "AGENTS.md").write_text("The agents should use NotebookLM for retrieval.")
        (tmp_path / "README.md").write_text("Welcome to the project. NotebookLM is integrated.")
        
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "adapter.py").write_text("def get_notebook(source_ids):\n    chat.ask(source_ids)")
        
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_adapter.py").write_text("def test_get_notebook():\n    # test get_notebook and chat.ask\n    pass")
        (tests_dir / "test_unrelated.py").write_text("def test_something_else():\n    pass")
        
        gen_dir = tmp_path / "src/generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "notebooklm_adapter.py").write_text("def get_notebook(source_ids):\n    chat.ask(source_ids)")
        
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "large_doc.md").write_text("This is a large low density documentation file with a lot of text. 'NotebookLM' is mentioned here once.\n" + ("a " * 1500))
        
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = \"test-project\"")

        # Create execution/ and agent/ directories and files
        execution_dir = tmp_path / "execution"
        execution_dir.mkdir(parents=True, exist_ok=True)
        (execution_dir / "notebooklm_adapter.py").write_text("def call_execution_script():\n    # NotebookLM source_ids workaround\n    pass")
        (execution_dir / "content_taxonomy.py").write_text("def normalize_post_type():\n    # infer_content_function taxonomy\n    pass")
        (execution_dir / "x_autoresearch_loop.py").write_text("def run_loop():\n    # execute search\n    pass")
        
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "_utils.py").write_text("def call_execution_script(script_name, action, input_data, cwd=None, timeout=300):\n    # helper for agents\n    pass")

        # Add x_source_discovery.py with discover_sources function
        (execution_dir / "x_source_discovery.py").write_text("def discover_sources():\n    # discover sources from arxiv and youtube\n    pass")

        # Create a large implementation file (>2000 chars) to test large_low_density penalty
        large_func_body = "    print('processing data')\n" * 250  # ~7500 chars
        (execution_dir / "virality_scorer.py").write_text(
            "import json\n\ndef score_virality(posts, source_point):\n"
            + large_func_body
            + "    return 0.85\n"
        )
        
        (tests_dir / "test_notebooklm_adapter.py").write_text("def test_notebooklm_adapter():\n    # test notebooklm_adapter and call_execution_script\n    pass")

        # Create a metadata-only file by writing a very large file under src/
        src_giant = tmp_path / "src" / "giant_bundle.js"
        src_giant.write_text("console.log('giant');\n" + ("a" * 600000))

        eng = KnowledgeEngine(project_path=tmp_path)
        eng.init_project(str(tmp_path))
        
        # Run indexing
        stats = eng.source.index()
        assert stats["status"] == "success"
        
        return eng

    def test_source_search_exact_function_match_beats_agents_md(self, indexed_engine):
        res = indexed_engine.source.search("notebooklm get_notebook timeout source_ids explicitly chat.ask", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        
        # src/adapter.py should rank above AGENTS.md (which is excluded under the new guardrails)
        assert "src/adapter.py" in rel_paths
        assert "AGENTS.md" not in rel_paths

    def test_source_search_identifier_multi_match_boosts_implementation_file(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook chat.ask", k=10)
        results = res["results"]
        assert results[0]["metadata"]["rel_path"] == "src/adapter.py"
        assert "matched identifiers (2)" in results[0]["metadata"]["source_diagnostics"]["ranking_reason"]

    def test_source_search_adapter_file_beats_broad_docs(self, indexed_engine):
        res = indexed_engine.source.search("notebooklm get_notebook", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/adapter.py" in rel_paths
        assert "README.md" not in rel_paths
        assert "docs/large_doc.md" not in rel_paths

    def test_source_search_unrelated_test_downranked(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/adapter.py" in rel_paths
        assert "tests/test_unrelated.py" not in rel_paths

    def test_source_search_relevant_test_can_rank_when_exact_identifier_matches(self, indexed_engine):
        res = indexed_engine.source.search("test get_notebook", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        # relevant test is test_adapter.py, it should rank first or near first
        assert "tests/test_adapter.py" in rel_paths[:2]

    def test_source_search_query_mentioning_tests_allows_relevant_test_to_rank(self, indexed_engine):
        res = indexed_engine.source.search("test get_notebook", k=10)
        results = res["results"]
        assert results[0]["metadata"]["source_diagnostics"]["query_intent"]["test_intent"] is True
        assert results[0]["metadata"]["rel_path"] == "tests/test_adapter.py"

    def test_source_search_generated_cache_downranked(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook chat.ask", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/adapter.py" in rel_paths
        assert "src/generated/notebooklm_adapter.py" in rel_paths
        src_idx = rel_paths.index("src/adapter.py")
        gen_idx = rel_paths.index("src/generated/notebooklm_adapter.py")
        assert src_idx < gen_idx
        
        # Check that generated file has penalty
        gen_result = next(r for r in results if r["metadata"]["rel_path"] == "src/generated/notebooklm_adapter.py")
        assert "generated_or_cache" in gen_result["metadata"]["source_diagnostics"]["ranking_penalties"]

    def test_source_search_large_doc_does_not_beat_exact_code_match(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/adapter.py" in rel_paths
        assert "docs/large_doc.md" not in rel_paths

    def test_source_search_ranking_diagnostics_present(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        for r in results:
            diag = r["metadata"]["source_diagnostics"]
            assert "source_type" in diag
            assert "base_score" in diag
            assert "final_score" in diag
            assert "ranking_reason" in diag
            assert "ranking_boosts" in diag
            assert "ranking_penalties" in diag

    def test_source_search_dotted_identifier_chat_ask_preserved(self, indexed_engine):
        res = indexed_engine.source.search("chat.ask", k=10)
        results = res["results"]
        assert "chat.ask" in results[0]["metadata"]["source_diagnostics"]["ranking_reason"][0]

    def test_source_search_snake_case_identifier_source_ids_preserved(self, indexed_engine):
        res = indexed_engine.source.search("source_ids", k=10)
        results = res["results"]
        assert any("source_ids" in r for r in results[0]["metadata"]["source_diagnostics"]["ranking_reason"])

    def test_source_search_uppercase_identifier_get_notebook_preserved(self, indexed_engine):
        res = indexed_engine.source.search("GET_NOTEBOOK", k=10)
        results = res["results"]
        assert any("GET_NOTEBOOK" in r for r in results[0]["metadata"]["source_diagnostics"]["ranking_reason"])

    def test_agents_md_penalized_for_implementation_query(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        assert "AGENTS.md" not in [r["metadata"]["rel_path"] for r in results]

    def test_bm25_base_cannot_overpower_exact_code_identifier_multi_match(self, indexed_engine):
        res = indexed_engine.source.search("notebooklm get_notebook source_ids chat.ask", k=10)
        results = res["results"]
        # README.md has lots of NotebookLM matches (high BM25), but src/adapter.py has exact identifier matches.
        # src/adapter.py must rank above README.md.
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "src/adapter.py" in rel_paths
        assert "README.md" not in rel_paths

    def test_source_search_identifier_matching_is_boundary_safe(self, indexed_engine):
        res = indexed_engine.source.search("ask", k=10)
        results = res["results"]
        # 'ask' query should not match 'chat.ask' as an exact identifier boost in src/adapter.py
        src_res = [r for r in results if r["metadata"]["rel_path"] == "src/adapter.py"]
        if src_res:
            boosts = src_res[0]["metadata"]["source_diagnostics"]["ranking_boosts"]
            assert "exact_identifier" not in boosts

    def test_source_search_generated_python_file_classified_as_generated_not_implementation(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        assert "src/generated/notebooklm_adapter.py" not in [r["metadata"]["rel_path"] for r in results]

    def test_source_search_impl_query_ranks_code_above_relevant_test(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        src_idx = rel_paths.index("src/adapter.py")
        test_idx = rel_paths.index("tests/test_adapter.py")
        assert src_idx < test_idx

    def test_source_search_agent_instruction_query_can_rank_agents_md(self, indexed_engine):
        res = indexed_engine.source.search("agent instruction notebooklm", k=10)
        results = res["results"]
        agents_res = next(r for r in results if r["metadata"]["rel_path"] == "AGENTS.md")
        # No agent_instruction penalty since doc_intent is active
        assert "agent_instruction" not in agents_res["metadata"]["source_diagnostics"]["ranking_penalties"]
        assert results[0]["metadata"]["rel_path"] == "AGENTS.md"

    def test_source_search_all_results_have_source_diagnostics(self, indexed_engine):
        res = indexed_engine.source.search("notebooklm", k=10)
        results = res["results"]
        for r in results:
            assert "source_diagnostics" in r["metadata"]

    def test_source_search_tie_breakers_are_deterministic(self, indexed_engine):
        res1 = indexed_engine.source.search("notebooklm", k=10)
        res2 = indexed_engine.source.search("notebooklm", k=10)
        assert len(res1["results"]) > 0
        assert [r["id"] for r in res1["results"]] == [r["id"] for r in res2["results"]]

    def test_source_search_symbol_definition_boosts_exact_function_definition(self, indexed_engine):
        # We search for "get_notebook".
        # src/adapter.py defines the function: "def get_notebook(source_ids):"
        # tests/test_adapter.py contains a broad mention in a comment: "# test get_notebook and chat.ask"
        # Since src/adapter.py has the symbol definition, it should get BOOST_SYMBOL_DEFINITION boost.
        res = indexed_engine.source.search("get_notebook", k=10)
        results = res["results"]
        
        # Verify diagnostics show the symbol definition boost for src/adapter.py
        src_diag = next(r for r in results if r["metadata"]["rel_path"] == "src/adapter.py")["metadata"]["source_diagnostics"]
        assert any("symbol definition: get_notebook" in reason for reason in src_diag["ranking_reason"])
        
        # Verify tests/test_adapter.py does not have the symbol definition boost
        test_diag = next(r for r in results if r["metadata"]["rel_path"] == "tests/test_adapter.py")["metadata"]["source_diagnostics"]
        assert not any("symbol definition" in reason for reason in test_diag["ranking_reason"])

    def test_source_index_includes_execution_python_files(self, indexed_engine):
        manifest = indexed_engine.source._load_manifest()
        files = manifest.get("files", {})
        assert "execution/notebooklm_adapter.py" in files
        assert "execution/content_taxonomy.py" in files
        assert "execution/x_autoresearch_loop.py" in files

    def test_source_index_includes_agent_python_files(self, indexed_engine):
        manifest = indexed_engine.source._load_manifest()
        files = manifest.get("files", {})
        assert "agent/_utils.py" in files

    def test_source_search_exact_file_query_returns_execution_file(self, indexed_engine):
        res = indexed_engine.source.search("execution/content_taxonomy.py")
        assert res["status"] == "success"
        assert len(res["results"]) > 0
        assert res["results"][0]["metadata"]["rel_path"] == "execution/content_taxonomy.py"

    def test_source_search_exact_file_query_returns_agent_file(self, indexed_engine):
        res = indexed_engine.source.search("agent/_utils.py")
        assert res["status"] == "success"
        assert len(res["results"]) > 0
        assert res["results"][0]["metadata"]["rel_path"] == "agent/_utils.py"

    def test_metadata_only_file_does_not_receive_implementation_boost_without_content_match(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook")
        results = res["results"]
        giant_res = [r for r in results if "giant_bundle" in r["metadata"]["rel_path"]]
        if giant_res:
            diag = giant_res[0]["metadata"]["source_diagnostics"]
            assert "implementation_code" not in diag["ranking_boosts"]
            assert "metadata_only result" in diag["ranking_reason"]

    def test_metadata_only_file_does_not_outrank_real_source_content(self, indexed_engine):
        res = indexed_engine.source.search("get_notebook")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]

        # Real source code should outrank the metadata-only giant file
        if "src/giant_bundle.js" in rel_paths:
            giant_idx = rel_paths.index("src/giant_bundle.js")
            src_idx = rel_paths.index("src/adapter.py")
            assert src_idx < giant_idx

    def test_implementation_code_boost_requires_positive_evidence(self, indexed_engine):
        # If we query an unrelated term that exists nowhere, normal results are empty
        res = indexed_engine.source.search("quantum entanglement laboratory measurements")
        assert res["status"] == "no_relevant_source_results"
        assert len(res["results"]) == 0

    def test_notebooklm_adapter_query_returns_execution_adapter_above_tests(self, indexed_engine):
        res = indexed_engine.source.search("NotebookLM source_ids workaround GET_NOTEBOOK")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        
        # execution/notebooklm_adapter.py should rank above tests/test_notebooklm_adapter.py
        exec_idx = rel_paths.index("execution/notebooklm_adapter.py")
        if "tests/test_notebooklm_adapter.py" in rel_paths:
            test_idx = rel_paths.index("tests/test_notebooklm_adapter.py")
            assert exec_idx < test_idx

    def test_content_taxonomy_query_returns_execution_file_above_docs(self, indexed_engine):
        res = indexed_engine.source.search("normalize_post_type infer_content_function")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        
        # execution/content_taxonomy.py should rank above docs/large_doc.md
        exec_idx = rel_paths.index("execution/content_taxonomy.py")
        if "docs/large_doc.md" in rel_paths:
            doc_idx = rel_paths.index("docs/large_doc.md")
            assert exec_idx < doc_idx

    def test_unrelated_query_returns_no_normal_results_or_below_threshold_only(self, indexed_engine):
        res = indexed_engine.source.search("unrelated random task with no project memory")
        assert res["status"] == "no_relevant_source_results"
        assert len(res["results"]) == 0

    def test_zero_and_negative_score_results_are_suppressed_from_normal_results(self, indexed_engine):
        res = indexed_engine.source.search("unrelated random task with no project memory")
        assert len(res["results"]) == 0

    def test_source_index_respects_explicit_exclude_for_execution_dir(self, tmp_path):
        from oem_knowledge.engine import KnowledgeEngine
        import yaml
        
        # Write config file excluding execution/
        oem_dir = tmp_path / ".oem"
        oem_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "version": 1,
            "include": ["src/**", "execution/**"],
            "exclude": ["execution/**", ".git/**"]
        }
        (oem_dir / "source_index_config.yml").write_text(yaml.safe_dump(config_data))
        
        execution_dir = tmp_path / "execution"
        execution_dir.mkdir(parents=True, exist_ok=True)
        (execution_dir / "test.py").write_text("def test(): pass")
        
        eng = KnowledgeEngine(project_path=tmp_path)
        eng.init_project(str(tmp_path))
        stats = eng.source.index()
        assert stats["status"] == "success"
        
        # Since execution/ is explicitly excluded, it should be in warnings and skipped files list
        discovery = eng.source.discover_files()
        assert any("implementation_directory_skipped:execution" in w for w in discovery.warnings)

    def test_exact_file_query_rejects_path_outside_project(self, indexed_engine):
        res = indexed_engine.source.search("../../outside.py")
        assert res["status"] == "no_relevant_source_results"

    def test_exact_file_query_normalizes_absolute_path_inside_project(self, indexed_engine):
        project_root = indexed_engine.source._project_root()
        abs_path = str((project_root / "execution/content_taxonomy.py").resolve())
        res = indexed_engine.source.search(abs_path)
        assert res["status"] == "success"
        assert len(res["results"]) > 0
        assert res["results"][0]["metadata"]["rel_path"] == "execution/content_taxonomy.py"

    def test_exact_file_query_beats_matching_test_file(self, indexed_engine):
        res = indexed_engine.source.search("execution/notebooklm_adapter.py")
        assert res["status"] == "success"
        assert len(res["results"]) > 0
        assert res["results"][0]["metadata"]["rel_path"] == "execution/notebooklm_adapter.py"
        
        # Ensure it outranks tests/test_notebooklm_adapter.py
        rel_paths = [r["metadata"]["rel_path"] for r in res["results"]]
        if "tests/test_notebooklm_adapter.py" in rel_paths:
            assert rel_paths.index("execution/notebooklm_adapter.py") < rel_paths.index("tests/test_notebooklm_adapter.py")

    def test_metadata_only_boosts_are_suppressed_before_source_type_boosts(self, indexed_engine):
        # Search for "giant bundle" (matches giant metadata-only file)
        res = indexed_engine.source.search("giant bundle")
        results = res["results"]
        giant_res = [r for r in results if "giant_bundle" in r["metadata"]["rel_path"]]
        if giant_res:
            diag = giant_res[0]["metadata"]["source_diagnostics"]
            assert "implementation_code" not in diag["ranking_boosts"]
            assert diag["ranking_penalties"]["metadata_only"] == 10.0

    def test_doc_query_can_return_readme_result(self, indexed_engine):
        res = indexed_engine.source.search("readme installation documentation")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "README.md" in rel_paths

    def test_agent_instruction_query_can_return_agents_md(self, indexed_engine):
        res = indexed_engine.source.search("agent instruction notebooklm")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "AGENTS.md" in rel_paths

    def test_config_query_can_return_config_file(self, indexed_engine):
        res = indexed_engine.source.search("pyproject.toml config settings")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "pyproject.toml" in rel_paths

    def test_execution_content_taxonomy_returns_impl_file_rank_one(self, indexed_engine):
        res = indexed_engine.source.search("execution/content_taxonomy.py")
        assert res["status"] == "success"
        assert len(res["results"]) > 0
        assert res["results"][0]["metadata"]["rel_path"] == "execution/content_taxonomy.py"

    def test_normalize_post_type_infer_content_function_returns_impl_before_tests(self, indexed_engine):
        res = indexed_engine.source.search("normalize_post_type infer_content_function content taxonomy implementation")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        exec_idx = rel_paths.index("execution/content_taxonomy.py")
        if "tests/test_notebooklm_adapter.py" in rel_paths:
            test_idx = rel_paths.index("tests/test_notebooklm_adapter.py")
            assert exec_idx < test_idx

    def test_agent_utils_call_execution_script_returns_impl_file(self, indexed_engine):
        res = indexed_engine.source.search("agent/_utils.py call_execution_script")
        assert res["status"] == "success"
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "agent/_utils.py" in rel_paths

    def test_x_source_discovery_returns_impl_file(self, indexed_engine):
        res = indexed_engine.source.search("execution/x_source_discovery.py discover sources")
        assert res["status"] == "success"
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "execution/x_source_discovery.py" in rel_paths

    def test_nonsense_source_query_returns_empty(self, indexed_engine):
        res = indexed_engine.source.search("asdfghjkl qwertyuiop nonexistent symbol")
        assert res["status"] == "no_relevant_source_results"
        assert len(res["results"]) == 0

    def test_obsidian_plugins_excluded_by_default(self, indexed_engine):
        manifest = indexed_engine.source._load_manifest()
        files = manifest.get("files", {})
        obsidian_files = [k for k in files if ".obsidian" in k]
        assert len(obsidian_files) == 0

    def test_stale_config_migrates_to_include_execution_and_agent(self, tmp_path):
        from oem_knowledge.engine import KnowledgeEngine
        import yaml

        oem_dir = tmp_path / ".oem"
        oem_dir.mkdir(parents=True, exist_ok=True)
        (oem_dir / "source_index_config.yml").write_text(yaml.safe_dump({
            "include": ["src/**", "packages/**", "tests/**", "docs/**",
                         "README.md", "AGENTS.md", "pyproject.toml", "package.json"],
            "max_read_lines": 400,
        }), encoding="utf-8")

        eng = KnowledgeEngine(project_path=tmp_path)
        eng.init_project(str(tmp_path))
        config = eng.source.load_config()
        assert "execution/**" in config.include
        assert "agent/**" in config.include

    def test_large_implementation_file_not_filtered_by_weak_result_threshold(self, indexed_engine):
        res = indexed_engine.source.search("score virality posts source point")
        results = res["results"]
        rel_paths = [r["metadata"]["rel_path"] for r in results]
        assert "execution/virality_scorer.py" in rel_paths

    def test_large_implementation_file_no_false_large_low_density_penalty(self, indexed_engine):
        res = indexed_engine.source.search("score virality posts")
        results = res["results"]
        for r in results:
            if "virality" in r["metadata"]["rel_path"]:
                diag = r["metadata"]["source_diagnostics"]
                assert "large_low_density" not in diag["ranking_penalties"], (
                    f"{r['metadata']['rel_path']} should not get "
                    "large_low_density penalty"
                )


class TestHasSymbolDefinition:
    def test_matches_python_def(self):
        from oem_knowledge.services.source_corpus import _has_symbol_definition
        assert _has_symbol_definition("my_func", "def my_func(): pass")
        assert not _has_symbol_definition("my_func", "def other_func(): pass")

    def test_matches_js_const_and_function(self):
        from oem_knowledge.services.source_corpus import _has_symbol_definition
        assert _has_symbol_definition("myClass", "class myClass {}")
        assert _has_symbol_definition("myVar", "const myVar = 1")
        assert _has_symbol_definition("myFunc", "function myFunc() {}")


class TestMatchedSourceIdentifiers:
    def test_matched_source_identifiers_empty(self):
        assert _matched_source_identifiers([], "any doc") == []
        assert _matched_source_identifiers(["foo"], "") == []

    def test_matched_source_identifiers_mixed(self):
        identifiers = ["foo", "bar", "baz"]
        doc = "foo matches, but bar matches too, not bazz"
        matched = _matched_source_identifiers(identifiers, doc)
        assert matched == ["foo", "bar"]
