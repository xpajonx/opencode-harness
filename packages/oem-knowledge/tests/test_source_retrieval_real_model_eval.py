from __future__ import annotations

import importlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import pytest

from oem_knowledge.engine import KnowledgeEngine
from oem_knowledge.services.source_corpus import (
    SOURCE_DENSE_NO_LEXICAL_EVIDENCE_THRESHOLD,
    _SourceIndexStore,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
REAL_FIXTURE = FIXTURE_DIR / "source_retrieval_real_model_eval.json"
R0_FIXTURE = FIXTURE_DIR / "source_retrieval_benchmark.json"

def _tokens(value: str) -> set[str]: return set(re.findall(r"[a-z0-9]+", value.lower()))
def _load_json(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def _materialize(corpus: dict[str, str], root: Path) -> None:
    for rel_path, content in corpus.items():
        path = root / rel_path; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")
def _paths(result: dict) -> list[str]: return [str(row.get("metadata", {}).get("rel_path")) for row in result.get("results", [])]
def _rank(paths: list[str], relevant: list[str]) -> int | None:
    for index, path in enumerate(paths[:5], 1):
        if path in relevant: return index
    return None
def _hit_at_5(paths: list[str], relevant: list[str]) -> bool: return bool(set(paths[:5]) & set(relevant))
def _top1_hit(paths: list[str], relevant: list[str]) -> bool: return bool(paths) and paths[0] in relevant
def _reciprocal_rank(paths: list[str], relevant: list[str]) -> float:
    rank = _rank(paths, relevant); return 1.0 / rank if rank is not None else 0.0
def _dense_scores(result: dict) -> list[float]:
    scores = []
    for row in result.get("results", []):
        try: scores.append(float(row.get("metadata", {}).get("source_diagnostics", {}).get("dense_score", 0.0)))
        except (TypeError, ValueError): scores.append(0.0)
    return scores

def _dense_diagnostics(result: dict) -> tuple[list[bool], list[float]]:
    flags, thresholds = [], []
    for row in result.get("results", []):
        diagnostic = row.get("metadata", {}).get("source_diagnostics", {})
        flags.append(bool(diagnostic.get("dense_evidence", False)))
        thresholds.append(float(diagnostic.get("dense_evidence_threshold", 0.0)))
    return flags, thresholds

def _raw_negative_dense_ceiling(engine: KnowledgeEngine, cases: list[dict], model, dimension: int) -> dict:
    negatives = [case for case in cases if case["kind"] == "negative"]
    store = _SourceIndexStore(engine.source._db_path())
    try:
        embeddings = store.load_embeddings(engine.resolve_embedding_model(), dimension)
        vectors = list(embeddings.values())
        maxima = []
        for case in negatives:
            query_vector = list(model.embed([case["query"]]))[0]
            maxima.append({"id": case["id"], "max_raw_dense_score": float(max(
                (engine.search.cosine_similarity(query_vector, vector) for vector in vectors),
                default=0.0,
            ))})
        overall = float(max((entry["max_raw_dense_score"] for entry in maxima), default=0.0))
        return {"per_negative": maxima, "overall_max": overall}
    finally:
        store.close()

def _validate_fixture(data: dict, corpus: dict[str, str]) -> dict[str, int]:
    assert set(data) == {"version", "corpus_fixture", "cases"}; assert data["version"] == 1; assert data["corpus_fixture"] == R0_FIXTURE.name; assert corpus
    corpus_tokens = set()
    for rel_path, content in corpus.items(): corpus_tokens.update(_tokens(rel_path)); corpus_tokens.update(_tokens(content))
    seen_ids = set(); counts = {"exact_identifier": 0, "semantic_paraphrase": 0, "negative": 0}
    for case in data["cases"]:
        assert set(case) == {"id", "kind", "query", "relevant_paths"}; assert isinstance(case["id"], str) and case["id"] and case["id"] not in seen_ids; seen_ids.add(case["id"]); assert case["kind"] in counts; assert isinstance(case["query"], str) and case["query"].strip(); assert isinstance(case["relevant_paths"], list); assert all(path in corpus for path in case["relevant_paths"]); counts[case["kind"]] += 1
        if case["kind"] == "exact_identifier": assert len(case["relevant_paths"]) == 1 and (case["query"].lower() in case["relevant_paths"][0].lower() or case["query"].lower() in corpus[case["relevant_paths"][0]].lower())
        elif case["kind"] == "semantic_paraphrase": assert len(case["relevant_paths"]) == 1 and not _tokens(case["query"]) & _tokens(Path(case["relevant_paths"][0]).stem)
        else: assert case["relevant_paths"] == [] and not _tokens(case["query"]) & corpus_tokens
    assert counts["exact_identifier"] >= 20; assert counts["semantic_paraphrase"] >= 40; assert counts["negative"] >= 20
    return {"exact": counts["exact_identifier"], "semantic": counts["semantic_paraphrase"], "negative": counts["negative"]}

def _run_cases(engine: KnowledgeEngine, cases: list[dict]) -> list[dict]:
    records = []
    for case in cases:
        result = engine.source.search(case["query"], k=5); assert result.get("status") in {"success", "no_relevant_source_results"}
        paths = _paths(result); relevant = case["relevant_paths"]; dense = _dense_scores(result)
        dense_flags, dense_thresholds = _dense_diagnostics(result)
        records.append({"paths": paths, "status": result.get("status"), "warnings": list(result.get("warnings", [])), "rank": _rank(paths, relevant), "hit_at_5": _hit_at_5(paths, relevant), "top1_hit": _top1_hit(paths, relevant), "reciprocal_rank": _reciprocal_rank(paths, relevant), "filtered_false_positive": case["kind"] == "negative" and bool(result.get("results", [])), "dense_scores": dense, "dense_evidence": dense_flags, "dense_evidence_thresholds": dense_thresholds, "dense_false_positive": case["kind"] == "negative" and any(dense_flags)})
    return records

def _metrics(cases: list[dict], records: list[dict]) -> dict:
    groups = {kind: [r for c, r in zip(cases, records) if c["kind"] == kind] for kind in ("exact_identifier", "semantic_paraphrase", "negative")}; exact, semantic, negative = groups.values()
    return {"exact_top1_hit_rate": sum(r["top1_hit"] for r in exact)/len(exact), "exact_mrr_at_5": sum(r["reciprocal_rank"] for r in exact)/len(exact), "semantic_hit_at_5": sum(r["hit_at_5"] for r in semantic)/len(semantic), "semantic_mrr_at_5": sum(r["reciprocal_rank"] for r in semantic)/len(semantic), "negative_filtered_false_positives": sum(r["filtered_false_positive"] for r in negative), "negative_dense_false_positives": sum(r["dense_false_positive"] for r in negative)}

def _bootstrap_ci(values: list[int], seed: int = 1729, resamples: int = 2000) -> list[float]:
    if not values: return [0.0, 0.0]
    rng = random.Random(seed); n = len(values); means = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples)); low = int(.025*len(means)); return [means[low], means[max(low, int(.975*len(means))-1)]]

def _semantic_transitions(cases, baseline, hybrid):
    transitions = {"baseline_hit_hybrid_hit": 0, "baseline_miss_hybrid_hit": 0, "baseline_hit_hybrid_miss": 0, "baseline_miss_hybrid_miss": 0}
    for case, b, h in zip(cases, baseline, hybrid):
        if case["kind"] != "semantic_paraphrase": continue
        transitions[("baseline_hit_" if b["hit_at_5"] else "baseline_miss_") + ("hybrid_hit" if h["hit_at_5"] else "hybrid_miss")] += 1
    return transitions

@pytest.mark.periodic_eval
def test_source_retrieval_real_model_eval(tmp_path: Path) -> None:
    if os.getenv("OEM_RUN_PERIODIC_EVAL") != "1": pytest.skip("requires OEM_RUN_PERIODIC_EVAL=1")
    real_fixture = _load_json(REAL_FIXTURE); r0_fixture = _load_json(R0_FIXTURE); counts = _validate_fixture(real_fixture, r0_fixture["corpus"]); cases = real_fixture["cases"]
    original_fastembed = sys.modules.pop("fastembed", None); engines = []
    try:
        try: fastembed = importlib.import_module("fastembed")
        except ImportError: pytest.skip("optional fastembed package unavailable")
        text_embedding = getattr(fastembed, "TextEmbedding", None); assert text_embedding is not None and not getattr(text_embedding, "__module__", "").startswith("conftest")
        baseline_root, hybrid_root = tmp_path / "baseline", tmp_path / "hybrid"; _materialize(r0_fixture["corpus"], baseline_root); _materialize(r0_fixture["corpus"], hybrid_root)
        baseline = KnowledgeEngine(project_path=baseline_root); engines.append(baseline); baseline.init_project(str(baseline_root)); baseline.search.set_retrieval_mode("bm25"); assert baseline.source.index().get("embedding_status") == "bm25"
        hybrid = KnowledgeEngine(project_path=hybrid_root); engines.append(hybrid); hybrid.init_project(str(hybrid_root)); hybrid.search.set_retrieval_mode("hybrid"); model = hybrid._load_local_model()
        if model is None: pytest.skip("cached fastembed model unavailable")
        assert model.__class__.__name__ != "MockTextEmbedding" and not model.__class__.__module__.startswith("conftest")
        probe = list(model.embed(["R2 embedding dimension probe"]))
        if (
            len(probe) != 1
            or len(probe[0]) == 0
            or any(not math.isfinite(float(value)) for value in probe[0])
        ): pytest.skip("cached fastembed model returned an unusable vector")
        dimension = len(probe[0]); hybrid_index = hybrid.source.index();
        if hybrid_index.get("embedding_status") != "ready": pytest.fail(f"real model loaded but source indexing was not ready: {hybrid_index.get('embedding_status')}")
        raw_negative = _raw_negative_dense_ceiling(hybrid, cases, model, dimension)
        baseline_records, hybrid_records = _run_cases(baseline, cases), _run_cases(hybrid, cases); bm, hm = _metrics(cases, baseline_records), _metrics(cases, hybrid_records); base = bm["semantic_hit_at_5"]; high = hm["semantic_hit_at_5"]; required = min(1.0, base + .15)
        pairs = [{"id": c["id"], "kind": c["kind"], "baseline_rank": b["rank"], "hybrid_rank": h["rank"], "baseline_hit_at_5": b["hit_at_5"], "hybrid_hit_at_5": h["hit_at_5"], "baseline_top1_hit": b["top1_hit"], "hybrid_top1_hit": h["top1_hit"], "baseline_paths": b["paths"], "hybrid_paths": h["paths"], "baseline_status": b["status"], "hybrid_status": h["status"], "baseline_warnings": b["warnings"], "hybrid_warnings": h["warnings"], "baseline_dense_scores": b["dense_scores"], "hybrid_dense_scores": h["dense_scores"], "baseline_dense_evidence": b["dense_evidence"], "hybrid_dense_evidence": h["dense_evidence"], "baseline_dense_evidence_thresholds": b["dense_evidence_thresholds"], "hybrid_dense_evidence_thresholds": h["dense_evidence_thresholds"]} for c, b, h in zip(cases, baseline_records, hybrid_records)]
        gates = {"baseline_semantic_hit_at_5_at_or_below_0_85": bool(base <= .85), "hybrid_exact_top1_is_1": bool(hm["exact_top1_hit_rate"] == 1.0), "hybrid_filtered_negative_false_positives_are_0": bool(hm["negative_filtered_false_positives"] == 0), "hybrid_dense_negative_false_positives_are_0": bool(hm["negative_dense_false_positives"] == 0), "hybrid_raw_negative_dense_ceiling_below_no_lexical_threshold": bool(raw_negative["overall_max"] < SOURCE_DENSE_NO_LEXICAL_EVIDENCE_THRESHOLD), "hybrid_semantic_hit_at_5_meets_threshold": bool(high >= required)}
        summary = {"fixture": {"version": real_fixture["version"], "corpus_fixture": real_fixture["corpus_fixture"]}, "case_counts": counts, "model": {"class": model.__class__.__name__, "module": model.__class__.__module__, "name": getattr(model, "model_name", hybrid.resolve_embedding_model()), "dimension": dimension}, "dense_thresholds": {"lexical_or_identifier": 0.25, "no_lexical": SOURCE_DENSE_NO_LEXICAL_EVIDENCE_THRESHOLD}, "raw_negative_dense_ceiling": raw_negative, "baseline": bm, "hybrid": hm, "required_hybrid_hit_at_5": required, "paired_uncertainty": {"semantic_hit_at_5_delta": high-base, "bootstrap_seed": 1729, "bootstrap_resamples": 2000, "bootstrap_95_ci": _bootstrap_ci([int(h["hit_at_5"])-int(b["hit_at_5"]) for c,b,h in zip(cases, baseline_records, hybrid_records) if c["kind"] == "semantic_paraphrase"])}, "semantic_transitions": _semantic_transitions(cases, baseline_records, hybrid_records), "paired_cases": pairs, "gate_checks": gates, "gate_status": "PASS" if all(gates.values()) else "FAIL"}
        print(json.dumps(summary, indent=2, sort_keys=True)); assert summary["gate_status"] == "PASS"
    finally:
        for engine in engines: engine.close()
        if original_fastembed is None: sys.modules.pop("fastembed", None)
        else: sys.modules["fastembed"] = original_fastembed
