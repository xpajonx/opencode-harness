# Source retrieval benchmark

Status: completed for the source-retrieval hybrid gate.

## Frozen fixture

- Fixture: `packages/oem-knowledge/tests/fixtures/source_retrieval_benchmark.json`
- Fixture version: 1
- Corpus: 26 synthetic source files with implementation, test, documentation, generated, agent, config, and unrelated entries
- Cases: 8 exact-identifier and 13 semantic-paraphrase queries; 10 semantic cases have eligible indexed targets
- Excluded fixture-only targets: `examples/demo.py`, `config/settings.yml`, and `generated/summary.txt`
- Metric cutoff: 5 results
- Command: `uv run pytest packages/oem-knowledge/tests/test_source_retrieval_benchmark.py -q -s`

The exact cases are the regression gate. Every labeled implementation target must remain at rank 1. Semantic hit@5 and MRR@5 are measured over the 10 eligible cases. Search timings use repeated `perf_counter` measurements. Source database size includes SQLite, WAL, and shared-memory sidecar files; the manifest size is measured separately.

## Decision rule

The approved hybrid acceptance target requires at least a 15 percentage-point semantic hit@5 improvement over BM25 with no exact top-1 regression. The precondition for attempting hybrid retrieval is a BM25 semantic hit@5 of 0.85 or lower. A semantic miss is a labeled target absent from the first five results.

## Measured result

| Metric | BM25 baseline | Hybrid proxy embedder |
| --- | ---: | ---: |
| Exact top-1 hit rate | 1.0000 | 1.0000 |
| Exact MRR@5 | 1.0000 | 1.0000 |
| Semantic hit@5 (10 eligible cases) | 0.5000 | 0.7000 |
| Semantic MRR@5 (10 eligible cases) | 0.4200 | 0.6000 |
| Semantic hit@5 gain | - | +0.2000 |
| Index latency (ms) | 117.06 | 101.30 |
| Median search latency (ms) | 4.55 | 6.01 |
| Source database bytes | 556016 | 617816 |
| Source manifest bytes | 15192 | 15386 |

BM25 misses 5 of 10 eligible semantic cases. The three excluded fixture-only targets are reported as coverage gaps, not retrieval misses. The hybrid proxy embedder improves eligible semantic hit@5 by 20 percentage points and keeps all exact cases at rank 1. These are proxy benchmark measurements only, not production improvement or real-model validation.

`eligible_target_coverage` is eligible labeled targets present in the source index divided by eligible labeled targets. `fixture_target_coverage` is eligible labeled targets divided by all labeled targets. Excluded fixture-only targets affect coverage reporting but are not scored as retrieval misses.

## Implementation behavior

- Source embeddings live in `source_embeddings` inside the source index database. Learned-memory vectors remain in `.oem/.local_vector_db`.
- Embeddings are keyed by source chunk, model generation, and content hash. Model or dimension changes create a new generation; the previous generation remains available for rollback.
- Source indexing writes text chunks first, then writes a validated embedding batch in a separate transaction. Manifest activation is atomic.
- Hybrid search combines BM25 and clamped cosine similarity. Existing exact identifier, path, symbol-definition, source-type, and weak-result rules remain active.
- A 0.25 threshold applies when normalized BM25, matched identifiers, or component lexical corroboration exists; 0.55 applies only when all are absent. `source_diagnostics.dense_evidence_threshold` and `lexical_component_match` explain the choice; BM25 scores themselves are unchanged.
- If the local model, vectors, dimensions, or query embedding are unavailable, search returns BM25 results with a `source_dense_fallback:*` warning and diagnostics. No model download is attempted.
- Preflight and session end remain read-only/offline with respect to source embedding generation.

## Proxy limitation

The hybrid column uses a deterministic noisy sparse hybrid proxy embedder with a small synonym map so the gate is local and reproducible. It validates source storage, generation handling, fallback behavior, ranking integration, and the threshold calculation. Measured gains are proxy benchmark evidence only: they do not claim production improvement or real-model validation. A periodic evaluation with the configured local model remains the quality check for real semantic recall.

The deterministic ranking regression exercises the actual hybrid branch with a fake source store and controlled two-dimensional vectors. It verifies that dense-only scores of 0.50 are filtered at 0.55, scores of 0.60 remain eligible, and lexical BM25 evidence uses the existing 0.25 threshold. This change adds no `SourceIndexConfig` setting.

## Storage follow-up

## R2 real-model periodic evaluation

Run `OEM_RUN_PERIODIC_EVAL=1 uv run pytest -q -m periodic_eval packages/oem-knowledge/tests/test_source_retrieval_real_model_eval.py`.

The opt-in fixture contains fixture-derived counts of at least 20 exact, 40 semantic, and 20 negative new-query cases and references the frozen R0 corpus. BM25 and real hybrid runs use paired temporary projects. The model uses `_load_local_model()` with `local_files_only=True`; no download is attempted. Missing package/cache/model causes an explicit skip, while an available model with a failing source index or quality gate fails the periodic evaluation.

The threshold is measured BM25 semantic hit@5 plus 0.15, with BM25 semantic hit@5 at or below 0.85, hybrid exact top-1 at 1.0, and zero hybrid filtered and dense negative false positives. Output includes fixture-derived counts, model class/name/dimension, per-case paired paths/ranks/dense evidence flags and selected thresholds, semantic transitions, and a deterministic seeded bootstrap 95% CI for the paired semantic hit-rate delta.

R2 also computes a cache-only raw dense ceiling for every negative query against the active source embeddings. The overall maximum must remain below 0.55. This gate validates the default cached model and surfaces embedding drift even when runtime filtering returns no rows; it does not download a model or alter production state.

R2 is distinct from the R0 proxy: it is limited to new queries on one shared corpus, and its small-sample bootstrap interval is not a population guarantee. The 40 semantic records include repeated variants of 10 base paraphrase templates, so they should not be treated as 40 independent semantic concepts.

The historical cached-model run on 2026-09-07 was the pre-fix result: BM25 semantic hit@5 was 0.4000 (MRR@5 0.3200) and hybrid semantic hit@5 was 0.8000 (MRR@5 0.6646), with hybrid exact top-1 1.0000. It returned filtered results and dense false positives for all 20 negative cases, so the R2 gate status was FAIL.

Post-fix BM25 exact top-1: 0.7500; hybrid exact top-1: 1.0000; BM25 semantic hit@5/MRR@5: 0.4000/0.3200; hybrid semantic hit@5/MRR@5: 0.8000/0.6646; filtered negative false positives: 0; dense negative false positives: 0; raw negative dense ceiling: 0.5309221148490906 versus 0.55; bootstrap 95% CI: [0.25, 0.55]; gate status PASS.

Old generations are retained by design until a later explicit cleanup policy is approved. Embeddings for files removed from the source corpus are deleted during source re-indexing. The benchmark reports the storage increase so model-generation retention is visible.
