# R3 Concurrency Hardening

## Scope

R3 hardens durable write serialization and source-generation publication. It
does not change MCP, event, or vector schemas; ranking or embedding behavior;
or CommitPipeline phase names, intent JSON, byte offsets, checksums, and
rollback rules.

## Lock table

| Lock | Protected operation | Timeout and ownership |
| --- | --- | --- |
| `.oem/commit.lock` | Complete durable `session_end` pipeline, including the isolated index wait, optional dream, and `complete_pipeline` | 60s; the public `session_end` wrapper acquires it before the legacy pipeline, so one `intent.json` owner exists for the whole operation |
| `concept_registry.lock` | Dream registry read-modify-write | Dream uses `lock=False` for nested state operations and `merge_concepts` |
| `source_manifest.lock` | Full `SourceCorpusService.index` cycle, including chunks, embeddings, SQLite generation writes, and atomic manifest activation | 60s |
| `.oem/.local_vector_db/index.lock` | Shared learned vector/file registry writes for `index_all`, concept-event upserts, and user-event delete/upserts | 60s; there is no external lock around `index_all_events` because it calls the locked per-concept method |
| Global `user_events.lock` | `UserStore` load/append and the user-event snapshot read | The snapshot lock is released before embedding; vector delete and upsert use `index.lock` |

## Ordering and recovery

`commit.lock` is the outer session owner. Registry and event locks are
internal phase locks. Source, learned-index, and global user locks are
independent. A call chain must never reacquire the same `FileLock`.

Stale-lock recovery uses the existing `FileLock` behavior. A 60-second timeout
limits contention waiting; it is not a guarantee that the pipeline completes
within 60 seconds.

## Source generation safety

SQLite generations can coexist. The manifest is atomically replaced only after
a serialized index cycle completes, so publication cannot expose a partially
completed generation.

## User snapshot semantics

The user-event snapshot is read under the global user lock, then that lock is
released before embedding. If an append occurs after the snapshot, it is
indexed on a later pass rather than extending the user-lock hold.

## Verification and outcome

Run the focused R3 suite:

```text
uv run pytest -q packages/oem-knowledge/tests/test_r3_concurrency.py
```

The existing crash-recovery and concurrency suites should also pass, followed
by the full gate:

```text
uv run pytest -q
```

The measurable outcome is serialized durable writes, safe source-generation
publication, and bounded user-lock holds; this document reports no unsupported
performance metrics.

No restart is required unless a runtime embeds this package; normally, none is
required.
