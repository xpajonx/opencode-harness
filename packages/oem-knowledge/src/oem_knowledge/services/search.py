from __future__ import annotations
import json
import math
import os
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oem_knowledge.markdown.frontmatter import parse_frontmatter
from oem_knowledge.source_classifier import classify_source
from oem_knowledge.vector_store import VectorStore
from oem_knowledge.memory_ranking import (
    rank_search_results,
    classify_memory_type,
    extract_query_targets,
    build_ranking_debug_report,
)
from oem_knowledge.retrieval import normalize_record_fields, parse_iso_window, record_in_window
from oem_knowledge.fs import FileLock

if TYPE_CHECKING:
    from oem_knowledge.engine import KnowledgeEngine

logger = logging.getLogger(__name__)

USER_EVENTS_SOURCE = "user_events.jsonl"


class SearchService:
    def __init__(self, engine: KnowledgeEngine):
        self.engine = engine
        self._vector_store = None

    def _index_lock(self) -> FileLock:
        lock_path = self.engine._resolve_harness() / ".local_vector_db" / "index.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(lock_path, timeout=60.0)

    @property
    def model(self):
        return self.engine.model

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            db_dir = self.engine._resolve_harness() / ".local_vector_db"
            db_dir.mkdir(parents=True, exist_ok=True)
            self._migrate_chroma_db_if_needed(db_dir)
            db_file = db_dir / "vectors.db"
            try:
                store = VectorStore(db_file)
                # Test query to ensure the DB is not corrupted
                store.conn.execute("SELECT COUNT(*) FROM chunks")
                self._vector_store = store
            except Exception as e:
                logger.warning(
                    "[OEM] Local SQLite database is corrupted or incompatible (%s). Recreating clean database...", e
                )
                try:
                    if db_file.exists():
                        db_file.unlink()
                except Exception as e2:
                    logger.warning("[OEM] Failed to delete corrupted SQLite database: %s", e2)
                self._vector_store = VectorStore(db_file)
        return self._vector_store

    def _migrate_chroma_db_if_needed(self, db_dir: Path):
        db_file = db_dir / "vectors.db"
        if db_file.exists():
            return

        # If there are files in db_dir but no vectors.db, it indicates existing ChromaDB data
        if db_dir.exists() and any(db_dir.iterdir()):
            backup_dir = db_dir / "chroma_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for item in db_dir.iterdir():
                if item == backup_dir or item == db_file:
                    continue
                try:
                    import shutil
                    shutil.move(str(item), str(backup_dir / item.name))
                except Exception as e:
                    logger.warning("Failed to move ChromaDB file %s to backup: %s", item.name, e)

    def get_retrieval_mode(self) -> str:
        config_file = self.engine._resolve_harness() / "config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                return data.get("retrieval_mode", "auto")
            except json.JSONDecodeError as e:
                logger.warning("Corrupt configuration file %s: %s", config_file, e)
            except OSError as e:
                logger.warning("Failed to read configuration file %s: %s", config_file, e)
            except Exception as e:
                logger.warning("Unexpected error reading config file %s: %s", config_file, e)
        return "auto"

    def set_retrieval_mode(self, mode: str):
        if mode not in ("auto", "bm25", "hybrid"):
            raise ValueError("Invalid retrieval mode. Use 'auto', 'bm25', or 'hybrid'.")
        harness = self.engine._resolve_harness()
        config_file = harness / "config.json"
        data = {}
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Corrupt configuration file %s during write fallback: %s", config_file, e)
            except OSError as e:
                logger.warning("Failed to read configuration file %s during write: %s", config_file, e)
            except Exception as e:
                logger.warning("Unexpected error reading config file %s: %s", config_file, e)
        data["retrieval_mode"] = mode
        config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def semantic_dependencies_available(self) -> bool:
        try:
            import fastembed  # noqa: F401
            return True
        except ImportError:
            return False

    def resolve_retrieval_mode(self) -> str:
        configured = self.get_retrieval_mode()
        if configured == "auto":
            try:
                from fastembed import TextEmbedding  # noqa: F401
            except ImportError:
                return "bm25"
            if self.engine.embedding_cache_ready():
                return "hybrid"
            return "bm25"
        return configured

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            raise ImportError(
                "Embedding model is not loaded. Ensure oem[semantic] is installed and "
                "retrieval mode is configured properly."
            )
        return [[float(x) for x in e] for e in self.model.embed(texts)]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def close(self) -> None:
        if self._vector_store is not None:
            self._vector_store.close()
            self._vector_store = None

    def calculate_sha256(self, filepath: Path) -> str:
        import hashlib
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha.update(block)
        return sha.hexdigest()


    def chunk_markdown(self, filepath: Path, rel_path: str) -> list[dict]:
        try:
            content = filepath.read_text()
        except OSError as e:
            logger.error("Failed to read file for chunking at %s: %s", filepath, e)
            raise
        except Exception as e:
            logger.error("Unexpected error reading file for chunking at %s: %s", filepath, e)
            raise
        if not content.strip():
            return []

        # Strip frontmatter once at document-load boundary
        parsed = parse_frontmatter(content, source_path=str(filepath))
        body = parsed.body if parsed.metadata or not parsed.warnings else content
        doc_metadata = parsed.metadata if parsed.metadata else {}

        # Log frontmatter warnings once per file (not per chunk)
        if parsed.warnings:
            reasons = [w.get("reason", "unknown") for w in parsed.warnings]
            logger.warning(
                "Frontmatter warning in %s: %s. Chunking from full content.",
                rel_path,
                ", ".join(reasons),
            )

        header_pattern = re.compile(r"^(#{1,4}\s+.*)$", re.MULTILINE)
        parts = header_pattern.split(body)
        chunks = []
        current_title = "Introduction"

        if parts[0].strip():
            chunks.append({"title": current_title, "text": parts[0].strip()})

        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            section_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if section_body:
                chunks.append(
                    {
                        "title": header.lstrip("#").strip(),
                        "text": f"{header}\n\n{section_body}",
                    }
                )

        if not chunks:
            chunks.append({"title": "Full Document", "text": body.strip()})

        formatted = []
        for idx, chunk in enumerate(chunks):
            section_text = (
                f"Document: {rel_path}\nSection: {chunk['title']}\n\n{chunk['text']}"
            )
            links = list(set(re.findall(r"\[\[([^\]]+)\]\]", chunk["text"])))
            chunk_meta = {
                "chunk_id": f"{rel_path}#chunk_{idx}",
                "text": section_text,
                "title": chunk["title"],
                "raw_body": chunk["text"],
                "linked_concepts": links,
            }
            if doc_metadata:
                chunk_meta["concept_id"] = doc_metadata.get("concept_id", "")
                chunk_meta["concept_status"] = doc_metadata.get("status", "")
            formatted.append(chunk_meta)

        return formatted

    def derive_importance(self, rel_path: str) -> str:
        lower = rel_path.lower()
        if "agents.md" in lower or "claude.md" in lower:
            return "critical"
        if "scratch" in lower:
            return "low"
        if "/" not in rel_path and "\\" not in rel_path:
            return "high"
        return "medium"

    def index_all(self, force: bool = False, progress_callback=None, budget_seconds: float | None = None, quiet: bool = False) -> dict:
        from oem_knowledge.fs import LockTimeoutError
        try:
            return self._index_all_impl(force, progress_callback, budget_seconds, quiet)
        except LockTimeoutError as e:
            logger.error("Lock acquisition failure during indexing: %s", e)
            return {
                "status": "error",
                "error": f"Lock acquisition failure during indexing: {e}",
                "failed_files": [],
            }

    def _index_all_impl(self, force: bool = False, progress_callback=None, budget_seconds: float | None = None, quiet: bool = False) -> dict:
        with self._index_lock():
            harness = self.engine._resolve_harness()
            wiki_dir = harness / "wiki"
        
            md_files = list(wiki_dir.rglob("*.md")) if wiki_dir.exists() else []
            for folder in ["skills", "skill_candidates"]:
                folder_dir = harness / folder
                if folder_dir.exists():
                    md_files.extend(folder_dir.rglob("*.md"))
            for f in harness.glob("*.md"):
                if f.is_file():
                    md_files.append(f)

            registry = {}
            reg_path = harness / "state" / "file_registry.json"
            if reg_path.exists():
                try:
                    registry = json.loads(reg_path.read_text())
                except (json.JSONDecodeError, OSError):
                    registry = {}

            # Detect relative path migration
            is_migration = False
            if registry:
                first_key = next(iter(registry.keys()))
                if os.path.isabs(first_key):
                    is_migration = True
                    if not quiet:
                        print("\n[OEM] Migrating search index registry to relative paths (one-time full re-index)...")
                    force = True

            stats = {
                "status": "success",
                "scanned": len(md_files),
                "new": 0,
                "updated": 0,
                "unchanged": 0,
                "failed": 0,
                "new_chunks": 0,
                "updated_chunks": 0,
                "unchanged_chunks": 0,
                "failed_chunks": 0,
                "failed_files": [],
                "deletes": 0,
                "count_time_s": 0.0,
                "chunk_time_s": 0.0,
                "embed_time_s": 0.0,
                "write_time_s": 0.0,
                "index_time_ms": 0,
            }
            start_index_time = time.time()
            active_paths = set()
            new_registry = {}
            to_index = []
            failed_files = []

            # 1. Count phase
            t0 = time.time()
            chunk_counts = {}
            store = None
            try:
                store = self.vector_store
                if store is None:
                    raise ValueError("Vector store failed to initialize")
                chunk_counts = store.count_chunks_by_source_batch()
            except Exception as e:
                logger.error("Failed to initialize vector store or count chunks: %s", e)
                stats["status"] = "error"
                stats["error"] = f"Vector store initialization error: {e}"
                stats["failed"] = len(md_files)
                stats["failed_files"] = [str(fp.relative_to(harness.parent)) if fp.is_relative_to(harness.parent) else fp.name for fp in md_files]
                return stats
            stats["count_time_s"] = time.time() - t0

            # 2. Chunk phase
            t_chunk_start = time.time()
            for fp in md_files:
                if budget_seconds is not None and (time.time() - start_index_time) >= budget_seconds:
                    stats["status"] = "partial"
                    stats["error"] = "Indexing budget exceeded"
                    return stats
                try:
                    rel_path = str(fp.relative_to(harness.parent))
                except Exception:
                    rel_path = fp.name
                path_str = rel_path
                active_paths.add(path_str)
                try:
                    cur_hash = self.calculate_sha256(fp)
                    old_hash = None if is_migration else registry.get(path_str)
                    new_registry[path_str] = cur_hash
                    if force or old_hash != cur_hash:
                        stats["new" if old_hash is None else "updated"] += 1
                        to_index.append((fp, path_str, old_hash, cur_hash))
                    else:
                        stats["unchanged"] += 1
                        stats["unchanged_chunks"] += chunk_counts.get(path_str, 0)
                except Exception as e:
                    logger.warning("Failed to scan/hash file %s: %s", path_str, e)
                    stats["failed"] += 1
                    failed_files.append(path_str)
                    if path_str in registry:
                        new_registry[path_str] = registry[path_str]

            chunks_to_upsert = []
            if to_index and store is not None:
                retrieval_mode = self.resolve_retrieval_mode()
                # Verify fastembed is installed for hybrid mode
                if retrieval_mode == "hybrid":
                    try:
                        from fastembed import TextEmbedding
                    except ImportError:
                        raise ImportError(
                            "Hybrid search requires fastembed. Please install it with 'uv tool install \"git+https://github.com/xpajonx/openempiric.git#subdirectory=packages/oem-knowledge[semantic]\"' "
                            "or switch to automatic/BM25 retrieval using 'oem config retrieval auto' or 'oem config retrieval bm25'."
                        )

                for idx, (fp, path_str, old_hash, cur_hash) in enumerate(to_index):
                    if budget_seconds is not None and (time.time() - start_index_time) >= budget_seconds:
                        stats["status"] = "partial"
                        stats["error"] = "Indexing budget exceeded"
                        return stats
                    if progress_callback is not None:
                        try:
                            progress_callback(idx + 1, len(to_index))
                        except Exception as e:
                            logger.warning("Progress callback failed: %s", e)

                    try:
                        rel_path = str(fp.relative_to(harness.parent))
                    except Exception as e:
                        logger.debug("Failed to calculate relative path for %s: %s", fp, e)
                        rel_path = fp.name

                    try:
                        content = fp.read_text(encoding="utf-8")
                        classification = classify_source(fp, content=content)
                        chunks = self.chunk_markdown(fp, rel_path)
                        if not chunks:
                            continue

                        mtime = os.path.getmtime(fp)
                        imp = self.derive_importance(rel_path)

                        # Compute memory_type for this file's chunks
                        doc_memory_type = "observation"
                        try:
                            from oem_knowledge.memory_ranking import classify_memory_type
                            doc_memory_type = classify_memory_type(document=content)
                        except Exception:
                            pass

                        for c in chunks:
                            meta = {
                                "source": path_str,
                                "source_path": rel_path,
                                "rel_path": rel_path,
                                "source_type": classification.source_type,
                                "ingestion_eligible": classification.ingestion_eligible,
                                "title": c["title"],
                                "content_hash": cur_hash,
                                "linked_concepts": ",".join(c["linked_concepts"]),
                                "created_at": str(mtime),
                                "updated_at": str(mtime),
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime)),
                                "importance": imp,
                                "memory_type": doc_memory_type,
                                 "embedding_model": self.engine.resolve_embedding_model(),
                            }
                            # Propagate document-level metadata from frontmatter
                            for fm_key in ("concept_id", "concept_status"):
                                if c.get(fm_key):
                                    meta[fm_key] = c[fm_key]
                            chunks_to_upsert.append({
                                "chunk_id": c["chunk_id"],
                                "text": c["text"],
                                "meta": meta,
                                "old_hash": old_hash
                            })
                    except Exception as e:
                        logger.warning("Failed to chunk file %s: %s", path_str, e)
                        failed_files.append(path_str)
                        stats["failed"] += 1
            stats["chunk_time_s"] = time.time() - t_chunk_start

            # 3. Embedding phase
            t_embed_start = time.time()
            retrieval_mode = self.resolve_retrieval_mode()
            stats["effective_retrieval_mode"] = retrieval_mode
            stats["model_available"] = self.engine.embedding_cache_ready() or self.model is not None
            embeddings = []
            if chunks_to_upsert and store is not None:
                if budget_seconds is not None and (time.time() - start_index_time) >= budget_seconds:
                    stats["status"] = "partial"
                    stats["error"] = "Indexing budget exceeded"
                    return stats
                if retrieval_mode == "hybrid":
                    if not stats["model_available"]:
                        stats["status"] = "partial"
                        stats["error"] = "Embedding model is not available in the local cache. Run `oem warmup`, then re-index."
                        return stats
                    try:
                        texts = [c["text"] for c in chunks_to_upsert]
                        embeddings = self.embed(texts)
                        for c, embedding in zip(chunks_to_upsert, embeddings):
                            if embedding is not None:
                                c["meta"]["embedding_dimension"] = len(embedding)
                    except Exception as e:
                        logger.warning("Batch embedding generation error: %s", e)
                        embeddings = [None] * len(chunks_to_upsert)
                else:
                    embeddings = [None] * len(chunks_to_upsert)
            stats["embed_time_s"] = time.time() - t_embed_start

            # 4. SQLite write phase
            t_write_start = time.time()
            write_error = None
            if store is not None:
                if budget_seconds is not None and (time.time() - start_index_time) >= budget_seconds:
                    stats["status"] = "partial"
                    stats["error"] = "Indexing budget exceeded"
                    return stats
                # Batch delete
                sources_to_delete = [path_str for _, path_str, old_hash, _ in to_index if old_hash is not None and path_str not in failed_files]
                if sources_to_delete:
                    try:
                        store.delete_by_sources(sources_to_delete)
                        stats["deletes"] += len(sources_to_delete)
                    except Exception as e:
                        logger.error("Failed to delete stale chunks from database: %s", e)
                        write_error = e

                # Batch upsert
                if chunks_to_upsert and not write_error:
                    batch_data = []
                    for idx, c in enumerate(chunks_to_upsert):
                        batch_data.append((
                            c["chunk_id"],
                            c["text"],
                            c["meta"],
                            embeddings[idx]
                        ))
                        if c["old_hash"] is None:
                            stats["new_chunks"] += 1
                        else:
                            stats["updated_chunks"] += 1

                    try:
                        store.upsert_batch(batch_data)
                    except Exception as e:
                        logger.error("Batch upsert error: %s", e)
                        stats["failed_chunks"] += len(chunks_to_upsert)
                        write_error = e

                # Handle deleted paths
                deleted_paths = set(registry.keys()) - active_paths
                if deleted_paths and not write_error:
                    try:
                        store.delete_by_sources(list(deleted_paths))
                        stats["deletes"] += len(deleted_paths)
                    except Exception as e:
                        logger.error("Failed to delete removed paths from database: %s", e)
                        write_error = e
            stats["write_time_s"] = time.time() - t_write_start

            # Only update registry if write succeeded
            if not write_error:
                try:
                    reg_path.parent.mkdir(parents=True, exist_ok=True)
                    reg_path.write_text(json.dumps(new_registry, indent=2))
                except Exception as e:
                    logger.error("Failed to write search file registry: %s", e)
                    write_error = e

            total_time = time.time() - start_index_time
            stats["index_time_ms"] = int(total_time * 1000)

            # Output the print instrumentation
            total_chunks = stats["new_chunks"] + stats["updated_chunks"] + stats["unchanged_chunks"]
            if not quiet:
                print(f"\nIndex Profile:")
                print(f"  Files scanned:       {stats['scanned']}")
                print(f"  Chunks total:        {total_chunks}")
                print(f"  New chunks:          {stats['new_chunks']}")
                print(f"  Updated chunks:      {stats['updated_chunks']}")
                print(f"  Unchanged chunks:    {stats['unchanged_chunks']}")
                print(f"  Deletes:             {stats['deletes']}")
                print(f"  Count phase:         {stats['count_time_s']:.2f}s")
                print(f"  Chunk phase:         {stats['chunk_time_s']:.2f}s")
                print(f"  Embedding phase:     {stats['embed_time_s']:.2f}s")
                print(f"  SQLite write phase:  {stats['write_time_s']:.2f}s")
                print(f"  Total:               {total_time:.2f}s")

            if write_error:
                stats["status"] = "error"
                stats["error"] = str(write_error)
                stats["failed_files"] = failed_files
            elif failed_files:
                stats["status"] = "partial"
                stats["failed_files"] = failed_files
            else:
                stats["status"] = "success"

            return stats

    def index_concept_events(self, concept_id: str, project: str | None = None) -> int:
        """Index all events for a concept as individual searchable chunks.
        Returns count of events indexed."""
        import time
        events = self.engine.state.get_events(project=project, concept=concept_id)
        if not events:
            return 0
        store = self.vector_store
        chunks_to_upsert = []
        for event in events:
            event_id = event.get("event_id") or event.get("id", "")
            chunk_id = f"{concept_id}#event_{event_id}"
            document = (
                f"Concept: {concept_id}\n"
                f"Event: {event.get('summary', event.get('evidence', ''))}\n"
                f"Type: {event.get('event_type', event.get('type', 'observation'))}\n"
                f"Source: {event.get('source', 'unknown')}"
            )
            meta = {
                "source": "events.jsonl",
                "concept_id": concept_id,
                "event_id": event_id,
                "event_type": event.get("event_type", event.get("type", "observation")),
                "importance": "medium",
                "created_at": str(event.get("timestamp", event.get("created_at", time.time()))),
                "scope": "project",
                "memory_type": event.get("event_type", event.get("type", "observation")),
                "timestamp": str(event.get("timestamp", event.get("created_at", time.time()))),
                "embedding_model": self.engine.resolve_embedding_model(),
            }
            chunks_to_upsert.append((chunk_id, document, meta))
        with self._index_lock():
            if chunks_to_upsert:
                texts = [c[1] for c in chunks_to_upsert]
                if self.resolve_retrieval_mode() == "hybrid":
                    embeddings = self.embed(texts)
                    for c, embedding in zip(chunks_to_upsert, embeddings):
                        if embedding is not None:
                            c[2]["embedding_dimension"] = len(embedding)
                else:
                    embeddings = [None] * len(texts)
                batch = [(cid, doc, meta, emb) for (cid, doc, meta), emb in zip(chunks_to_upsert, embeddings)]
                store.upsert_batch(batch)
        return len(chunks_to_upsert)

    def index_all_events(self, project: str | None = None) -> dict:
        """Index all events across all concepts. Returns {per_concept: {cid: count}, total: int}."""
        registry = self.engine.state._load_registry(project)
        results = {}
        total = 0
        for cid in registry:
            count = self.index_concept_events(cid, project)
            results[cid] = count
            total += count
        return {"per_concept": results, "total": total}

    def index_user_events(self, project: str | None = None, budget_seconds: float | None = None) -> dict:
        """Index identity-scoped user events from the global user store.

        Idempotent by event_id (chunk id `user#<event_id>`). Stale user
        chunks are removed first (source-level sweep). Command-log-like,
        malformed, or missing-id records are rejected. Budget-bounded and
        exception-safe: never raises.
        """
        from oem_knowledge.services.state import is_ingestion_noise_event
        start = time.time()
        stats = {"status": "success", "indexed": 0, "rejected": 0, "reason": None}
        # The user events path is identity-independent; existence is the gate
        # (writes already required a resolved identity). Avoids a git subprocess
        # probe on every session/index path.
        user_path = Path.home() / ".config" / "openempiric" / "user_events.jsonl"
        if user_path is None or not user_path.exists():
            stats["reason"] = "no_user_events"
            return stats
        events = []
        rejected = 0
        with FileLock(user_path.with_suffix(".lock"), timeout=60.0):
            with open(user_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        rejected += 1
                        continue
                    if not isinstance(ev, dict):
                        rejected += 1
                        continue
                    ev_id = ev.get("event_id") or ev.get("id")
                    if not ev_id:
                        rejected += 1
                        continue
                    summary = str(ev.get("summary") or ev.get("evidence") or "")
                    if not summary or is_ingestion_noise_event(ev):
                        rejected += 1
                        continue
                    memory_type = ev.get("event_type") or ev.get("type") or "observation"
                    timestamp = str(ev.get("timestamp") or "")
                    events.append((
                        f"user#{ev_id}",
                        f"Type: {memory_type}\n{summary}",
                        {
                            "source": USER_EVENTS_SOURCE,
                            "scope": "user",
                            "memory_type": memory_type,
                            "timestamp": timestamp,
                            "created_at": timestamp,
                            "project": str(ev.get("project", "") or ""),
                            "session_id": str(ev.get("session_id", "") or ""),
                            "provenance": "user_events",
                            "embedding_model": self.engine.resolve_embedding_model(),
                        },
                    ))
        if budget_seconds is not None and (time.time() - start) >= budget_seconds:
            stats["status"] = "partial"
            stats["reason"] = "Indexing budget exceeded"
            stats["rejected"] = rejected
            return stats
        if not events:
            try:
                with self._index_lock():
                    store = self.vector_store
                    store.delete_by_source(USER_EVENTS_SOURCE)
            except Exception as e:
                stats["status"] = "partial"
                stats["reason"] = f"user chunk cleanup failed: {e}"
                return stats
            stats["rejected"] = rejected
            return stats
        try:
            if self.resolve_retrieval_mode() == "hybrid":
                embeddings = self.embed([e[1] for e in events])
                for event, embedding in zip(events, embeddings):
                    if embedding is not None:
                        event[2]["embedding_dimension"] = len(embedding)
            else:
                embeddings = [None] * len(events)
            with self._index_lock():
                store = self.vector_store
                store.delete_by_source(USER_EVENTS_SOURCE)
                store.upsert_batch([(cid, doc, meta, emb) for (cid, doc, meta), emb in zip(events, embeddings)])
            stats["indexed"] = len(events)
            stats["rejected"] = rejected
        except Exception as e:
            stats["status"] = "partial"
            stats["reason"] = f"user events indexing failed: {e}"
        return stats

    def _collect_raw_candidates(self, query: str, hybrid: bool = True, auto_index: bool = True) -> list[dict]:
        """Collect raw candidates from vector search with retrieval scores (no ranking applied)."""
        store = self.vector_store
        if store.count() == 0:
            if not auto_index:
                return []
            harness = self.engine._resolve_harness()
            wiki_dir = harness / "wiki"
            has_md_files = (wiki_dir.exists() and any(wiki_dir.glob("*.md"))) or any(harness.glob("*.md"))
            if has_md_files:
                import logging
                logging.info("[OEM] Vector database is empty. Auto-indexing existing wiki concepts...")
                self.index_all(force=True, quiet=True)
            else:
                return []

        retrieval_mode = self.resolve_retrieval_mode()
        if retrieval_mode == "hybrid":
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "Hybrid search requires fastembed. Please install it with 'uv tool install \"git+https://github.com/xpajonx/openempiric.git#subdirectory=packages/oem-knowledge[semantic]\"' "
                    "or switch to automatic/BM25 retrieval using 'oem config retrieval auto' or 'oem config retrieval bm25'."
                )

        chunks = store.all_chunks()
        if not chunks:
            return []

        doc_texts = [c["document"] for c in chunks]
        bm25_scores = self._compute_bm25(query, doc_texts)

        if retrieval_mode == "hybrid":
            query_vec = self.embed([query])[0]
            candidates = []
            current_model = self.engine.resolve_embedding_model()
            warned_incompatible = False
            for idx, chunk in enumerate(chunks):
                meta = chunk["metadata"]
                embedding = chunk["embedding"]
                compatible = (embedding and meta.get("embedding_model") == current_model
                              and meta.get("embedding_dimension") == len(query_vec)
                              and len(embedding) == len(query_vec))
                if not compatible and embedding and not warned_incompatible:
                    logger.warning("Skipping dense scoring for incompatible or legacy vectors; re-index with model '%s'.", current_model)
                    warned_incompatible = True
                dense = self.cosine_similarity(query_vec, embedding) if compatible else 0.0
                sparse = bm25_scores[idx]
                recency = self._recency_score(meta.get("created_at", str(time.time())))
                importance = self._importance_score(meta.get("importance", "medium"))
                final = (
                    (0.45 * dense)
                    + (0.30 * sparse)
                    + (0.15 * recency)
                    + (0.10 * importance)
                )
                candidates.append({
                    "id": chunk["id"],
                    "document": chunk["document"],
                    "metadata": meta,
                    "score": final,
                })
        else:
            candidates = []
            for idx, chunk in enumerate(chunks):
                sparse = bm25_scores[idx]
                meta = chunk["metadata"]
                recency = self._recency_score(meta.get("created_at", str(time.time())))
                importance = self._importance_score(meta.get("importance", "medium"))
                final = (
                    (0.75 * sparse)
                    + (0.15 * recency)
                    + (0.10 * importance)
                )
                candidates.append({
                    "id": chunk["id"],
                    "document": chunk["document"],
                    "metadata": meta,
                    "score": final,
                })

        return candidates

    def _collect_raw_fallback_candidates(self, query: str) -> list[dict]:
        """Collect raw candidates from the concept registry (no ranking applied)."""
        try:
            registry = self.engine.state._load_registry()
        except Exception as e:
            logger.warning("Registry search fallback failed because the registry could not be loaded: %s", e)
            return []

        query_lower = query.lower().strip()
        query_terms = [t for t in re.findall(r"\w+", query_lower) if len(t) > 1]
        candidates = []
        concepts_dir = self.engine._concepts_dir()

        for cid, cdata in registry.items():
            canonical = cdata.get("canonical_name", cid).lower()
            aliases = [a.lower() for a in cdata.get("aliases", [])]
            wiki_path = concepts_dir / f"{cid}.md"
            wiki_text = ""
            if wiki_path.exists():
                try:
                    wiki_text = wiki_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed to read concept wiki file %s: %s", wiki_path, e)

            score = 0.0
            if canonical == query_lower:
                score = 1.0
            elif query_lower in aliases:
                score = 0.85
            else:
                text_to_search = f"{canonical} {' '.join(aliases)} {wiki_text}".lower()
                matched_terms = 0
                for term in query_terms:
                    if term in text_to_search:
                        matched_terms += 1
                if query_terms:
                    score = 0.5 * (matched_terms / len(query_terms))
                max_sim = 0.0
                for term in [canonical] + aliases:
                    sim = self._string_similarity(query_lower, term)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim >= 0.80:
                    score = max(score, 0.50 + 0.35 * max_sim)

            if score > 0.1 or (not query_terms and not query_lower):
                document = wiki_text if wiki_text else f"Concept: {cdata.get('canonical_name', cid)}\nDescription: {cdata.get('description', '')}"
                h = self.engine._resolve_harness()
                try:
                    rel_path = str(wiki_path.relative_to(h.parent))
                except Exception as e:
                    logger.debug("Failed to calculate relative path for %s: %s", wiki_path, e)
                    rel_path = f".oem/wiki/{cid}.md"
                imp = self.derive_importance(rel_path)
                importance_val = self._importance_score(imp)
                final_score = 0.8 * score + 0.2 * importance_val
                candidates.append({
                    "id": f"{cid}#fallback",
                    "document": document,
                    "metadata": {
                        "source": str(wiki_path),
                        "rel_path": rel_path,
                        "title": cdata.get("canonical_name", cid),
                        "content_hash": "",
                        "linked_concepts": "",
                        "importance": imp,
                        "scope": "project",
                    },
                    "score": final_score,
                })

        return candidates

    def search(self, query: str, k: int = 3, hybrid: bool = True,
               scope: str | None = None, memory_type: str | None = None,
               since: str | None = None, until: str | None = None,
               record_references: bool = True, auto_index: bool = True) -> list[dict]:
        """Search indexed memory with normalized metadata filters.

        Filters are applied AFTER metadata normalization (fields promoted from
        metadata to top level). A record missing `scope` counts as project
        scope. Invalid since/until returns an empty result with a logged
        warning (the MCP tool surfaces a controlled error before this point).
        """
        try:
            candidates = self._collect_raw_candidates(query, hybrid, auto_index=auto_index)
            if not candidates:
                return []
            candidates.sort(key=lambda x: x["score"], reverse=True)
            registry = self.engine.state._load_registry()
            ranked = rank_search_results(query, candidates, registry)
            filtered_ranked = []
            for r in ranked:
                is_memory = (
                    r.get("memory_type") in ("decision", "failure", "outcome") or
                    "active_work_signal" in r.get("ranking_boosts", {})
                )
                eligible = r.get("eligible_for_type_boost", True)
                if is_memory and not eligible:
                    continue
                if r.get("score", 0.0) <= 0.0:
                    continue
                filtered_ranked.append(normalize_record_fields(r))
            # Scope filter (missing scope counts as project for backward compatibility)
            if scope:
                filtered_ranked = [
                    r for r in filtered_ranked
                    if r.get("scope") == scope or (r.get("scope") is None and scope == "project")
                ]
            # Memory type filter (authoritative index-time type, classify fallback)
            if memory_type:
                filtered_ranked = [r for r in filtered_ranked if r.get("memory_type") == memory_type]
            # Temporal window filter (strict on invalid input)
            if since or until:
                since_dt, until_dt, win_err = parse_iso_window(since, until)
                if win_err:
                    logger.warning("Invalid search window (%s): returning no results for since=%r until=%r", win_err, since, until)
                    filtered_ranked = []
                else:
                    filtered_ranked = [
                        r for r in filtered_ranked
                        if record_in_window(
                            r.get("timestamp") or r.get("created_at") or r.get("updated_at"),
                            since_dt, until_dt,
                        )
                    ]
            results = filtered_ranked[:k]
            if record_references:
                try:
                    concept_ids = self.engine.state.concept_ids_from_retrieval_results(results)
                    self.engine.state.record_concept_references(concept_ids, source="search")
                except Exception as ref_err:
                    logger.warning("Failed to record concept references for search results: %s", ref_err)
            return results
        except Exception as e:
            import logging
            logging.warning("Vector database search failed, falling back to registry-only: %s", e)
            return self._search_registry_fallback(query, k, scope, memory_type, since, until, record_references=record_references)

    def _search_registry_fallback(self, query: str, k: int = 3, scope: str | None = None,
                                  memory_type: str | None = None,
                                  since: str | None = None, until: str | None = None,
                                  record_references: bool = True) -> list[dict]:
        candidates = self._collect_raw_fallback_candidates(query)
        if not candidates:
            return []
        candidates.sort(key=lambda x: x["score"], reverse=True)
        registry = self.engine.state._load_registry()
        ranked = rank_search_results(query, candidates, registry)
        filtered_ranked = []
        for r in ranked:
            is_memory = (
                r.get("memory_type") in ("decision", "failure", "outcome") or
                "active_work_signal" in r.get("ranking_boosts", {})
            )
            eligible = r.get("eligible_for_type_boost", True)
            if is_memory and not eligible:
                continue
            if r.get("score", 0.0) <= 0.0:
                continue
            filtered_ranked.append(r)
        if scope:
            filtered_ranked = [
                r for r in filtered_ranked
                if r.get("scope") == scope or (r.get("scope") is None and scope == "project")
            ]
        if memory_type:
            filtered_ranked = [r for r in filtered_ranked if r.get("memory_type") == memory_type]
        if since or until:
            since_dt, until_dt, win_err = parse_iso_window(since, until)
            if win_err:
                logger.warning("Invalid search window (%s): returning no results for since=%r until=%r", win_err, since, until)
                filtered_ranked = []
            else:
                filtered_ranked = [
                    r for r in filtered_ranked
                    if record_in_window(
                        r.get("timestamp") or r.get("created_at") or r.get("updated_at"),
                        since_dt, until_dt,
                    )
                ]
        results = filtered_ranked[:k]
        if record_references:
            try:
                concept_ids = self.engine.state.concept_ids_from_retrieval_results(results)
                self.engine.state.record_concept_references(concept_ids, source="search")
            except Exception as ref_err:
                logger.warning("Failed to record concept references for fallback search results: %s", ref_err)

        return results

    def debug_ranking(self, query: str, k: int = 3, hybrid: bool = True) -> dict[str, Any]:
        used_fallback = False
        try:
            candidates = self._collect_raw_candidates(query, hybrid)
        except Exception as e:
            import logging
            logging.warning("Vector search failed, using fallback for debug ranking: %s", e)
            candidates = self._collect_raw_fallback_candidates(query)
            used_fallback = True

        if not candidates:
            candidates = []

        targets = extract_query_targets(query)
        raw_before_sort = list(candidates)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        pool_size = len(candidates)
        ranked = rank_search_results(query, candidates)

        return build_ranking_debug_report(
            query=query,
            targets=targets,
            raw_candidates=raw_before_sort,
            reranked_candidates=ranked,
            k=k,
            candidate_pool_size=pool_size,
            used_fallback=used_fallback,
        )

    def _string_similarity(self, s1: str, s2: str) -> float:
        import difflib
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _compute_bm25(self, query: str, documents: list[str]) -> list[float]:
        query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
        if not query_terms or not documents:
            return [0.0] * len(documents)

        doc_terms = [[t.lower() for t in re.findall(r"\w+", d)] for d in documents]
        doc_count = len(documents)
        df: dict[str, int] = {}
        for terms in doc_terms:
            for t in set(terms):
                df[t] = df.get(t, 0) + 1

        k1, b = 1.5, 0.75
        avg_dl = sum(len(t) for t in doc_terms) / max(doc_count, 1)

        scores = []
        for terms in doc_terms:
            score = 0.0
            doc_len = len(terms)
            tf = Counter(terms)
            for q in query_terms:
                if q in tf:
                    n = df.get(q, 0)
                    idf = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)
                    freq = tf[q]
                    score += (
                        idf
                        * (freq * (k1 + 1))
                        / (freq + k1 * (1 - b + b * doc_len / max(avg_dl, 1)))
                    )
            scores.append(score)

        max_s = max(scores) if scores else 0
        return [s / max_s for s in scores] if max_s > 0 else scores

    def _recency_score(self, created_at_str: str) -> float:
        try:
            age_days = (time.time() - float(created_at_str)) / (3600 * 24)
            return math.exp(-0.05 * max(0.0, age_days))
        except Exception as e:
            logger.debug("Failed to calculate recency score for %s: %s", created_at_str, e)
            return 1.0

    def _importance_score(self, imp: str) -> float:
        match str(imp).lower():
            case "critical":
                return 1.0
            case "high":
                return 0.7
            case "medium":
                return 0.5
            case "low":
                return 0.2
            case _:
                return 0.5

    def stats(self) -> dict:
        try:
            store = self.vector_store
            total_chunks = store.count()
        except Exception as e:
            logger.warning("Failed to count chunks for stats: %s", e)
            total_chunks = 0
        db_size = 0
        db_path = self.engine._resolve_harness() / ".local_vector_db"
        if db_path.exists():
            for f in db_path.rglob("*"):
                if f.is_file():
                    db_size += f.stat().st_size

        return {
            "total_chunks": total_chunks,
            "db_size_mb": db_size / (1024 * 1024),
            "harness_path": str(self.engine._resolve_harness()),
        }
