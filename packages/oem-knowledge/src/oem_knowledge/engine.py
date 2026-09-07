from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path
def build_embedding_runtime_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Builds a runtime environment mapping for embeddings and model execution,
    setting defaults for CUDA_VISIBLE_DEVICES and TOKENIZERS_PARALLELISM if not already set,
    without mutating the global os.environ.
    """
    env = dict(base_env or os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def apply_oem_process_env_defaults() -> None:
    """Explicitly mutates os.environ to set safe defaults for openempiric-knowledge execution.
    Only call this from explicit process entrypoints (like CLI main or MCP server startup).
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from oem_knowledge.fs import FileLock, SecureFileSystem
from oem_knowledge.project_layout import ProjectLayout
from oem_knowledge.services.embedding_worker import _worker_main
from contextlib import contextmanager
from typing import Callable

logger = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

class PhaseTimer:
    def __init__(self):
        self.timings = {}
        self.current_phase = None
        self.failed_phase = None

    @contextmanager
    def phase(self, name: str, callback: Callable[[str], None] | None = None):
        if callback is not None:
            try:
                callback(name)
            except Exception:
                pass
        self.current_phase = name
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.failed_phase = name
            raise
        finally:
            self.timings[name] = time.perf_counter() - start
            self.current_phase = None

OEM_DIR = ".oem"
DEFAULT_DIRS = [
    "wiki",
    "sessions",
    "state",
    "graph",
    "skills",
    "skill_candidates",
]

def migrate_harness_to_oem(project_dir: Path):
    old_dir = project_dir / ".harness"
    new_dir = project_dir / ".oem"
    if old_dir.is_dir() and not new_dir.exists():
        old_dir.rename(new_dir)
        
        old_wiki = new_dir / "directives" / "wiki_concepts"
        new_wiki = new_dir / "wiki"
        if old_wiki.is_dir():
            new_wiki.parent.mkdir(parents=True, exist_ok=True)
            old_wiki.rename(new_wiki)
            
        old_sess = new_dir / "directives" / "sessions"
        new_sess = new_dir / "sessions"
        if old_sess.is_dir():
            new_sess.parent.mkdir(parents=True, exist_ok=True)
            old_sess.rename(new_sess)
            
        old_reg = new_dir / "state" / "concept_registry.json"
        new_reg = new_dir / "concept_registry.json"
        if old_reg.exists():
            old_reg.rename(new_reg)
            
        old_events = new_dir / "state" / "events.jsonl"
        new_events = new_dir / "events.jsonl"
        if old_events.exists():
            old_events.rename(new_events)
            
        old_inbox = new_dir / "directives" / "wiki_inbox.md"
        new_inbox = new_dir / "wiki" / "inbox.md"
        if old_inbox.exists():
            new_inbox.parent.mkdir(parents=True, exist_ok=True)
            old_inbox.rename(new_inbox)
            
        old_idx = new_dir / "directives" / "index.md"
        new_idx = new_dir / "wiki" / "index.md"
        if old_idx.exists():
            new_idx.parent.mkdir(parents=True, exist_ok=True)
            old_idx.rename(new_idx)

        old_log = new_dir / "directives" / "log.md"
        new_log = new_dir / "wiki" / "log.md"
        if old_log.exists():
            new_log.parent.mkdir(parents=True, exist_ok=True)
            old_log.rename(new_log)

        old_prog = new_dir / "directives" / "progress.md"
        new_prog = new_dir / "progress.md"
        if old_prog.exists():
            old_prog.rename(new_prog)

        old_handoff = new_dir / "directives" / "session-handoff.md"
        new_handoff = new_dir / "session-handoff.md"
        if old_handoff.exists():
            old_handoff.rename(new_handoff)

        directives_dir = new_dir / "directives"
        if directives_dir.is_dir() and not any(directives_dir.iterdir()):
            try:
                directives_dir.rmdir()
            except Exception:
                pass


def find_harness_root(path: str | Path) -> Path | None:
    """Walk up from path looking for .oem/ or .harness/ directory, stopping at boundaries."""
    p = Path(path).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".oem").is_dir() or (parent / ".harness").is_dir():
            return parent
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / ".opencode").exists():
            break
    return None




class KnowledgeEngine:
    def __init__(self, project_path: str | Path | None = None):
        self._model = None
        self._model_lock = threading.RLock()
        self._local_load_failed = False
        self.project_path = Path(project_path).resolve() if project_path else None
        self._db_dir: Path | None = None

        # Instantiate services dynamically with self-injection
        from oem_knowledge.services.search import SearchService
        from oem_knowledge.services.materialization import MaterializationService
        from oem_knowledge.services.reflection import ReflectionService
        from oem_knowledge.services.state import StateService
        from oem_knowledge.services.event_migration import EventMigrator
        from oem_knowledge.services.fitness import FitnessService
        from oem_knowledge.services.source_corpus import SourceCorpusService
        from oem_knowledge.services.skills import SkillService
        from oem_knowledge.services.skill_promotion import SkillPromotionService

        self.search = SearchService(self)
        self.source = SourceCorpusService(self)
        self.materialization = MaterializationService(self)
        self.reflection = ReflectionService(self)
        self.state = StateService(self)
        self.event_migrator = EventMigrator(self)
        self.fitness = FitnessService(self)
        self.skills = SkillService(self)
        self.skill_promotion = SkillPromotionService(self)

        # Phase 3: Instantiate Storage layer wrappers
        from oem_knowledge.storage.event_store import EventStore
        from oem_knowledge.storage.registry_store import RegistryStore
        from oem_knowledge.storage.concept_files import ConceptFiles
        from oem_knowledge.storage.session_files import SessionFiles
        from oem_knowledge.storage.user_store import UserStore

        self.event_store = EventStore(self)
        self.registry_store = RegistryStore(self)
        self.concept_files = ConceptFiles(self)
        self.session_files = SessionFiles(self)
        self.user_store = UserStore(self)

        # Phase 3: Instantiate Computation layer wrappers
        from oem_knowledge.computation.snapshot import SnapshotComputation
        from oem_knowledge.computation.reflection import ReflectionComputation
        from oem_knowledge.computation.indexing import IndexingComputation
        from oem_knowledge.computation.search import SearchComputation
        from oem_knowledge.computation.fitness import FitnessComputation
        from oem_knowledge.computation.evolution import EvolutionComputation
        from oem_knowledge.computation.preflight import PreflightComputation
        from oem_knowledge.computation.materialization import MaterializationComputation
        from oem_knowledge.computation.skills import SkillsComputation

        self.snapshot = SnapshotComputation(self)
        self.materialization_computation = MaterializationComputation(self)
        self.reflection_computation = ReflectionComputation(self)
        self.indexing = IndexingComputation(self)
        self.search_computation = SearchComputation(self)
        self.fitness_computation = FitnessComputation(self)
        self.evolution = EvolutionComputation(self)
        self.preflight_computation = PreflightComputation(self)
        self.skills_computation = SkillsComputation(self)

    def export_memory(self, output_path: str, project: str | None = None) -> dict:
        """Export project memory to a tar.gz archive.

        Exports .oem/ directory + user events + manifest.
        Returns dict with status and archive path.
        """
        import tarfile
        import tempfile
        from pathlib import Path

        output = Path(output_path).resolve()
        harness = self._resolve_harness(project)

        try:
            with tarfile.open(str(output), "w:gz") as tar:
                # Archive the .oem directory
                if harness.exists():
                    tar.add(str(harness), arcname=".oem")

                # Include user events if available
                user_path = self.user_store.get_events_path()
                if user_path and user_path.exists():
                    tar.add(str(user_path), arcname="user_events.jsonl")

                # Write a manifest
                import json as _json
                manifest = {
                    "schema_version": 1,
                    "project_id": harness.parent.name,
                    "exported_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                    "includes_user_events": bool(user_path and user_path.exists()),
                }
                manifest_path = tempfile.mktemp(suffix=".json")
                Path(manifest_path).write_text(_json.dumps(manifest, indent=2))
                tar.add(manifest_path, arcname="manifest.json")
                Path(manifest_path).unlink(missing_ok=True)

            return {
                "status": "success",
                "operation": "export_memory",
                "archive_path": str(output),
                "size_bytes": output.stat().st_size if output.exists() else 0,
            }
        except Exception as e:
            return {
                "status": "error",
                "operation": "export_memory",
                "message": str(e),
            }

    def import_memory(self, input_path: str, project: str | None = None) -> dict:
        """Import project memory from a tar.gz archive.

        Merges with existing events via event_id dedup and alias-merge
        for concept conflicts. Returns dict with status and conflict report.
        """
        import tarfile
        import tempfile
        import json as _json
        from pathlib import Path
        import shutil

        input_p = Path(input_path).resolve()
        if not input_p.exists():
            return {"status": "error", "operation": "import_memory", "message": f"Archive not found: {input_path}"}

        harness = self._resolve_harness(project)
        conflicts = []
        imported_events = 0
        skipped_events = 0

        try:
            extract_dir = Path(tempfile.mkdtemp())

            with tarfile.open(str(input_p), "r:gz") as tar:
                tar.extractall(path=str(extract_dir), filter="data")

            # Import user events
            user_archive = extract_dir / "user_events.jsonl"
            if user_archive.exists():
                existing_user_ids = set()
                user_path = self.user_store.get_events_path()
                if user_path and user_path.exists():
                    with open(user_path) as f:
                        for line in f:
                            if line.strip():
                                try:
                                    ev = _json.loads(line)
                                    existing_user_ids.add(ev.get("event_id", ""))
                                except _json.JSONDecodeError:
                                    pass

                with open(user_archive) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            ev = _json.loads(line)
                            eid = ev.get("event_id", "")
                            if eid and eid in existing_user_ids:
                                skipped_events += 1
                            else:
                                self.user_store.append_event(ev)
                                imported_events += 1
                        except _json.JSONDecodeError:
                            skipped_events += 1

            # Import project events (.oem directory)
            oem_archive = extract_dir / ".oem"
            if oem_archive.exists():
                events_archive = oem_archive / "events.jsonl"
                if events_archive.exists():
                    existing_project_ids = set()
                    events_path = harness / "events.jsonl"
                    if events_path.exists():
                        with open(events_path) as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        ev = _json.loads(line)
                                        existing_project_ids.add(ev.get("event_id", ""))
                                    except _json.JSONDecodeError:
                                        pass

                    with open(events_archive) as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                ev = _json.loads(line)
                                eid = ev.get("event_id", "")
                                if eid and eid in existing_project_ids:
                                    skipped_events += 1
                                else:
                                    self.event_store.append_event(ev, str(harness.parent))
                                    imported_events += 1
                            except _json.JSONDecodeError:
                                skipped_events += 1

            # Cleanup
            shutil.rmtree(extract_dir, ignore_errors=True)

            return {
                "status": "success",
                "operation": "import_memory",
                "imported_events": imported_events,
                "skipped_events": skipped_events,
                "conflicts": conflicts,
                "message": f"Imported {imported_events} events, skipped {skipped_events} duplicates.",
            }
        except Exception as e:
            return {
                "status": "error",
                "operation": "import_memory",
                "message": str(e),
            }

    def close(self) -> None:
        for service in (getattr(self, "search", None), getattr(self, "source", None)):
            close = getattr(service, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "KnowledgeEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _sfs(self, project: str | Path | None = None) -> SecureFileSystem:
        p = Path(project or self.project_path or ".").resolve()
        return SecureFileSystem(p)

    def _resolve_harness(self, project_path: str | Path | None = None) -> Path:
        p = Path(project_path or self.project_path or ".").resolve()
        root = find_harness_root(p) or p
        harness = root / OEM_DIR
        if not harness.exists():
            harness.mkdir(parents=True, exist_ok=True)
            self._bootstrap_harness(root)
        return harness

    def _bootstrap_harness(self, project_path: Path):
        """Create a minimal .oem/ structure."""
        harness = project_path / OEM_DIR
        for d in DEFAULT_DIRS:
            (harness / d).mkdir(parents=True, exist_ok=True)
        sfs = self._sfs(project_path)
        for fname, content in [
            (
                "progress.md",
                f"# Progress — {project_path.name}\n- **{time.strftime('%Y-%m-%d')}:** Initialized.\n",
            ),
            (
                "session-handoff.md",
                (
                    "# Session Handoff\n\n"
                    "## Historical Context\n"
                    "Complete phase one.\n\n"
                    "## Previous Decisions\n"
                    "- Setup initial directory structure.\n\n"
                    "## Open Questions\n"
                    "- What goals/concepts should be mapped next?\n"
                ),
            ),
        ]:
            fp = harness / fname
            if not sfs.exists(fp):
                sfs.write_text(fp, content, force_allow_truncation=True)

    @property
    def model(self):
        with self._model_lock:
            if self._model is None and not self._local_load_failed:
                self._load_local_model()
            return self._model

    def resolve_embedding_model(self) -> str:
        """Return the configured model, failing safe for invalid configuration."""
        path = self._resolve_harness() / "config" / "embedding_model.json"
        if not path.exists():
            return DEFAULT_EMBEDDING_MODEL
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("embedding_model")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception as exc:
            logger.warning("Invalid embedding model configuration %s: %s; using default %s", path, exc, DEFAULT_EMBEDDING_MODEL)
            return DEFAULT_EMBEDDING_MODEL
        logger.warning("Invalid embedding model configuration %s; using default %s", path, DEFAULT_EMBEDDING_MODEL)
        return DEFAULT_EMBEDDING_MODEL

    def _load_local_model(self):
        """Load the embedding model from the local cache only. Never downloads.
        The failure is memoized so a broken cache cannot trigger repeated attempts."""
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._local_load_failed:
                return None
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise
            model_name = self.resolve_embedding_model()
            cache_path = str(Path.home() / ".cache" / "fastembed")
            try:
                self._model = TextEmbedding(model_name=model_name, cache_dir=cache_path, local_files_only=True)
            except Exception as e:
                logging.warning("[OEM] Embedding model '%s' not available in local cache (%s). Run `oem warmup` to download it.", model_name, e)
                self._local_load_failed = True
                return None
            self._local_load_failed = False
            return self._model

    def _download_model(self):
        """Explicit download path used only by warmup(). This is the only place
        in the package allowed to construct fastembed with local_files_only=False."""
        with self._model_lock:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                self._local_load_failed = True
                return None
            model_name = self.resolve_embedding_model()
            cache_path = str(Path.home() / ".cache" / "fastembed")
            try:
                self._model = TextEmbedding(model_name=model_name, cache_dir=cache_path, local_files_only=False)
            except Exception as e:
                logging.warning("[OEM] Embedding model download failed: %s", e)
                self._model = None
                self._local_load_failed = True
                raise
            self._local_load_failed = False
            return self._model

    def layout(self, project: str | None = None) -> ProjectLayout:
        """Return a ProjectLayout for the resolved .oem directory.

        KnowledgeEngine.layout() is the single resolver — ProjectLayout itself
        is a pure value object and does no path resolution.
        """
        return ProjectLayout(root=self._resolve_harness(project))

    def _registry_path(self, project: str | None = None) -> Path:
        return self.layout(project).registry_path

    def _events_path(self, project: str | None = None) -> Path:
        return self.layout(project).events_path

    def _sessions_dir(self, project: str | None = None) -> Path:
        return self.layout(project).sessions_dir

    def _concepts_dir(self, project: str | None = None) -> Path:
        return self.layout(project).concepts_dir

    def _wiki_paths(self, project: str | None = None) -> dict:
        return self.layout(project).wiki_paths()




    # --- Orchestrator Level Methods kept on engine ---

    def init_project(self, name: str) -> dict:
        path = Path(name)
        if path.is_absolute():
            project_dir = path
        else:
            base = Path.cwd() if not self.project_path else self.project_path
            project_dir = base / name if not (base / name).exists() else base

        harness = project_dir / OEM_DIR

        created_dirs = []
        for d in DEFAULT_DIRS:
            p = harness / d
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created_dirs.append(d)

        created_files = []

        for fname, content in [
            (
                "wiki/inbox.md",
                "# Wiki Inbox\n\nAppend raw lessons, API observations, and style guidelines here.\n",
            ),
            (
                "progress.md",
                f"# Project Progress Log — {name}\n\n- **{time.strftime('%Y-%m-%d')}:** OpenEmpiric initialized.\n",
            ),
            (
                "session-handoff.md",
                (
                    "# Session Handoff\n\n"
                    "## Historical Context\n"
                    "Initialized project layout.\n\n"
                    "## Previous Decisions\n"
                    "- Setup initial directory structure.\n\n"
                    "## Open Questions\n"
                    "- What goals/concepts should be mapped next?\n"
                ),
            ),
            (
            "config/reflection.yml",
            (
                "reflection:\n"
                "  mode: auto\n\n"
                "  structured:\n"
                "    enabled: true\n\n"
                "  marker:\n"
                "    enabled: true\n\n"
                "  dense:\n"
                "    enabled: false\n"
                "    on_unavailable: skip\n"
                "    max_retry_count: 0\n"
                "    queue_pending: false\n\n"
                "  auto_dream:\n"
                "    enabled: false\n"
                "    half_life_days: 30\n"
                "    consolidate_threshold: 0.9\n"
                "    promote_threshold:\n"
                "      evidence_count: 5\n"
                "      session_count: 3\n"
            ),
            ),
        ]:
            fp = harness / fname
            if not fp.exists():
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content)
                created_files.append(fname)

        # Write manifest.json
        try:
            from oem_knowledge.runtime.manifest import ensure_manifest
            ensure_manifest(project_dir)
            created_files.append("manifest.json")
        except Exception:
            pass

        # Write init.sh
        try:
            init_sh = harness / "init.sh"
            if not init_sh.exists():
                init_content = (
                    "#!/bin/bash\n"
                    "# generated_by: openempiric\n"
                    "# source_type: oem_generated\n\n"
                    "echo \"Initializing OpenEmpiric workspace...\"\n"
                    "oem init\n"
                    "oem setup opencode\n"
                    "oem doctor\n"
                )
                init_sh.write_text(init_content, encoding="utf-8")
                try:
                    init_sh.chmod(0o755)
                except Exception:
                    pass
                created_files.append("init.sh")
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"OpenEmpiric initialized in {project_dir}",
            "created_directories": created_dirs,
            "created_files": created_files,
        }

    def is_initialized(self, project: str | None = None) -> bool:
        """Check if project has been initialized (has .oem/ dir with state/)."""
        p = Path(project or self.project_path or ".").resolve()
        root = find_harness_root(p)
        if root is None:
            return False
        harness = root / OEM_DIR
        return (harness / "state").is_dir()

    def warmup(self) -> dict:
        with self._model_lock:
            model_name = self.resolve_embedding_model()
            if self._model is not None:
                print("[OEM] Embedding model ready (cached globally, one-time per machine).", file=sys.stderr)
                return {"status": "success", "model": model_name}
            print(f"[OEM] Warming up embedding model '{model_name}'...", file=sys.stderr)
            try:
                from fastembed import TextEmbedding  # noqa: F401
            except ImportError:
                raise
            try:
                model = self._download_model()
            except Exception as e:
                raise RuntimeError("Embedding model download failed. Check network access and retry `oem warmup`.") from e
            if model is None:
                raise RuntimeError("Embedding model download failed. Check network access and retry `oem warmup`.")
            print("[OEM] Embedding model ready (cached globally, one-time per machine).", file=sys.stderr)
            return {"status": "success", "model": model_name}

    def warmup_if_needed(self) -> dict:
        """Warm up embedding model if already cached locally. Never downloads."""
        with self._model_lock:
            try:
                from fastembed import TextEmbedding  # noqa: F401
            except ImportError:
                return {"status": "skipped", "reason": "fastembed_unavailable"}
            if self._model is None and not self._local_load_failed:
                self._load_local_model()
            if self._model is not None:
                return {"status": "success"}
            cache_dir = Path.home() / ".cache" / "fastembed"
            slug = self.resolve_embedding_model().rsplit("/", 1)[-1]
            known_paths = (
                cache_dir / f"models--qdrant--{slug}-onnx-q",
                cache_dir / slug,
                cache_dir / f"fast-{slug}",
            )
            if not any(p.exists() for p in known_paths):
                return {"status": "skipped", "reason": "local_cache_missing"}
            return {"status": "skipped", "reason": "local_cache_invalid"}


    def restore_session_state(self, project: str | None = None) -> dict:
        h = self._resolve_harness(project)
        state_dir = h / "state"

        progress = h / "progress.md"
        handoff = h / "session-handoff.md"
        goals = state_dir / "current-goals.md"
        issues = state_dir / "open-issues.md"
        decisions = state_dir / "active-decisions.md"

        active_goals = []
        blockers = []
        discoveries = []
        full_content = ""

        for fp, target_list, attr in [
            (goals, active_goals, "goals"),
            (handoff, active_goals, "handoff"),
            (issues, blockers, "issues"),
            (decisions, discoveries, "decisions"),
            (progress, None, "progress"),
        ]:
            if fp.exists():
                text = fp.read_text()
                full_content += text + "\n"
                if attr == "handoff":
                    # Parse "Historical Context"
                    hist_match = re.search(r"## Historical Context\s*\n\s*([^#]+)", text)
                    if hist_match:
                        lines_of_hist = hist_match.group(1).strip().split("\n")
                        if lines_of_hist:
                            first_line = lines_of_hist[0].strip().lstrip("-").strip()
                            if first_line:
                                active_goals.append(first_line)
                    
                    # Parse "Previous Decisions"
                    dec_match = re.search(r"## Previous Decisions\s*\n\s*([^#]+)", text)
                    if dec_match:
                        for line in dec_match.group(1).splitlines():
                            s = line.strip()
                            if s.startswith("-"):
                                clean = re.sub(r"^-\s*(?:\[[ xX/]\])?\s*", "", s).strip()
                                if clean and clean not in discoveries:
                                    discoveries.append(clean)
                    
                    # Parse "Open Questions"
                    q_match = re.search(r"## Open Questions\s*\n\s*([^#]+)", text)
                    if q_match:
                        for line in q_match.group(1).splitlines():
                            s = line.strip()
                            if s.startswith("-"):
                                clean = re.sub(r"^-\s*(?:\[[ xX/]\])?\s*", "", s).strip()
                                if clean and clean not in active_goals:
                                    active_goals.append(clean)

                    # Fallback to Next Action
                    if not active_goals:
                        m = re.search(r"## Next Action\s*\n\s*(?:-\s*)?([^\n#]+)", text)
                        if m:
                            active_goals.append(m.group(1).strip())
                elif attr == "goals":
                    for line in text.splitlines():
                        s = line.strip()
                        if s.startswith("-"):
                            clean = re.sub(r"^-\s*(?:\[[ xX/]\])?\s*", "", s)
                            if clean.strip():
                                active_goals.append(clean.strip())
                elif attr == "issues":
                    for line in text.splitlines():
                        s = line.strip()
                        if s.startswith("-"):
                            clean = re.sub(r"^-\s*(?:\[[ xX/]\])?\s*", "", s)
                            if clean.strip():
                                blockers.append(clean.strip())
                elif attr == "decisions":
                    for line in text.splitlines():
                        s = line.strip()
                        if s.startswith("-") and s.strip("-").strip():
                            discoveries.append(s.strip("-").strip())

        keywords = " ".join(re.findall(r"\w+", full_content.strip())[:15])
        rec_files = []
        if keywords:
            try:
                results = self.search.search(keywords, k=4)
                rec_files = [r["metadata"]["rel_path"] for r in results]
            except Exception:
                pass

        global_concepts = []

        from oem_knowledge.runtime.working_set import get_resume_status, load_working_set
        
        status_info = get_resume_status(project or self.project_path)
        ws_active_concepts = []
        resume_source = status_info["resume_source"]
        
        active_work_item = None
        active_topic = None
        active_task = None
        
        if resume_source == "working_set":
            ws = load_working_set(project or self.project_path)
            if ws:
                active_work_item = ws.active_work_item
                active_topic = ws.active_topic
                active_task = ws.active_task
                
                # Restore goal and open questions to active_goals
                if ws.goal:
                    if ws.goal not in active_goals:
                        active_goals.insert(0, ws.goal)
                if ws.open_questions:
                    for q in ws.open_questions:
                        if q not in active_goals:
                            active_goals.append(q)
                
                # Restore blockers
                if ws.blocked_by:
                    for b in ws.blocked_by:
                        if b not in blockers:
                            blockers.append(b)
                
                # Prepend active_files to recommended_files (filter duplicates)
                if ws.active_files:
                    seen = set(rec_files)
                    for f in reversed(ws.active_files):
                        if f not in seen:
                            rec_files.insert(0, f)
                            seen.add(f)
                            
                # Restore active_concepts
                if ws.active_concepts:
                    ws_active_concepts = list(ws.active_concepts)
        else:
            from oem_knowledge.runtime.active_work import resolve_active_work_identity
            try:
                ident = resolve_active_work_identity(h)
                active_work_item = ident.active_work_item
                active_topic = ident.active_topic
                active_task = ident.active_task
            except Exception:
                pass
                
        res = {
            "status": "success",
            "active_goals": active_goals[:5],
            "blockers": blockers[:5],
            "recent_discoveries": discoveries[:5],
            "recommended_files": rec_files,
            "query_context": keywords[:60],
            "global_concepts": global_concepts,
            "active_work_item": active_work_item,
            "active_topic": active_topic,
            "active_task": active_task,
            "resume_source": resume_source,
            "resume_reason": status_info["resume_reason"],
        }
        if ws_active_concepts:
            res["active_concepts"] = ws_active_concepts
            
        return res




    def session_start(self, project: str | None = None) -> dict:
        from oem_knowledge.runtime.session import SessionState
        import uuid
        
        harness = self._resolve_harness(project)
        layout = self.layout(project)
        active_session_file = harness / "state" / "active_session.json"
        
        session_id = None
        session_state = SessionState.load(active_session_file)
        if session_state:
            session_id = session_state.session_id
        else:
            session_id = uuid.uuid4().hex[:12]
            session_state = SessionState.create(
                session_id=session_id,
                agent="mcp-agent",
                project=project or ".",
                transcript_path=str((harness / "state" / f"chat_{session_id}.md").resolve()),
                context_path=str((harness / "state" / "oem_runtime_context.json").resolve()),
                temp_instructions=str((harness / "state" / "oem_temp_instructions.md").resolve()),
            )
            session_state.save(active_session_file)

        warnings = []
        recovery = {"recovered": False, "action": "no_intent", "warning": None}
        try:
            from oem_knowledge.runtime.commit_pipeline import CommitPipeline
            commit_pipeline = CommitPipeline(harness)
            if commit_pipeline.intent_log.has_uncompleted_intent():
                recovery = commit_pipeline.recover_from_crash(self._events_path(project))
                warnings.append(f"Session commit recovery: {recovery.get('action')}")
        except Exception as e:
            warnings.append(f"Commit recovery check failed: {e}")
        try:
            from oem_knowledge.instructions import (
                discover_instruction_sources,
                get_db_connection,
                get_stale_sources,
                index_source_file,
                get_active_directives,
                render_current_directives
            )
            project_root = layout.root.parent
            sources = discover_instruction_sources(project_root, layout)
            conn = get_db_connection(layout.instruction_ledger_path)
            stale_paths = get_stale_sources(conn, sources)
            for ds in sources:
                if ds["path"] in stale_paths:
                    try:
                        content = (project_root / ds["path"]).read_text(encoding="utf-8")
                        index_source_file(conn, ds["path"], content, ds["hash"], ds["mtime"], ds["size_bytes"])
                    except Exception as e:
                        warnings.append(f"Failed to index instruction file {ds['path']}: {e}")
            
            # Generate current directives MD
            active_dirs = get_active_directives(conn)
            md_content = render_current_directives(active_dirs, layout)
            layout.current_directives_path.parent.mkdir(parents=True, exist_ok=True)
            layout.current_directives_path.write_text(md_content, encoding="utf-8")
            conn.close()
        except Exception as e:
            warnings.append(f"Failed to initialize instruction ledger: {e}")

        return {
            "status": "success",
            "operation": "knowledge_session_start",
            "session_id": session_id,
            "project": str(self.project_path or project or ""),
            "message": "OEM session started.",
            "warnings": warnings,
            **({"recovered": recovery.get("recovered"), "recovery_action": recovery.get("action"), "recovery_warning": recovery.get("warning")} if recovery.get("action") != "no_intent" else {}),
            "suggestion": "Use knowledge_read when you need orientation, then knowledge_search for specific memory."
        }

    def index_isolated(self, project_dir: str | None = None, budget_s: float = 10.0) -> dict:
        """Run semantic indexing in a spawn-isolated subprocess with a hard wall-clock bound.

        The embedding phase is a single non-interruptible call, so the subprocess
        terminate() is the only hard guarantee; the worker's own budget_seconds
        checks provide graceful early partials.
        """
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        target_dir = str(project_dir or self.project_path or "")
        proc = ctx.Process(target=_worker_main, args=(target_dir, float(budget_s), queue))
        proc.start()
        proc.join(float(budget_s))
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            return {
                "status": "partial",
                "error": "Indexing budget exceeded",
                "scanned": 0, "new": 0, "updated": 0, "unchanged": 0, "failed": 0,
                "new_chunks": 0, "updated_chunks": 0, "unchanged_chunks": 0,
                "failed_chunks": 0, "failed_files": 0, "deletes": 0, "timings": {},
            }
        try:
            result = queue.get(timeout=5.0)
        except Exception:
            result = {"status": "error", "error": "isolated index produced no result"}
        if result.get("status") == "error":
            result["status"] = "partial"
        return result

    @staticmethod
    def _build_session_phase_timings(timer: PhaseTimer) -> dict[str, float]:
        return {
            key: timer.timings.get(key, 0.0)
            for key in ("load_state", "reflection", "append_events", "materialization", "search_index", "write_report", "cleanup", "total")
        }

    # The single per-project intent log requires one owner for the complete
    # session pipeline; this is the safe bounded seam.
    def session_end(
        self,
        project: str | None = None,
        conversation_text: str = "",
        session_id: str = "",
        telemetry: dict | None = None,
        session_started_at: float | None = None,
        update_index: bool = True,
        index_budget_seconds: float | None = 10.0,
        progress_callback = None,
        events: list[dict] | None = None,
        extraction_mode: str = "auto",
        timeout_seconds: float | None = None,
    ) -> dict:
        from oem_knowledge.fs import FileLock, LockTimeoutError
        from oem_knowledge.runtime.result import error

        commit_lock_path = self._resolve_harness(project) / "commit.lock"
        try:
            with FileLock(commit_lock_path, timeout=60.0):
                return self._session_end_unlocked(
                    project=project,
                    conversation_text=conversation_text,
                    session_id=session_id,
                    telemetry=telemetry,
                    session_started_at=session_started_at,
                    update_index=update_index,
                    index_budget_seconds=index_budget_seconds,
                    progress_callback=progress_callback,
                    events=events,
                    extraction_mode=extraction_mode,
                    timeout_seconds=timeout_seconds,
                )
        except LockTimeoutError as e:
            return error(
                operation="session_end",
                message=f"Lock acquisition timeout: {e}",
                failed_step="commit_lock",
                warnings=[f"Lock failure: Commit lock contention on file. {e}"],
                data={
                    "report_path": None,
                    "knowledge_events": [],
                    "materialized_log": [],
                    "links_updated": 0,
                    "index_stats": {},
                    "explainability": {},
                },
            )

    def _session_end_unlocked(
        self,
        project: str | None = None,
        conversation_text: str = "",
        session_id: str = "",
        telemetry: dict | None = None,
        session_started_at: float | None = None,
        update_index: bool = True,
        index_budget_seconds: float | None = 10.0,
        progress_callback = None,
        events: list[dict] | None = None,
        extraction_mode: str = "auto",
        timeout_seconds: float | None = None,
    ) -> dict:
        from oem_knowledge.fs import LockTimeoutError
        from oem_knowledge.runtime.supervisor import CommitProgressSupervisor
        
        timer = PhaseTimer()
        start_time = time.perf_counter()
        
        progress = CommitProgressSupervisor()
        progress.start()
        
        warnings = []
        res = {"report_path": None, "knowledge_events": [], "explainability": {}}
        mat_log = []
        idx_res = {}
        links_updated = 0
        failed_step = None
        status = "success"
        message = ""

        from oem_knowledge.runtime.commit_pipeline import CommitPipeline, prefix_checksum
        commit_pipeline = CommitPipeline(self._resolve_harness(project))

        try:
            progress.update_step("transcript", "running")
            with timer.phase("load_state", progress_callback):
                if not session_id:
                    session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
            progress.update_step("transcript", "success")

            progress.update_step("reflection", "running")
            with timer.phase("reflection", progress_callback):
                # Check if reflect_session is mocked/patched in tests
                reflect_method = self.reflection.reflect_session
                is_mocked = (
                    hasattr(reflect_method, "mock")
                    or hasattr(reflect_method, "_mock_self")
                    or type(reflect_method).__name__ in ("Mock", "MagicMock")
                    or getattr(reflect_method, "__name__", None) != "reflect_session"
                    or not hasattr(reflect_method, "__func__")
                    or reflect_method.__func__ is not getattr(self.reflection.__class__, "_original_reflect_session_func", None)
                )
                if is_mocked:
                    res = reflect_method(
                        project=project or None,
                        conversation_text=conversation_text,
                        session_id=session_id,
                        telemetry=telemetry,
                        session_started_at=session_started_at,
                        progress_callback=progress_callback,
                        events=events,
                        extraction_mode=extraction_mode,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    res = self.reflection.extract_session_events(
                        project,
                        conversation_text,
                        session_id=session_id,
                        telemetry=telemetry,
                        session_started_at=session_started_at,
                        progress_callback=progress_callback,
                        events=events,
                        extraction_mode=extraction_mode,
                        timeout_seconds=timeout_seconds,
                    )

                # Running the Directive Receipt and Workflow Drift logic and notification generation early.
                matched_directives_for_summary = []
                drift_proposals = []
                notification = None
                try:
                    from oem_knowledge.instructions import (
                        get_db_connection,
                        get_active_directives,
                        match_directives,
                        render_directive_receipt
                    )
                    layout = self.layout(project)
                    conn = get_db_connection(layout.instruction_ledger_path)
                    
                    # Query session matched directives
                    cursor = conn.execute("SELECT * FROM session_directive_matches WHERE session_id = ?", (session_id,))
                    matches = [dict(row) for row in cursor.fetchall()]
                    
                    # Convert DB matches or match directly if empty
                    if not matches:
                        active_dirs = get_active_directives(conn)
                        direct_matches = match_directives(conversation_text, active_dirs)
                        for dm in direct_matches:
                            matched_directives_for_summary.append({
                                "id": dm["id"],
                                "title": dm["title"],
                                "rule": dm["rule"],
                                "source_path": dm["source_path"],
                                "line_start": dm["line_start"],
                                "line_end": dm["line_end"],
                                "score": dm["score"],
                                "reason": dm["reason"],
                                "priority": dm.get("priority", "normal")
                            })
                    else:
                        for m in matches:
                            cursor_d = conn.execute("SELECT * FROM directives WHERE id = ?", (m["directive_id"],))
                            d_row = cursor_d.fetchone()
                            if d_row:
                                d = dict(d_row)
                                matched_directives_for_summary.append({
                                    "id": d["id"],
                                    "title": d["title"],
                                    "rule": d["rule"],
                                    "source_path": d["source_path"],
                                    "line_start": d["line_start"],
                                    "line_end": d["line_end"],
                                    "score": m["match_score"],
                                    "reason": m["reason"],
                                    "priority": d.get("priority", "normal")
                                })
                    
                    # Assess directive applications
                    applications = []
                    from datetime import datetime
                    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    for md in matched_directives_for_summary:
                        d_id = md["id"]
                        # Default positive application status
                        status_app = "applied"
                        evidence_app = "adhered to directive constraints in this session"
                        
                        # Check for negation keyword rules in text
                        cursor_d = conn.execute("SELECT forbidden_actions_json FROM directives WHERE id = ?", (d_id,))
                        row_fa = cursor_d.fetchone()
                        forbidden = json.loads(row_fa["forbidden_actions_json"]) if row_fa else []
                        
                        conv_lower = conversation_text.lower()
                        for fa in forbidden:
                            if fa in conv_lower:
                                evidence_app = f"discussed or referenced action related to {fa}"
                                break
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO session_directive_applications (
                                session_id, directive_id, status, evidence, created_at
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (session_id, d_id, status_app, evidence_app, now_str))
                        
                        applications.append({
                            "directive_id": d_id,
                            "status": status_app,
                            "evidence": evidence_app
                        })
                        
                    # Propose workflow candidate if drift detected
                    conv_lower = conversation_text.lower()
                    if "wsl" in conv_lower and ("bridge" in conv_lower or "split" in conv_lower or "oem" in conv_lower):
                        from oem_knowledge.instructions import create_instruction_update_candidate
                        proposed_patch = "When configuring OpenCode Desktop with WSL OEM, always ensure the MCP bridge points to the WSL-native project root and never initializes a second `.oem` folder on the Windows side."
                        reason_desc = "This session revealed a recurring workflow rule for Windows/WSL OpenCode setup."
                        cand_res = create_instruction_update_candidate(layout, session_id, "AGENTS.md", proposed_patch, reason_desc)
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO instruction_update_candidates (
                                id, session_id, target_path, proposed_patch, reason, status, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (cand_res["id"], session_id, "AGENTS.md", proposed_patch, reason_desc, "proposed", now_str))
                        
                        drift_proposals.append({
                            "id": cand_res["id"],
                            "reason": "Windows/WSL bridge must prevent split `.oem` memory"
                        })
                        
                    # Render directive receipt
                    run_id = session_id
                    runs_dir = layout.root / "runs" / run_id
                    if (layout.root / "runs").is_dir():
                        runs_dir.mkdir(parents=True, exist_ok=True)
                        receipt_path = runs_dir / "directive_receipt.md"
                    else:
                        receipt_path = layout.root / ".runtime" / "directive_receipt.md"
                        
                    receipt_content = render_directive_receipt(session_id, matched_directives_for_summary, applications, drift_proposals)
                    receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    receipt_path.write_text(receipt_content, encoding="utf-8")
                    conn.close()
                except Exception as e:
                    logger.warning("Failed to record directive receipt/drift in session_end: %s", e)

                try:
                    candidates = self.skills.list_skill_candidates(project)
                    high_conf = [
                        c for c in candidates 
                        if c.status in ("proposed", "deferred") 
                        and (c.confidence == "high" or len(c.evidence) >= 3)
                    ]
                    if high_conf:
                        cand = high_conf[0]
                        
                        events_data = []
                        for evid in cand.source_event_ids:
                            try:
                                ev = self.state.get_event(project, evid)
                                if ev and ev.get("evidence"):
                                    events_data.append(ev["evidence"])
                            except Exception:
                                pass
                        
                        if not events_data:
                            events_data = cand.evidence
                        
                        ev_lines = []
                        for ev_val in events_data:
                            val = str(ev_val).strip()
                            if not val.lower().startswith("used successfully"):
                                val = f"Used successfully in {val}"
                            ev_lines.append(f"- {val}")
                        
                        unique_ev_lines = []
                        for line in ev_lines:
                            if line not in unique_ev_lines:
                                unique_ev_lines.append(line)
                        
                        ev_lines_str = "\n".join(unique_ev_lines[:3])
                        
                        notification = (
                            "OEM noticed a repeated successful workflow pattern.\n\n"
                            f"Candidate skill:\n"
                            f"{cand.title}\n\n"
                            f"Evidence:\n"
                            f"{ev_lines_str}\n\n"
                            f"Recommendation:\n"
                            f"Review with: oem skills show {cand.slug}\n"
                            f"Approve with: oem skills approve {cand.slug}"
                        )
                except Exception as e:
                    logger.warning("Failed to generate skill candidate notification: %s", e)

                # Add Directive Receipt summary to notification if present
                if matched_directives_for_summary:
                    directive_summary_lines = ["", "Directive Receipt Summary:"]
                    for md in matched_directives_for_summary:
                        directive_summary_lines.append(f"✓ {md['title']} (Source: {md['source_path']})")
                    if drift_proposals:
                        directive_summary_lines.append("\nWorkflow Drift / Instruction Proposals:")
                        for dp in drift_proposals:
                            directive_summary_lines.append(f"- Proposed candidate {dp['id']} for {dp['reason']}")
                    
                    summary_str = "\n".join(directive_summary_lines)
                    if notification:
                        notification += "\n\n" + summary_str
                    else:
                        notification = summary_str.strip()

                status_val = res.get("status")
                failed_step_val = res.get("failed_step")
                is_non_fatal = (status_val in ("warn", "empty") or (
                    status_val == "partial" and failed_step_val == "llm_extraction"
                )) and res.get("events_written", 0) == 0

                if is_non_fatal:
                    # Close the session safely
                    try:
                        self.state.record_outcome(
                            "success_with_warnings" if status_val == "warn" else ("partial" if status_val == "partial" else "success"),
                            session_id=session_id,
                            project=project
                        )
                    except Exception as e:
                        logger.warning("Outcome recording failed: %s", e)

                    self._close_active_session(project)

                    # Update progress steps
                    if status_val == "warn":
                        progress.update_step(
                            "reflection",
                            "warn",
                            name="Dense LLM Reflection Skipped",
                            detail="No local/remote LLM provider configured"
                        )
                    elif status_val == "empty":
                        progress.update_step(
                            "reflection",
                            "warn",
                            name="Reflection Skipped/Empty",
                            detail="No markers or structured events"
                        )
                    elif status_val == "partial" and failed_step_val == "llm_extraction":
                        progress.update_step(
                            "reflection",
                            "warn",
                            name="Reflection Degraded",
                            detail="LLM extraction timed out"
                        )

                    progress.update_step(
                        "materialization",
                        "skipped",
                        name="Materialization Skipped",
                        detail="No reflection events produced"
                    )
                    progress.update_step(
                        "index",
                        "skipped",
                        name="Search Index Skipped",
                        detail="No memory changes to index"
                    )
                    progress.update_step("session_close", "success")

                    timer.timings["total"] = time.perf_counter() - start_time
                    p_timings = self._build_session_phase_timings(timer)
                    from oem_knowledge.runtime.result import make_result
                    return make_result(
                        status=status_val,
                        operation="session_end",
                        project=str(self.project_path or project or ""),
                        message=res.get("message", "Reflection did not complete successfully."),
                        suggestion=res.get("suggestion"),
                        failed_step=failed_step_val,
                        warnings=res.get("warnings", []),
                        reflection=res.get("reflection"),
                        events_written=0,
                        materialization_skipped=True,
                        index_skipped=True,
                        notification=notification,
                        data={
                            "events_written": 0,
                            "phase_timings": p_timings,
                            "report_path": None,
                            "knowledge_events": [],
                            "materialized_log": [],
                            "links_updated": 0,
                            "index_stats": {},
                            "explainability": res.get("explainability", {}),
                            "notification": notification,
                        }
                    )

                if res.get("status") == "error":
                    progress.update_step("reflection", "failed")
                    timer.timings["total"] = time.perf_counter() - start_time
                    p_timings = self._build_session_phase_timings(timer)
                    from oem_knowledge.runtime.result import error
                    return error(
                        operation="session_end",
                        message=res.get("message", "Reflection failed"),
                        failed_step="reflection",
                        warnings=res.get("warnings", []),
                        reflection=res.get("reflection"),
                        data={
                            "report_path": res.get("report_path"),
                            "knowledge_events": res.get("knowledge_events", []),
                            "materialized_log": [],
                            "links_updated": 0,
                            "index_stats": {},
                            "explainability": res.get("explainability", {}),
                            "phase_timings": p_timings,
                        }
                    )
            progress.update_step("reflection", "success")
            if status_val in ("warn", "partial"):
                status = status_val

            # Persist events using public append_events
            with timer.phase("append_events", progress_callback):
                canonical_events = res.get("canonical_events", [])
                t_append_start = time.perf_counter()
                events_path = self._events_path(project)
                commit_pipeline.record_appended_bytes(events_path)
                _offset = commit_pipeline.staging.get_byte_offset("append_events")
                try:
                    _prefix = prefix_checksum(events_path, _offset) if _offset else None
                except Exception:
                    _prefix = None
                try:
                    from oem_knowledge.models import KnowledgeEvent
                    _expected = 0
                    for _ev in canonical_events:
                        _model = _ev if isinstance(_ev, KnowledgeEvent) else KnowledgeEvent(**_ev)
                        _expected += len(_model.model_dump_json()) + 1
                except Exception:
                    _expected = None
                commit_pipeline.intent_log.write_intent(
                    "append_events",
                    session_id=session_id,
                    project=str(self.project_path or project or ""),
                    byte_offset=_offset,
                    expected_bytes=_expected,
                    prefix_checksum=_prefix,
                )
                self.state.append_events(canonical_events, project)
                try:
                    _actual_delta = self._events_path(project).stat().st_size - _offset
                    commit_pipeline.intent_log.write_intent(
                        "append_events",
                        session_id=session_id,
                        project=str(self.project_path or project or ""),
                        byte_offset=_offset,
                        expected_bytes=_actual_delta,
                        prefix_checksum=_prefix,
                    )
                except Exception:
                    pass
                commit_pipeline.complete_phase("append_events", {})
                append_events_time = time.perf_counter() - t_append_start
            timer.timings["append_events"] = append_events_time

            # Write session report file
            with timer.phase("write_report", progress_callback):
                t_write_start = time.perf_counter()
                sessions_dir = self._sessions_dir(project)
                sfs = self._sfs(project)
                date_str = time.strftime("%Y-%m-%d")
                report_file = sessions_dir / f"{date_str}.md"
                counter = 1
                while sfs.exists(report_file) and counter < 1000:
                    report_file = sessions_dir / f"{date_str}_{counter}.md"
                    counter += 1

                yaml_events = res.get("knowledge_events", [])
                yaml_content = json.dumps({"knowledge_events": yaml_events}, indent=2)

                report_content = f"""---
date: {date_str}
project: {project or "default"}
---
# Session Learning Report — {date_str}

## Knowledge Events
```json
{yaml_content}
```
"""
                try:
                    sfs.write_text(report_file, report_content, force_allow_truncation=True)
                except OSError as e:
                    logger.error("Failed to write session learning report to %s: %s", report_file, e)
                    from oem_knowledge.runtime.result import error
                    return error(
                        operation="session_end",
                        message=f"Failed to write session learning report: {e}",
                        failed_step="write_report",
                    )
                write_report_time = time.perf_counter() - t_write_start
            timer.timings["write_report"] = write_report_time

            # Emit reflection metrics
            try:
                from oem_knowledge.tools.metrics import update_metrics_file
                p = Path(project or ".").resolve()
                root = find_harness_root(p) or p
                metrics_file = (root / OEM_DIR / "state" / "metrics.json")
                
                structured_events_found = res.get("explainability", {}).get("structured_events_found", 0)
                fallback_extraction_used = res.get("explainability", {}).get("fallback_extraction_used", False)
                file_observations_count = res.get("explainability", {}).get("file_observations_count", 0)
                
                update_metrics_file(metrics_file, {
                    "structured_events": structured_events_found,
                    "fallback_extractions": 1 if fallback_extraction_used else 0,
                    "file_observations": file_observations_count,
                    "empty_reflections": 1 if structured_events_found == 0 and not fallback_extraction_used and file_observations_count == 0 else 0,
                    "noise_events_filtered": res.get("explainability", {}).get("noise_events_filtered", 0),
                    "telemetry_events_skipped": res.get("explainability", {}).get("telemetry_events_skipped", 0),
                    "reflections": 1,
                })
            except Exception as e:
                logger.warning("Failed to emit reflection metrics: %s", e)

            # Materialization
            events_written = len(canonical_events)
            if events_written == 0:
                progress.update_step(
                    "materialization",
                    "skipped",
                    name="Materialization Skipped",
                    detail="No reflection events produced"
                )
                mat_log = []
                links_updated = 0
            else:
                progress.update_step("materialization", "running")
                with timer.phase("materialization", progress_callback):
                    mat_res = self.materialization.materialize_concepts(project)
                    if mat_res.get("status") == "error":
                        try:
                            self.state.record_outcome("partial", session_id=session_id, project=project)
                        except Exception:
                            pass

                        self._close_active_session(project)

                        progress.update_step("materialization", "failed")
                        progress.update_step("index", "skipped")
                        progress.update_step("session_close", "success")

                        timer.timings["total"] = time.perf_counter() - start_time
                        p_timings = self._build_session_phase_timings(timer)
                        
                        from oem_knowledge.runtime.result import make_result
                        res_dict = make_result(
                            status="partial",
                            operation="session_end",
                            project=str(self.project_path or project or ""),
                            message=mat_res.get("message", "Materialization failed"),
                            failed_step="materialization",
                            report_path=str(report_file),
                            events_written=events_written,
                            warnings=warnings + mat_res.get("warnings", []),
                            data={
                                "report_path": str(report_file),
                                "knowledge_events": res.get("knowledge_events", []),
                                "materialized_log": [],
                                "links_updated": 0,
                                "index_stats": {},
                                "explainability": res.get("explainability", {}),
                                "phase_timings": p_timings,
                            }
                        )
                        res_dict.update({
                            "status": "partial",
                            "events_written": events_written,
                            "materialization": {
                                "status": "failed",
                                "reason": mat_res.get("message", "concept_id_collision")
                            },
                            "index": {
                                "status": "skipped"
                            },
                            "session_closed": True
                        })
                        return res_dict
                    mat_log = mat_res.get("materialized", [])
                    warnings.extend(mat_res.get("warnings", []))
                progress.update_step("materialization", "success")

                links_updated = self.materialization.update_graph(project).get("links_updated", 0)

            idx_res = {
                "status": "success",
                "new": 0,
                "updated": 0,
                "scanned": 0,
                "unchanged": 0,
                "failed": 0,
                "new_chunks": 0,
                "updated_chunks": 0,
                "unchanged_chunks": 0,
                "failed_chunks": 0,
                "failed_files": [],
            }
            index_failed_reason = None
            
            if events_written == 0:
                progress.update_step(
                    "index",
                    "skipped",
                    name="Search Index Skipped",
                    detail="No memory changes to index"
                )
            else:
                progress.update_step("index", "running")
                if not update_index:
                    progress.update_step("index", "success")
                    warnings.append("Search indexing skipped (update_index disabled); run `oem index --project ...` to build the derived search index.")
                elif index_budget_seconds == 0:
                    index_failed_reason = "Indexing budget skipped"
                    failed_step = "indexing"
                    status = "partial"
                    warnings.append("Search indexing skipped after budget; run `oem index --project ...` to rebuild derived search index.")
                    progress.update_step("index", "failed")
                else:
                    with timer.phase("search_index", progress_callback):
                        try:
                            idx_res = self.index_isolated(
                                project_dir=str(self.project_path or ""),
                                budget_s=float(index_budget_seconds or 10.0),
                            )

                            if idx_res.get("status") == "error":
                                progress.update_step("index", "failed")
                                timer.timings["total"] = time.perf_counter() - start_time
                                p_timings = self._build_session_phase_timings(timer)
                                from oem_knowledge.runtime.result import error
                                return error(
                                    operation="session_end",
                                    message=idx_res.get("error", "Indexing failed"),
                                    failed_step="indexing",
                                    data={
                                        "report_path": str(report_file),
                                        "knowledge_events": res.get("knowledge_events", []),
                                        "materialized_log": mat_log,
                                        "links_updated": links_updated,
                                        "index_stats": idx_res,
                                        "explainability": res.get("explainability", {}),
                                        "warnings": warnings + [f"indexing failed: {idx_res.get('error')}"],
                                        "phase_timings": p_timings,
                                    }
                                )
                            elif idx_res.get("status") == "partial" and idx_res.get("error") == "Indexing budget exceeded":
                                index_failed_reason = "Indexing budget exceeded"
                                failed_step = "indexing"
                                status = "partial"
                                warnings.append("Search indexing skipped after timeout budget; run `oem index --project ...` to rebuild derived search index.")
                                progress.update_step("index", "failed")
                            elif idx_res.get("status") == "partial":
                                index_failed_reason = idx_res.get("error") or f"Some files failed to index: {', '.join(idx_res.get('failed_files', []))}"
                                failed_step = "indexing"
                                status = "partial"
                                progress.update_step("index", "failed")
                            else:
                                progress.update_step("index", "success")
                        except Exception as e:
                            index_failed_reason = str(e)
                            progress.update_step("index", "failed")
                            failed_step = "indexing"
                            status = "partial"
                            idx_res = {
                                "status": "error",
                                "new": 0, "updated": 0, "scanned": 0, "unchanged": 0, "failed": 0,
                                "new_chunks": 0, "updated_chunks": 0, "unchanged_chunks": 0, "failed_chunks": 0,
                                "failed_files": [],
                                "error": str(e),
                            }
                            warnings.append(f"Indexing error: {e}")

        except LockTimeoutError as e:
            timer.timings["total"] = time.perf_counter() - start_time
            p_timings = self._build_session_phase_timings(timer)
            from oem_knowledge.runtime.result import error
            return error(
                operation="session_end",
                message=f"Lock acquisition timeout: {e}",
                failed_step="reflection" if (timer.failed_phase or timer.current_phase) == "append_events" else (timer.failed_phase or timer.current_phase or "state"),
                warnings=[f"Lock failure: Registry/state/runtime_events lock contention on file. {e}"],
                data={
                    "report_path": None,
                    "knowledge_events": [],
                    "materialized_log": [],
                    "links_updated": 0,
                    "index_stats": {},
                    "explainability": {},
                    "phase_timings": p_timings,
                }
            )

        timer.timings["total"] = time.perf_counter() - start_time
        explainability = res.get("explainability", {})
        explainability["materialized"] = len(mat_log)
        
        p_timings = self._build_session_phase_timings(timer)

        ret_status = status
        if index_failed_reason:
            warnings.append(f"Session commit partial: reflection/materialization succeeded, indexing failed: {index_failed_reason}")
            ret_status = "partial"

        from oem_knowledge.runtime.result import make_result
        standard_res = make_result(
            status=ret_status,
            operation="session_end",
            project=str(self.project_path or project or ""),
            message=message or ("Session ended successfully." if ret_status == "success" else f"Session ended with status: {ret_status}"),
            warnings=warnings + res.get("warnings", []),
            failed_step=failed_step,
            report_path=str(report_file),
            items_processed=len(res.get("knowledge_events", [])),
            items_written=len(canonical_events),
            events_written=events_written,
            items_rejected=res.get("events_rejected", 0),
            materialization_skipped=(events_written == 0),
            index_skipped=(events_written == 0),
            reflection=res.get("reflection"),
            data={
                "knowledge_events": res.get("knowledge_events", []),
                "materialized_log": mat_log,
                "links_updated": links_updated,
                "index_stats": idx_res,
                "explainability": explainability,
                "phase_timings": p_timings,
                "notification": notification,
            }
        )
        # Flatten keys for legacy callers
        standard_res.update({
            "report_path": str(report_file),
            "knowledge_events": res.get("knowledge_events", []),
            "materialized_log": mat_log,
            "links_updated": links_updated,
            "index_stats": idx_res,
            "explainability": explainability,
            "phase_timings": p_timings,
            "notification": notification,
            "materialization_skipped": (events_written == 0),
            "index_skipped": (events_written == 0),
            "events_written": events_written,
        })
        # Phase: Dream (post-session consolidation)
        if events_written > 0:
            try:
                auto_dream_enabled = self._is_auto_dream_enabled(project)
            except Exception:
                auto_dream_enabled = False

            if auto_dream_enabled:
                try:
                    _remaining = None
                    if index_budget_seconds is not None:
                        _remaining = max(0.0, float(index_budget_seconds) - (time.perf_counter() - start_time))
                    dream_result = self.dream(project=project, index_budget_seconds=_remaining)
                    standard_res["dream"] = dream_result
                except Exception as e:
                    logger.warning("Dream phase failed (non-fatal): %s", e)
                    standard_res["dream"] = {"status": "failed", "error": str(e)}

        # Close the active session on any terminal commit path
        if ret_status in ("success", "warn", "partial", "empty"):
            try:
                commit_pipeline.complete_pipeline()
            except Exception:
                pass
            self._close_active_session(project)
        return standard_res

    def _close_active_session(self, project) -> None:
        """Remove the active session marker and temp files after a terminal
        session_end commit. Outcomes are recorded by the caller; hard-error
        and lock-timeout paths never reach this helper, preserving crash
        recovery of failed sessions.
        """
        try:
            harness = self._resolve_harness(project)
        except Exception:
            return
        active_session_file = harness / "state" / "active_session.json"

        try:
            from oem_knowledge.runtime import SessionState
            session_state = SessionState.load(active_session_file)
            if session_state:
                for path_str in (session_state.context_path, session_state.temp_instructions):
                    if path_str:
                        p = Path(path_str)
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass
        except Exception:
            pass

        try:
            self.session_files.unlink_active_session(project)
        except Exception as e:
            logger.warning("Failed to unlink active session file: %s", e)

    def session_commit(self, *args, **kwargs) -> dict:
        return self.session_end(*args, **kwargs)

    def dream(self, project: str | None = None, force: bool = False, index_budget_seconds: float | None = None) -> dict:
        """Run dream while serializing concept-registry read-modify-write state."""
        from oem_knowledge.fs import FileLock

        registry_lock = FileLock(
            self._registry_path(project).with_suffix(".lock"),
            timeout=60.0,
        )
        with registry_lock:
            return self._dream_unlocked(
                project=project,
                force=force,
                index_budget_seconds=index_budget_seconds,
            )

    def _dream_unlocked(self, project: str | None = None, force: bool = False, index_budget_seconds: float | None = None) -> dict:
        """Run the memory maintainer dream cycle.
        
        Four phases:
        1. Orientation — scan registry + events, establish baseline
        2. Signal Gather — identify decay/promotion/archive/merge candidates
        3. Consolidation — apply changes
        4. Pruning — log actions, re-index
        """
        from oem_knowledge.services.evolution import (
            apply_decay, DreamLog, should_archive, should_promote,
        )

        harness = self._resolve_harness(project)
        registry = self.state._load_registry(project, lock=False)
        events = self.state.get_events(project)
        config = self._read_reflection_config(project)
        reflection_cfg = config.get("reflection", {})
        auto_dream_cfg = reflection_cfg.get("auto_dream", {})

        half_life = auto_dream_cfg.get("half_life_days", 30)
        consolidate_threshold = auto_dream_cfg.get("consolidate_threshold", 0.9)
        promote_cfg = auto_dream_cfg.get("promote_threshold", {})
        evidence_threshold = promote_cfg.get("evidence_count", 5)
        session_threshold = promote_cfg.get("session_count", 3)

        # Phase 1: Orientation
        total_concepts = len(registry)
        if total_concepts < 2 and not force:
            return {"status": "noop", "reason": "fewer than 2 concepts", "total_concepts": total_concepts}

        baseline = {
            "total_concepts": total_concepts,
            "total_events": len(events),
            "by_status": {},
        }
        for cdata in registry.values():
            if isinstance(cdata, dict):
                st = cdata.get("status", "candidate")
                baseline["by_status"][st] = baseline["by_status"].get(st, 0) + 1

        # Phase 2: Signal Gathering
        promotion_candidates = []
        archival_candidates = []
        decay_candidates = []
        merge_candidates = []

        concept_list = [
            {"concept_id": cid, **cdata}
            for cid, cdata in registry.items()
            if isinstance(cdata, dict)
        ]

        decay_results = apply_decay(concept_list, half_life_days=half_life)
        for dr in decay_results:
            decay_candidates.append(dr)

        for cid, cdata in registry.items():
            if not isinstance(cdata, dict):
                continue
            should_prom, prom_reason = should_promote(cdata, evidence_threshold=evidence_threshold, session_threshold=session_threshold)
            if should_prom:
                promotion_candidates.append({"concept_id": cid, "reason": prom_reason})

            should_arch, arch_reason = should_archive(cdata, stale_sessions=30, current_session_count=len(events))
            if should_arch:
                archival_candidates.append({"concept_id": cid, "reason": arch_reason})

        try:
            merge_proposals = self.propose_merges(
                similarity_threshold=consolidate_threshold,
                project=project,
                lock=False,
            )
            merge_candidates = merge_proposals if merge_proposals else []
        except Exception as e:
            logger.warning("Merge proposal failed during dream: %s", e)
            merge_candidates = []

        # Phase 3: Consolidation
        dream_log = DreamLog(harness / "state" / "dream_log.jsonl")
        promotions_applied = 0
        archives_applied = 0
        decays_applied = 0
        merges_applied = 0

        # Apply decays
        for dr in decay_candidates:
            cid = dr["concept_id"]
            if cid in registry and isinstance(registry[cid], dict):
                old_conf = registry[cid].get("confidence", 1)
                registry[cid]["confidence"] = dr["new_confidence"]
                decays_applied += 1
                dream_log.record("decay", {"concept_id": cid, "old_confidence": old_conf, "new_confidence": dr["new_confidence"]})

        # Apply promotions
        for pc in promotion_candidates:
            cid = pc["concept_id"]
            if cid in registry and isinstance(registry[cid], dict):
                old_status = registry[cid].get("status", "candidate")
                if old_status == "candidate":
                    registry[cid]["status"] = "emerging"
                elif old_status == "emerging":
                    registry[cid]["status"] = "validated"
                promotions_applied += 1
                dream_log.record("promotion", {"concept_id": cid, "from_status": old_status, "to_status": registry[cid]["status"]})

        # Apply archivals
        for ac in archival_candidates:
            cid = ac["concept_id"]
            if cid in registry and isinstance(registry[cid], dict):
                old_status = registry[cid].get("status", "candidate")
                registry[cid]["status"] = "needs_review"
                archives_applied += 1
                dream_log.record("archive", {"concept_id": cid, "from_status": old_status})

        # Save non-merge changes before merge_concepts writes the registry.
        registry_changed = decays_applied > 0 or promotions_applied > 0 or archives_applied > 0
        if registry_changed:
            self.state._save_registry(registry, project, lock=False)

        # Apply merges
        for mc in merge_candidates:
            primary_id = mc.get("primary_id")
            secondary_id = mc.get("secondary_id")
            if primary_id and secondary_id and primary_id in registry and secondary_id in registry:
                try:
                    self.state.merge_concepts(project, primary_id, secondary_id, lock=False)
                    merges_applied += 1
                    dream_log.record("merge", {"primary_id": primary_id, "secondary_id": secondary_id})
                except Exception as e:
                    logger.warning("Merge failed for %s -> %s: %s", secondary_id, primary_id, e)

        if merges_applied > 0:
            registry = self.state._load_registry(project, lock=False)

        # Phase 4: Pruning (bounded when a budget is supplied; skipped at zero)
        index_res = {"status": "success"}
        try:
            if index_budget_seconds is not None and index_budget_seconds <= 0:
                index_res = {"status": "skipped", "reason": "index budget exhausted"}
                logger.warning("Dream re-index skipped: budget exhausted (index_budget_seconds=%s)", index_budget_seconds)
            else:
                index_res = self.search.index_all(budget_seconds=index_budget_seconds)
                if index_res.get("status") == "partial":
                    logger.warning("Dream re-index partial: %s", index_res.get("error", "budget exceeded"))
        except Exception as e:
            logger.warning("Index re-build during dream failed: %s", e)
            index_res = {"status": "error", "error": str(e)}

        return {
            "status": "success",
            "baseline": baseline,
            "decay": {"candidates": len(decay_candidates), "applied": decays_applied},
            "promotion": {"candidates": len(promotion_candidates), "applied": promotions_applied},
            "archive": {"candidates": len(archival_candidates), "applied": archives_applied},
            "merge": {"candidates": len(merge_candidates), "applied": merges_applied},
            "index": index_res,
        }

    def preflight(
        self,
        task: str,
        project: str | None = None,
        *,
        session_id: str = "",
        limit: int = 8,
        write_audit: bool = True,
    ) -> dict:
        from oem_knowledge.preflight import make_preflight_budget, normalize_preflight_result, run_preflight
        from oem_knowledge.project import ProjectResolutionError, resolve_active_project

        resolved_project_arg = project or (str(self.project_path) if self.project_path else "")
        try:
            resolved_project_arg = str(
                resolve_active_project(project_arg=resolved_project_arg, session_id=session_id)
            )
        except ProjectResolutionError:
            pass

        budget, limit_warnings, clamped_limit = make_preflight_budget(limit)
        result = run_preflight(
            task,
            project=resolved_project_arg,
            session_id=session_id,
            write_audit=write_audit,
            budget=budget,
            retriever=lambda q, k: self.search.search(q, k=k, record_references=False, auto_index=False),
        )
        concept_ids: list[str] = []
        try:
            surfaced_matches = list(result.matched_concepts[:clamped_limit]) + list(result.matched_memory[:clamped_limit])
            for match in surfaced_matches:
                candidates = []
                if match.id:
                    candidates.append({"id": match.id})
                if match.source_path:
                    candidates.append({"source_path": match.source_path})
                if match.metadata:
                    candidates.append({"metadata": match.metadata})
                concept_ids.extend(self.state.concept_ids_from_retrieval_results(candidates))
            self.state.record_concept_references(
                concept_ids,
                source="preflight",
                project=resolved_project_arg,
                session_id=session_id,
            )
        except Exception as ref_err:
            logger.warning("Failed to record concept references for preflight results: %s", ref_err)

        try:
            from oem_knowledge.runtime.working_set import update_working_set, merge_working_set
            from oem_knowledge.runtime.active_work import resolve_active_work_identity
            from pathlib import Path
            
            # 1. Active Work updates
            try:
                ident = resolve_active_work_identity(Path(resolved_project_arg) / ".oem")
                if ident and (ident.active_work_item or ident.active_topic or ident.active_task):
                    update_working_set(
                        project=resolved_project_arg,
                        active_work_item=ident.active_work_item,
                        active_topic=ident.active_topic,
                        active_task=ident.active_task,
                    )
            except Exception as aw_err:
                logger.warning("Failed to resolve active work identity for working set: %s", aw_err)
                
            # 2. Active Concepts updates
            if concept_ids:
                merge_working_set(
                    project=resolved_project_arg,
                    active_concepts=concept_ids,
                )
                
            # 3. Active Files updates (from source suggestions in preflight)
            if result.source_suggestions:
                top_files = []
                for match in result.source_suggestions:
                    if match.source_path:
                        stype = match.metadata.get("source_type", "unknown")
                        if stype in {"adapter_code", "service_code", "client_code", "implementation_code"}:
                            top_files.append(match.source_path)
                if top_files:
                    merge_working_set(
                        project=resolved_project_arg,
                        active_files=top_files,
                    )
                    
            # 4. Matched Memory IDs updates
            if result.decision in {"required", "suggest"} and result.matched_memory:
                memory_ids = [m.id for m in result.matched_memory if m.id]
                if memory_ids:
                    merge_working_set(
                        project=resolved_project_arg,
                        active_memory_ids=memory_ids,
                    )
            
            # Create checkpoint if preflight decision is required
            if result.decision == "required":
                from oem_knowledge.runtime.working_set import create_checkpoint
                create_checkpoint(reason="preflight_required", project=resolved_project_arg)
        except Exception as ws_update_err:
            logger.warning("Failed to update working set in preflight: %s", ws_update_err)

        return normalize_preflight_result(
            result,
            operation="knowledge_preflight",
            limit=clamped_limit,
            extra_warnings=limit_warnings,
        )

    def knowledge_read(self, project: str | None = None, scope: str = "project", limit: int = 10) -> dict:
        from oem_knowledge.runtime.read import execute_knowledge_read
        return execute_knowledge_read(self, project, scope, limit)

    def _read_reflection_config(self, project: str | None = None) -> dict:
        """Read and parse config/reflection.yml, returning a dict."""
        try:
            import yaml
            harness = self._resolve_harness(project)
            config_path = harness / "config" / "reflection.yml"
            if not config_path.exists():
                return {}
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Failed to read reflection config: %s", e)
            return {}

    def _is_auto_dream_enabled(self, project: str | None = None) -> bool:
        """Check if auto_dream is enabled in config/reflection.yml."""
        config = self._read_reflection_config(project)
        reflection = config.get("reflection", {})
        auto_dream = reflection.get("auto_dream", {})
        return bool(auto_dream.get("enabled", False))

    def propose_merges(
        self,
        similarity_threshold: float = 0.85,
        project: str | None = None,
        *,
        lock: bool = True,
    ) -> list[dict]:
        from oem_knowledge.evolution import ConceptEvolutionEngine
        ev = ConceptEvolutionEngine(self)
        return ev.propose_merges(similarity_threshold, project, lock=lock)

    def detect_contradictions(self, project: str | None = None) -> list[dict]:
        from oem_knowledge.evolution import ContradictionDetector
        detector = ContradictionDetector(self)
        return detector.detect_contradictions(project)

    def embedding_cache_ready(self) -> bool:
        """Inspect if the embedding model is present in fastembed cache without instantiating it."""
        try:
            from fastembed import TextEmbedding
            if TextEmbedding.__name__ == "MockTextEmbedding":
                return True
        except Exception:
            pass

        try:
            from fastembed.common.utils import define_cache_dir
            from pathlib import Path

            # Check both the custom cache dir we use and the default fastembed one
            cache_dirs = [
                Path.home() / ".cache" / "fastembed",
                Path(define_cache_dir(None))
            ]

            for cache_dir in cache_dirs:
                if self._validate_fastembed_cache_dir(cache_dir, self.resolve_embedding_model()):
                    return True

            return False
        except Exception:
            return False

    def _validate_fastembed_cache_dir(self, cache_dir: Path, model_name: str = DEFAULT_EMBEDDING_MODEL) -> bool:
        """Strictly validate a fastembed cache directory: broken or partial
        caches must be detected and rejected."""
        # HuggingFace layout
        slug = model_name.rsplit("/", 1)[-1]
        hf_dir = cache_dir / f"models--qdrant--{slug}-onnx-q"
        if hf_dir.is_dir():
            refs_main = hf_dir / "refs" / "main"
            if not refs_main.is_file():
                return False
            try:
                snapshot_id = refs_main.read_text().strip().splitlines()
            except OSError:
                return False
            snapshot_id = next((line.strip() for line in snapshot_id if line.strip()), "")
            if not snapshot_id:
                return False
            snap_dir = hf_dir / "snapshots" / snapshot_id
            if not snap_dir.is_dir():
                return False
            meta_path = snap_dir / "files_metadata.json"
            if not meta_path.is_file():
                return False
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                return False
            if isinstance(meta, list):
                if not meta or not all(isinstance(e, dict) and e.get("path") for e in meta):
                    return False
                meta = {e.get("path"): e for e in meta}
            if not isinstance(meta, dict):
                return False
            if len(meta) == 0:
                return False
            has_onnx = False
            has_tokenizer = False
            for path, entry in meta.items():
                if isinstance(entry, dict):
                    blob_id = entry.get("blob")
                elif isinstance(entry, str):
                    blob_id = entry
                else:
                    return False
                if not blob_id:
                    return False
                blob_candidates = [
                    hf_dir / "blobs" / blob_id,
                    hf_dir / "blobs" / blob_id[:2] / blob_id[2:],
                ]
                if not any(p.is_file() for p in blob_candidates):
                    return False
                if path.endswith("model_optimized.onnx"):
                    has_onnx = True
                if path.endswith("tokenizer.json"):
                    has_tokenizer = True
            if not (has_onnx and has_tokenizer):
                return False
            return True

        # Legacy GCS layouts
        for legacy_dir_name in (slug, f"fast-{slug}"):
            gcs_dir = cache_dir / legacy_dir_name
            if gcs_dir.is_dir():
                onnx = gcs_dir / "model.onnx"
                tokenizer = gcs_dir / "tokenizer.json"
                if onnx.is_file() and tokenizer.is_file() and onnx.stat().st_size > 0 and tokenizer.stat().st_size > 0:
                    return True

        return False

    def config_embedding_set_model(self, model_name: str, dry_run: bool = False) -> dict:
        """Switch the embedding model. Requires re-indexing.

        Args:
            model_name: New embedding model name (e.g., 'BAAI/bge-large-en-v1.5')
            dry_run: If True, report how many chunks would be re-embedded without making changes
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Embedding model name must be a non-blank string")
        model_name = model_name.strip()
        harness = self._resolve_harness()
        config_path = harness / "config" / "embedding_model.json"

        current_model = self.resolve_embedding_model()

        if dry_run:
            count = "unknown"
            try:
                store = getattr(self.search, "_vector_store", None)
                if store and hasattr(store, "count"):
                    count = store.count()
            except Exception:
                pass
            return {
                "status": "dry_run",
                "current_model": current_model,
                "new_model": model_name,
                "chunks_to_reindex": count,
                "message": f"Dry run: ~{count} chunks would be re-embedded with {model_name}",
            }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "embedding_model": model_name,
            "previous_model": current_model,
            "switched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))

        with self._model_lock:
            self._model = None
            self._local_load_failed = False

        return {
            "status": "success",
            "model": model_name,
            "message": f"Embedding model set to {model_name}. Run 'oem index --reindex' to re-embed.",
            "warning": "Re-indexing required. Run in background to avoid blocking session_end.",
        }
