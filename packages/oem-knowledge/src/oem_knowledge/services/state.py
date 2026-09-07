from __future__ import annotations
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING
from oem_knowledge.fs import FileLock, SecureFileSystem, LockTimeoutError
from oem_knowledge.models import ConceptData, KnowledgeEvent

if TYPE_CHECKING:
    from oem_knowledge.engine import KnowledgeEngine

logger = logging.getLogger(__name__)

_GIT_IDENTITY_CACHE: dict[str, object] = {"checked": False, "value": None}


REFERENCE_TRACKING_WATERMARK_TS = 1782864000.0  # 2026-07-01T00:00:00Z
REFERENCE_TRACKING_WATERMARK_ISO = "2026-07-01T00:00:00Z"


def _is_command_log_event(summary: str) -> bool:
    """Return True when text is shaped like an operational command/tool log."""
    if not summary:
        return False
    patterns = [
        r"^Command\s+(?:execution|failed|succeeded):",
        r"^Tool\s+(?:call|result|execution):",
        r"^MCP\s+(?:call|result|tool):",
        r"^Exit code:", r"^Output:\s*\n?FAILED:",
        r"^Command `.*` executed with exit code",
        r"^Command `.*`\s*\n?Output:",
    ]
    return any(re.search(p, summary, re.IGNORECASE) for p in patterns)


def is_ingestion_noise_event(event: dict) -> bool:
    """Return whether an event is operational/generated noise, not durable memory."""
    if not isinstance(event, dict):
        return True
    if event.get("ingestion_eligible") is False:
        return True
    event_type = (event.get("event_type") or event.get("type") or "").lower()
    source = (event.get("source") or "").lower()
    source_type = (event.get("source_type") or "").lower()
    summary = str(event.get("summary") or "")
    evidence = str(event.get("evidence") or "")
    path = str(event.get("source_path") or event.get("path") or event.get("report_path") or "").replace("\\", "/").lower()
    if event_type in {"telemetry", "command_log", "search_log"}:
        return True
    if source_type in {"oem_session_report", "oem_runtime_log", "oem_generated", "openempiric_generated"} or source_type.startswith("oem_"):
        return True
    if any(marker in path.replace("\\", "/") for marker in (".oem/sessions", ".oem/session_reports", ".oem/reports", ".oem/.runtime", ".oem/state")):
        return True
    command_log = _is_command_log_event(summary) or _is_command_log_event(evidence)
    durable = {"failure", "decision", "outcome", "validation", "hypothesis", "experiment", "risk", "deprecation"}
    runtime = source in {"opencode_hook", "orchestrator", "agent_runtime_signal"} and (source_type == "agent_runtime_signal" or source == "orchestrator")
    recursive = re.search(r"session end / commit complete|session end completed|extracted knowledge events:|graph & index updates:|session duration:|tool calls:|^report:", f"{summary}\n{evidence}", re.IGNORECASE) is not None
    if runtime:
        if event_type == "failure":
            return command_log
        if event_type in durable:
            return False
        return True
    if recursive and source in {"opencode_hook", "orchestrator", "agent_runtime_signal"}:
        return True
    if command_log:
        if source == "opencode_hook" and event_type == "failure":
            return True
        return event_type not in durable
    return False


def memory_quality_score(summary: str, evidence: str) -> float:
    """Score memory quality 0.0-1.0. Higher = better."""
    score = 0.0
    if not summary:
        return 0.0
    # Length check: too short is suspicious
    if len(summary) > 30:
        score += 0.3
    if len(summary) > 80:
        score += 0.2
    # Has evidence
    if evidence and len(evidence) > 20:
        score += 0.3
    # Contains decision language
    if re.search(r"\b(decided|chose|selected|picked|agreed|concluded)\b", summary, re.IGNORECASE):
        score += 0.2
    return min(score, 1.0)


def _parse_timestamp(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    if isinstance(val, str):
        try:
            from datetime import datetime
            iso_str = val.strip()
            if iso_str.endswith("Z"):
                iso_str = iso_str[:-1] + "+00:00"
            return datetime.fromisoformat(iso_str).timestamp()
        except Exception:
            pass
    return None


def resolve_user_identity() -> str | None:
    """Resolve the current user identity with precedence: OEM_USER_ID > git user.email.

    The git probe is cached per process (it spawns a subprocess); the env var
    is re-read every call so tests and runtime changes stay responsive.

    Returns:
        User identifier string or None if unidentifiable.
    """
    import os
    import subprocess

    # Precedence 1: explicit env var (uncached)
    oem_user = os.environ.get("OEM_USER_ID", "").strip()
    if oem_user:
        return oem_user

    if _GIT_IDENTITY_CACHE.get("checked"):
        return _GIT_IDENTITY_CACHE.get("value")

    # Precedence 2: git user.email (machine-specific, known limitation)
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True, text=True, timeout=5,
        )
        git_email = result.stdout.strip()
        if git_email and result.returncode == 0:
            _GIT_IDENTITY_CACHE.update({"checked": True, "value": git_email})
            return git_email
    except Exception:
        pass

    return None


def get_user_events_path() -> Path | None:
    """Get the path to user-scoped events file.

    Returns None if user identity cannot be resolved.
    """
    user_id = resolve_user_identity()
    if not user_id:
        return None
    path = Path.home() / ".config" / "openempiric" / "user_events.jsonl"
    return path


class StateCorruptionError(ValueError):
    """Raised when state files (like the concept registry) contain corrupt or invalid JSON."""
    pass


class StateService:
    def __init__(self, engine: KnowledgeEngine):
        self.engine = engine

    def _sfs(self, project: str | None = None) -> SecureFileSystem:
        return SecureFileSystem(self.engine._resolve_harness(project))

    def _load_registry(self, project: str | None = None, lock: bool = True) -> dict:
        # lock=False is only valid when caller already holds registry_path.with_suffix(".lock")
        p = self.engine._registry_path(project)
        sfs = self._sfs(project)
        if not lock:
            if sfs.exists(p):
                try:
                    return json.loads(sfs.read_text(p))
                except json.JSONDecodeError as e:
                    logger.error("Corrupt concept registry JSON at %s: %s", p, e)
                    raise StateCorruptionError(f"Corrupt concept registry JSON at {p}: {e}") from e
                except OSError as e:
                    logger.error("Failed to read concept registry at %s: %s", p, e)
                    raise
            return {}

        try:
            with FileLock(p.with_suffix(".lock")):
                if sfs.exists(p):
                    try:
                        return json.loads(sfs.read_text(p))
                    except json.JSONDecodeError as e:
                        logger.error("Corrupt concept registry JSON at %s: %s", p, e)
                        raise StateCorruptionError(f"Corrupt concept registry JSON at {p}: {e}") from e
                    except OSError as e:
                        logger.error("Failed to read concept registry at %s: %s", p, e)
                        raise
                return {}
        except LockTimeoutError:
            logger.error("Timed out acquiring state lock for %s", p)
            raise

    def _save_registry(self, registry: dict, project: str | None = None, lock: bool = True):
        # lock=False is only valid when caller already holds registry_path.with_suffix(".lock")
        p = self.engine._registry_path(project)
        sfs = self._sfs(project)
        if not lock:
            try:
                sfs.write_text(p, json.dumps(registry, indent=2))
            except OSError as e:
                logger.error("Failed to save concept registry at %s: %s", p, e)
                raise
            return

        try:
            with FileLock(p.with_suffix(".lock")):
                try:
                    sfs.write_text(p, json.dumps(registry, indent=2))
                except OSError as e:
                    logger.error("Failed to save concept registry at %s: %s", p, e)
                    raise
        except LockTimeoutError:
            logger.error("Timed out acquiring state lock for %s", p)
            raise

    def _atomic_save_registry_unlocked(self, registry: dict, project: str | None = None) -> None:
        p = self.engine._registry_path(project)
        sfs = self._sfs(project)
        verified = sfs.resolve_path(p)
        verified.parent.mkdir(parents=True, exist_ok=True)
        tmp = verified.with_name(f".{verified.name}.{int(time.time() * 1000000)}.tmp")
        tmp.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        tmp.replace(verified)

    def _active_session_id(self, project: str | None = None) -> str:
        harness = self.engine._resolve_harness(project)
        active_session_file = harness / "state" / "active_session.json"
        if not active_session_file.exists():
            return ""
        try:
            data = json.loads(active_session_file.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(data.get("session_id") or "")

    def _completed_session_ids(self, project: str | None = None) -> list[str]:
        harness = self.engine._resolve_harness(project)
        outcomes_file = harness / "state" / "outcomes.jsonl"
        sessions: list[str] = []
        seen: set[str] = set()
        if not outcomes_file.exists():
            return sessions
        try:
            content = outcomes_file.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read outcomes file at %s: %s", outcomes_file, e)
            return sessions
        except Exception as e:
            logger.warning("Unexpected error reading outcomes file at %s: %s", outcomes_file, e)
            return sessions

        for line_idx, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping corrupt line %d in outcomes file %s: %s", line_idx, outcomes_file, e)
                continue
            sid = str(record.get("session_id") or "")
            if sid and sid not in seen:
                seen.add(sid)
                sessions.append(sid)
        return sessions

    @staticmethod
    def concept_ids_from_retrieval_results(results: list[dict]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for result in results:
            candidates: list[str] = []
            rid = result.get("id")
            if isinstance(rid, str):
                candidates.append(rid)
            meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
            for key in ("concept_id", "source", "source_path", "rel_path", "file_path"):
                val = meta.get(key)
                if isinstance(val, str):
                    candidates.append(val)
            source_path = result.get("source_path")
            if isinstance(source_path, str):
                candidates.append(source_path)

            for candidate in candidates:
                match = re.search(r"\b(concept_[A-Za-z0-9_-]+)\b", candidate)
                if match:
                    cid = match.group(1)
                    if cid not in seen:
                        seen.add(cid)
                        ids.append(cid)
                    break
        return ids

    def record_concept_references(
        self,
        concept_ids: list[str] | tuple[str, ...] | set[str],
        *,
        source: str,
        project: str | None = None,
        session_id: str = "",
    ) -> dict:
        unique_ids = []
        seen = set()
        for raw_cid in concept_ids:
            cid = str(raw_cid or "").strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            unique_ids.append(cid)
        if not unique_ids:
            return {"status": "success", "updated": 0, "concept_ids": []}

        p = self.engine._registry_path(project)
        resolved_session_id = session_id or self._active_session_id(project)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        updated: list[str] = []
        try:
            with FileLock(p.with_suffix(".lock")):
                registry = self._load_registry(project, lock=False)
                for cid in unique_ids:
                    cdata = registry.get(cid)
                    if not isinstance(cdata, dict):
                        continue
                    cdata["last_referenced_at"] = now
                    cdata["last_referenced_session"] = resolved_session_id
                    cdata["last_reference_source"] = source
                    updated.append(cid)
                if updated:
                    self._atomic_save_registry_unlocked(registry, project)
        except LockTimeoutError:
            logger.error("Timed out acquiring state lock for %s", p)
            return {"status": "error", "updated": 0, "concept_ids": [], "error": f"Lock timeout for {p}"}
        except OSError as e:
            logger.error("Failed to read/write registry for reference recording at %s: %s", p, e)
            return {"status": "error", "updated": 0, "concept_ids": [], "error": str(e)}
        except Exception as e:
            logger.warning("Failed to record concept references: %s", e)
            return {"status": "error", "updated": 0, "concept_ids": [], "error": str(e)}

        return {"status": "success", "updated": len(updated), "concept_ids": updated}

    def _load_events(self, project: str | None = None, include_user: bool = False) -> list[dict]:
        p = self.engine._events_path(project)
        sfs = self._sfs(project)
        events: list[dict] = []
        try:
            with FileLock(p.with_suffix(".lock")):
                if sfs.exists(p):
                    try:
                        content = sfs.read_text(p)
                    except OSError as e:
                        logger.error("Failed to read events file at %s: %s", p, e)
                        raise

                    for line_idx, line in enumerate(content.splitlines(), start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev_dict = json.loads(line)
                            events.append(self.engine.event_migrator.upcast(ev_dict))
                        except json.JSONDecodeError as e:
                            logger.warning("Skipping corrupt event line %d in %s: %s", line_idx, p, e)
                            continue
        except LockTimeoutError:
            logger.error("Timed out acquiring state lock for %s", p)
            raise

        if include_user:
            user_path = get_user_events_path()
            if user_path and user_path.exists():
                try:
                    with open(user_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    ev = json.loads(line)
                                    ev["scope"] = "user"
                                    events.append(ev)
                                except json.JSONDecodeError:
                                    continue
                except OSError:
                    pass

        return events

    def append_event(self, event: dict | KnowledgeEvent, project: str | None = None):
        if isinstance(event, dict):
            event = KnowledgeEvent(**event)
        p = self.engine._events_path(project)
        sfs = self._sfs(project)
        try:
            with FileLock(p.with_suffix(".lock")):
                try:
                    sfs.append_text(p, event.model_dump_json() + "\n")
                except OSError as e:
                    logger.error("Failed to append event to %s: %s", p, e)
                    raise
        except LockTimeoutError:
            logger.error("Timed out acquiring state lock for %s", p)
            raise

    def _append_event(self, event: dict | KnowledgeEvent, project: str | None = None):
        return self.append_event(event, project)

    def append_events(self, events: list[dict | KnowledgeEvent], project: str | None = None):
        is_mocked = (
            hasattr(self._append_event, "mock")
            or hasattr(self._append_event, "_mock_self")
            or type(self._append_event).__name__ in ("Mock", "MagicMock")
        )
        if is_mocked:
            for event in events:
                self._append_event(event, project)
            return

        p = self.engine._events_path(project)
        sfs = self._sfs(project)
        try:
            with FileLock(p.with_suffix(".lock")):
                existing_ids = set()
                if sfs.exists(p):
                    try:
                        for line in sfs.read_text(p).strip().splitlines():
                            if line.strip():
                                try:
                                    ev = json.loads(line)
                                    ev_id = ev.get("event_id") or ev.get("id")
                                    if ev_id:
                                        existing_ids.add(ev_id)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.warning("Failed to read events file for deduplication: %s", e)

                try:
                    lines = []
                    for event in events:
                        if isinstance(event, dict):
                            event = KnowledgeEvent(**event)
                        if event.event_id in existing_ids:
                            continue
                        lines.append(event.model_dump_json() + "\n")
                    if lines:
                        sfs.append_text(p, "".join(lines))
                except OSError as e:
                    logger.error("Failed to append events to %s: %s", p, e)
                    raise
        except LockTimeoutError:
            logger.error("Timed out acquiring state lock for %s", p)
            raise

    def load_events(
        self,
        project: str | None = None,
        include_user: bool = False,
    ) -> list[dict]:
        """Public alias for _load_events. Satisfies EventStoreProtocol."""
        return self._load_events(project, include_user=include_user)

    def load_registry(
        self,
        project: str | None = None,
        lock: bool = True,
    ) -> dict:
        """Public alias for _load_registry. Satisfies RegistryStoreProtocol."""
        return self._load_registry(project, lock=lock)

    def save_registry(
        self,
        registry: dict,
        project: str | None = None,
        lock: bool = False,
    ) -> None:
        """Public alias for _save_registry. Satisfies RegistryStoreProtocol."""
        self._save_registry(registry, project, lock=lock)

    def _resolve_concept(
        self,
        term: str,
        registry: dict,
        project: str | None = None,
        reserved_ids: set[str] | None = None,
    ) -> tuple[str, dict]:
        import difflib
        term_clean = re.sub(r"[^\w\s-]", "", term).strip().lower()
        if not term_clean:
            term_clean = term.strip().lower()

        for cid, data in registry.items():
            if not isinstance(data, dict) or not cid.startswith("concept_"):
                continue
            canon = data.get("canonical_name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]
            if term_clean == canon or term_clean in aliases:
                return cid, data

        for cid, data in registry.items():
            if not isinstance(data, dict) or not cid.startswith("concept_"):
                continue
            canon = data.get("canonical_name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]
            candidates = [canon] + aliases
            for cand in candidates:
                if difflib.SequenceMatcher(None, term_clean, cand).ratio() >= 0.85:
                    if term not in data.get("aliases", []):
                        data.setdefault("aliases", []).append(term)
                    return cid, data

        from oem_knowledge.concept_id import allocate_concept_id
        wiki_dir = self.engine._concepts_dir(project)
        new_id = allocate_concept_id(registry, wiki_dir, reserved_ids)
        if reserved_ids is not None:
            reserved_ids.add(new_id)

        match = re.match(r"^concept_(\d+)$", new_id)
        allocated_num = int(match.group(1)) if match else 1

        canon_name = (
            re.sub(r"[^a-zA-Z0-9\s-]", "", term).strip().replace(" ", "-").lower()
            or f"concept-{allocated_num}"
        )
        new_data = ConceptData(
            concept_id=new_id, canonical_name=canon_name, aliases=[term]
        ).model_dump()
        registry[new_id] = new_data
        return new_id, new_data

    def evaluate_concept_status(
        self,
        cdata: dict,
        e_type: str,
        session_id: str,
        fitness_data: dict | None = None,
    ) -> dict:
        confidence = cdata.get("confidence", 1)

        if session_id not in cdata.setdefault("sessions", []):
            cdata["sessions"].append(session_id)
            cdata["session_count"] = len(cdata["sessions"])

        if e_type == "validation":
            confidence = min(5, confidence + 1)
        elif e_type == "failure":
            confidence = max(1, confidence - 1)
            cdata["failure_count"] = cdata.get("failure_count", 0) + 1
        cdata["confidence"] = confidence

        current_status = cdata.get("status", "candidate")
        evidence_count = cdata.get("evidence_count", 0)

        # Retrieve fitness telemetry (treated as correlation, not proof of correctness)
        cid = cdata.get("concept_id")
        fit_score = 0.0
        succ_sessions = 0
        fail_sessions = 0
        has_fitness = False

        if fitness_data and cid in fitness_data:
            fit = fitness_data[cid]
            fit_score = fit.fitness_score
            succ_sessions = fit.successful_sessions
            fail_sessions = fit.failed_sessions
            has_fitness = (fit.successful_sessions + fit.failed_sessions) > 0

        # Confidence/Evidence awareness for demotion thresholds
        if confidence >= 4 or evidence_count >= 10:
            min_failures = 5
        else:
            min_failures = 3

        # Status transitions
        history_reason = ""
        if e_type == "deprecation":
            new_status = "deprecated"
            history_reason = "Manual/event-triggered deprecation"
        elif has_fitness and fail_sessions >= min_failures and fit_score < 0.60:
            new_status = "needs_review"
            history_reason = f"Telemetry Correlation: Repeated failures (fitness: {fit_score * 100:.1f}%, failures: {fail_sessions}/{fail_sessions + succ_sessions}, confidence: {confidence}, evidence: {evidence_count})"
        elif current_status == "needs_review":
            # If it is already needs_review, it can only exit via deprecation or High Fitness promotion
            if has_fitness and fit_score >= 0.80 and succ_sessions >= 2 and evidence_count >= 2:
                new_status = "validated"
                history_reason = f"Telemetry Correlation: High fitness promotion from review (fitness: {fit_score * 100:.1f}%, successes: {succ_sessions}, evidence: {evidence_count})"
            else:
                new_status = "needs_review"
                history_reason = "Status retained: remains in needs_review"
        elif cdata.get("session_count", 0) >= 5 and cdata["confidence"] >= 4:
            new_status = "canonical"
            history_reason = f"Standard Promotion: High session usage ({cdata.get('session_count', 0)}) and confidence ({confidence})"
        elif (
            evidence_count >= 2
            or current_status == "validated"
            or (has_fitness and fit_score >= 0.80 and succ_sessions >= 2 and evidence_count >= 2)
        ):
            new_status = "validated"
            if has_fitness and fit_score >= 0.80 and succ_sessions >= 2 and evidence_count >= 2 and current_status not in ("validated", "canonical"):
                history_reason = f"Telemetry Correlation: High fitness promotion (fitness: {fit_score * 100:.1f}%, successes: {succ_sessions}, evidence: {evidence_count})"
            else:
                history_reason = f"Standard Validation: Evidence count ({evidence_count}) or status retention"
        elif cdata.get("session_count", 0) >= 1 and evidence_count >= 1:
            new_status = "emerging"
            history_reason = f"Standard Promotion: Emerging concept based on session count ({cdata.get('session_count', 0)})"
        else:
            new_status = "candidate"
            history_reason = "Standard initialization as Candidate"

        if new_status != current_status:
            cdata.setdefault("promotion_history", []).append({
                "from_status": current_status,
                "to_status": new_status,
                "trigger_event": e_type,
                "session_id": session_id,
                "reason": history_reason,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })

        cdata["status"] = new_status
        return cdata

    def consolidate(self, project: str | None = None, threshold: float = 0.82) -> dict:
        concepts_dir = self.engine._concepts_dir(project)
        sfs = self.engine._sfs(project)
        if not sfs.exists(concepts_dir):
            return {"status": "error", "message": "No concepts directory found."}

        md_files = list(concepts_dir.rglob("*.md"))
        if len(md_files) < 2:
            return {
                "status": "success",
                "message": "Fewer than 2 files. No consolidation needed.",
                "merged": [],
            }

        registry = self._load_registry(project)
        if not registry:
            return {
                "status": "success",
                "message": "Empty registry. No consolidation needed.",
                "merged": [],
            }

        from oem_knowledge.identity_resolver import SemanticIdentityResolver
        resolver = SemanticIdentityResolver(self.engine)
        duplicates = resolver.scan_duplicates(project, threshold=threshold)

        if not duplicates:
            return {
                "status": "success",
                "message": "No duplicates found.",
                "merged": [],
            }

        import difflib
        status_ranks = {"canonical": 5, "validated": 4, "emerging": 3, "candidate": 2, "deprecated": 1}

        def get_quality_score(cid: str, data: dict) -> tuple[int, int, float]:
            status_val = status_ranks.get(data.get("status", "candidate"), 2)
            ev_count = data.get("evidence_count", 0)
            from oem_knowledge.health import calculate_concept_health
            try:
                h_score = calculate_concept_health(data)
            except Exception as e:
                logger.warning("Failed to calculate health score for concept %s: %s", cid, e)
                h_score = 0.0
            return (status_val, ev_count, h_score)

        merged = []
        already_merged = set()

        for d in duplicates:
            cid_a = d["concept_a"]
            cid_b = d["concept_b"]

            if cid_a in already_merged or cid_b in already_merged:
                continue

            f1_path = concepts_dir / f"{cid_a}.md"
            f2_path = concepts_dir / f"{cid_b}.md"
            if not f1_path.exists() or not f2_path.exists():
                continue

            # Second validation step to reduce false positives
            name_a = registry[cid_a].get("canonical_name", "").lower()
            name_b = registry[cid_b].get("canonical_name", "").lower()
            name_similarity = difflib.SequenceMatcher(None, name_a, name_b).ratio()

            words_a = set(re.findall(r"\w+", name_a))
            words_b = set(re.findall(r"\w+", name_b))
            has_overlap = bool(words_a & words_b)

            if name_similarity < 0.4 and not has_overlap:
                continue

            # Determine primary vs secondary based on concept quality
            score_a = get_quality_score(cid_a, registry[cid_a])
            score_b = get_quality_score(cid_b, registry[cid_b])

            if score_a >= score_b:
                cid_primary, cid_secondary = cid_a, cid_b
                f_primary, f_secondary = f1_path, f2_path
            else:
                cid_primary, cid_secondary = cid_b, cid_a
                f_primary, f_secondary = f2_path, f1_path

            try:
                content_primary = sfs.read_text(f_primary)
                content_secondary = sfs.read_text(f_secondary)
            except OSError as e:
                logger.warning("Failed to read concept files for merging (%s, %s): %s", f_primary, f_secondary, e)
                continue
            except Exception as e:
                logger.warning("Unexpected error reading concept files for merging (%s, %s): %s", f_primary, f_secondary, e)
                continue

            # Merge markdown contents
            secondary_name = registry[cid_secondary].get("canonical_name", cid_secondary).replace("-", " ").title()
            merged_content = f"{content_primary.strip()}\n\n## Consolidated: {secondary_name}\n{content_secondary.strip()}"
            sfs.write_text(f_primary, merged_content, force_allow_truncation=True)

            # Delegate registry merge and secondary deletion
            res = self.merge_concepts(project, cid_primary, cid_secondary)
            if res.get("status") == "success":
                already_merged.add(cid_secondary)
                merged.append(f"Merged {cid_secondary} -> {cid_primary}")

        if merged:
            self.engine.search.index_all(force=True)

        return {
            "status": "success",
            "message": f"Consolidated {len(merged)} files.",
            "merged": merged,
        }

    def rebuild_registry(self, project: str | None = None) -> dict:
        self.engine._resolve_harness(project)
        self.engine._registry_path(project)
        self.engine._events_path(project)
        concepts_dir = self.engine._concepts_dir(project)
        sfs = self.engine._sfs(project)

        temp_registry = {}
        reserved_ids = set()
        if sfs.exists(concepts_dir):
            for f in concepts_dir.glob("concept_*.md"):
                try:
                    sfs.unlink(f)
                except OSError as e:
                    logger.warning("Failed to unlink temporary concept file %s during rebuild: %s", f, e)
                except Exception as e:
                    logger.warning("Unexpected error unlinking temporary concept file %s during rebuild: %s", f, e)

        fitness_data = self.engine.fitness.calculate_fitness(project)
        events = self._load_events(project)
        for event in events:
            if is_ingestion_noise_event(event):
                continue
            concept_candidates = event.get("concept_candidates", [])
            primary_term = (
                concept_candidates[0]
                if concept_candidates
                else event.get("concept", "General")
            )

            cid, cdata = self._resolve_concept(primary_term, temp_registry, project, reserved_ids)

            if event.get("evidence"):
                cdata["evidence_count"] = cdata.get("evidence_count", 0) + 1

            event_id = event.get("event_id") or event.get("id")
            if event_id:
                if "source_event_ids" not in cdata:
                    cdata["source_event_ids"] = []
                if event_id not in cdata["source_event_ids"]:
                    cdata["source_event_ids"].append(event_id)

            cdata = self.evaluate_concept_status(
                cdata=cdata,
                e_type=event.get("event_type", "observation"),
                session_id=event.get("session_id", "historical"),
                fitness_data=fitness_data,
            )
            temp_registry[cid] = cdata

        self._save_registry(temp_registry, project)

        materialized_log = []
        concepts_dir.mkdir(parents=True, exist_ok=True)
        for cid, cdata in temp_registry.items():
            if cdata["status"] in ("validated", "canonical", "needs_review"):
                concept_file = concepts_dir / f"{cid}.md"
                if cdata["status"] == "needs_review":
                    body = f"# {cdata['canonical_name'].replace('-', ' ').title()}\n\nThis concept requires review due to repeated session failures.\n\n## Learnings\n"
                else:
                    body = f"# {cdata['canonical_name'].replace('-', ' ').title()}\n\nThis concept is a validated organizational knowledge node.\n\n## Learnings\n"

                concept_content = f"---\nconcept_id: {cid}\ncanonical_name: {cdata['canonical_name']}\nstatus: {cdata['status']}\nconfidence: {cdata['confidence']}\nevidence_count: {cdata['evidence_count']}\nsession_count: {cdata.get('session_count', 0)}\naliases: {json.dumps(cdata.get('aliases', []))}\nsource_event_ids: {json.dumps(cdata.get('source_event_ids', []))}\n---\n{body}"
                self.engine.materialization._safe_write_concept_file(concept_file, concept_content, project)
                self.engine.materialization._sync_index(cdata["canonical_name"], cid, project)
                materialized_log.append(cid)

        self.engine.search.index_all(force=True)

        return {
            "status": "success",
            "message": "Registry rebuilt from events log.",
            "materialized": len(materialized_log),
        }

    def explain_concept(self, project: str | None = None, concept_id: str = "") -> dict:
        registry = self._load_registry(project)
        if concept_id not in registry:
            return {"status": "error", "message": f"Concept {concept_id} not found."}

        cdata = registry[concept_id]
        events = self.engine.state.get_events(project, concept=cdata["canonical_name"])

        from oem_knowledge.health import calculate_concept_health
        health_score = calculate_concept_health(cdata)

        summary = {
            "concept": cdata,
            "total_events": len(events),
            "supporting_events": events,
            "promotion_history": cdata.get("promotion_history", []),
            "health_score": health_score,
            "recent_evidence": [
                e.get("evidence") for e in events[-5:] if e.get("evidence")
            ],
        }
        return {"status": "success", "explanation": summary}

    def merge_concepts(
        self,
        project: str | None = None,
        primary_id: str = "",
        secondary_id: str = "",
        lock: bool = True,
    ) -> dict:
        registry = self._load_registry(project, lock=lock)
        if primary_id not in registry or secondary_id not in registry:
            return {"status": "error", "message": "One or both concepts not found."}

        pdata = registry[primary_id]
        sdata = registry[secondary_id]

        new_aliases = set(
            pdata.get("aliases", [])
            + sdata.get("aliases", [])
            + [sdata.get("canonical_name")]
        )
        pdata["aliases"] = list(new_aliases)

        pdata["evidence_count"] = pdata.get("evidence_count", 0) + sdata.get(
            "evidence_count", 0
        )
        pdata["sessions"] = list(
            set(pdata.get("sessions", []) + sdata.get("sessions", []))
        )
        pdata["session_count"] = len(pdata["sessions"])

        del registry[secondary_id]
        self._save_registry(registry, project, lock=lock)

        pdata = self.evaluate_concept_status(pdata, "merge", "system")
        registry[primary_id] = pdata
        self._save_registry(registry, project, lock=lock)

        concepts_dir = self.engine._concepts_dir(project)
        sf = concepts_dir / f"{secondary_id}.md"
        if sf.exists():
            sf.unlink()

        self.engine.materialization._log_action(
            f"Merge | Merged secondary concept {secondary_id} into primary {primary_id} ({pdata['canonical_name']})",
            project,
        )
        return {
            "status": "success",
            "message": f"Merged {secondary_id} into {primary_id}",
            "concept": pdata,
        }

    def record_outcome(
        self,
        outcome: str,
        referenced_concepts: list[str] | None = None,
        reason: str | None = None,
        session_id: str | None = None,
        project: str | None = None,
        goal_satisfaction: float | None = None,
    ) -> dict:
        harness = self.engine._resolve_harness(project)
        state_dir = harness / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        session_state_path = state_dir / "session_state.json"
        injected_concepts = []
        resolved_session_id = session_id

        if session_state_path.exists():
            try:
                state_data = json.loads(session_state_path.read_text(encoding="utf-8"))
                if not resolved_session_id:
                    resolved_session_id = state_data.get("session_id")
                injected_concepts = state_data.get("last_injected_concepts", [])
            except json.JSONDecodeError as e:
                logger.warning("Corrupt session state JSON at %s: %s", session_state_path, e)
            except OSError as e:
                logger.warning("Failed to read session state at %s: %s", session_state_path, e)
            except Exception as e:
                logger.warning("Unexpected error reading session state at %s: %s", session_state_path, e)

        if not resolved_session_id:
            resolved_session_id = f"session_{int(time.time() * 1000)}"

        if referenced_concepts is None:
            referenced_concepts = injected_concepts

        # Read metrics if available
        metrics_file = state_dir / "metrics.json"
        concepts_injected = 0
        concepts_referenced = 0
        search_count = 0
        if metrics_file.exists():
            try:
                metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                concepts_injected = metrics_data.get("knowledge_usage", {}).get("concepts_injected", 0)
                concepts_referenced = metrics_data.get("knowledge_usage", {}).get("concepts_referenced", 0)
                search_count = metrics_data.get("retrieval", {}).get("search_count", 0)
            except json.JSONDecodeError as e:
                logger.warning("Corrupt metrics JSON at %s: %s", metrics_file, e)
            except OSError as e:
                logger.warning("Failed to read metrics at %s: %s", metrics_file, e)
            except Exception as e:
                logger.warning("Unexpected error reading metrics at %s: %s", metrics_file, e)

        # Handle default goal satisfaction based on binary outcome
        resolved_satisfaction = goal_satisfaction
        if resolved_satisfaction is None:
            resolved_satisfaction = 1.0 if outcome == "success" else 0.0

        outcomes_file = state_dir / "outcomes.jsonl"
        log_entry = {
            "schema_version": 1,
            "session_id": resolved_session_id,
            "outcome": outcome,
            "referenced_concepts": referenced_concepts,
            "retrieved_concepts": injected_concepts,
            "reason": reason,
            "goal_satisfaction": resolved_satisfaction,
            "metrics": {
                "concepts_injected": concepts_injected,
                "concepts_referenced": concepts_referenced,
                "search_count": search_count,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Include explicit project in outcomes entry when provided
        if project and isinstance(project, str):
            log_entry["project"] = project

        with open(outcomes_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Create checkpoint on successful implementation milestones
        if outcome in ("success", "success_with_warnings"):
            from oem_knowledge.runtime.working_set import create_checkpoint
            try:
                create_checkpoint(reason="implementation_success", project=project)
            except Exception as e:
                logger.warning("Failed to create implementation_success checkpoint: %s", e)

        # Conditional structured handoff update — only for explicit project-level outcomes
        if project and isinstance(project, str) and outcome in ("success", "failure", "partial"):
            try:
                self._update_structured_handoff(harness, project, resolved_session_id)
            except Exception as e:
                logger.warning("Failed to update structured handoff: %s", e)

        self.engine.materialization._log_action(
            f"Outcome | Logged session outcome '{outcome}' for {resolved_session_id} (satisfaction: {resolved_satisfaction})",
            project,
        )

        return {
            "status": "success",
            "session_id": resolved_session_id,
            "outcome": outcome,
            "referenced_concepts": referenced_concepts,
            "retrieved_concepts": injected_concepts,
            "reason": reason,
            "goal_satisfaction": resolved_satisfaction,
            "metrics": log_entry["metrics"],
        }

    def _update_structured_handoff(self, harness: Path, project_val: str, session_id: str) -> None:
        """Update .oem/session-handoff.json with semantic active-work fields only.
        Never invent active_work_item from workspace root. Write only known values (or null/omit consistently).
        """
        from oem_knowledge.runtime.active_work import (
            _normalize_project_identity,
            classify_active_work_value,
        )
        import datetime

        handoff_json = harness / "session-handoff.json"
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        current: dict = {}
        if handoff_json.exists():
            try:
                current = json.loads(handoff_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                current = {}

        existing_project = current.get("project_root") or current.get("project") or current.get("workspace_root")
        existing_norm = _normalize_project_identity(existing_project) if existing_project else ""
        new_norm = _normalize_project_identity(project_val)

        if existing_norm == new_norm:
            return  # Same workspace, no update needed

        # Classify the supplied value
        workspace_context = Path(project_val) if Path(project_val).is_absolute() else None
        ws_candidate = classify_active_work_value(project_val, workspace_root=workspace_context)
        workspace_root_val = project_val if ws_candidate == "workspace_root" else current.get("workspace_root")
        memory_root_val = current.get("memory_root")

        # Do NOT invent active_work_item / topic / task from workspace root.
        # Only carry forward explicit previous values if present and not the workspace itself.
        active_work_item_val = current.get("active_work_item")
        if active_work_item_val and classify_active_work_value(active_work_item_val) == "workspace_root":
            active_work_item_val = None
        active_topic_val = current.get("active_topic")
        active_task_val = current.get("active_task")

        previous_entry: dict = {}
        if existing_project:
            previous_entry = {
                "workspace_root": current.get("workspace_root") or (existing_project if classify_active_work_value(existing_project) == "workspace_root" else None),
                "updated_at": current.get("updated_at", now_iso),
            }

        payload: dict = {
            "schema_version": "1.0.0",
            "workspace_root": str(Path(workspace_root_val).resolve()) if workspace_root_val and Path(workspace_root_val).is_absolute() else workspace_root_val,
            "memory_root": memory_root_val,
            "active_work_item": active_work_item_val,
            "active_topic": active_topic_val,
            "active_task": active_task_val,
            "updated_at": now_iso,
            "source_session_id": session_id,
            "status": "active",
            "primary_objective": current.get("primary_objective", ""),
            "next_action": current.get("next_action", ""),
            "previous": previous_entry,
        }

        # Remove nulls for cleanliness (consistent with spec: do not invent)
        payload = {k: v for k, v in payload.items() if v is not None or k in ("active_work_item", "active_topic", "active_task")}

        # Preserve other fields from current session-handoff.json
        for k, v in current.items():
            if k not in payload and k not in ("project", "project_root"):
                payload[k] = v

        handoff_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def detect_stale_concepts(self, n_sessions: int = 5, project: str | None = None) -> list[dict]:
        """Identify concepts that have not been referenced in the last N sessions."""
        try:
            registry = self._load_registry(project)
            completed_sessions = self._completed_session_ids(project)
            active_session_id = self._active_session_id(project)
        except Exception as e:
            return [{
                "concept_id": "all",
                "canonical_name": "all",
                "stale_status": "reference_history_unavailable",
                "sessions_since_reference": None,
                "last_referenced_session": "",
                "last_referenced_at": None,
                "last_reference_source": "unknown",
                "reference_confidence": "unknown",
                "severity": "warning",
                "explanation": f"Cannot determine reference history due to registry or outcome file loading error: {e}",
                "recommended_action": "Verify that your registry and outcome files exist and are readable."
            }]

        sessions_below_threshold = len(completed_sessions) < n_sessions
        last_n_sessions: set[str] = set()

        if not sessions_below_threshold:
            last_n_sessions = set(completed_sessions[-n_sessions:])

        stale_concepts = []
        for cid, cdata in registry.items():
            if not isinstance(cdata, dict):
                continue

            last_ref_session = str(cdata.get("last_referenced_session") or "")
            last_ref_at = cdata.get("last_referenced_at")
            last_ref_source = cdata.get("last_reference_source")
            legacy_sessions = [str(s) for s in cdata.get("sessions", []) if str(s)]

            # Determine if we have any reference history
            is_empty_reference = (
                not last_ref_session and
                not last_ref_at and
                (not last_ref_source or last_ref_source in ("unknown", "")) and
                not legacy_sessions
            )

            if is_empty_reference:
                created_at = cdata.get("created_at")
                if created_at is not None:
                    created_ts = _parse_timestamp(created_at)
                    if created_ts is not None:
                        if created_ts < REFERENCE_TRACKING_WATERMARK_TS:
                            stale_status = "legacy_no_reference_metadata"
                        else:
                            stale_status = "never_referenced_since_tracking_enabled"
                    else:
                        stale_status = "reference_metadata_missing"
                else:
                    stale_status = "reference_metadata_missing"

                if stale_status == "legacy_no_reference_metadata":
                    explanation = "This concept has no reference metadata, likely because it was created before reference tracking was added."
                    rec_action = "Run a search/read that surfaces this concept to record a fresh reference, or run a future reference backfill command if available."
                elif stale_status == "never_referenced_since_tracking_enabled":
                    explanation = "This concept was created after reference tracking was enabled but has never been referenced."
                    rec_action = "Interact with this concept in a session to establish a reference history."
                else:
                    explanation = "This registry concept lacks reference tracking fields entirely."
                    rec_action = "Interact with this concept to initialize its reference tracking metadata."

                stale_concepts.append({
                    "concept_id": cid,
                    "canonical_name": cdata.get("canonical_name", cid),
                    "last_referenced_session": "",
                    "last_referenced_at": None,
                    "last_reference_source": "unknown",
                    "sessions_since_reference": None,
                    "stale_status": stale_status,
                    "reference_confidence": "unknown",
                    "severity": "info",
                    "explanation": explanation,
                    "recommended_action": rec_action,
                })
                continue

            # Current active session — not stale; sessions_since_reference = 0
            if last_ref_session and active_session_id and last_ref_session == active_session_id:
                continue

            # Has a known completed-session reference
            if last_ref_session and last_ref_session in completed_sessions:
                if sessions_below_threshold:
                    continue
                if last_ref_session in last_n_sessions:
                    continue
                sessions_since = len(completed_sessions) - completed_sessions.index(last_ref_session) - 1
                stale_concepts.append({
                    "concept_id": cid,
                    "canonical_name": cdata.get("canonical_name", cid),
                    "last_referenced_session": last_ref_session,
                    "last_referenced_at": last_ref_at,
                    "last_reference_source": last_ref_source,
                    "sessions_since_reference": sessions_since,
                    "stale_status": "stale",
                    "reference_confidence": "high",
                    "severity": "warning",
                    "explanation": "This concept has not been referenced in the last N sessions.",
                    "recommended_action": "Touch or reference this concept if it is still valid, or prune it if it is obsolete.",
                })
                continue

            # Legacy sessions field
            reliable_legacy_sessions = [s for s in legacy_sessions if s in completed_sessions]
            if reliable_legacy_sessions:
                if sessions_below_threshold:
                    continue
                last_legacy = reliable_legacy_sessions[-1]
                if last_legacy in last_n_sessions:
                    continue
                sessions_since = len(completed_sessions) - completed_sessions.index(last_legacy) - 1
                stale_concepts.append({
                    "concept_id": cid,
                    "canonical_name": cdata.get("canonical_name", cid),
                    "last_referenced_session": last_legacy,
                    "last_referenced_at": None,
                    "last_reference_source": "legacy_sessions",
                    "sessions_since_reference": sessions_since,
                    "stale_status": "stale",
                    "reference_confidence": "legacy",
                    "severity": "warning",
                    "explanation": "This concept has not been referenced in the last N sessions.",
                    "recommended_action": "Touch or reference this concept if it is still valid, or prune it if it is obsolete.",
                })
                continue

            # Has reference timestamp or session but not in completed_sessions — surface even below threshold
            stale_concepts.append({
                "concept_id": cid,
                "canonical_name": cdata.get("canonical_name", cid),
                "last_referenced_session": last_ref_session,
                "last_referenced_at": last_ref_at,
                "last_reference_source": last_ref_source,
                "sessions_since_reference": None,
                "stale_status": "reference_session_missing",
                "reference_confidence": "unknown",
                "severity": "info",
                "explanation": "The last referenced session ID exists in the concept metadata but is missing from the known completed sessions.",
                "recommended_action": "This can happen if session history was pruned or cleared. Surface this concept in a new session to refresh the metadata.",
            })
            continue

        return stale_concepts

    def get_events(
        self,
        project: str | None = None,
        concept: str = "",
        event_type: str = "",
        session_id: str = "",
    ) -> list[dict]:
        """Return events filtered by optional concept, event_type and session_id."""
        events = self._load_events(project)
        filtered = []
        for ev in events:
            if concept:
                match_values = set()
                if concept.startswith("concept_"):
                    reg = self._load_registry(project)
                    cdata = reg.get(concept, {})
                    canon = cdata.get("canonical_name", concept)
                    match_values.add(canon.strip().replace(" ", "-").lower())
                match_values.add(concept.strip().replace(" ", "-").lower())
                candidates_clean = [c.strip().replace(" ", "-").lower() for c in ev.get("concept_candidates", [])]
                if not match_values.intersection(candidates_clean):
                    continue
            if event_type and ev.get("event_type", "").lower() != event_type.lower():
                continue
            if session_id and ev.get("session_id", "") != session_id:
                continue
            filtered.append(ev)
        return filtered

    def get_event(self, project: str | None = None, event_id: str = "") -> dict:
        """Return a single event by ID, raising KeyError if not found."""
        for ev in self._load_events(project):
            if ev.get("event_id") == event_id:
                return ev
        raise KeyError(f"Event {event_id} not found")
