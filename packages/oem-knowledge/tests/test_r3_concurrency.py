"""Deterministic coverage for R3 file-lock boundaries."""

import json
import threading
from collections import Counter
from unittest.mock import patch

import pytest

from oem_knowledge.engine import KnowledgeEngine
from oem_knowledge.fs import FileLock, LockTimeoutError


@pytest.fixture(autouse=True)
def fast_lock_timeouts():
    """Bound production lock waits in contention tests without changing production code."""
    original_init = FileLock.__init__

    def init(lock, lock_path, timeout=10.0, stale_timeout=300.0, poll_interval=0.1, *args, **kwargs):
        if timeout >= 1.0:
            timeout = 0.1
        if poll_interval >= 0.05:
            poll_interval = 0.01
        return original_init(lock, lock_path, timeout, stale_timeout, poll_interval, *args, **kwargs)

    with patch.object(FileLock, "__init__", init):
        yield


@pytest.fixture
def engine(tmp_path):
    eng = KnowledgeEngine(project_path=tmp_path)
    eng.init_project(str(tmp_path))
    yield eng
    eng.close()


def test_session_end_commit_lock_contention_is_visible(engine, tmp_path):
    lock_path = tmp_path / ".oem" / "commit.lock"
    with FileLock(lock_path, timeout=0.1, poll_interval=0.01):
        result = engine.session_end(
            project=str(tmp_path), conversation_text="", session_id="", events=[]
        )

    assert result["status"] == "error"
    assert result["failed_step"] == "commit_lock"
    assert any("Commit lock" in warning for warning in result["warnings"])


def test_dream_merge_preserves_merge_and_pending_registry_state(engine, tmp_path):
    registry = {
        "concept_001": {
            "concept_id": "concept_001",
            "canonical_name": "primary-concept",
            "aliases": ["primary-alias"],
            "evidence_count": 2,
            "sessions": ["session-primary"],
            "status": "candidate",
            "confidence": 1,
        },
        "concept_002": {
            "concept_id": "concept_002",
            "canonical_name": "secondary-concept",
            "aliases": ["secondary-alias"],
            "evidence_count": 3,
            "sessions": ["session-secondary"],
            "status": "candidate",
            "confidence": 1,
        },
    }
    engine.state._save_registry(registry, str(tmp_path))
    proposal = {
        "primary_id": "concept_001",
        "secondary_id": "concept_002",
        "primary_name": "primary-concept",
        "secondary_name": "secondary-concept",
        "reason": "test merge",
    }
    with patch.object(engine, "propose_merges", return_value=[proposal]), patch.object(
        engine.search, "index_all", return_value={"status": "success"}
    ):
        result = engine.dream(force=True, index_budget_seconds=0)

    assert result["status"] == "success"
    updated = engine.state._load_registry(str(tmp_path))
    assert "concept_002" not in updated
    primary = updated["concept_001"]
    assert "secondary-concept" in primary["aliases"]
    assert primary["evidence_count"] == 5
    assert {"session-primary", "session-secondary"}.issubset(set(primary["sessions"]))


def test_dream_real_merge_proposal_does_not_reacquire_registry_lock(engine, tmp_path):
    registry = {
        "concept_primary": {
            "concept_id": "concept_primary",
            "canonical_name": "shared-concept",
            "aliases": ["shared-primary-alias"],
            "evidence_count": 3,
            "sessions": ["session-primary"],
            "status": "candidate",
            "confidence": 1,
        },
        "concept_secondary": {
            "concept_id": "concept_secondary",
            "canonical_name": "shared-concep",
            "aliases": ["shared-secondary-alias"],
            "evidence_count": 2,
            "sessions": ["session-secondary"],
            "status": "candidate",
            "confidence": 1,
        },
    }
    engine.state._save_registry(registry, str(tmp_path))

    with patch.object(engine.search, "index_all", return_value={"status": "success"}):
        result = engine.dream(force=True, index_budget_seconds=0)

    assert result["status"] == "success"
    assert result["merge"]["candidates"] >= 1
    assert result["merge"]["applied"] >= 1
    updated = engine.state._load_registry(str(tmp_path))
    assert "concept_secondary" not in updated


def test_source_index_lock_contention(engine, tmp_path):
    lock_path = engine.source._manifest_path().with_suffix(".lock")
    with FileLock(lock_path, timeout=0.1, poll_interval=0.01):
        with pytest.raises(LockTimeoutError):
            engine.source.index(dry_run=True)
    assert not lock_path.is_symlink()


def test_learned_index_lock_contention_returns_error(engine):
    lock = engine.search._index_lock()
    with lock:
        result = engine.search.index_all(quiet=True)

    assert result["status"] == "error"
    text = json.dumps(result).lower()
    assert "lock" in text and ("fail" in text or "timeout" in text or "acqui" in text)


def test_concurrent_learned_index_writers_keep_registry_and_chunks_consistent(engine, tmp_path):
    engine.search.set_retrieval_mode("bm25")
    wiki = tmp_path / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "note.md").write_text("# A small note\nThis is indexed content.\n", encoding="utf-8")
    barrier = threading.Barrier(3)
    results = []
    errors = []

    def index():
        try:
            barrier.wait()
            results.append(engine.search.index_all(force=True, quiet=True))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=index) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not errors
    assert len(results) == 2
    assert all(result["status"] == "success" for result in results)
    registry = json.loads((tmp_path / ".oem" / "state" / "file_registry.json").read_text())
    assert isinstance(registry, dict)
    rows = engine.search.vector_store.conn.execute("SELECT id FROM chunks").fetchall()
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids))


def test_concurrent_user_store_appends_are_valid_jsonl(engine, tmp_path, monkeypatch):
    user_path = tmp_path / "user_events.jsonl"
    monkeypatch.setenv("OEM_USER_ID", "r3-test-user")
    monkeypatch.setattr(engine.user_store, "get_events_path", lambda: user_path)
    expected = [f"event-{i}" for i in range(10)]
    barrier = threading.Barrier(3)
    errors = []

    def append(event_ids):
        try:
            barrier.wait()
            for event_id in event_ids:
                engine.user_store.append_event({"event_id": event_id})
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=append, args=(expected[:5],)),
        threading.Thread(target=append, args=(expected[5:],)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not errors
    records = [json.loads(line) for line in user_path.read_text().splitlines() if line.strip()]
    assert Counter(record["event_id"] for record in records) == Counter(expected)
