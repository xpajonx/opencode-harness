from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is available in the runtime today
    yaml = None

try:
    import pathspec
except ImportError:  # pragma: no cover - allow fallback matching in lightweight installs
    pathspec = None

if TYPE_CHECKING:
    from oem_knowledge.engine import KnowledgeEngine

logger = logging.getLogger(__name__)

@dataclass
class SourceIndexConfig:
    include: list[str]
    exclude: list[str]
    exclude_globs: list[str]
    max_file_size_bytes: int
    chunk_lines: int
    chunk_overlap_lines: int
    max_read_lines: int
    max_read_characters: int

@dataclass
class SourceFileClassification:
    path: Path
    rel_path: str
    eligible: bool
    reason: str
    file_type: str
    language: str
    size_bytes: int
    metadata_only: bool = False

@dataclass
class SourceDiscoveryResult:
    project_root: str
    files_included: int
    files_excluded: int
    excluded_reasons: dict[str, int]
    warnings: list[str]
    discovered_files: list[SourceFileClassification]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "files_included": self.files_included,
            "files_excluded": self.files_excluded,
            "excluded_reasons": self.excluded_reasons,
            "warnings": self.warnings
        }

DEFAULT_SOURCE_CONFIG: dict[str, Any] = {
    "version": 1,
    "include": [
        "src/**",
        "packages/**",
        "tests/**",
        "docs/**",
        "execution/**",
        "agent/**",
        "README.md",
        "AGENTS.md",
        "pyproject.toml",
        "package.json"
    ],
    "exclude": [
        ".git/**",
        ".oem/**",
        "node_modules/**",
        "dist/**",
        "build/**",
        ".venv/**",
        ".obsidian/plugins/**",
        "*.png",
        "*.jpg",
        "*.pdf",
        ".env"
    ],
    "chunk_lines": 120,
    "chunk_overlap_lines": 20,
    "max_file_size_bytes": 524288,
    "max_read_lines": 400,
    "max_read_characters": 50000,
    "exclude_globs": [],
}

# Historical migration sentinel: old default max_read_lines value.
# Used to detect projects that manually customized max_read_lines while still using
# old include patterns that need upgrading to new defaults (adds execution/** and agent/**).
_OLD_DEFAULT_MAX_READ_LINES = 200

MANDATORY_EXCLUDED_DIRS = {
    ".git",
    ".oem",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
}
MANDATORY_EXCLUDED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".jar",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".pyc",
    ".pyo",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}
LOCKFILE_NAMES = {
    "uv.lock",
    "poetry.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "Pipfile.lock",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".sh": "shell",
    ".ps1": "powershell",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".css": "css",
    ".html": "html",
    ".sql": "sql",
}


def _estimate_tokens(characters: int) -> int:
    if characters <= 0:
        return 0
    return max(1, math.ceil(characters / 4))


def _deep_copy_defaults() -> dict[str, Any]:
    return {
        "version": DEFAULT_SOURCE_CONFIG["version"],
        "include": list(DEFAULT_SOURCE_CONFIG["include"]),
        "exclude": list(DEFAULT_SOURCE_CONFIG["exclude"]),
        "chunk_lines": DEFAULT_SOURCE_CONFIG["chunk_lines"],
        "chunk_overlap_lines": DEFAULT_SOURCE_CONFIG["chunk_overlap_lines"],
        "max_file_size_bytes": DEFAULT_SOURCE_CONFIG["max_file_size_bytes"],
        "max_read_lines": DEFAULT_SOURCE_CONFIG["max_read_lines"],
        "max_read_characters": DEFAULT_SOURCE_CONFIG["max_read_characters"],
        "exclude_globs": list(DEFAULT_SOURCE_CONFIG["exclude_globs"]),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_rel_path(path: str | Path) -> str:
    return str(PurePosixPath(str(path).replace("\\", "/")))


def _path_tokens(rel_path: str) -> set[str]:
    pieces = re.split(r"[^A-Za-z0-9_]+", rel_path.lower())
    tokens = {piece for piece in pieces if len(piece) > 1}
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        expanded.update(part for part in token.split("_") if len(part) > 1)
    return expanded


def _source_component_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for original in re.findall(r"\w+", text):
        lowered = original.lower()
        if len(lowered) > 1 and lowered not in SOURCE_STOPWORDS:
            tokens.add(lowered)
        for underscore_part in original.split("_"):
            for component in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+|\d+",
                underscore_part,
            ):
                lowered_component = component.lower()
                if len(lowered_component) > 1 and lowered_component not in SOURCE_STOPWORDS:
                    tokens.add(lowered_component)
    return tokens


def _extract_symbols(text: str) -> list[str]:
    symbols = []
    seen = set()
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text):
        symbol = match.group(0)
        lowered = symbol.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        symbols.append(symbol)
        if len(symbols) >= 128:
            break
    return symbols


def _guess_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def _guess_file_type(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in LOCKFILE_NAMES or name.endswith(".lock"):
        return "lockfile"
    if suffix in {".md", ".rst", ".txt"}:
        return "document"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "config"
    if suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".css", ".html", ".sql", ".sh", ".ps1"}:
        return "code"
    return "text"


# Ranking Score Weights & Constants
BOOST_DOTTED_IDENTIFIER = 4.0
BOOST_SNAKE_IDENTIFIER = 3.0
BOOST_UPPERCASE_IDENTIFIER = 3.0
BOOST_CAMEL_IDENTIFIER = 3.0
BOOST_SYMBOL_DEFINITION = 4.0
BOOST_PATH_MATCH_MULTIPLIER = 2.0

BOOST_IDENTIFIER_MULTI_MATCH_1 = 2.0
BOOST_IDENTIFIER_MULTI_MATCH_2 = 4.0
BOOST_IDENTIFIER_MULTI_MATCH_3_PLUS = 6.0

BOOST_CODE_QUERY_ADAPTER = 6.0
BOOST_CODE_QUERY_IMPLEMENTATION = 5.0
BOOST_CODE_QUERY_RELEVANT_TEST_ACTIVE = 12.0
BOOST_CODE_QUERY_RELEVANT_TEST_INACTIVE = 2.0

BOOST_DOC_QUERY_AGENT = 5.0
BOOST_DOC_QUERY_DOC = 2.0

PENALTY_AGENT_INSTRUCTION = 10.0
PENALTY_BROAD_DOC = 8.0
PENALTY_UNRELATED_TEST_ACTIVE = 6.0
PENALTY_UNRELATED_TEST_INACTIVE = 10.0
PENALTY_CONFIG_FILE = 5.0
PENALTY_GENERATED_OR_CACHE = 15.0
PENALTY_LARGE_LOW_DENSITY = 5.0

# Explicit score literals used in ranking logic
BOOST_EXACT_PATH_QUERY = 100.0
PENALTY_METADATA_ONLY = 10.0
MIN_POSITIVE_RESULT_SCORE = 0.0
WEAK_RESULT_MIN_SCORE = 5.0
SOURCE_DENSE_EVIDENCE_THRESHOLD = 0.25
SOURCE_DENSE_NO_LEXICAL_EVIDENCE_THRESHOLD = 0.55
SCORE_TIE_EPSILON = 1e-9

# BM25 normalization parameters (used in _bm25_scores)
BM25_K1 = 1.5
BM25_B = 0.75

LARGE_DOCUMENT_THRESHOLD_CHARS = 2000

SOURCE_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "from",
    "by", "at", "as", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "than", "that", "this", "it", "its",
    "not", "no", "how", "why", "what", "where", "who", "which",
    "do", "does", "did", "will", "would", "could", "should", "can", "may",
}

_UPPER_NOISE = {"THE", "AND", "FOR", "ARE", "NOT", "ALL", "GET", "SET", "PUT", "HOW", "WHY", "CAN", "YOU", "HAS", "HAD", "WAS", "WERE"}

CODE_SUFFIXES = {".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".java", ".kt"}


def _extract_source_identifiers(query: str) -> list[str]:
    # Extract dotted identifiers (e.g., chat.ask)
    dotted = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+\b', query)
    dotted = [d for d in dotted if not any(
        d.lower().endswith(f".{ext}") for ext in
        ["md", "py", "ts", "js", "json", "yaml", "yml", "toml", "txt", "css", "html", "sh"])]
    dotted = [d for d in dotted if all(len(seg) >= 2 for seg in d.split("."))]

    # Extract snake_case identifiers (e.g. source_ids, get_notebook)
    snake_case = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b', query)

    # Extract uppercase constants (e.g., GET_NOTEBOOK)
    upper_case = re.findall(r'\b[A-Z][A-Z_0-9]+\b', query)
    upper_case = [u for u in upper_case if u not in _UPPER_NOISE]

    # Extract CamelCase/PascalCase (e.g. NotebookLM)
    pascal_case = re.findall(r'\b[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*\b', query)

    # Extract camelCase (e.g. getNotebook)
    lower_camel = re.findall(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b', query)

    all_idents = list(set(dotted + snake_case + upper_case + pascal_case + lower_camel))
    
    cleaned = []
    for i in all_idents:
        if len(i) < 3:
            continue
        if i.lower() in SOURCE_STOPWORDS:
            continue
        cleaned.append(i)
        
    return sorted(cleaned)


def _has_boundary_identifier_match(identifier: str, document: str) -> bool:
    escaped = re.escape(identifier)
    if re.search(rf'(?<![\w.]){escaped}(?!\w)', document, re.IGNORECASE):
        return True
    if re.search(rf'(?<!\w)_{escaped}(?!\w)', document, re.IGNORECASE):
        return True
    return False


def _has_symbol_definition(identifier: str, document: str) -> bool:
    name = identifier
    if "." in identifier:
        name = identifier.split(".")[-1]
    
    escaped = re.escape(name)
    patterns = [
        rf'\bdef\s+{escaped}\b',
        rf'\bclass\s+{escaped}\b',
        rf'\bfunction\s+{escaped}\b',
        rf'\bconst\s+{escaped}\b',
        rf'\blet\s+{escaped}\b',
        rf'\bvar\s+{escaped}\b',
        rf'\binterface\s+{escaped}\b',
        rf'\btype\s+{escaped}\b',
    ]
    for pattern in patterns:
        if re.search(pattern, document):
            return True
    return False


def _matched_source_identifiers(identifiers: list[str], document: str) -> list[str]:
    return [ident for ident in identifiers if _has_boundary_identifier_match(ident, document)]


def _detect_source_query_intent(query: str) -> dict:
    q = query.strip()
    q_lower = q.lower()

    identifiers = _extract_source_identifiers(q)

    words = re.findall(r'\w+', q_lower)
    ident_lower = {i.lower() for i in identifiers}
    domain_terms = [
        w for w in words
        if w not in SOURCE_STOPWORDS and w not in ident_lower and len(w) > 2
    ]

    error_signals = {"timeout", "error", "exception", "fail", "failed", "failure",
                     "crash", "broken", "bug", "regression"}
    error_terms = [w for w in words if w in error_signals]

    debug_signals = {"debug", "fix", "repair", "trace", "workaround", "patch"}
    has_debug = any(w in words for w in debug_signals) or bool(error_terms)

    test_signals = {"test", "regression", "pytest", "unittest", "failing"}
    has_test = any(w in words for w in test_signals)

    doc_intent = any(w in words for w in
                     ["agent", "instruction", "instructions", "documentation", "docs", "readme", "help", "claude"])
    config_intent = any(w in words for w in ["config", "configuration", "settings", "setup", "pyproject", "package"])

    has_identifiers = len(identifiers) > 0
    source_intent = (has_identifiers or has_debug or (not doc_intent and not config_intent)) and (has_identifiers or bool(words))
    debug_intent = has_debug
    test_intent = has_test

    return {
        "source_intent": source_intent,
        "debug_intent": debug_intent,
        "test_intent": test_intent,
        "doc_intent": doc_intent,
        "config_intent": config_intent,
        "identifiers": identifiers,
        "domain_terms": domain_terms,
        "error_terms": error_terms,
    }


def _classify_source_result(rel_path: str, document: str, matched_identifiers: list[str]) -> str:
    rel_lower = rel_path.lower()
    prefixed = f"/{rel_lower}"
    name = PurePosixPath(rel_lower).name

    # Priority 1: generated_or_cache
    gen_cache_patterns = ["/generated/", "generated/", "/cache/", "cache/", "/.cache/", ".cache/", "/__pycache__/", "__pycache__/", "/build/", "build/", "/dist/", "dist/", "/.obsidian/", ".obsidian/"]
    if any(p in prefixed or rel_lower.startswith(p) for p in gen_cache_patterns):
        return "generated_or_cache"

    # Priority 2: agent_instruction
    if name in ("agents.md", "claude.md") or "/.agents/" in prefixed or rel_lower.startswith(".agents/"):
        return "agent_instruction"

    # Priority 3: readme_doc / project_doc
    if name in ("readme.md", "readme.txt") or name.startswith("readme."):
        return "readme_doc"
    if "/docs/" in prefixed or rel_lower.startswith("docs/"):
        return "project_doc"

    # Priority 4: config_file
    suffix = PurePosixPath(rel_lower).suffix
    if suffix in (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"):
        return "config_file"

    # Priority 5: relevant_test / unrelated_test
    # Priority 6: adapter_code / service_code / client_code
    # Priority 7: implementation_code
    if suffix in CODE_SUFFIXES:
        is_test = ("/tests/" in prefixed or rel_lower.startswith("tests/")
                   or name.startswith("test_")
                   or name.endswith("_test.py")
                   or name.endswith("_test.ts")
                   or name.endswith("_test.js")
                   or name.endswith("_test.tsx"))
        if is_test:
            if matched_identifiers:
                return "relevant_test"
            return "unrelated_test"

        is_adapter = "/adapter" in prefixed or rel_lower.startswith("adapter")
        is_service = "/service" in prefixed or rel_lower.startswith("service")
        is_client = "/client" in prefixed or rel_lower.startswith("client")
        if is_adapter:
            return "adapter_code"
        if is_service:
            return "service_code"
        if is_client:
            return "client_code"
        return "implementation_code"

    return "unknown"


class _IgnoreMatcher:
    def __init__(self, patterns: list[str]):
        self.patterns = self._clean_patterns(patterns)
        self.spec = None
        if pathspec is not None and self.patterns:
            try:
                self.spec = pathspec.PathSpec.from_lines("gitwildmatch", self.patterns)
            except Exception:
                self.spec = None

    def _clean_patterns(self, patterns: list[str]) -> list[str]:
        cleaned = []
        for raw in patterns:
            line = str(raw).strip()
            if not line or line.startswith("#"):
                continue
            cleaned.append(line.replace("\\", "/"))
        return cleaned

    def matches(self, rel_path: str, is_dir: bool = False) -> bool:
        normalized = rel_path.replace("\\", "/")
        candidate = normalized + ("/" if is_dir and not normalized.endswith("/") else "")
        if self.spec is not None:
            return self.spec.match_file(candidate)

        ignored = False
        basename = PurePosixPath(normalized).name
        for pattern in self.patterns:
            negate = pattern.startswith("!")
            pat = pattern[1:] if negate else pattern
            pat = pat.lstrip("/")
            matched = False
            if pat.endswith("/"):
                base = pat.rstrip("/")
                matched = normalized == base or normalized.startswith(base + "/")
            elif "/" not in pat:
                matched = fnmatch.fnmatch(basename, pat) or fnmatch.fnmatch(normalized, pat)
            elif "**" in pat:
                segments = pat.split("/")
                regex_parts = []
                for idx, segment in enumerate(segments):
                    if segment == "**":
                        if idx == 0:
                            regex_parts.append("(?:|.*/)")
                        elif idx == len(segments) - 1:
                            regex_parts.append("(?:/.*)?")
                        else:
                            regex_parts.append("(?:/[^/]*(?:/[^/]*)*)?")
                    else:
                        seg_regex = ""
                        i = 0
                        while i < len(segment):
                            c = segment[i]
                            if c == "*":
                                seg_regex += "[^/]*"
                                i += 1
                            elif c == "?":
                                seg_regex += "[^/]"
                                i += 1
                            else:
                                seg_regex += re.escape(c)
                                i += 1
                        if idx > 0 and segments[idx - 1] != "**":
                            regex_parts.append("/")
                        regex_parts.append(seg_regex)
                matched = re.match("^" + "".join(regex_parts) + "$", normalized) is not None
            else:
                matched = fnmatch.fnmatch(normalized, pat)
            if matched:
                ignored = not negate
        return ignored


class _SourceIndexStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_chunks (
                    id TEXT PRIMARY KEY,
                    rel_path TEXT NOT NULL,
                    document TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    path_text TEXT NOT NULL,
                    symbols_text TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    metadata TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_files (
                    rel_path TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    line_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_chunks_rel_path ON source_chunks(rel_path)"
            )
            self.conn.execute("""CREATE TABLE IF NOT EXISTS source_embeddings (
                chunk_id TEXT NOT NULL, embedding_generation TEXT NOT NULL,
                embedding_model TEXT NOT NULL, embedding_dimension INTEGER NOT NULL,
                content_hash TEXT NOT NULL, embedding TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(chunk_id, embedding_generation)
            )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source_embeddings_model_dim ON source_embeddings(embedding_model, embedding_dimension)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source_embeddings_chunk ON source_embeddings(chunk_id)")

    def upsert_embeddings(self, rows: list[dict[str, Any]]) -> None:
        with self.conn:
            self.conn.executemany("""INSERT OR REPLACE INTO source_embeddings
                (chunk_id, embedding_generation, embedding_model, embedding_dimension, content_hash, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", [(
                r["chunk_id"], r["embedding_generation"], r["embedding_model"], r["embedding_dimension"],
                r["content_hash"], r["embedding"], r["created_at"]) for r in rows])

    def load_embeddings(self, model: str, dimension: int) -> dict[tuple[str, str], list[float]]:
        rows = self.conn.execute("SELECT chunk_id, content_hash, embedding FROM source_embeddings WHERE embedding_model=? AND embedding_dimension=?", (model, dimension)).fetchall()
        return {(r["chunk_id"], r["content_hash"]): json.loads(r["embedding"]) for r in rows}

    def replace_file(self, file_record: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        rel_path = file_record["rel_path"]
        with self.conn:
            self.conn.execute("DELETE FROM source_chunks WHERE rel_path = ?", (rel_path,))
            self.conn.execute(
                """
                INSERT OR REPLACE INTO source_files (
                    rel_path, status, size_bytes, line_count, chunk_count, content_hash, mtime_ns, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_path,
                    file_record["status"],
                    int(file_record["size_bytes"]),
                    int(file_record["line_count"]),
                    int(file_record["chunk_count"]),
                    file_record["content_hash"],
                    int(file_record["mtime_ns"]),
                    json.dumps(file_record["metadata"]),
                ),
            )
            if chunks:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO source_chunks (
                        id, rel_path, document, snippet, path_text, symbols_text, start_line, end_line, metadata, content_hash, mtime_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk["id"],
                            rel_path,
                            chunk["document"],
                            chunk["snippet"],
                            chunk["path_text"],
                            chunk["symbols_text"],
                            chunk.get("start_line"),
                            chunk.get("end_line"),
                            json.dumps(chunk["metadata"]),
                            chunk["content_hash"],
                            int(chunk["mtime_ns"]),
                        )
                        for chunk in chunks
                    ],
                )

    def remove_files(self, rel_paths: list[str]) -> None:
        if not rel_paths:
            return
        with self.conn:
            self.conn.executemany(
                "DELETE FROM source_embeddings WHERE chunk_id IN ("
                "SELECT id FROM source_chunks WHERE rel_path = ?)",
                [(item,) for item in rel_paths],
            )
            self.conn.executemany("DELETE FROM source_chunks WHERE rel_path = ?", [(item,) for item in rel_paths])
            self.conn.executemany("DELETE FROM source_files WHERE rel_path = ?", [(item,) for item in rel_paths])

    def iter_chunks(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, rel_path, document, snippet, path_text, symbols_text, start_line, end_line, metadata, content_hash, mtime_ns FROM source_chunks"
        ).fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "rel_path": row["rel_path"],
                    "document": row["document"],
                    "snippet": row["snippet"],
                    "path_text": row["path_text"],
                    "symbols_text": row["symbols_text"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "metadata": json.loads(row["metadata"]),
                    "content_hash": row["content_hash"],
                    "mtime_ns": row["mtime_ns"],
                }
            )
        return results

    def close(self) -> None:
        if self.conn is None:
            return
        self.conn.close()
        self.conn = None


class SourceCorpusService:
    def __init__(self, engine_or_path: "KnowledgeEngine" | Path | str):
        if hasattr(engine_or_path, "project_path"):
            self.engine = engine_or_path
            self.project_root = Path(engine_or_path.project_path or ".").resolve()
        else:
            self.engine = None
            self.project_root = Path(engine_or_path).resolve()
        self.memory_root = self.project_root / ".oem"
        self._store: _SourceIndexStore | None = None

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def _project_root(self) -> Path:
        return self.project_root

    def _memory_root(self) -> Path:
        return self.memory_root

    def _layout(self):
        if self.engine is not None:
            return self.engine.layout(str(self.project_root))
        from oem_knowledge.project_layout import ProjectLayout
        return ProjectLayout(self.memory_root)

    def _store_for_write(self) -> _SourceIndexStore:
        if self._store is None:
            db_path = self._layout().source_index_db_path
            self._store = _SourceIndexStore(db_path)
        return self._store

    def _manifest_path(self) -> Path:
        return self.memory_root / "source_manifest.json"

    def _config_path(self) -> Path:
        return self.memory_root / "source_index_config.yml"

    def _db_path(self) -> Path:
        return self.memory_root / "indexes" / "source_index.sqlite"

    def _load_config(self) -> dict[str, Any]:
        config = _deep_copy_defaults()
        config_path = self._config_path()
        if not config_path.exists() or yaml is None:
            return config
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to read source config %s: %s", config_path, exc)
            return config
        if not isinstance(loaded, dict):
            return config
        for key in (
            "chunk_lines",
            "chunk_overlap_lines",
            "max_file_size_bytes",
            "max_read_lines",
            "max_read_characters",
        ):
            if key in loaded:
                try:
                    config[key] = int(loaded[key])
                except Exception:
                    pass
        if isinstance(loaded.get("exclude_globs"), list):
            config["exclude_globs"] = [str(item) for item in loaded["exclude_globs"] if str(item).strip()]
        if isinstance(loaded.get("include"), list):
            config["include"] = [str(item) for item in loaded["include"]]
        if isinstance(loaded.get("exclude"), list):
            config["exclude"] = [str(item) for item in loaded["exclude"]]
        return config

    def _ensure_config_written(self, config: dict[str, Any]) -> None:
        config_path = self._config_path()
        if config_path.exists() or yaml is None:
            return
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to read source manifest %s: %s", manifest_path, exc)
            return {}

    def _read_ignore_file(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            logger.warning("Failed to read ignore file %s: %s", path, exc)
            return []

    def _build_ignore_matcher(self, config: dict[str, Any]) -> _IgnoreMatcher:
        project_root = self.project_root
        patterns = []
        patterns.extend(self._read_ignore_file(project_root / ".gitignore"))
        patterns.extend(self._read_ignore_file(project_root / ".oemignore"))
        patterns.extend(config.get("exclude_globs", []))
        return _IgnoreMatcher(patterns)

    def _mandatory_exclusion_reason(self, rel_path: str, is_dir: bool) -> str | None:
        pure = PurePosixPath(rel_path)
        parts = pure.parts
        if any(part in MANDATORY_EXCLUDED_DIRS for part in parts):
            return "mandatory_excluded_directory"
        name = pure.name.lower()
        if is_dir:
            return None
        if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
            return "secret_env_file"
        if name.endswith(".log"):
            return "log_file"
        if pure.suffix.lower() in MANDATORY_EXCLUDED_EXTENSIONS:
            return "unsupported_binary_or_media"
        return None

    def load_config(self) -> SourceIndexConfig:
        config_dict = self._load_config()
        include = config_dict.get("include")
        exclude = config_dict.get("exclude")
        if exclude is None:
            exclude = list(DEFAULT_SOURCE_CONFIG["exclude"])
        else:
            exclude = list(exclude)

        if include is None:
            include = list(DEFAULT_SOURCE_CONFIG["include"])
        else:
            include = list(include)
            old_defaults_1 = {"src/**", "tests/**"}
            old_defaults_2 = {"src/**", "packages/**", "tests/**", "docs/**", "README.md", "AGENTS.md", "pyproject.toml", "package.json"}
            is_old_defaults = (set(include) == old_defaults_1 or set(include) == old_defaults_2) and config_dict.get("max_read_lines") != _OLD_DEFAULT_MAX_READ_LINES
            if is_old_defaults:
                include = list(DEFAULT_SOURCE_CONFIG["include"])
            
        return SourceIndexConfig(
            include=[str(i) for i in include],
            exclude=[str(e) for e in exclude],
            exclude_globs=[str(e) for e in config_dict.get("exclude_globs", DEFAULT_SOURCE_CONFIG["exclude_globs"])],
            max_file_size_bytes=int(config_dict.get("max_file_size_bytes", DEFAULT_SOURCE_CONFIG["max_file_size_bytes"])),
            chunk_lines=int(config_dict.get("chunk_lines", DEFAULT_SOURCE_CONFIG["chunk_lines"])),
            chunk_overlap_lines=int(config_dict.get("chunk_overlap_lines", DEFAULT_SOURCE_CONFIG["chunk_overlap_lines"])),
            max_read_lines=int(config_dict.get("max_read_lines", DEFAULT_SOURCE_CONFIG["max_read_lines"])),
            max_read_characters=int(config_dict.get("max_read_characters", DEFAULT_SOURCE_CONFIG["max_read_characters"])),
        )

    def classify_file(self, path: Path) -> SourceFileClassification:
        project_root = self.project_root.resolve()
        
        # 1. Path Safety Check (traversal, absolute path outside, symlink escaping root)
        try:
            # Check traversal and absolute path outside root by resolving and checking relative
            resolved = path.resolve()
            resolved.relative_to(project_root)
            is_inside = True
        except (ValueError, RuntimeError):
            is_inside = False

        if not is_inside:
            rel_path = str(path)
            try:
                rel_path = _normalize_rel_path(path.relative_to(project_root))
            except Exception:
                pass
            return SourceFileClassification(
                path=path,
                rel_path=rel_path,
                eligible=False,
                reason="outside_project_symlink",
                file_type="unknown",
                language="unknown",
                size_bytes=0
            )

        rel_path = _normalize_rel_path(resolved.relative_to(project_root))
        
        # 2. Mandatory Exclusions check (these always override includes)
        reason = self._mandatory_exclusion_reason(rel_path, is_dir=resolved.is_dir())
        if reason is not None:
            if reason == "mandatory_excluded_directory":
                reason = "mandatory_excluded"
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason=reason,
                file_type="unknown",
                language="unknown",
                size_bytes=resolved.stat().st_size if resolved.is_file() else 0
            )

        name_lower = resolved.name.lower()
        if any(name_lower.endswith(ext) for ext in (".pem", ".key", ".crt", ".p12", ".pfx", ".sqlite", ".db")):
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason="mandatory_excluded",
                file_type="unknown",
                language="unknown",
                size_bytes=resolved.stat().st_size if resolved.is_file() else 0
            )

        if resolved.is_dir():
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason="directory",
                file_type="directory",
                language="unknown",
                size_bytes=0
            )

        # 3. Ignore Matcher check (.gitignore / .oemignore)
        config = self.load_config()
        ignore_matcher = self._build_ignore_matcher({
            "exclude_globs": config.exclude_globs
        })
        if ignore_matcher.matches(rel_path, is_dir=False):
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason="gitignored",
                file_type="unknown",
                language="unknown",
                size_bytes=resolved.stat().st_size
            )

        # 4. Exclude configuration check (before lockfile/large so explicit exclude wins)
        if config.exclude:
            exclude_matcher = _IgnoreMatcher(config.exclude)
            if exclude_matcher.matches(rel_path, is_dir=False):
                return SourceFileClassification(
                    path=resolved,
                    rel_path=rel_path,
                    eligible=False,
                    reason="excluded",
                    file_type="unknown",
                    language="unknown",
                    size_bytes=resolved.stat().st_size
                )

        # 5. Include configuration check (before lockfile/large/binary so files
        # outside the include set never become metadata-only search rows)
        if config.include:
            include_matcher = _IgnoreMatcher(config.include)
            if not include_matcher.matches(rel_path, is_dir=False):
                return SourceFileClassification(
                    path=resolved,
                    rel_path=rel_path,
                    eligible=False,
                    reason="not_included",
                    file_type="unknown",
                    language="unknown",
                    size_bytes=resolved.stat().st_size
                )

        # 6. Lockfile/large file/binary check
        size_bytes = resolved.stat().st_size
        file_type = _guess_file_type(resolved)
        language = _guess_language(resolved)

        if file_type == "lockfile":
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason="lockfile_metadata_only",
                file_type=file_type,
                language=language,
                size_bytes=size_bytes,
                metadata_only=True
            )

        if size_bytes > config.max_file_size_bytes:
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason="large_file",
                file_type=file_type,
                language=language,
                size_bytes=size_bytes,
                metadata_only=True
            )

        if not self._is_text_like(resolved):
            return SourceFileClassification(
                path=resolved,
                rel_path=rel_path,
                eligible=False,
                reason="binary",
                file_type="binary",
                language=language,
                size_bytes=size_bytes
            )

        return SourceFileClassification(
            path=resolved,
            rel_path=rel_path,
            eligible=True,
            reason="eligible",
            file_type=file_type,
            language=language,
            size_bytes=size_bytes
        )

    def discover_files(self) -> SourceDiscoveryResult:
        project_root = self.project_root.resolve()
        config = self.load_config()
        
        discovered_files: list[SourceFileClassification] = []
        excluded_reasons: dict[str, int] = {
            "mandatory_excluded": 0,
            "gitignored": 0,
            "binary": 0,
            "large_file": 0,
            "outside_project_symlink": 0,
            "lockfile_metadata_only": 0,
            "excluded": 0,
            "not_included": 0,
        }
        
        for current_root, dirnames, filenames in os.walk(project_root):
            current_path = Path(current_root)
            
            kept_dirs = []
            for dirname in dirnames:
                dir_path = current_path / dirname
                classification = self.classify_file(dir_path)
                
                if classification.eligible or classification.reason not in (
                    "mandatory_excluded",
                    "gitignored",
                    "outside_project_symlink"
                ):
                    kept_dirs.append(dirname)
                else:
                    reason = classification.reason
                    if reason in excluded_reasons:
                        excluded_reasons[reason] += 1
            dirnames[:] = kept_dirs

            for filename in filenames:
                file_path = current_path / filename
                classification = self.classify_file(file_path)
                discovered_files.append(classification)
                
                if not classification.eligible:
                    reason = classification.reason
                    if reason in excluded_reasons:
                        excluded_reasons[reason] += 1

        files_included = sum(1 for c in discovered_files if c.eligible)
        files_excluded = len(discovered_files) - files_included
        
        warnings = []
        if not self._config_path().exists():
            warnings.append("config_missing_using_defaults")
        
        # Check if execution/ or agent/ is explicitly excluded or skipped by config
        config = self.load_config()
        exclude_matcher = _IgnoreMatcher(config.exclude)
        include_matcher = _IgnoreMatcher(config.include)
        for path_check in ("execution/test.py", "agent/test.py"):
            if not include_matcher.matches(path_check, is_dir=False) or exclude_matcher.matches(path_check, is_dir=False):
                warnings.append(f"implementation_directory_skipped:{path_check.split('/')[0]}")

        # Also check if gitignore/oemignore patterns would block these dirs
        ignore_matcher = self._build_ignore_matcher({
            "exclude_globs": config.exclude_globs
        })
        for path_check in ("execution/test.py", "agent/test.py"):
            if ignore_matcher.matches(path_check, is_dir=False):
                warnings.append(f"implementation_directory_gitignored:{path_check.split('/')[0]}")
            
        return SourceDiscoveryResult(
            project_root=str(project_root),
            files_included=files_included,
            files_excluded=files_excluded,
            excluded_reasons=excluded_reasons,
            warnings=warnings,
            discovered_files=discovered_files
        )

    def _discover_files(self, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        res = self.discover_files()
        discovered = []
        for c in res.discovered_files:
            if c.eligible or c.reason in ("lockfile_metadata_only", "large_file"):
                discovered.append({
                    "absolute_path": Path(c.path),
                    "rel_path": c.rel_path,
                    "size_bytes": c.size_bytes,
                    "mtime_ns": Path(c.path).stat().st_mtime_ns if Path(c.path).exists() else 0,
                    "content_hash": _sha256_file(Path(c.path)) if Path(c.path).is_file() else "",
                    "language": c.language,
                    "file_type": c.file_type,
                    "mode": "metadata_only" if c.reason in ("lockfile_metadata_only", "large_file") else "index",
                    "reason": c.reason,
                })
        
        counters = {
            "excluded": sum(1 for c in res.discovered_files if not c.eligible and c.reason not in ("lockfile_metadata_only", "large_file")),
            "lockfile_metadata_only": res.excluded_reasons.get("lockfile_metadata_only", 0),
            "skipped_large_file": res.excluded_reasons.get("large_file", 0),
        }
        discovered.sort(key=lambda item: item["rel_path"])
        return discovered, counters

    def estimate_tokens(self, text: str) -> int:
        return _estimate_tokens(len(text))

    def _is_text_like(self, path: Path) -> bool:
        try:
            sample = path.read_bytes()[:4096]
        except Exception:
            return False
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def _chunk_file(self, file_info: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        abs_path = file_info["absolute_path"]
        rel_path = file_info["rel_path"]
        content = abs_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        line_count = len(lines)
        metadata = {
            "corpus": "source",
            "memory_ingestion_eligible": False,
            "source_retrieval_eligible": True,
            "source": rel_path,
            "rel_path": rel_path,
            "language": file_info["language"],
            "file_type": file_info["file_type"],
            "indexing_reason": file_info["reason"],
        }

        if file_info["mode"] == "metadata_only":
            chunk = {
                "id": f"{rel_path}#metadata",
                "document": f"Path: {rel_path}\nIndexing mode: metadata_only\nReason: {file_info['reason']}",
                "snippet": f"{rel_path} ({file_info['reason']})",
                "path_text": " ".join(sorted(_path_tokens(rel_path))),
                "symbols_text": " ".join(sorted(_path_tokens(rel_path))),
                "start_line": None,
                "end_line": None,
                "metadata": {**metadata, "indexing_mode": "metadata_only"},
                "content_hash": file_info["content_hash"],
                "mtime_ns": file_info["mtime_ns"],
            }
            record = {
                "rel_path": rel_path,
                "status": "metadata_only",
                "size_bytes": file_info["size_bytes"],
                "line_count": line_count,
                "chunk_count": 1,
                "content_hash": file_info["content_hash"],
                "mtime_ns": file_info["mtime_ns"],
                "metadata": {
                    **metadata,
                    "indexing_mode": "metadata_only",
                    "reason": file_info["reason"],
                    "size_bytes": file_info["size_bytes"],
                    "line_count": line_count,
                },
            }
            return record, [chunk], 0

        chunk_lines = max(20, int(config["chunk_lines"]))
        overlap = min(chunk_lines - 1, max(0, int(config["chunk_overlap_lines"])))
        chunks = []
        char_count = 0
        start_index = 0
        while start_index < line_count or (line_count == 0 and start_index == 0):
            end_index = min(line_count, start_index + chunk_lines)
            chunk_body_lines = lines[start_index:end_index]
            if line_count == 0:
                chunk_body_lines = []
            body = "\n".join(chunk_body_lines)
            start_line = start_index + 1
            end_line = end_index if end_index > 0 else 1
            doc = f"Path: {rel_path}\nLines: {start_line}-{end_line}\n\n{body}"
            snippet = body[:500] or f"{rel_path} (empty file)"
            symbols = _extract_symbols(body or rel_path)
            chunk_meta = {
                **metadata,
                "start_line": start_line,
                "end_line": end_line,
                "indexing_mode": "content",
            }
            chunks.append(
                {
                    "id": f"{rel_path}#L{start_line}-{end_line}",
                    "document": doc,
                    "snippet": snippet,
                    "path_text": " ".join(sorted(_path_tokens(rel_path))),
                    "symbols_text": " ".join(symbols),
                    "start_line": start_line,
                    "end_line": end_line,
                    "metadata": chunk_meta,
                    "content_hash": file_info["content_hash"],
                    "mtime_ns": file_info["mtime_ns"],
                }
            )
            char_count += len(body)
            if line_count == 0 or end_index >= line_count:
                break
            start_index = end_index - overlap

        record = {
            "rel_path": rel_path,
            "status": "indexed",
            "size_bytes": file_info["size_bytes"],
            "line_count": line_count,
            "chunk_count": len(chunks),
            "content_hash": file_info["content_hash"],
            "mtime_ns": file_info["mtime_ns"],
            "metadata": {
                **metadata,
                "indexing_mode": "content",
                "reason": file_info["reason"],
                "size_bytes": file_info["size_bytes"],
                "line_count": line_count,
            },
        }
        return record, chunks, char_count

    def index(self, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
        from oem_knowledge.fs import FileLock
        manifest_lock = FileLock(
            self._manifest_path().with_suffix(".lock"),
            timeout=60.0,
        )
        with manifest_lock:
            project_root = self._project_root()
            effective_config = self.load_config()
            config = {
                "version": DEFAULT_SOURCE_CONFIG["version"],
                "include": list(effective_config.include),
                "exclude": list(effective_config.exclude),
                "chunk_lines": effective_config.chunk_lines,
                "chunk_overlap_lines": effective_config.chunk_overlap_lines,
                "max_file_size_bytes": effective_config.max_file_size_bytes,
                "max_read_lines": effective_config.max_read_lines,
                "max_read_characters": effective_config.max_read_characters,
                "exclude_globs": list(effective_config.exclude_globs),
            }
            previous_manifest = self._load_manifest()
            previous_files = previous_manifest.get("files", {}) if isinstance(previous_manifest, dict) else {}
            previous_config = previous_manifest.get("config", {}) if isinstance(previous_manifest, dict) else {}
            if not force and previous_config and previous_config != config:
                force = True

            if not dry_run and not self._memory_root().exists():
                self.engine.init_project(str(project_root))
            if not dry_run:
                self._ensure_config_written(config)

            discovered, counters = self._discover_files(config)
            discovered_map = {item["rel_path"]: item for item in discovered}
            removed_files = sorted(set(previous_files) - set(discovered_map))

            stats = {
                "status": "success",
                "operation": "knowledge_source_index",
                "mode": "dry_run" if dry_run else "write",
                "scanned_files": len(discovered),
                "indexed_files": 0,
                "metadata_only_files": 0,
                "new_files": 0,
                "updated_files": 0,
                "unchanged_files": 0,
                "removed_files": len(removed_files),
                "skipped_large_files": counters["skipped_large_file"],
                "lockfile_metadata_only_files": counters["lockfile_metadata_only"],
                "excluded_files": counters["excluded"],
                "total_chunks": 0,
                "estimated_source_tokens": 0,
            }
            manifest_files: dict[str, Any] = {}
            total_chars = 0

            store = None if dry_run else self._store_for_write()
            for rel_path, file_info in discovered_map.items():
                record, chunks, indexed_chars = self._chunk_file(file_info, config)
                previous = previous_files.get(rel_path)
                is_unchanged = (
                    not force
                    and isinstance(previous, dict)
                    and previous.get("content_hash") == record["content_hash"]
                    and int(previous.get("mtime_ns", -1)) == int(record["mtime_ns"])
                    and previous.get("status") == record["status"]
                    and int(previous.get("chunk_count", -1)) == int(record["chunk_count"])
                )
                if is_unchanged:
                    stats["unchanged_files"] += 1
                elif previous is None:
                    stats["new_files"] += 1
                else:
                    stats["updated_files"] += 1

                if record["status"] == "indexed":
                    stats["indexed_files"] += 1
                else:
                    stats["metadata_only_files"] += 1

                stats["total_chunks"] += record["chunk_count"]
                total_chars += indexed_chars
                manifest_files[rel_path] = {
                    "status": record["status"],
                    "content_hash": record["content_hash"],
                    "mtime_ns": record["mtime_ns"],
                    "chunk_count": record["chunk_count"],
                    "size_bytes": record["size_bytes"],
                    "line_count": record["line_count"],
                    "metadata": record["metadata"],
                }
                if store is not None and not is_unchanged:
                    store.replace_file(record, chunks)

            if store is not None and removed_files:
                store.remove_files(removed_files)

            stats["estimated_source_tokens"] = _estimate_tokens(total_chars)
            summary = {
                "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "project_root": str(project_root),
                "indexed_files": stats["indexed_files"],
                "metadata_only_files": stats["metadata_only_files"],
                "removed_files": stats["removed_files"],
                "skipped_large_files": stats["skipped_large_files"],
                "lockfile_metadata_only_files": stats["lockfile_metadata_only_files"],
                "excluded_files": stats["excluded_files"],
                "total_chunks": stats["total_chunks"],
                "estimated_source_tokens": stats["estimated_source_tokens"],
                "max_file_size_bytes": int(config["max_file_size_bytes"]),
                "max_read_lines": int(config["max_read_lines"]),
                "max_read_characters": int(config["max_read_characters"]),
            }

            embedding_status = None
            embedding_failure = None
            embedding_manifest = previous_manifest.get("embedding") if isinstance(previous_manifest, dict) else None
            if not dry_run:
                # Embeddings are deliberately a second transaction after text indexing.
                try:
                    mode = self.engine.search.resolve_retrieval_mode() if self.engine is not None else "bm25"
                    if mode == "hybrid":
                        if store is None:
                            raise RuntimeError("source_store_unavailable")
                        model_name = self.engine.resolve_embedding_model()
                        current_chunks = store.iter_chunks()
                        if current_chunks:
                            active = previous_manifest.get("embedding", {}) if isinstance(previous_manifest, dict) else {}
                            active_model = active.get("model")
                            try:
                                active_dimension = int(active.get("dimension") or 0)
                            except (TypeError, ValueError):
                                active_dimension = 0
                            existing_rows = store.conn.execute(
                                "SELECT chunk_id, content_hash, embedding_dimension "
                                "FROM source_embeddings WHERE embedding_model = ?",
                                (model_name,),
                            ).fetchall()
                            existing_dimensions = {
                                (row["chunk_id"], row["content_hash"]): int(row["embedding_dimension"])
                                for row in existing_rows
                            }
                            probe = self.engine.search.embed([current_chunks[0]["document"]])
                            if len(probe) != 1 or not probe[0]:
                                raise ValueError("invalid_embedding_probe")
                            observed_dimension = len(probe[0])
                            if observed_dimension <= 0:
                                raise ValueError("invalid_embedding_dimension")
                            dimension = observed_dimension
                            missing = [
                                chunk
                                for chunk in current_chunks
                                if existing_dimensions.get((chunk["id"], chunk["content_hash"])) != dimension
                            ]
                            if active_model != model_name or active_dimension != dimension:
                                missing = current_chunks
                            if missing:
                                vectors = self.engine.search.embed([chunk["document"] for chunk in missing])
                                if len(vectors) != len(missing) or not vectors:
                                    raise ValueError("invalid_embedding_batch")
                                if any(not vector or len(vector) != dimension for vector in vectors):
                                    raise ValueError("inconsistent_embedding_dimensions")
                                if any(
                                    not math.isfinite(float(value))
                                    for vector in vectors
                                    for value in vector
                                ):
                                    raise ValueError("non_finite_embedding")
                                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                                generation = f"{model_name}:{dimension}"
                                rows_to_write = [
                                    {
                                        "chunk_id": chunk["id"],
                                        "embedding_generation": generation,
                                        "embedding_model": model_name,
                                        "embedding_dimension": dimension,
                                        "content_hash": chunk["content_hash"],
                                        "embedding": json.dumps([float(value) for value in vector]),
                                        "created_at": now,
                                    }
                                    for chunk, vector in zip(missing, vectors)
                                ]
                                store.upsert_embeddings(rows_to_write)
                            generation = f"{model_name}:{dimension}"
                            stored = store.load_embeddings(model_name, dimension)
                            if not all(
                                (chunk["id"], chunk["content_hash"]) in stored
                                for chunk in current_chunks
                            ):
                                raise ValueError("incomplete_embedding_generation")
                            count = store.conn.execute(
                                "SELECT COUNT(*) FROM source_embeddings "
                                "WHERE embedding_generation = ?",
                                (generation,),
                            ).fetchone()[0]
                            embedding_manifest = {
                                "active_generation": generation,
                                "model": model_name,
                                "dimension": dimension,
                                "status": "ready",
                                "embedded_chunks": count,
                            }
                            embedding_status = "ready"
                        else:
                            embedding_status = "no_chunks"
                    else:
                        embedding_status = "bm25"
                except Exception as exc:
                    embedding_status = "bm25_fallback"
                    embedding_failure = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(exc))[:80] or "embedding_unavailable"
                manifest = {
                    "version": "1.0.1",
                    "corpus": "source",
                    "config": config,
                    "summary": summary,
                    "files": manifest_files,
                }
                if embedding_manifest:
                    manifest["embedding"] = embedding_manifest
                manifest_path = self._manifest_path()
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = manifest_path.with_name(manifest_path.name + ".tmp")
                try:
                    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                    os.replace(temporary, manifest_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()

            stats["summary"] = summary
            stats["warnings"] = []
            if embedding_status:
                stats["embedding_status"] = embedding_status
            if embedding_failure:
                stats["embedding_failure"] = embedding_failure
                stats["warnings"].append(f"source_embedding_fallback:{embedding_failure}")
            if dry_run:
                stats["warnings"].append("dry_run_no_files_written")
            return stats

    def _trim_content(self, content: str, max_characters: int) -> tuple[str, bool]:
        if len(content) <= max_characters:
            return content, False
        return content[:max_characters], True

    def _resolve_source_path(self, path: str) -> tuple[Path, str]:
        project_root = self._project_root()
        requested = Path(path)
        candidate = requested if requested.is_absolute() else project_root / requested
        verified = self.engine._sfs(project_root).resolve_path(candidate)
        rel_path = _normalize_rel_path(verified.relative_to(project_root))
        return verified, rel_path

    def read(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
        project_root = self._project_root()
        config = self._load_config()
        max_lines = max(1, int(config["max_read_lines"]))
        max_chars = max(1000, int(config["max_read_characters"]))
        try:
            verified, rel_path = self._resolve_source_path(path)
        except PermissionError:
            return {
                "status": "error",
                "reason": "path_outside_project",
                "message": "Requested source path is outside the project boundary.",
                "suggestion": "Request a path inside the active project root.",
            }
        if not verified.exists() or not verified.is_file():
            return {
                "status": "error",
                "reason": "source_not_found",
                "message": "Requested source path was not found.",
                "suggestion": "Verify the relative path from the project root.",
            }
        try:
            content = verified.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {
                "status": "error",
                "reason": "source_not_text",
                "message": "Requested source path is not readable as UTF-8 text.",
                "suggestion": "Use a text source path inside the indexed corpus.",
            }
        lines = content.splitlines()
        total_lines = len(lines)
        start = max(1, int(start_line or 1))
        requested_end = int(end_line) if end_line is not None else total_lines
        if requested_end < start:
            return {
                "status": "error",
                "reason": "invalid_line_range",
                "message": "end_line must be greater than or equal to start_line.",
                "suggestion": "Request a valid inclusive line range.",
            }
        capped_end = min(total_lines, requested_end, start + max_lines - 1)
        selected = "\n".join(lines[start - 1 : capped_end])
        trimmed, char_truncated = self._trim_content(selected, max_chars)
        line_truncated = capped_end < requested_end or (end_line is None and capped_end < total_lines)
        warnings = []
        reason = None
        if not self._db_path().exists() or not self._manifest_path().exists():
            warnings.append("source_index_missing")
            reason = "source_index_missing"
        manifest = self._load_manifest()
        file_entry = manifest.get("files", {}).get(rel_path) if manifest else None
        current_hash = _sha256_file(verified)
        if file_entry and file_entry.get("content_hash") != current_hash:
            warnings.append("source_index_stale")
            if reason is None:
                reason = "source_index_stale"
        if line_truncated or char_truncated:
            warnings.append("content_truncated")
            reason = "content_truncated"

        returned_end = start - 1 + len(trimmed.splitlines()) if trimmed else start - 1
        status = "warning" if warnings else "success"
        suggestion = None
        if "content_truncated" in warnings:
            suggestion = "Request a narrower line range."
        elif "source_index_missing" in warnings:
            suggestion = "Run `oem source index` to enable searchable source retrieval."

        return {
            "status": status,
            "reason": reason,
            "path": rel_path,
            "project_root": str(project_root),
            "line_range": {
                "start": start,
                "end": returned_end,
                "requested_end": requested_end,
                "total_lines": total_lines,
            },
            "content": trimmed,
            "warnings": warnings,
            "suggestion": suggestion,
        }

    def _bm25_scores(self, query: str, documents: list[str]) -> list[float]:
        # Returns normalized scores in [0.0, 1.0]
        query_terms = [term.lower() for term in re.findall(r"\w+", query) if len(term) > 1]
        if not query_terms or not documents:
            return [0.0] * len(documents)
        tokenized = [re.findall(r"\w+", doc.lower()) for doc in documents]
        doc_count = len(tokenized)
        df: dict[str, int] = {}
        for terms in tokenized:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        avg_dl = sum(len(terms) for terms in tokenized) / max(doc_count, 1)
        scores = []
        for terms in tokenized:
            tf: dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            doc_len = len(terms)
            for query_term in query_terms:
                if query_term not in tf:
                    continue
                freq = tf[query_term]
                n = df.get(query_term, 0)
                idf = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)
                score += idf * (freq * (BM25_K1 + 1)) / (freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avg_dl, 1)))
            scores.append(score)
        max_score = max(scores) if scores else 0.0
        return [score / max_score if max_score > 0 else 0.0 for score in scores]

    def search(self, query: str, k: int = 5) -> dict[str, Any]:
        project_root = self._project_root()
        query_cleaned = query.strip().strip("'\"")
        warnings: list[str] = []
        retrieval_mode = self.engine.search.resolve_retrieval_mode() if self.engine is not None else "bm25"
        dense_vectors: dict[tuple[str, str], list[float]] = {}
        query_vector: list[float] | None = None
        dense_reason = None
        
        # Path safety/traversal check
        if "/" in query_cleaned or "\\" in query_cleaned or ".." in query_cleaned:
            try:
                test_path = Path(project_root / query_cleaned).resolve()
                test_path.relative_to(project_root.resolve())
            except (ValueError, RuntimeError):
                return {
                    "status": "no_relevant_source_results",
                    "results": [],
                    "warnings": [],
                    "project_root": str(project_root),
                }

        if not query.strip():
            return {
                "status": "success",
                "results": [],
                "warnings": [],
                "project_root": str(project_root),
            }
        manifest = self._load_manifest()
        if not self._db_path().exists() or not self._manifest_path().exists():
            return {
                "status": "error",
                "reason": "source_index_missing",
                "message": "Source index is missing.",
                "suggestion": "Run `oem source index` before using source search.",
                "results": [],
            }
        store = _SourceIndexStore(self._db_path())
        try:
            rows = store.iter_chunks()
            if retrieval_mode == "hybrid":
                embedding = manifest.get("embedding", {})
                model_name = self.engine.resolve_embedding_model()
                dimension = int(embedding.get("dimension", 0))
                if (
                    embedding.get("status") != "ready"
                    or embedding.get("model") != model_name
                    or dimension <= 0
                ):
                    dense_reason = "embedding_generation_unavailable"
                else:
                    dense_vectors = store.load_embeddings(model_name, dimension)
                    if not dense_vectors:
                        dense_reason = "embedding_content_stale"
                    else:
                        query_vector = self.engine.search.embed([query])[0]
                        if len(query_vector) != dimension:
                            dense_vectors = {}
                            dense_reason = "embedding_dimension_mismatch"
        except Exception:
            dense_vectors = {}
            dense_reason = "embedding_unavailable"
        finally:
            store.close()
        dense_available = bool(dense_vectors and query_vector)
        if retrieval_mode == "hybrid" and not dense_available:
            warnings.append(f"source_dense_fallback:{dense_reason or 'embedding_unavailable'}")

        # Check if the query is an exact path query
        is_exact_path_query = False
        potential_path = None
        
        try:
            test_path = Path(project_root / query_cleaned).resolve()
            if test_path.is_file() and test_path.relative_to(project_root.resolve()):
                is_exact_path_query = True
                potential_path = str(test_path.relative_to(project_root.resolve()))
        except Exception:
            pass

        if not is_exact_path_query:
            try:
                test_path = Path(query_cleaned).resolve()
                if test_path.is_file() and test_path.relative_to(project_root.resolve()):
                    is_exact_path_query = True
                    potential_path = str(test_path.relative_to(project_root.resolve()))
            except Exception:
                pass

        if is_exact_path_query:
            has_indexed_row = any(row["rel_path"] == potential_path for row in rows)
            if not has_indexed_row:
                return {
                    "status": "not_indexed",
                    "results": [],
                    "warnings": [f"File {potential_path} exists but is not indexed."],
                    "project_root": str(project_root),
                }

        # _bm25_scores returns normalized scores in [0.0, 1.0]
        normalized_base_scores = self._bm25_scores(query, [row["document"] for row in rows])

        intent = _detect_source_query_intent(query)
        identifiers = intent["identifiers"]
        source_intent = intent["source_intent"]
        debug_intent = intent["debug_intent"]
        test_intent = intent["test_intent"]
        doc_intent = intent["doc_intent"]
        config_intent = intent["config_intent"]

        ranked = []
        for idx, row in enumerate(rows):
            rel_path = row["rel_path"]
            document = row["document"]
            normalized_base_score = normalized_base_scores[idx]
            dense_score = 0.0
            if dense_available:
                vector = dense_vectors.get((row["id"], row["content_hash"]))
                if vector is not None:
                    dense_score = min(1.0, max(0.0, self.engine.search.cosine_similarity(query_vector, vector)))
            combined_score = 0.35 * normalized_base_score + 0.65 * dense_score if dense_available else normalized_base_score

            matched_identifiers = _matched_source_identifiers(identifiers, document)
            matched_identifier_count = len(matched_identifiers)
            lexical_component_match = bool(
                _source_component_tokens(query) & _source_component_tokens(document)
            )
            dense_evidence_threshold = (
                SOURCE_DENSE_EVIDENCE_THRESHOLD
                if normalized_base_score > 0.0
                or matched_identifier_count > 0
                or lexical_component_match
                else SOURCE_DENSE_NO_LEXICAL_EVIDENCE_THRESHOLD
            )
            dense_evidence = (
                dense_available
                and dense_score >= dense_evidence_threshold
                and (not identifiers or matched_identifier_count > 0)
            )

            source_type = _classify_source_result(rel_path, document, matched_identifiers)

            rel_lower = f"/{rel_path.lower()}"
            path_matches = 0
            for ident in identifiers:
                if re.search(rf'(?<!\w){re.escape(ident.lower())}(?!\w)', rel_lower):
                    path_matches += 1
            for term in intent.get("domain_terms", []):
                if re.search(rf'(?<!\w){re.escape(term.lower())}(?!\w)', rel_lower):
                    path_matches += 1

            boosts = {}
            reasons = []
            penalties = {}

            # 1. Metadata-Only detection & demotion before normal boosts
            metadata = dict(row["metadata"])
            indexing_mode = metadata.get("indexing_mode")
            is_metadata_only = (indexing_mode == "metadata_only")
            
            exact_metadata_match = False
            if is_metadata_only:
                # check if query exactly matches path or filename
                exact_metadata_match = (
                    query_cleaned.lower() == rel_path.lower() or
                    query_cleaned.lower() == PurePosixPath(rel_path).name.lower()
                )

            # Symbol definition match
            has_symbol_def = False
            for ident in matched_identifiers:
                if _has_symbol_definition(ident, document):
                    has_symbol_def = True
            for word in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query_cleaned):
                if _has_symbol_definition(word, document):
                    has_symbol_def = True

            # Positive evidence gating
            name = PurePosixPath(rel_path).name
            name_no_ext = PurePosixPath(rel_path).stem
            query_lower = query_cleaned.lower()
            exact_path_or_filename_match = (
                query_lower == rel_path.lower() or 
                query_lower == name.lower() or 
                query_lower == name_no_ext.lower() or
                re.search(rf'(?<!\w){re.escape(name_no_ext.lower())}(?!\w)', query_lower) is not None
            )
            
            has_boost_evidence = (
                normalized_base_score > 0.0
                or dense_evidence
                or matched_identifier_count > 0
                or has_symbol_def
                or (is_exact_path_query and rel_path == potential_path)
                or exact_path_or_filename_match
            )

            # Apply boosts if not metadata-only, or if metadata-only is rescued by exact match
            if not is_metadata_only or exact_metadata_match:
                
                # Exact path query boost
                if is_exact_path_query and rel_path == potential_path:
                    boosts["exact_path_query"] = BOOST_EXACT_PATH_QUERY
                    reasons.append("exact path query match")

                # Exact identifiers
                exact_ident_boost = 0.0
                symbol_definition_boost = 0.0
                for ident in matched_identifiers:
                    if "." in ident:
                        exact_ident_boost += BOOST_DOTTED_IDENTIFIER
                        reasons.append(f"dotted identifier: {ident}")
                    elif "_" in ident:
                        exact_ident_boost += BOOST_SNAKE_IDENTIFIER
                        reasons.append(f"exact identifier: {ident}")
                    else:
                        exact_ident_boost += BOOST_CAMEL_IDENTIFIER
                        reasons.append(f"exact identifier: {ident}")
                    
                    if _has_symbol_definition(ident, document):
                        symbol_definition_boost += BOOST_SYMBOL_DEFINITION
                        reasons.append(f"symbol definition: {ident}")
                
                if exact_ident_boost > 0.0:
                    boosts["exact_identifier"] = round(exact_ident_boost, 2)
                if symbol_definition_boost > 0.0:
                    boosts["symbol_definition"] = round(symbol_definition_boost, 2)

                if matched_identifier_count >= 3:
                    boosts["identifier_multi_match"] = BOOST_IDENTIFIER_MULTI_MATCH_3_PLUS
                    reasons.append("matched identifiers (3+)")
                elif matched_identifier_count == 2:
                    boosts["identifier_multi_match"] = BOOST_IDENTIFIER_MULTI_MATCH_2
                    reasons.append("matched identifiers (2)")
                elif matched_identifier_count == 1:
                    boosts["identifier_multi_match"] = BOOST_IDENTIFIER_MULTI_MATCH_1
                    reasons.append("matched identifiers (1)")

                if path_matches > 0:
                    boosts["path_match"] = round(BOOST_PATH_MATCH_MULTIPLIER * path_matches, 2)
                    reasons.append(f"path match ({path_matches})")

                if source_intent or debug_intent or test_intent:
                    if has_boost_evidence:
                        if source_type in ("adapter_code", "service_code", "client_code"):
                            boosts[source_type] = BOOST_CODE_QUERY_ADAPTER
                            reasons.append(f"{source_type.replace('_code', '')} path match")
                            reasons.append("implementation code file")
                        elif source_type == "implementation_code":
                            boosts["implementation_code"] = BOOST_CODE_QUERY_IMPLEMENTATION
                            reasons.append("implementation code file")
                        elif source_type == "relevant_test":
                            if test_intent:
                                boosts["relevant_test"] = BOOST_CODE_QUERY_RELEVANT_TEST_ACTIVE
                                reasons.append("relevant test")
                            else:
                                boosts["relevant_test"] = BOOST_CODE_QUERY_RELEVANT_TEST_INACTIVE
                                reasons.append("relevant test")

                if doc_intent:
                    if source_type == "agent_instruction":
                        boosts["agent_instruction"] = BOOST_DOC_QUERY_AGENT
                        reasons.append("agent instruction query boost")
                    elif source_type in ("readme_doc", "project_doc"):
                        boosts[source_type] = BOOST_DOC_QUERY_DOC
                        reasons.append("doc query boost")

            # Penalties
            if is_metadata_only and not exact_metadata_match:
                penalties["metadata_only"] = PENALTY_METADATA_ONLY
                reasons.append("metadata_only result")
                reasons.append("implementation boost suppressed: no content evidence")
            else:
                if source_type == "agent_instruction" and not doc_intent:
                    penalties["agent_instruction"] = PENALTY_AGENT_INSTRUCTION
                    reasons.append("agent_instruction penalty")
                    reasons.append("broad doc / non-implementation penalty")
                elif source_type == "agent_instruction" and doc_intent:
                    pass
                elif source_type in ("readme_doc", "project_doc") and not doc_intent:
                    penalties[source_type] = PENALTY_BROAD_DOC
                    reasons.append("broad doc penalty")
                
                if source_type == "unrelated_test":
                    if not test_intent:
                        penalties["unrelated_test"] = PENALTY_UNRELATED_TEST_INACTIVE
                        reasons.append("unrelated test penalty")
                    else:
                        penalties["unrelated_test"] = PENALTY_UNRELATED_TEST_ACTIVE
                        reasons.append("unrelated test penalty")

                if source_type == "config_file" and not config_intent:
                    penalties["config_file"] = PENALTY_CONFIG_FILE
                    reasons.append("config file penalty")

                if source_type == "generated_or_cache":
                    penalties["generated_or_cache"] = PENALTY_GENERATED_OR_CACHE
                    reasons.append("generated/cache penalty")

                _CODE_SOURCE_TYPES = {"implementation_code", "adapter_code", "service_code", "client_code", "relevant_test"}
                if len(document) > LARGE_DOCUMENT_THRESHOLD_CHARS and matched_identifier_count == 0 and source_type not in _CODE_SOURCE_TYPES:
                    penalties["large_low_density"] = PENALTY_LARGE_LOW_DENSITY
                    reasons.append("large low density penalty")

            total_boost = sum(boosts.values())
            total_penalty = sum(penalties.values())
            final_score = round(combined_score + total_boost - total_penalty, 4)

            metadata["source_type"] = source_type
            metadata["source_diagnostics"] = {
                "source_type": source_type,
                "base_score": round(normalized_base_score, 4),
                "retrieval_mode": retrieval_mode,
                "dense_available": dense_available,
                "bm25_score": round(normalized_base_score, 4),
                "dense_score": round(dense_score, 4),
                "dense_evidence": dense_evidence,
                "dense_evidence_threshold": dense_evidence_threshold,
                "lexical_component_match": lexical_component_match,
                "final_score": final_score,
                "ranking_reason": reasons,
                "ranking_boosts": boosts,
                "ranking_penalties": penalties,
                "exact_identifier_count": matched_identifier_count,
                "exact_path_query": (is_exact_path_query and rel_path == potential_path),
                "query_intent": {
                    "source_intent": source_intent,
                    "debug_intent": debug_intent,
                    "test_intent": test_intent,
                }
            }
            if dense_reason:
                metadata["source_diagnostics"]["dense_fallback_reason"] = dense_reason
            metadata["rel_path"] = rel_path
            metadata["start_line"] = row["start_line"]
            metadata["end_line"] = row["end_line"]

            file_entry = manifest.get("files", {}).get(rel_path, {}) if manifest else {}
            metadata["status"] = file_entry.get("status", metadata.get("indexing_mode", "indexed"))

            ranked.append({
                "id": row["id"],
                "document": row["snippet"],
                "metadata": metadata,
                "score": final_score,
            })

        TYPE_PRIORITY = {
            "adapter_code": 10,
            "service_code": 10,
            "client_code": 10,
            "implementation_code": 9,
            "relevant_test": 8,
            "config_file": 7,
            "project_doc": 6,
            "readme_doc": 5,
            "agent_instruction": 4,
            "unrelated_test": 3,
            "generated_or_cache": 2,
            "unknown": 1,
        }

        from functools import cmp_to_key

        def compare_results(a, b):
            score_diff = a["score"] - b["score"]
            if abs(score_diff) > SCORE_TIE_EPSILON:
                return 1 if score_diff > 0 else -1
                
            p_a = TYPE_PRIORITY.get(a["metadata"]["source_type"], 0)
            p_b = TYPE_PRIORITY.get(b["metadata"]["source_type"], 0)
            if p_a != p_b:
                return 1 if p_a > p_b else -1
                
            c_a = a["metadata"]["source_diagnostics"].get("exact_identifier_count", 0)
            c_b = b["metadata"]["source_diagnostics"].get("exact_identifier_count", 0)
            if c_a != c_b:
                return 1 if c_a > c_b else -1
                
            path_a = a["metadata"]["rel_path"]
            path_b = b["metadata"]["rel_path"]
            if path_a != path_b:
                return 1 if path_a < path_b else -1
                
            id_a = str(a["id"])
            id_b = str(b["id"])
            if id_a != id_b:
                return 1 if id_a < id_b else -1
                
            return 0

        ranked.sort(key=cmp_to_key(compare_results), reverse=True)

        filtered_ranked = []
        for r in ranked:
            score = r["score"]
            diag = r["metadata"]["source_diagnostics"]
            source_type = r["metadata"]["source_type"]
            
            # Check if it has positive evidence
            has_positive_evidence = (
                diag["exact_identifier_count"] > 0
                or any("symbol definition" in reason for reason in diag["ranking_reason"])
                or any("path match" in reason for reason in diag["ranking_reason"])
                or diag.get("exact_path_query", False)
                or diag.get("dense_evidence", False)
            )

            # Exclude zero or negative scores
            if score <= MIN_POSITIVE_RESULT_SCORE:
                continue

            # Skip demotion filtering for legitimate doc/config queries
            is_legit_doc_config_query = (
                (doc_intent and source_type in ("agent_instruction", "readme_doc", "project_doc"))
                or (config_intent and source_type == "config_file")
            )

            # For very weak topic-only results:
            if not has_positive_evidence and not is_legit_doc_config_query:
                # Exclude if score is low, unless it's a single-word query
                if score < WEAK_RESULT_MIN_SCORE and len(query_cleaned.split()) > 1:
                    continue

            filtered_ranked.append(r)

        if not filtered_ranked:
            return {
                "status": "no_relevant_source_results",
                "results": [],
                "warnings": warnings,
                "project_root": str(project_root),
            }

        return {
            "status": "success",
            "results": filtered_ranked[: max(1, int(k))],
            "warnings": warnings,
            "project_root": str(project_root),
        }

    def stats(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        if not manifest:
            return {
                "status": "warning",
                "reason": "source_index_missing",
                "warnings": ["source_index_missing"],
                "summary": {
                    "estimated_source_tokens": 0,
                    "estimated_baseline_tokens": 0,
                    "estimated_retrieval_tokens": 0,
                    "estimated_savings": 0,
                    "indexed_files": 0,
                    "metadata_only_files": 0,
                    "skipped_large_files": 0,
                    "total_chunks": 0,
                },
            }
        summary = dict(manifest.get("summary", {}))
        baseline = int(summary.get("estimated_source_tokens", 0))
        total_chunks = int(summary.get("total_chunks", 0))
        retrieval = min(baseline, max(1, total_chunks) * 250) if baseline else 0
        summary["estimated_baseline_tokens"] = baseline
        summary["estimated_retrieval_tokens"] = retrieval
        summary["estimated_savings"] = max(0, baseline - retrieval)
        db_size_bytes = 0
        db_path = self._db_path()
        if db_path.exists():
            db_size_bytes = int(db_path.stat().st_size)
        summary["db_size_mb"] = round(db_size_bytes / (1024 * 1024), 4)
        return {
            "status": "success",
            "summary": summary,
            "warnings": [],
        }
