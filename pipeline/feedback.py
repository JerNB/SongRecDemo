"""FeedbackStore -- P3 exposure + interaction logging.

A local SQLite log of *what was recommended* and *what the user did with
it*. This is the raw material a learned ranker would later train on, so it
is captured now -- without changing the P0 ranking formula.

Three event layers
------------------
* ``recommendation_request`` -- one row per ``/api/recommend`` call: the
  request controls, profile summary, funnel counts, latency and the
  algorithm version stamps.
* ``recommendation_item``    -- one row per returned card: its rank, the P0
  score breakdown, source types, pick type, and ``was_impressed`` (True
  because returning it to the client counts as an exposure today; a future
  viewport tracker can refine this).
* ``user_feedback``          -- one row per explicit UI interaction
  (click / like / dislike / ...). Naming is deliberately honest: there is
  no real listen-completion signal yet.

Design rules
------------
* Stdlib ``sqlite3`` only, thread-safe (``check_same_thread=False`` + a
  lock), ``":memory:"`` supported for hermetic tests.
* Fire-and-forget: every public method swallows its own errors and returns
  a falsy / no-op result rather than raising, so logging can never break the
  recommendation path.
* ``log_user_feedback`` is fail-soft on an unknown ``request_id`` -- the row
  is still written (the frontend may post feedback for a request the server
  has since forgotten); only the ``event_type`` whitelist is enforced, and
  that enforcement is surfaced to the caller so the API can return 400.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendation_request (
    request_id              TEXT PRIMARY KEY,
    timestamp               REAL NOT NULL,
    profile_summary_json    TEXT NOT NULL DEFAULT '{}',
    liked_song_ids_json     TEXT NOT NULL DEFAULT '[]',
    liked_artists_json      TEXT NOT NULL DEFAULT '[]',
    genres_json             TEXT NOT NULL DEFAULT '[]',
    moods_json              TEXT NOT NULL DEFAULT '[]',
    tags_json               TEXT NOT NULL DEFAULT '[]',
    excluded_song_ids_json  TEXT NOT NULL DEFAULT '[]',
    content_weight          REAL,
    novelty                 REAL,
    diversity               REAL,
    k                       INTEGER,
    num_raw_candidates      INTEGER,
    num_filtered_candidates INTEGER,
    num_enriched_candidates INTEGER,
    num_final_candidates    INTEGER,
    embedding_recall_enabled INTEGER,
    num_embedding_candidates INTEGER,
    latency_ms              REAL,
    model_version           TEXT,
    ranking_config_version  TEXT
);

CREATE TABLE IF NOT EXISTS recommendation_item (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id              TEXT NOT NULL,
    timestamp               REAL NOT NULL,
    song_id                 INTEGER NOT NULL,
    rank_position           INTEGER,
    final_score             REAL,
    rank_score              REAL,
    content_score           REAL,
    retrieval_score         REAL,
    multi_source_agreement  REAL,
    quality_score           REAL,
    novelty_score           REAL,
    source_types_json       TEXT NOT NULL DEFAULT '[]',
    reasons_json            TEXT NOT NULL DEFAULT '[]',
    pick_type               TEXT,
    was_impressed           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_item_request ON recommendation_item(request_id);

CREATE TABLE IF NOT EXISTS user_feedback (
    event_id                TEXT PRIMARY KEY,
    request_id              TEXT,
    timestamp               REAL NOT NULL,
    song_id                 INTEGER,
    rank_position           INTEGER,
    event_type              TEXT NOT NULL,
    event_value             REAL,
    extra_json              TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_feedback_request ON user_feedback(request_id);
CREATE INDEX IF NOT EXISTS idx_feedback_song ON user_feedback(song_id);
"""


class FeedbackEventError(ValueError):
    """Raised for an invalid feedback payload (e.g. unknown event_type)."""


class FeedbackStore:
    """SQLite-backed exposure + interaction log."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        allowed_event_types: Optional[frozenset] = None,
    ) -> None:
        if path is None:
            import config
            path = config.FEEDBACK_STORE_PATH
        if allowed_event_types is None:
            import config
            allowed_event_types = config.FEEDBACK_EVENT_TYPES
        self._path = path
        self._allowed_event_types = frozenset(allowed_event_types)
        self._lock = threading.Lock()

        # --- P4: minimal write observability. -------------------------
        # Logging stays fire-and-forget (failures never break recall), but
        # these counters let a developer confirm the log is actually being
        # written. They are bumped per successful / failed write op.
        self._write_success_count = 0
        self._write_failure_count = 0
        self._last_error: Optional[str] = None

        if str(path) == ":memory:":
            db_target: Any = ":memory:"
        else:
            p = Path(path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback_store: could not create dir for %s: %s", p, exc)
            db_target = str(p)

        self._conn = sqlite3.connect(db_target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def allowed_event_types(self) -> frozenset:
        return self._allowed_event_types

    # ------------------------------------------------------------------
    # P4: write health diagnostics
    # ------------------------------------------------------------------

    def _record_write_success(self) -> None:
        with self._lock:
            self._write_success_count += 1

    def _record_write_failure(self, exc: Exception) -> None:
        with self._lock:
            self._write_failure_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"

    @property
    def write_success_count(self) -> int:
        with self._lock:
            return int(self._write_success_count)

    @property
    def write_failure_count(self) -> int:
        with self._lock:
            return int(self._write_failure_count)

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_health(self) -> dict[str, Any]:
        """Snapshot of write health so a developer can tell whether feedback
        is actually being persisted. ``healthy`` is True until the first
        write failure is observed."""
        with self._lock:
            return {
                "healthy": self._write_failure_count == 0,
                "write_success_count": int(self._write_success_count),
                "write_failure_count": int(self._write_failure_count),
                "last_error": self._last_error,
            }

    def count(self, table: str) -> int:
        if table not in {"recommendation_request", "recommendation_item", "user_feedback"}:
            return 0
        with self._lock:
            try:
                row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608 (table is whitelisted)
                return int(row["n"]) if row else 0
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback_store.count(%s) failed: %s", table, exc)
                return 0

    # ------------------------------------------------------------------
    # Writes -- exposure
    # ------------------------------------------------------------------

    def log_request(self, payload: dict[str, Any]) -> bool:
        """Insert (or replace) one recommendation_request row."""
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO recommendation_request (
                        request_id, timestamp, profile_summary_json,
                        liked_song_ids_json, liked_artists_json, genres_json,
                        moods_json, tags_json, excluded_song_ids_json,
                        content_weight, novelty, diversity, k,
                        num_raw_candidates, num_filtered_candidates,
                        num_enriched_candidates, num_final_candidates,
                        embedding_recall_enabled, num_embedding_candidates,
                        latency_ms, model_version, ranking_config_version
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(payload.get("request_id") or ""),
                        float(payload.get("timestamp") or time.time()),
                        _dumps(payload.get("profile_summary")),
                        _dumps(payload.get("liked_song_ids")),
                        _dumps(payload.get("liked_artists")),
                        _dumps(payload.get("genres")),
                        _dumps(payload.get("moods")),
                        _dumps(payload.get("tags")),
                        _dumps(payload.get("excluded_song_ids")),
                        _f(payload.get("content_weight")),
                        _f(payload.get("novelty")),
                        _f(payload.get("diversity")),
                        _i(payload.get("k")),
                        _i(payload.get("num_raw_candidates")),
                        _i(payload.get("num_filtered_candidates")),
                        _i(payload.get("num_enriched_candidates")),
                        _i(payload.get("num_final_candidates")),
                        1 if payload.get("embedding_recall_enabled") else 0,
                        _i(payload.get("num_embedding_candidates")),
                        _f(payload.get("latency_ms")),
                        str(payload.get("model_version") or ""),
                        str(payload.get("ranking_config_version") or ""),
                    ),
                )
                self._conn.commit()
            self._record_write_success()
            return True
        except Exception as exc:  # noqa: BLE001 -- logging must never break recall
            self._record_write_failure(exc)
            log.warning("feedback_store.log_request failed: %s", exc)
            return False

    def log_items(self, request_id: str, items: list[dict[str, Any]]) -> int:
        """Insert one recommendation_item row per card. Returns rows written."""
        if not items:
            return 0
        now = time.time()
        rows = []
        for it in items:
            rows.append((
                str(request_id or ""),
                float(it.get("timestamp") or now),
                _i(it.get("song_id")) or 0,
                _i(it.get("rank_position")),
                _f(it.get("final_score")),
                _f(it.get("rank_score")),
                _f(it.get("content_score")),
                _f(it.get("retrieval_score")),
                _f(it.get("multi_source_agreement")),
                _f(it.get("quality_score")),
                _f(it.get("novelty_score")),
                _dumps(it.get("source_types")),
                _dumps(it.get("reasons")),
                str(it.get("pick_type") or ""),
                1 if it.get("was_impressed", True) else 0,
            ))
        try:
            with self._lock:
                self._conn.executemany(
                    """
                    INSERT INTO recommendation_item (
                        request_id, timestamp, song_id, rank_position,
                        final_score, rank_score, content_score, retrieval_score,
                        multi_source_agreement, quality_score, novelty_score,
                        source_types_json, reasons_json, pick_type, was_impressed
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    rows,
                )
                self._conn.commit()
            self._record_write_success()
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            self._record_write_failure(exc)
            log.warning("feedback_store.log_items failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Writes -- user feedback
    # ------------------------------------------------------------------

    def log_user_feedback(
        self,
        *,
        event_type: str,
        request_id: Optional[str] = None,
        song_id: Optional[int] = None,
        rank_position: Optional[int] = None,
        event_value: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        """Record one user feedback event. Returns the generated event_id.

        Raises :class:`FeedbackEventError` if ``event_type`` is not in the
        whitelist (so the API can answer 400). An unknown ``request_id`` is
        accepted (fail-soft) -- the row is still written.
        """
        et = str(event_type or "").strip().lower()
        if et not in self._allowed_event_types:
            raise FeedbackEventError(
                f"unknown event_type {event_type!r}; "
                f"allowed: {sorted(self._allowed_event_types)}"
            )
        event_id = str(uuid.uuid4())
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO user_feedback (
                        event_id, request_id, timestamp, song_id,
                        rank_position, event_type, event_value, extra_json
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        event_id,
                        (str(request_id) if request_id else None),
                        time.time(),
                        _i(song_id),
                        _i(rank_position),
                        et,
                        _f(event_value),
                        _dumps(extra or {}),
                    ),
                )
                self._conn.commit()
            self._record_write_success()
        except FeedbackEventError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._record_write_failure(exc)
            log.warning("feedback_store.log_user_feedback failed: %s", exc)
        return event_id

    # ------------------------------------------------------------------
    # Reads (handy for tests / future eval joins)
    # ------------------------------------------------------------------

    def get_request(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM recommendation_request WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_items(self, request_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM recommendation_item WHERE request_id = ? ORDER BY rank_position",
                (str(request_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_feedback(self, request_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM user_feedback WHERE request_id = ? ORDER BY timestamp",
                (str(request_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Bulk reads (training data builder)
    # ------------------------------------------------------------------

    def get_all_requests(self) -> dict[str, dict[str, Any]]:
        """All recommendation_request rows keyed by request_id."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM recommendation_request"
                ).fetchall()
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback_store.get_all_requests failed: %s", exc)
                return {}
        return {str(r["request_id"]): dict(r) for r in rows}

    def get_all_items(self) -> list[dict[str, Any]]:
        """All recommendation_item rows (one per exposed card)."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM recommendation_item ORDER BY request_id, rank_position"
                ).fetchall()
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback_store.get_all_items failed: %s", exc)
                return []
        return [dict(r) for r in rows]

    def get_all_feedback(self) -> list[dict[str, Any]]:
        """All user_feedback rows."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM user_feedback ORDER BY timestamp"
                ).fetchall()
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback_store.get_all_feedback failed: %s", exc)
                return []
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dumps(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else [], ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"


def _i(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
