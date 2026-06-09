"""SongFeatureStore -- P2 local catalogue of every NetEase song seen.

This is the durable memory that lets the recommender stop depending
entirely on live NetEase ``/search``. Every song that flows through the
pipeline (from search recall or deep enrichment) is ``upsert``-ed here
with a normalised feature row and a *stable* content text
(``text_for_embedding``). The embedding recall channel
(:mod:`embedding_retrieval`) reads this store to find songs that are
semantically close to the user's profile.

Design rules
------------
* One row per ``song_id`` -- repeated sightings update ``last_seen_at`` and
  bump ``source_seen_count``; they never insert duplicates.
* ``text_for_embedding`` is built only from *stable* song content: title,
  artist names, album, any known tags, the retrieval *query texts* that
  found the song, and the *source types* that found it. The user's live
  profile text is deliberately NOT persisted into a song's text (that would
  let one session's query pollute the shared catalogue). The one exception
  is the retrieval query of a non-embedding SourceHit, which legitimately
  describes how the song was found.
* The store is dependency-free (stdlib ``sqlite3`` only) and safe to share
  across threads (``check_same_thread=False`` + a lock); it never raises into
  the request path -- on any error it degrades to a no-op.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from . import scoring
from .text import _maybe_float, _maybe_int
from .types import Candidate

log = logging.getLogger(__name__)


# Source types that should never be persisted as a song's "how it was found"
# query text, because their query string is the live user profile rather than
# a stable description of the song.
_NON_PERSISTED_SOURCE_TYPES = frozenset({"embedding"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    song_id                 INTEGER PRIMARY KEY,
    title                   TEXT    NOT NULL DEFAULT '',
    artists                 TEXT    NOT NULL DEFAULT '[]',
    artist_names            TEXT    NOT NULL DEFAULT '',
    album                   TEXT    NOT NULL DEFAULT '',
    cover_url               TEXT    NOT NULL DEFAULT '',
    netease_url             TEXT    NOT NULL DEFAULT '',
    duration_ms             INTEGER,
    playable                INTEGER,
    comment_count           INTEGER,
    hot_comment_count       INTEGER,
    liked_count             INTEGER,
    artist_followers        INTEGER,
    audio_quality           REAL,
    metadata_quality_score  REAL    NOT NULL DEFAULT 0.0,
    popularity_score        REAL    NOT NULL DEFAULT 0.0,
    artist_authority_score  REAL    NOT NULL DEFAULT 0.0,
    source_seen_count       INTEGER NOT NULL DEFAULT 0,
    first_seen_at           REAL    NOT NULL DEFAULT 0.0,
    last_seen_at            REAL    NOT NULL DEFAULT 0.0,
    tags                    TEXT    NOT NULL DEFAULT '[]',
    query_texts             TEXT    NOT NULL DEFAULT '[]',
    source_types            TEXT    NOT NULL DEFAULT '[]',
    raw_json                TEXT    NOT NULL DEFAULT '{}',
    text_for_embedding      TEXT    NOT NULL DEFAULT ''
);
"""


@dataclass
class SongFeatureRecord:
    """A normalised row in the song feature store."""

    song_id: int
    title: str = ""
    artists: list[str] = field(default_factory=list)
    artist_names: str = ""
    album: str = ""
    cover_url: str = ""
    netease_url: str = ""
    duration_ms: Optional[int] = None
    playable: Optional[bool] = None
    comment_count: Optional[int] = None
    hot_comment_count: Optional[int] = None
    liked_count: Optional[int] = None
    artist_followers: Optional[int] = None
    audio_quality: Optional[float] = None
    metadata_quality_score: float = 0.0
    popularity_score: float = 0.0
    artist_authority_score: float = 0.0
    source_seen_count: int = 0
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    tags: list[str] = field(default_factory=list)
    query_texts: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)
    text_for_embedding: str = ""

    def to_track_dict(self) -> dict[str, Any]:
        """Shape compatible with NetEase search hits / TrackRef inputs."""
        return {
            "netease_song_id": int(self.song_id),
            "title": self.title,
            "artist": self.artists[0] if self.artists else "",
            "artists": list(self.artists),
            "album": self.album,
            "cover_url": self.cover_url,
            "duration_ms": self.duration_ms,
        }


class SongFeatureStore:
    """SQLite-backed catalogue of seen songs + their stable content text."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        """Open (or create) the store.

        ``path`` may be ``":memory:"`` for a hermetic in-memory store (tests),
        a filesystem path, or ``None`` to fall back to
        ``config.FEATURE_STORE_PATH``.
        """
        if path is None:
            import config
            path = config.FEATURE_STORE_PATH
        self._path = path
        self._lock = threading.Lock()

        if str(path) == ":memory:":
            db_target: Any = ":memory:"
        else:
            p = Path(path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("feature_store: could not create dir for %s: %s", p, exc)
            db_target = str(p)

        self._conn = sqlite3.connect(db_target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            try:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM songs").fetchone()
                return int(row["n"]) if row else 0
            except Exception as exc:  # noqa: BLE001
                log.warning("feature_store.count failed: %s", exc)
                return 0

    def get_song(self, song_id: int) -> Optional[SongFeatureRecord]:
        try:
            sid = int(song_id)
        except (TypeError, ValueError):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM songs WHERE song_id = ?", (sid,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def all_songs(self) -> list[SongFeatureRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM songs").fetchall()
        return [self._row_to_record(r) for r in rows]

    def iter_text_corpus(self) -> tuple[list[int], list[str]]:
        """Return ``(ids, texts)`` aligned for fitting an embedding index.

        Songs whose content text is empty are skipped -- they carry no
        semantic signal and would only add noise to the index.
        """
        ids: list[int] = []
        texts: list[str] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT song_id, text_for_embedding FROM songs"
            ).fetchall()
        for r in rows:
            txt = str(r["text_for_embedding"] or "").strip()
            if not txt:
                continue
            ids.append(int(r["song_id"]))
            texts.append(txt)
        return ids, texts

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_candidates(self, candidates: Iterable[Candidate]) -> int:
        """Upsert every candidate. Returns the number of rows written.

        Each candidate either inserts a new row (first_seen) or updates an
        existing one (bumping ``source_seen_count`` and ``last_seen_at`` and
        merging in any newly-learned query texts / source types / platform
        signals). Same ``song_id`` never duplicates.
        """
        written = 0
        now = time.time()
        for cand in candidates:
            try:
                if self._upsert_one(cand, now):
                    written += 1
            except Exception as exc:  # noqa: BLE001 -- never break the request path
                log.warning("feature_store.upsert failed for a candidate: %s", exc)
        return written

    def _upsert_one(self, cand: Candidate, now: float) -> bool:
        sid = _maybe_int(getattr(cand.track, "netease_song_id", None))
        if not sid:
            return False

        track = cand.track
        artists = list(track.artists) if track.artists else (
            [track.artist] if track.artist else []
        )
        artists = [a for a in artists if a]
        artist_names = ", ".join(artists) if artists else (track.artist or "")

        # Stable "how it was found" signals -- exclude the embedding channel
        # so the user's live profile text never leaks into the song's text.
        new_query_texts = [
            h.query for h in cand.source_hits
            if h.query and h.source_type not in _NON_PERSISTED_SOURCE_TYPES
        ]
        new_source_types = [
            h.source_type for h in cand.source_hits if h.source_type
        ]

        enr = cand.enrichment
        meta_q = scoring.metadata_quality(track)
        pop = scoring.popularity_score(enr)
        authority = scoring.artist_authority_score(enr)

        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM songs WHERE song_id = ?", (sid,)
            ).fetchone()

            if existing is None:
                tags: list[str] = []
                query_texts = _dedupe_keep_order(new_query_texts)
                source_types = _dedupe_keep_order(new_source_types)
                first_seen = now
                source_seen_count = 1
                duration_ms = _maybe_int(getattr(track, "duration_ms", None))
            else:
                tags = _json_list(existing["tags"])
                query_texts = _dedupe_keep_order(
                    _json_list(existing["query_texts"]) + new_query_texts
                )
                source_types = _dedupe_keep_order(
                    _json_list(existing["source_types"]) + new_source_types
                )
                first_seen = float(existing["first_seen_at"] or now)
                source_seen_count = int(existing["source_seen_count"] or 0) + 1
                duration_ms = _maybe_int(getattr(track, "duration_ms", None))
                if duration_ms is None:
                    duration_ms = _maybe_int(existing["duration_ms"])

            # Platform signals: prefer freshly enriched values, else keep
            # whatever the row already had.
            def _pick_int(new_val: Any, col: str) -> Optional[int]:
                v = _maybe_int(new_val)
                if v is not None:
                    return v
                return _maybe_int(existing[col]) if existing is not None else None

            def _pick_float(new_val: Any, col: str) -> Optional[float]:
                v = _maybe_float(new_val)
                if v is not None:
                    return v
                return _maybe_float(existing[col]) if existing is not None else None

            comment_count = _pick_int(enr.comment_count if enr else None, "comment_count")
            hot_comment_count = _pick_int(enr.hot_comment_count if enr else None, "hot_comment_count")
            liked_count = _pick_int(enr.song_red_count if enr else None, "liked_count")
            artist_followers = _pick_int(enr.artist_follow_count if enr else None, "artist_followers")
            audio_quality = _pick_float(enr.audio_quality if enr else None, "audio_quality")

            if enr is not None and enr.playable is not None:
                playable: Optional[int] = 1 if enr.playable else 0
            elif existing is not None and existing["playable"] is not None:
                playable = int(existing["playable"])
            else:
                playable = None

            netease_url = track.netease_url or (
                f"https://music.163.com/#/song?id={sid}"
            )

            text_for_embedding = _build_text_for_embedding(
                title=track.title,
                artist_names=artist_names,
                album=track.album,
                tags=tags,
                query_texts=query_texts,
                source_types=source_types,
            )

            raw_json = {
                "song_id": sid,
                "title": track.title,
                "artists": artists,
                "album": track.album,
                "cover_url": track.cover_url,
                "enrichment": enr.to_cache_payload() if enr is not None else None,
            }

            self._conn.execute(
                """
                INSERT INTO songs (
                    song_id, title, artists, artist_names, album, cover_url,
                    netease_url, duration_ms, playable, comment_count,
                    hot_comment_count, liked_count, artist_followers,
                    audio_quality, metadata_quality_score, popularity_score,
                    artist_authority_score, source_seen_count, first_seen_at,
                    last_seen_at, tags, query_texts, source_types, raw_json,
                    text_for_embedding
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(song_id) DO UPDATE SET
                    title=excluded.title,
                    artists=excluded.artists,
                    artist_names=excluded.artist_names,
                    album=excluded.album,
                    cover_url=excluded.cover_url,
                    netease_url=excluded.netease_url,
                    duration_ms=excluded.duration_ms,
                    playable=excluded.playable,
                    comment_count=excluded.comment_count,
                    hot_comment_count=excluded.hot_comment_count,
                    liked_count=excluded.liked_count,
                    artist_followers=excluded.artist_followers,
                    audio_quality=excluded.audio_quality,
                    metadata_quality_score=excluded.metadata_quality_score,
                    popularity_score=excluded.popularity_score,
                    artist_authority_score=excluded.artist_authority_score,
                    source_seen_count=excluded.source_seen_count,
                    last_seen_at=excluded.last_seen_at,
                    tags=excluded.tags,
                    query_texts=excluded.query_texts,
                    source_types=excluded.source_types,
                    raw_json=excluded.raw_json,
                    text_for_embedding=excluded.text_for_embedding
                """,
                (
                    sid, track.title, json.dumps(artists, ensure_ascii=False),
                    artist_names, track.album, track.cover_url, netease_url,
                    duration_ms, playable, comment_count, hot_comment_count,
                    liked_count, artist_followers, audio_quality,
                    float(meta_q), float(pop), float(authority),
                    int(source_seen_count), float(first_seen), float(now),
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(query_texts, ensure_ascii=False),
                    json.dumps(source_types, ensure_ascii=False),
                    json.dumps(raw_json, ensure_ascii=False),
                    text_for_embedding,
                ),
            )
            self._conn.commit()
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> SongFeatureRecord:
        return SongFeatureRecord(
            song_id=int(row["song_id"]),
            title=str(row["title"] or ""),
            artists=_json_list(row["artists"]),
            artist_names=str(row["artist_names"] or ""),
            album=str(row["album"] or ""),
            cover_url=str(row["cover_url"] or ""),
            netease_url=str(row["netease_url"] or ""),
            duration_ms=_maybe_int(row["duration_ms"]),
            playable=(None if row["playable"] is None else bool(row["playable"])),
            comment_count=_maybe_int(row["comment_count"]),
            hot_comment_count=_maybe_int(row["hot_comment_count"]),
            liked_count=_maybe_int(row["liked_count"]),
            artist_followers=_maybe_int(row["artist_followers"]),
            audio_quality=_maybe_float(row["audio_quality"]),
            metadata_quality_score=float(row["metadata_quality_score"] or 0.0),
            popularity_score=float(row["popularity_score"] or 0.0),
            artist_authority_score=float(row["artist_authority_score"] or 0.0),
            source_seen_count=int(row["source_seen_count"] or 0),
            first_seen_at=float(row["first_seen_at"] or 0.0),
            last_seen_at=float(row["last_seen_at"] or 0.0),
            tags=_json_list(row["tags"]),
            query_texts=_json_list(row["query_texts"]),
            source_types=_json_list(row["source_types"]),
            raw_json=_json_obj(row["raw_json"]),
            text_for_embedding=str(row["text_for_embedding"] or ""),
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        s = str(raw or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _build_text_for_embedding(
    *,
    title: str,
    artist_names: str,
    album: str,
    tags: Iterable[str],
    query_texts: Iterable[str],
    source_types: Iterable[str],
) -> str:
    """Stable content text for one song.

    Concatenates only durable, song-describing signals. Order is fixed so
    the same inputs always yield the same text (deterministic index).
    """
    parts: list[str] = [title, artist_names, album]
    parts.extend(tags)
    parts.extend(query_texts)
    parts.extend(source_types)
    return " ".join(p for p in (str(x).strip() for x in parts) if p)


def _json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value) if isinstance(value, (str, bytes)) else value
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _json_obj(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value) if isinstance(value, (str, bytes)) else value
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
