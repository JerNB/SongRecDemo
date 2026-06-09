"""QueryPlanner + Retriever -- stage 2 of the pipeline.

``QueryPlanner`` turns a :class:`UserProfile` into a list of labelled
:class:`RetrievalQuery` objects (one per artist / genre / mood / tag /
seed-song / discovery channel). ``Retriever`` runs those queries through
the cache-first NetEase ``/search`` and merges the hits into standard
:class:`Candidate` objects, recording for every hit its ``query_text``,
``source_type``, ``reliability`` and ``rank position`` in
:class:`SourceHit`.

The Retriever performs **no** filtering -- it only recalls and labels.
Filtering is the ``CandidateFilter``'s job, which keeps recall and
precision concerns cleanly separated and makes it easy to bolt on an
embedding recall channel later (just append more Candidates).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .text import _maybe_int, _tokens
from .types import (
    Candidate,
    RealSongRequest,
    RetrievalQuery,
    SourceHit,
    TrackRef,
    UserProfile,
    _merge_track,
    _NeteaseClient,
    _QueryCache,
)


log = logging.getLogger(__name__)


class QueryPlanner:
    """Builds the labelled retrieval queries for one request."""

    def __init__(
        self,
        *,
        max_artist_queries: int = 6,
        max_tag_queries: int = 4,
        max_title_queries: int = 3,
    ) -> None:
        self._max_artist_queries = int(max_artist_queries)
        self._max_tag_queries = int(max_tag_queries)
        self._max_title_queries = int(max_title_queries)

    def plan(
        self,
        profile: UserProfile,
        req: RealSongRequest,
        per_query: int,
    ) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        seen: set[tuple[str, str]] = set()

        def add(source_type: str, query: str, reliability: float, limit: int) -> None:
            q = " ".join(str(query or "").split())
            if not q:
                return
            key = (source_type, q.lower())
            if key in seen:
                return
            seen.add(key)
            queries.append(RetrievalQuery(
                source_name=f"{source_type}:{q}",
                source_type=source_type,
                query=q,
                reliability=float(reliability),
                limit=max(1, int(limit)),
            ))

        top_genre = profile.preferred_genres[0] if profile.preferred_genres else ""
        top_mood = profile.preferred_moods[0] if profile.preferred_moods else ""
        top_tag = profile.preferred_tags[0] if profile.preferred_tags else (
            profile.tag_phrases[0] if profile.tag_phrases else ""
        )

        for artist in profile.liked_artists_display[: self._max_artist_queries]:
            add("artist", artist, 0.85, per_query)
            if top_genre:
                add("artist_context", f"{artist} {top_genre}", 0.80, per_query)
            if top_mood:
                add("artist_context", f"{artist} {top_mood}", 0.80, per_query)

        for genre in profile.preferred_genres[: self._max_tag_queries]:
            add("genre", genre, 0.75, per_query)
        for mood in profile.preferred_moods[: self._max_tag_queries]:
            add("mood", mood, 0.65, per_query)
        for tag in profile.preferred_tags[: self._max_tag_queries]:
            add("tag", tag, 0.75, per_query)
        if top_genre and top_mood:
            add("genre_mood", f"{top_genre} {top_mood}", 0.75, per_query)
        if len(profile.preferred_tags) >= 2:
            add("tag_combo", " ".join(profile.preferred_tags[:2]), 0.72, per_query)

        for song in req.liked_songs[: self._max_title_queries]:
            if song.title and song.artist:
                add("seed_song", f"{song.title} {song.artist}", 0.78, per_query)
            album_keyword = self._album_keyword(song.album)
            if song.artist and album_keyword:
                add("seed_album", f"{song.artist} {album_keyword}", 0.70, per_query)
            if song.title:
                add("title", song.title, 0.45, max(4, min(per_query, 8)))

        if int(req.discovery_limit) > 0:
            dlim = int(req.discovery_limit)
            if top_genre and top_mood:
                add("discovery", f"{top_genre} {top_mood}", 0.55, dlim)
            if top_tag:
                add("discovery", top_tag, 0.55, dlim)
            if profile.liked_artists_display and top_tag:
                add("discovery", f"{profile.liked_artists_display[0]} {top_tag}", 0.55, dlim)
            mixed = " ".join(profile.query_intent_terms[:4])
            if mixed:
                add("discovery", mixed, 0.55, dlim)

        return queries

    @staticmethod
    def _album_keyword(album: str) -> str:
        toks = _tokens(album)
        return " ".join(toks[:2])


class Retriever:
    """Cache-first multi-channel candidate recall.

    Output is a ``{song_id: Candidate}`` dict plus a retrieval stats
    dict (per-bucket hit counts + cache hit/miss info for the trace).
    """

    def __init__(
        self,
        client: _NeteaseClient,
        cache: Optional[_QueryCache] = None,
        *,
        max_per_query: int = 12,
        planner: Optional[QueryPlanner] = None,
    ) -> None:
        self._client = client
        self._cache = cache
        self._max_per_query = int(max_per_query)
        self._planner = planner or QueryPlanner()
        # Reset on each retrieve(); exposed for the recommendation trace.
        self.last_cache_hits = 0
        self.last_cache_misses = 0

    def retrieve(
        self, profile: UserProfile, req: RealSongRequest,
    ) -> tuple[dict[int, Candidate], dict[str, int]]:
        """Issue focused /search calls, dedup, and label sources."""
        per_query = max(1, min(self._max_per_query, int(req.candidates_per_signal)))
        self.last_cache_hits = 0
        self.last_cache_misses = 0

        candidates: dict[int, Candidate] = {}
        route_counts = {"artist": 0, "tag": 0, "title": 0, "discovery": 0}
        retrieved_total = 0

        for q in self._planner.plan(profile, req, per_query):
            hits = self._search_cached(q.query, q.limit)
            retrieved_total += len(hits)
            bucket = self._summary_bucket(q.source_type)
            route_counts[bucket] = route_counts.get(bucket, 0) + len(hits)
            self._merge_hits(
                candidates,
                hits,
                source_name=q.source_name,
                source_type=q.source_type,
                query=q.query,
                reliability=q.reliability,
            )

        retrieval_stats = {
            "artist":          int(route_counts.get("artist", 0)),
            "tag":             int(route_counts.get("tag", 0)),
            "title":           int(route_counts.get("title", 0)),
            "discovery":       int(route_counts.get("discovery", 0)),
            "retrieved_total": int(retrieved_total),
            "after_dedup":     int(len(candidates)),
        }
        return candidates, retrieval_stats

    def _search_cached(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Cache-first NetEase /search. Returns [] on empty / failure."""
        q = (query or "").strip()
        if not q:
            return []

        cache_key = f"songrec_demo:{q}::{int(limit)}"
        if self._cache is not None:
            hit = self._cache.get_query(cache_key)
            if hit is not None:
                self.last_cache_hits += 1
                return hit

        self.last_cache_misses += 1
        try:
            hits = self._client.search_songs(q, limit=int(limit)) or []
        except Exception as exc:                  # noqa: BLE001 -- never crash the demo
            log.warning("NetEase search %r failed (%s); returning no hits.", q, exc)
            hits = []

        if self._cache is not None:
            try:
                self._cache.set_query(cache_key, hits)
            except Exception as exc:              # noqa: BLE001
                log.debug("NetEase cache write failed for %r: %s", q, exc)
        return hits

    @staticmethod
    def _summary_bucket(source_type: str) -> str:
        if source_type in {"artist", "artist_context"}:
            return "artist"
        if source_type in {"genre", "mood", "tag", "genre_mood", "tag_combo"}:
            return "tag"
        if source_type in {"seed_song", "seed_album", "title"}:
            return "title"
        return "discovery"

    @staticmethod
    def _merge_hits(
        candidates: dict[int, Candidate],
        hits: list[dict[str, Any]],
        *,
        source_name: str,
        source_type: str,
        query: str,
        reliability: float,
    ) -> None:
        for pos, h in enumerate(hits):
            try:
                sid = int(h.get("netease_song_id"))
            except (TypeError, ValueError):
                continue
            if not sid:
                continue
            track = TrackRef(
                netease_song_id=sid,
                title=str(h.get("title") or "").strip(),
                artist=str(h.get("artist") or "").strip(),
                artists=[str(a).strip() for a in (h.get("artists") or []) if a],
                album=str(h.get("album") or "").strip(),
                cover_url=str(h.get("cover_url") or "").strip(),
            )
            artist_ids = [
                int(aid) for aid in (h.get("artist_ids") or [])
                if _maybe_int(aid) is not None
            ]
            cand = candidates.get(sid)
            if cand is None:
                cand = Candidate(track=track, artist_ids=artist_ids)
                candidates[sid] = cand
            else:
                # Prefer the richer track payload when both exist.
                cand.track = _merge_track(cand.track, track)
                if not cand.artist_ids and artist_ids:
                    cand.artist_ids = artist_ids
            if source_name not in cand.sources:
                cand.sources.append(source_name)
            # Keep the earliest position seen for a given source label.
            cand.positions.setdefault(source_name, pos)
            cand.source_hits.append(SourceHit(
                source_name=source_name,
                source_type=source_type,
                query=query,
                reliability=float(reliability),
                position=int(pos),
            ))
