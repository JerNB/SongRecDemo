"""EmbeddingRetriever -- P2 second recall channel.

An OPTIONAL retrieval channel that complements (never replaces) the
NetEase ``/search`` channel. It reads the local :class:`SongFeatureStore`,
builds / refreshes an in-memory :class:`Embedder` index, and finds the
songs whose stable content text is semantically closest to the user's
profile text. Each hit becomes a standard :class:`Candidate` carrying a
``SourceHit`` with ``source_type == "embedding"``.

This stage does recall only. It assigns no final scores -- ranking stays
entirely with the P0 :class:`Ranker`. If the store holds fewer than
``min_corpus_size`` songs (the cold-start / small-sample case) the channel
quietly returns nothing instead of mis-recalling or raising.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .embedding import Embedder
from .feature_store import SongFeatureStore
from .types import Candidate, RealSongRequest, SourceHit, TrackRef, UserProfile

log = logging.getLogger(__name__)


class EmbeddingRetriever:
    """Semantic recall over the local song feature store."""

    def __init__(
        self,
        store: SongFeatureStore,
        embedder: Optional[Embedder] = None,
        *,
        reliability: float = 0.68,
        top_k: int = 30,
        min_corpus_size: int = 20,
        svd_dim: int = 64,
        model_type: str = "tfidf_svd",
    ) -> None:
        self._store = store
        self._embedder = embedder or Embedder(model_type=model_type, svd_dim=svd_dim)
        self._reliability = float(reliability)
        self._top_k = int(top_k)
        self._min_corpus_size = int(min_corpus_size)
        # Signature of the corpus the index was last fitted on, so we only
        # re-fit when the store actually changed.
        self._fitted_ids: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _ensure_index(self) -> bool:
        """(Re)build the in-memory index from the store if needed.

        Returns True when a usable index is ready.
        """
        ids, texts = self._store.iter_text_corpus()
        if len(ids) < self._min_corpus_size:
            self._fitted_ids = ()
            return False

        signature = tuple(sorted(ids))
        if signature != self._fitted_ids or not self._embedder.ready:
            self._embedder.fit(texts, ids)
            self._fitted_ids = signature if self._embedder.ready else ()
        return self._embedder.ready

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        profile: UserProfile,
        req: RealSongRequest,
        *,
        exclude_ids: Optional[set[int]] = None,
    ) -> tuple[dict[int, Candidate], dict[str, Any]]:
        """Recall semantically-similar songs from the local store.

        Returns ``(candidates_by_id, stats)``. ``stats`` always carries the
        observability fields the trace needs, even when the channel is
        skipped, so the caller never has to special-case an empty store.
        """
        t0 = time.perf_counter()
        store_count = self._store.count()
        stats: dict[str, Any] = {
            "num_feature_store_songs": int(store_count),
            "num_embedding_candidates": 0,
            "embedding_index_ready": False,
            "embedding_latency_ms": 0.0,
        }
        exclude = set(exclude_ids or set())

        index_ready = self._ensure_index()
        stats["embedding_index_ready"] = bool(index_ready)
        if not index_ready:
            stats["embedding_latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return {}, stats

        query_text = (profile.user_profile_text or "").strip()
        if not query_text:
            stats["embedding_latency_ms"] = (time.perf_counter() - t0) * 1000.0
            return {}, stats

        matches = self._embedder.search(query_text, self._top_k)

        candidates: dict[int, Candidate] = {}
        for match in matches:
            sid = int(match.song_id)
            if sid in exclude or sid in candidates:
                continue
            record = self._store.get_song(sid)
            if record is None:
                continue

            track = TrackRef(
                netease_song_id=sid,
                title=record.title,
                artist=record.artists[0] if record.artists else "",
                artists=list(record.artists),
                album=record.album,
                cover_url=record.cover_url,
            )
            source_name = f"embedding:{sid}"
            cand = Candidate(track=track)
            cand.sources.append(source_name)
            cand.positions[source_name] = int(match.rank)
            cand.source_hits.append(SourceHit(
                source_name=source_name,
                source_type="embedding",
                query=query_text,
                reliability=self._reliability,
                position=int(match.rank),
            ))
            candidates[sid] = cand

        stats["num_embedding_candidates"] = len(candidates)
        stats["embedding_latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return candidates, stats
