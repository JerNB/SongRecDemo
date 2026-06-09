"""NeteaseRecommender -- the thin orchestrator that wires the stages.

    profile building -> retrieval -> filtering -> enrichment
        -> ranking -> reranking -> explanation -> trace

Each stage is an independently testable component (see the sibling
modules). This class only sequences them, assembles the response /
candidate_summary, and records a :class:`RecommendationTrace`. The P0
scoring behaviour is unchanged -- it lives in :mod:`ranking` / :mod:`scoring`.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from .enrichment import FeatureEnricher
from .explain import Explainer
from .filtering import CandidateFilter
from .profile import ProfileBuilder
from .ranking import Ranker
from .reranking import Reranker
from .retrieval import QueryPlanner, Retriever
from .scoring import _W
from .trace import RecommendationTrace
from .text import _clip01
from .types import (
    Candidate,
    RealSongRequest,
    RealSongResponse,
    UserProfile,
    _NeteaseClient,
    _QueryCache,
)


log = logging.getLogger(__name__)


class NeteaseRecommender:
    """Real-song recommendation pipeline backed by NetEase /search.

    Construction
    ------------
    >>> rec = NeteaseRecommender(client=NeteaseAPIClient(...), cache=NeteaseCache(...))
    >>> resp = rec.recommend(RealSongRequest(...))

    The optional ``cache`` parameter (any object with the small
    :class:`_QueryCache` shape) dramatically speeds up repeat calls
    -- repeated runs of the demo with the same artists / tags will
    short-circuit on the SQLite cache instead of re-hitting NetEase.
    """

    MODEL_NAME = "NetEase-Pipeline-v1"

    def __init__(
        self,
        client: _NeteaseClient,
        cache: Optional[_QueryCache] = None,
        *,
        max_per_query: int = 12,
        max_artist_queries: int = 6,
        max_tag_queries: int = 4,
        max_title_queries: int = 3,
        per_artist_cap: int = 2,
    ) -> None:
        self._client = client
        self._cache = cache
        self._max_per_query = int(max_per_query)
        self._per_artist_cap = int(per_artist_cap)
        self._enrich_top_n = 16
        self._live_deep_enrichment = False
        self._min_comment_count = 10
        self._min_artist_follow_count = 77

        # --- Stage components (each independently testable). -----------
        self._profile_builder = ProfileBuilder()
        self._planner = QueryPlanner(
            max_artist_queries=max_artist_queries,
            max_tag_queries=max_tag_queries,
            max_title_queries=max_title_queries,
        )
        self._retriever = Retriever(
            client=client,
            cache=cache,
            max_per_query=self._max_per_query,
            planner=self._planner,
        )
        self._filter = CandidateFilter(
            min_comment_count=self._min_comment_count,
            min_artist_follow_count=self._min_artist_follow_count,
        )
        self._enricher = FeatureEnricher(
            client=client,
            cache=cache,
            enrich_top_n=self._enrich_top_n,
            live_deep_enrichment=self._live_deep_enrichment,
        )
        self._ranker = Ranker()
        self._reranker = Reranker()
        self._explainer = Explainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, req: RealSongRequest) -> RealSongResponse:
        t_start = time.perf_counter()
        request_id = req.request_id or str(uuid.uuid4())

        # Stage 1: profile building.
        profile = self._profile_builder.build(req)

        # Hard validation: a real-song demo needs at least one signal.
        has_any = bool(
            profile.liked_track_ids
            or profile.liked_artists_norm
            or profile.tag_tokens
        )
        if not has_any:
            trace = self._build_trace(
                request_id, profile, req,
                num_raw=0, num_filtered=0, num_enriched=0, num_final=0,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                stage_latencies={},
            )
            log.info(trace.log_line())
            return RealSongResponse(
                request_id=request_id,
                items=[],
                control=self._control_echo(req),
                candidate_summary={"artist": 0, "tag": 0, "title": 0,
                                   "discovery": 0, "total_unique": 0},
                profile=self._profile_echo(profile),
                model_info=self._model_info(),
                fallback_used="no_input",
                trace=trace.to_dict(),
            )

        stage_latencies: dict[str, float] = {}

        # Stage 2: multi-channel candidate retrieval (recall only).
        t = time.perf_counter()
        candidates, retrieval_stats = self._retriever.retrieve(profile, req)
        num_raw = int(retrieval_stats["after_dedup"])
        stage_latencies["retrieval"] = (time.perf_counter() - t) * 1000.0

        # Stage 3a: pre-enrichment filtering (metadata-only rules).
        t = time.perf_counter()
        pre_stats = self._filter.pre_enrichment(candidates, profile)
        stage_latencies["filter_pre"] = (time.perf_counter() - t) * 1000.0

        # Stage 5 (round 1): lightweight score to pick what to enrich.
        t = time.perf_counter()
        scored = self._ranker.score_all(candidates, profile, req)
        scored.sort(key=lambda x: (-x[0], x[1].track.title or "", x[1].track.netease_song_id))
        stage_latencies["rank_round1"] = (time.perf_counter() - t) * 1000.0

        # Stage 4: deep enrichment of the strongest candidates.
        t = time.perf_counter()
        enrich_n = self._enricher.enrichment_budget(req)
        enriched_count = self._enricher.enrich([cand for _score, cand, _bd in scored[:enrich_n]])
        stage_latencies["enrichment"] = (time.perf_counter() - t) * 1000.0

        # Stage 3b: post-enrichment filtering (signal-dependent rules).
        t = time.perf_counter()
        post_stats = self._filter.post_enrichment(candidates)
        stage_latencies["filter_post"] = (time.perf_counter() - t) * 1000.0

        # Stage 5 (round 2): re-score after enrichment so popularity,
        # authority, playability, and trust affect ranking + explanations.
        t = time.perf_counter()
        scored = self._ranker.score_all(candidates, profile, req)
        scored.sort(key=lambda x: (-x[0], x[1].track.title or "", x[1].track.netease_song_id))
        stage_latencies["rank_round2"] = (time.perf_counter() - t) * 1000.0

        # Stage 6: MMR rerank for diversity + per-artist / per-album cap.
        t = time.perf_counter()
        ranked = self._reranker.rerank(scored, profile, req)
        stage_latencies["rerank"] = (time.perf_counter() - t) * 1000.0

        # Stage 7: build cards with explanations + pick type.
        t = time.perf_counter()
        cards = self._explainer.build_cards(ranked, profile)
        stage_latencies["explain"] = (time.perf_counter() - t) * 1000.0

        summary = {
            "artist":                     int(retrieval_stats["artist"]),
            "tag":                        int(retrieval_stats["tag"]),
            "title":                      int(retrieval_stats["title"]),
            "discovery":                  int(retrieval_stats["discovery"]),
            "retrieved_total":            int(retrieval_stats["retrieved_total"]),
            "after_dedup":                int(retrieval_stats["after_dedup"]),
            "filtered_liked":             int(pre_stats["filtered_liked"]),
            "filtered_same_title":        int(pre_stats["filtered_same_title"]),
            "filtered_tag_title":         int(pre_stats["filtered_tag_title"]),
            "filtered_duplicate_version": int(pre_stats["filtered_duplicate_version"]),
            "filtered_missing_metadata":  int(pre_stats["filtered_missing_metadata"]),
            "enriched_count":             int(enriched_count),
            "filtered_unplayable":        int(post_stats["filtered_unplayable"]),
            "filtered_low_trust":         int(post_stats["filtered_low_trust"]),
            "final_candidate_count":      int(len(candidates)),
            "total_unique":               int(len(candidates)),
        }

        # Stage 8: trace.
        trace = self._build_trace(
            request_id, profile, req,
            num_raw=num_raw,
            num_filtered=int(len(candidates)),
            num_enriched=int(enriched_count),
            num_final=int(len(cards)),
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            stage_latencies=stage_latencies,
        )
        log.info(trace.log_line())

        return RealSongResponse(
            request_id=request_id,
            items=cards,
            control=self._control_echo(req),
            candidate_summary=summary,
            profile=self._profile_echo(profile),
            model_info=self._model_info(),
            fallback_used=None if cards else "no_candidates",
            trace=trace.to_dict(),
        )

    # ------------------------------------------------------------------
    # Trace + echo helpers
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        request_id: str,
        profile: UserProfile,
        req: RealSongRequest,
        *,
        num_raw: int,
        num_filtered: int,
        num_enriched: int,
        num_final: int,
        latency_ms: float,
        stage_latencies: dict[str, float],
    ) -> RecommendationTrace:
        return RecommendationTrace(
            request_id=request_id,
            profile_summary={
                "liked_songs":    len(profile.liked_track_ids),
                "liked_artists":  list(profile.liked_artists_display),
                "tags":           list(profile.tag_phrases),
                "tag_tokens":     list(profile.tag_tokens),
                "excluded_songs": len(profile.excluded_track_ids),
            },
            num_raw_candidates=int(num_raw),
            num_filtered_candidates=int(num_filtered),
            num_enriched_candidates=int(num_enriched),
            num_final_candidates=int(num_final),
            content_weight=float(_clip01(req.content_weight)),
            novelty=float(_clip01(req.novelty)),
            diversity=float(_clip01(req.diversity)),
            latency_ms=float(latency_ms),
            cache_info={
                "search_cache_hits":   int(self._retriever.last_cache_hits),
                "search_cache_misses": int(self._retriever.last_cache_misses),
                "cache_enabled":       self._cache is not None,
            },
            stage_latencies_ms=dict(stage_latencies),
        )

    def _control_echo(self, req: RealSongRequest) -> dict[str, Any]:
        return {
            "content_weight": float(_clip01(req.content_weight)),
            "novelty":        float(_clip01(req.novelty)),
            "diversity":      float(_clip01(req.diversity)),
            "k":              int(max(1, min(50, int(req.k)))),
        }

    def _profile_echo(self, profile: UserProfile) -> dict[str, Any]:
        return {
            "liked_song_ids":   sorted(profile.liked_track_ids),
            "liked_artists":    list(profile.liked_artists_display),
            "tags":             list(profile.tag_phrases),
            "preferred_genres":  list(profile.preferred_genres),
            "preferred_moods":   list(profile.preferred_moods),
            "preferred_tags":    list(profile.preferred_tags),
            "tag_tokens":       list(profile.tag_tokens),
            "title_tokens":     sorted(profile.title_tokens),
            "query_intent_terms": list(profile.query_intent_terms),
        }

    def _model_info(self) -> dict[str, Any]:
        return {
            "name":                         self.MODEL_NAME,
            "model_type":                   "real_song_hybrid_retrieval_ranking",
            "uses_netease_api":             True,
            "trained_collaborative_filtering": False,
            # multi_source_agreement is a retrieval-consensus signal, NOT
            # collaborative filtering. The legacy `collaborative_proxy_used`
            # flag is retained for backward compatibility only.
            "multi_source_agreement_used":  True,
            "collaborative_proxy_used":     True,  # legacy alias; prefer multi_source_agreement_used
            "candidate_enrichment_used":    True,
            "ranking_weights_version":      "v1",
            "ranking_weights":              _W,
            "pipeline_stages": [
                "profile", "retrieval", "filtering", "enrichment",
                "ranking", "reranking", "explanation", "trace",
            ],
            "content_weight_meaning": (
                "Higher content_weight leans on user text / liked songs / "
                "artists / tags / genres; lower content_weight leans on "
                "retrieval confidence and multi-source agreement."
            ),
            "quality_thresholds": {
                "soft_min_comment_count": self._min_comment_count,
                "soft_min_artist_follow_count": self._min_artist_follow_count,
            },
            "enrichment_live_budget": self._enrich_top_n,
            "deep_live_enrichment": self._live_deep_enrichment,
            "research_layer":               "KGRec ALS/content/popularity evaluation remains separate",
            "source":                       "NetEase /search",
        }


# ---------------------------------------------------------------------------
# Test double for hermetic smoke tests
# ---------------------------------------------------------------------------

class FakeNeteaseClient:
    """Deterministic in-memory NetEase client used by tests.

    Construct with a mapping ``query_lower -> list[song dict]`` and
    optionally a fallback list returned for any unknown query. The
    canned songs match the shape returned by
    :class:`NeteaseAPIClient.search_songs` so the recommender can be
    swapped against this without code changes.
    """

    def __init__(
        self,
        responses: Optional[dict[str, list[dict[str, Any]]]] = None,
        *,
        default: Optional[list[dict[str, Any]]] = None,
        enrichments: Optional[dict[int, dict[str, Any]]] = None,
        alive: bool = True,
    ) -> None:
        self._responses = {k.lower(): list(v) for k, v in (responses or {}).items()}
        self._default = list(default or [])
        self._enrichments = {int(k): dict(v) for k, v in (enrichments or {}).items()}
        self._alive = bool(alive)
        self.calls: list[tuple[str, int]] = []

    def search_songs(self, keywords: str, limit: int = 5) -> list[dict[str, Any]]:
        self.calls.append((keywords, int(limit)))
        if not self._alive:
            return []
        key = (keywords or "").strip().lower()
        hits = self._responses.get(key, self._default)
        return list(hits)[: int(limit)]

    def ping(self) -> bool:
        return self._alive

    def enrich_song(self, song_id: int) -> dict[str, Any]:
        return dict(self._enrichments.get(int(song_id), {
            "comment_count": 120,
            "hot_comment_count": 3,
            "song_red_count": 300,
            "artist_follow_count": 1200,
            "playable": True,
            "audio_quality": 0.8,
            "similar_song_ids": [],
        }))
