"""Backward-compatible facade for the NetEase recommendation pipeline.

The implementation was refactored (P1) from a single monolithic class
into a clearly staged package -- :mod:`SongRecDemo.pipeline`:

    profile building -> retrieval -> filtering -> enrichment
        -> ranking -> reranking -> explanation -> trace

This module is kept as a thin re-export so existing imports
(``from SongRecDemo.netease_pipeline import NeteaseRecommender`` and the
hermetic smoke test's ``from netease_pipeline import _norm_title,
FakeNeteaseClient``) keep working unchanged. Prefer importing from
``SongRecDemo.pipeline`` going forward.
"""

from __future__ import annotations

# Public surface + stage components + standard data objects.
from SongRecDemo.pipeline import (  # noqa: F401
    Candidate,
    CandidateEnrichment,
    CandidateFilter,
    Embedder,
    EmbeddingMatch,
    EmbeddingRetriever,
    Explainer,
    FakeNeteaseClient,
    FeatureEnricher,
    NeteaseRecommender,
    ProfileBuilder,
    QueryPlanner,
    Ranker,
    RealSongCard,
    RealSongRequest,
    RealSongResponse,
    RecommendationTrace,
    Reranker,
    Retriever,
    RetrievalQuery,
    SongFeatureRecord,
    SongFeatureStore,
    SourceHit,
    TrackRef,
    UserProfile,
    merge_candidates_into,
)

# Backward-compatible internal aliases (pre-refactor names).
from SongRecDemo.pipeline.types import (  # noqa: F401
    _Candidate,
    _CandidateEnrichment,
    _NeteaseClient,
    _Profile,
    _QueryCache,
    _RetrievalQuery,
    _SourceHit,
    _merge_track,
)

# Tokenisation / vector helpers used by the smoke test and any callers
# that reached into the old module-level helpers.
from SongRecDemo.pipeline.text import (  # noqa: F401
    _clip01,
    _contains_tokens,
    _cosine,
    _dedupe_phrases,
    _jaccard,
    _maybe_float,
    _maybe_int,
    _norm_artist,
    _norm_title,
    _profile_text,
    _raw_tokens,
    _source_preference_tokens,
    _starts_with_tokens,
    _strip_profile_tag_terms,
    _tfidf_vectors,
    _token_set,
    _tokens,
)

__all__ = [
    "TrackRef",
    "RealSongRequest",
    "RealSongCard",
    "RealSongResponse",
    "NeteaseRecommender",
    "FakeNeteaseClient",
    "ProfileBuilder",
    "QueryPlanner",
    "Retriever",
    "CandidateFilter",
    "FeatureEnricher",
    "Ranker",
    "Reranker",
    "Explainer",
    "EmbeddingRetriever",
    "RecommendationTrace",
    "SongFeatureStore",
    "SongFeatureRecord",
    "Embedder",
    "EmbeddingMatch",
    "UserProfile",
    "Candidate",
    "CandidateEnrichment",
    "RetrievalQuery",
    "SourceHit",
    "merge_candidates_into",
]
