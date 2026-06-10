"""Production-inspired NetEase recommendation pipeline.

Stages (each in its own module, independently testable):

    ProfileBuilder   (profile.py)    -- normalise user input -> UserProfile
    QueryPlanner     (retrieval.py)  -- UserProfile -> labelled queries
    Retriever        (retrieval.py)  -- queries -> Candidate recall
    CandidateFilter  (filtering.py)  -- dedup / filter / soft-penalty
    FeatureEnricher  (enrichment.py) -- NetEase deep enrichment (+cache)
    Ranker           (ranking.py)    -- P0 hybrid relevance scoring
    Reranker         (reranking.py)  -- MMR diversification (final vs rank)
    Explainer        (explain.py)    -- reasons / pick_type / explanation
    RecommendationTrace (trace.py)   -- end-to-end observability

The :class:`NeteaseRecommender` orchestrator (recommender.py) sequences
them and assembles the response.
"""

from __future__ import annotations

from .embedding import Embedder, EmbeddingMatch
from .embedding_retrieval import EmbeddingRetriever
from .enrichment import FeatureEnricher
from .explain import Explainer
from .feature_store import SongFeatureRecord, SongFeatureStore
from .feedback import FeedbackEventError, FeedbackStore
from .filtering import CandidateFilter
from .profile import ProfileBuilder
from .ranking import Ranker
from .recommender import FakeNeteaseClient, NeteaseRecommender
from .reranking import Reranker
from .retrieval import QueryPlanner, Retriever
from .trace import RecommendationTrace
from .types import (
    Candidate,
    CandidateEnrichment,
    RealSongCard,
    RealSongRequest,
    RealSongResponse,
    RetrievalQuery,
    SourceHit,
    TrackRef,
    UserProfile,
    merge_candidates_into,
)

__all__ = [
    # Public request/response surface.
    "TrackRef",
    "RealSongRequest",
    "RealSongCard",
    "RealSongResponse",
    # Orchestrator + test double.
    "NeteaseRecommender",
    "FakeNeteaseClient",
    # Stage components.
    "ProfileBuilder",
    "QueryPlanner",
    "Retriever",
    "EmbeddingRetriever",
    "CandidateFilter",
    "FeatureEnricher",
    "Ranker",
    "Reranker",
    "Explainer",
    "RecommendationTrace",
    # P2: feature store + embedding.
    "SongFeatureStore",
    "SongFeatureRecord",
    "Embedder",
    "EmbeddingMatch",
    # P3: feedback logging.
    "FeedbackStore",
    "FeedbackEventError",
    # Standard data objects.
    "UserProfile",
    "Candidate",
    "CandidateEnrichment",
    "RetrievalQuery",
    "SourceHit",
    "merge_candidates_into",
]
