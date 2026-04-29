"""
Personalized recommendation layer built on top of the ALS backbone.

Public entry point: :class:`service.RecommendationService`.

The rest of this subpackage is internal detail:
  * ``interface``    — JSON-friendly request/response dataclasses
  * ``profile``      — seed / favourite / tag inputs -> ALS fold-in vector
                       and tag-TF-IDF taste profile
  * ``engine``       — retrieval + reranking + novelty + MMR diversity
  * ``explanations`` — per-item "why this track?" strings
  * ``enrichment``   — pluggable metadata Protocol (display fields)

The interface is deliberately frontend-agnostic: the same dataclasses
serialise straight to JSON for an HTTP/gRPC/whatever transport, and the
service can be imported directly by a Streamlit/FastAPI/Flask demo.
"""

from src.personalization.interface import (
    SeedInput,
    RecommendationRequest,
    ScoredItem,
    RecommendationResponse,
)
from src.personalization.service import RecommendationService
from src.personalization.enrichment import (
    MetadataEnricher,
    NullEnricher,
    InternalFeaturesEnricher,
)
from src.personalization.netease_enrichment import (
    NeteaseAPIClient,
    NeteaseAPIError,
    NeteaseCache,
    NeteaseMetadataEnricher,
)

__all__ = [
    "SeedInput",
    "RecommendationRequest",
    "ScoredItem",
    "RecommendationResponse",
    "RecommendationService",
    "MetadataEnricher",
    "NullEnricher",
    "InternalFeaturesEnricher",
    "NeteaseMetadataEnricher",
    "NeteaseAPIClient",
    "NeteaseAPIError",
    "NeteaseCache",
]
