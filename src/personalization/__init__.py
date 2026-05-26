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

Implementation note
---------------------
:class:`RecommendationService` is imported **lazily** (see :func:`__getattr__`)
so that ``from src.personalization.netease_enrichment import NeteaseAPIClient``
does not pull in scikit-learn / SciPy at package import time.  That keeps
lightweight demos (e.g. ``SongRecDemo/app.py``) fast until code actually
references ``RecommendationService``.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.personalization.interface import (
    SeedInput,
    RecommendationRequest,
    ScoredItem,
    RecommendationResponse,
)
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


def __getattr__(name: str) -> Any:
    """Lazy-load :class:`RecommendationService` (heavy sklearn/scipy import)."""
    if name == "RecommendationService":
        mod = importlib.import_module("src.personalization.service")
        svc = getattr(mod, "RecommendationService")
        globals()["RecommendationService"] = svc
        return svc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {k for k in globals() if not k.startswith("_")})
