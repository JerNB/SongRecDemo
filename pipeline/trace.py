"""RecommendationTrace -- stage 8 (cross-cutting observability).

A lightweight, dependency-free record of how one recommendation flowed
from input to output: profile summary, candidate counts at each stage,
the control sliders, end-to-end latency, and cache hit/miss info. No
database -- it is attached to the response metadata and logged so a
developer can see the funnel (profile -> retrieval -> filtering ->
enrichment -> ranking) at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RecommendationTrace:
    request_id: str
    profile_summary: dict[str, Any] = field(default_factory=dict)
    num_raw_candidates: int = 0
    num_filtered_candidates: int = 0
    num_enriched_candidates: int = 0
    num_final_candidates: int = 0
    content_weight: float = 0.0
    novelty: float = 0.0
    diversity: float = 0.0
    latency_ms: float = 0.0
    cache_info: dict[str, Any] = field(default_factory=dict)
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id":              self.request_id,
            "profile_summary":         dict(self.profile_summary),
            "num_raw_candidates":      int(self.num_raw_candidates),
            "num_filtered_candidates": int(self.num_filtered_candidates),
            "num_enriched_candidates": int(self.num_enriched_candidates),
            "num_final_candidates":    int(self.num_final_candidates),
            "content_weight":          float(self.content_weight),
            "novelty":                 float(self.novelty),
            "diversity":               float(self.diversity),
            "latency_ms":              round(float(self.latency_ms), 2),
            "cache_info":              dict(self.cache_info),
            "stage_latencies_ms":      {k: round(float(v), 2) for k, v in self.stage_latencies_ms.items()},
        }

    def log_line(self) -> str:
        return (
            f"trace request_id={self.request_id} "
            f"raw={self.num_raw_candidates} "
            f"filtered={self.num_filtered_candidates} "
            f"enriched={self.num_enriched_candidates} "
            f"final={self.num_final_candidates} "
            f"content_weight={self.content_weight:.2f} "
            f"novelty={self.novelty:.2f} diversity={self.diversity:.2f} "
            f"latency_ms={self.latency_ms:.1f} "
            f"cache={self.cache_info}"
        )
