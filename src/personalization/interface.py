"""
Public request/response contract for the personalized recommender.

These dataclasses form the stable interface that an external caller --
the future website demo, a notebook, another Python module, or a REST
layer -- will use to talk to :class:`RecommendationService`.

Design notes
------------
* Every field has a default so a caller can build a request with just
  the fields they care about.
* All types are JSON-serialisable primitives (str, int, float, list,
  dict, bool, None). ``asdict(resp)`` is enough to hand the response to
  ``json.dumps`` without a custom encoder.
* ``ScoredItem.metadata`` is reserved for a future enrichment layer
  (cleaner titles, artist names, cover art, etc.). The core engine
  leaves it empty; a pluggable :class:`MetadataEnricher` fills it in.
* Control knobs live in :class:`RecommendationRequest` so the UI can
  expose sliders without touching model internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------

@dataclass
class SeedInput:
    """What the interactive user has told us about their taste.

    All fields are optional. An empty :class:`SeedInput` is legal and
    triggers the popularity cold-start fallback.

    Fields
    ------
    item_ids : list[str]
        Raw item IDs the user has clicked / played / indicated casual
        interest in. Treated as soft positives (confidence = 1.0).
    favorite_ids : list[str]
        Raw item IDs the user has explicitly starred / favourited.
        Treated as strong positives (confidence = 2.0 by default).
    tags : list[str]
        Free-form tag tokens the user typed or selected from a
        suggestion chip list, e.g. ``["indie", "mellow", "80s"]``.
        Normalised and matched against the trained tag vocabulary.
    exclude_ids : list[str]
        Items the user has marked "not interested" or has seen recently
        and doesn't want back. Subtracted after ranking.
    """

    item_ids: list[str] = field(default_factory=list)
    favorite_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)


@dataclass
class RecommendationRequest:
    """Full request envelope with control knobs.

    Control knobs (all floats are clipped into their documented range
    before use; out-of-range values are not errors):

    novelty : float in [0, 1]
        0 = ranking is unmodified (ALS-like, head-heavy);
        1 = strong popularity demotion, pushes long-tail items up.
    content_weight : float in [0, 1]
        0 = ranking is pure ALS (latent factors only);
        1 = ranking is pure tag-TF-IDF cosine to the taste profile.
        Middle values linearly blend the two normalised scores.
    diversity : float in [0, 1]
        MMR lambda applied after scoring.
        0 = no diversification (top-N by score);
        1 = each subsequent pick maximises distance to previous picks
        in tag-TF-IDF space.

    Other fields
    ------------
    n : int
        Number of recommendations to return.
    candidate_pool : int
        How many items to carry from the retrieval stage into reranking.
        Larger pool = better reranking, slower response. Clamped to
        ``[n, n_items]``.
    fold_in_tag_seeds : int
        When the user provides free-text ``tags`` but few/no seed items,
        the tag vector is matched against the catalogue and the top-N
        matches are injected as pseudo-seeds into the ALS fold-in. This
        turns tag-only input into a latent-space query.
    request_id : str | None
        Optional ID to echo back for logging / caching. If None, the
        service generates one.
    """

    seeds: SeedInput = field(default_factory=SeedInput)
    n: int = 20
    novelty: float = 0.2
    content_weight: float = 0.25
    diversity: float = 0.0
    candidate_pool: int = 500
    fold_in_tag_seeds: int = 25
    request_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------

@dataclass
class ScoredItem:
    """One recommended track, fully explained.

    ``score_breakdown`` carries the individual components that went into
    ``score`` so a debug / "why" panel in the UI can show them. Keys are
    stable:

      - ``als``              : normalised ALS latent-match score in [0,1]
      - ``content``          : normalised tag-TF-IDF cosine in [0,1]
      - ``blended``          : (1-w) * als + w * content
      - ``popularity``       : normalised train-count in [0,1]
      - ``novelty_penalty``  : novelty * popularity  (subtracted)
      - ``raw``              : ``blended - novelty_penalty`` (pre-MMR)

    ``metadata`` is filled by the :class:`MetadataEnricher`; the core
    engine leaves it empty. A web frontend should prefer fields in
    ``metadata`` when present and only fall back to ``item_id`` for
    display.
    """

    item_id: str
    rank: int
    score: float
    score_breakdown: dict[str, float]
    explanation: str
    reasons: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationResponse:
    """Full response envelope.

    Fields
    ------
    request_id : str
        Echoes ``RecommendationRequest.request_id`` (or the generated
        one).
    items : list[ScoredItem]
        Ranked recommendations, length <= ``request.n``.
    seed_summary : dict
        What the service actually used after validation:
          - ``accepted_item_ids`` / ``rejected_item_ids``
          - ``accepted_favorite_ids`` / ``rejected_favorite_ids``
          - ``matched_tags`` / ``unknown_tags``
          - ``tag_fold_in_item_ids``  (pseudo-seeds from tag matching)
    control : dict
        The effective control knobs after clipping. Useful for the UI
        to display the active settings in the response.
    model_info : dict
        Identity of the retrieval model (name, d, alpha, lambda, iters)
        so the frontend can display a "Powered by ALS ..." footer and
        log which model version produced the response.
    fallback_used : str | None
        None on a regular call.  One of:
          - ``"popularity_cold_start"`` -- seeds + tags all empty or
            unresolvable; returned globally popular items.
          - ``"tag_only_fold_in"``      -- no seed items, only tags;
            pseudo-seeds were generated from the tag vector.
    """

    request_id: str
    items: list[ScoredItem]
    seed_summary: dict[str, Any]
    control: dict[str, float]
    model_info: dict[str, Any]
    fallback_used: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
