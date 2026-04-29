"""
Top-level service for the personalized recommender.

:class:`RecommendationService` is the single public entry point the
future website / CLI / notebook calls. It wires together:

  * persisted ALS state  (item factors + hyperparams)
  * preprocessed tag TF-IDF matrix + fitted vectoriser
  * popularity counts derived from training
  * the :class:`ProfileBuilder`, :class:`PersonalizedEngine`, and
    :class:`Explainer`
  * an optional :class:`MetadataEnricher` for display fields

The service is immutable after construction. A single instance can
serve many concurrent recommendation requests; the per-request state
lives on the stack.

Usage
-----

::

    from src.personalization import RecommendationService, SeedInput, RecommendationRequest

    svc = RecommendationService.from_artifacts()

    req = RecommendationRequest(
        seeds=SeedInput(
            item_ids=["1679", "2076"],
            tags=["indie", "mellow"],
        ),
        n=10,
        novelty=0.3,
        content_weight=0.25,
    )
    resp = svc.recommend(req)
    for item in resp.items:
        print(item.rank, item.item_id, item.explanation)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import config
from src.data.artifacts import ProcessedArtifacts, load_processed_artifacts
from src.personalization.engine import PersonalizedEngine, _Candidate
from src.personalization.enrichment import (
    InternalFeaturesEnricher,
    MetadataEnricher,
    NullEnricher,
)
from src.personalization.explanations import Explainer
from src.personalization.interface import (
    RecommendationRequest,
    RecommendationResponse,
    ScoredItem,
    SeedInput,
)
from src.personalization.profile import ProfileBuilder, TasteProfile
from src.recommenders.collaborative import CollaborativeFilteringRecommender

log = logging.getLogger(__name__)


class RecommendationService:
    """Stateless-per-request orchestration layer.

    Build once at process startup; call :meth:`recommend` many times.
    """

    MODEL_NAME = "ALS-Personalized-v1"

    def __init__(
        self,
        item_ids: list[str],
        item_factors: np.ndarray,
        tag_matrix: sp.csr_matrix,
        tag_vectorizer,
        popularity: np.ndarray,
        item_tag_tokens: dict[str, list[str]],
        alpha: float,
        reg: float,
        factors: int,
        iterations: int,
        enricher: Optional[MetadataEnricher] = None,
        favorite_weight: float = 2.0,
    ) -> None:
        # Everything is kept in the ALS canonical order.
        n = len(item_ids)
        if item_factors.shape[0] != n or tag_matrix.shape[0] != n or popularity.shape[0] != n:
            raise ValueError("All item-indexed arrays must share the same length.")

        self._item_ids = list(item_ids)
        self._id_to_col = {iid: i for i, iid in enumerate(self._item_ids)}
        self._factors = int(factors)
        self._alpha = float(alpha)
        self._reg = float(reg)
        self._iterations = int(iterations)

        self._tag_matrix = tag_matrix.tocsr()
        self._tag_vectorizer = tag_vectorizer
        self._popularity = np.asarray(popularity, dtype=np.int64)

        self._profile_builder = ProfileBuilder(
            item_factors=item_factors,
            item_id_to_col=self._id_to_col,
            tag_vectorizer=tag_vectorizer,
            tag_item_matrix=self._tag_matrix,
            alpha=self._alpha,
            reg=self._reg,
            favorite_weight=favorite_weight,
        )

        self._engine = PersonalizedEngine(
            item_factors=item_factors,
            item_tag_matrix=self._tag_matrix,
            item_ids=self._item_ids,
            popularity=self._popularity,
        )

        self._explainer = Explainer(
            item_tag_matrix=self._tag_matrix,
            item_id_to_col=self._id_to_col,
            item_tag_tokens=item_tag_tokens,
        )

        self._enricher: MetadataEnricher = enricher or NullEnricher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Run the full personalization pipeline for one request."""
        seeds = request.seeds or SeedInput()

        profile = self._profile_builder.build(
            item_ids=seeds.item_ids,
            favorite_ids=seeds.favorite_ids,
            tags=seeds.tags,
            fold_in_tag_seeds=max(0, int(request.fold_in_tag_seeds)),
        )

        candidates, fallback = self._engine.rank(
            profile=profile,
            n=request.n,
            content_weight=request.content_weight,
            novelty=request.novelty,
            diversity=request.diversity,
            candidate_pool=request.candidate_pool,
            exclude_ids=list(seeds.exclude_ids),
        )

        # Control echo (post-clipping) for the UI.
        control = {
            "novelty": float(np.clip(request.novelty, 0.0, 1.0)),
            "content_weight": float(np.clip(request.content_weight, 0.0, 1.0)),
            "diversity": float(np.clip(request.diversity, 0.0, 1.0)),
            "n": int(max(1, request.n)),
            "candidate_pool": int(max(request.n, request.candidate_pool)),
        }

        items = self._build_items(candidates, profile, control)

        resp = RecommendationResponse(
            request_id=request.request_id or str(uuid.uuid4()),
            items=items,
            seed_summary={
                "accepted_item_ids": profile.accepted_item_ids,
                "rejected_item_ids": profile.rejected_item_ids,
                "accepted_favorite_ids": profile.accepted_favorite_ids,
                "rejected_favorite_ids": profile.rejected_favorite_ids,
                "matched_tags": profile.matched_tags,
                "unknown_tags": profile.unknown_tags,
                "tag_fold_in_item_ids": profile.tag_fold_in_item_ids,
            },
            control=control,
            model_info={
                "name": self.MODEL_NAME,
                "factors": self._factors,
                "alpha": self._alpha,
                "reg": self._reg,
                "iterations": self._iterations,
                "n_items": len(self._item_ids),
            },
            fallback_used=fallback,
        )
        return resp

    # ------------------------------------------------------------------
    # Inspection helpers (useful for a "search items" UI)
    # ------------------------------------------------------------------

    def has_item(self, item_id: str) -> bool:
        return str(item_id) in self._id_to_col

    def catalogue_size(self) -> int:
        return len(self._item_ids)

    def vocabulary_size(self) -> int:
        return len(self._tag_vectorizer.vocabulary_)

    def item_tags(self, item_id: str) -> list[str]:
        """Return normalised tag tokens for an item (for the UI picker).

        Enrichment goes through the enricher; this is a raw pass-through.
        """
        md = self._enricher.enrich(str(item_id))
        tags = md.get("tags")
        if tags:
            return list(tags)
        # Fall back to explainer's internal cache.
        return list(self._explainer._tags.get(str(item_id), []))   # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        als_state_path: Optional[Path] = None,
        enricher: Optional[MetadataEnricher] = None,
        enrich_with_internal_features: bool = True,
    ) -> "RecommendationService":
        """Build a service from the on-disk preprocessing + ALS artefacts.

        If ``als_state_path`` is None, uses ``config.ALS_STATE_FILE``.
        If that file is missing, re-trains ALS on the fly. This keeps
        the demo a single command to start up, but a production
        deployment should always pre-train (``run_train_personalized.py``).

        If ``enricher`` is None and ``enrich_with_internal_features`` is
        True (default), a lightweight enricher is attached that fills
        ``tags`` + ``description`` per item from the preprocessed
        ``item_features`` DataFrame. A future external enricher
        (MusicBrainz, Last.fm, Spotify) can be passed here instead --
        the rest of the pipeline does not need to change.
        """
        arts = load_processed_artifacts()
        state = cls._load_or_train_als_state(als_state_path, arts)

        # ------------------------------------------------------------------
        # Align tag matrix row order with ALS column order
        # ------------------------------------------------------------------
        # ALS column order is sorted(train items as strings): "1", "10",
        # "100", ... -- it's lexicographic because the CF trainer does
        # sorted(unique()). The preprocessor's TF-IDF matrix uses
        # item_features.index order, which is numeric/arrival. Both are
        # the same *set* of 8,640 ids but different permutations, so we
        # compute a permutation once and keep everything in ALS order.
        tag_index_pos = {iid: i for i, iid in enumerate(arts.tfidf_item_index)}
        permutation = np.array(
            [tag_index_pos[iid] for iid in state["item_ids"]],
            dtype=np.int64,
        )
        # Reorder + L2-normalise the tag matrix so cosine == dot.
        tag_matrix = normalize(
            arts.tfidf_matrix[permutation, :],
            norm="l2",
            axis=1,
        ).tocsr()

        # Popularity (training counts) in ALS column order.
        popularity_series = arts.train["item_id_raw"].value_counts()
        popularity = np.array(
            [int(popularity_series.get(iid, 0)) for iid in state["item_ids"]],
            dtype=np.int64,
        )

        # Tag token lookup for the explainer.
        # The item_features index contains all catalogue items; any
        # missing ids (shouldn't happen in this dataset) default to [].
        tf = arts.item_features["tags_normalised"]
        item_tag_tokens: dict[str, list[str]] = {}
        for iid in state["item_ids"]:
            if iid in tf.index:
                item_tag_tokens[iid] = list(tf.loc[iid])
            else:
                item_tag_tokens[iid] = []

        # Default enricher: pull tags + description from item_features.
        if enricher is None and enrich_with_internal_features:
            enricher = InternalFeaturesEnricher(arts.item_features)

        return cls(
            item_ids=list(state["item_ids"]),
            item_factors=np.asarray(state["item_factors"], dtype=np.float32),
            tag_matrix=tag_matrix,
            tag_vectorizer=arts.tfidf_vectorizer,
            popularity=popularity,
            item_tag_tokens=item_tag_tokens,
            alpha=float(state["alpha"]),
            reg=float(state["reg"]),
            factors=int(state["factors"]),
            iterations=int(state["iterations"]),
            enricher=enricher,
        )

    @staticmethod
    def _load_or_train_als_state(
        path: Optional[Path],
        arts: ProcessedArtifacts,
    ) -> dict:
        path = Path(path) if path is not None else config.ALS_STATE_FILE
        if path.exists():
            log.info("Loading ALS state from %s", path)
            return CollaborativeFilteringRecommender.load_state(path)

        log.warning(
            "ALS state not found at %s — training on the fly. "
            "For production, run `python run_train_personalized.py` "
            "once up front.", path,
        )
        model = CollaborativeFilteringRecommender(
            factors=config.CF_FACTORS,
            regularization=config.CF_REGULARIZATION,
            iterations=config.CF_ITERATIONS,
            alpha=config.CF_ALPHA,
            random_state=config.SPLIT_SEED,
        )
        model.fit(arts.train, arts.item_features)
        model.save_state(path)
        return CollaborativeFilteringRecommender.load_state(path)

    # ------------------------------------------------------------------
    # Internals: candidate -> ScoredItem with explanations + metadata
    # ------------------------------------------------------------------

    def _build_items(
        self,
        candidates: list[_Candidate],
        profile: TasteProfile,
        control: dict[str, float],
    ) -> list[ScoredItem]:
        if not candidates:
            return []

        # Enrich in batch for efficiency (enrichers may hit a cache/API).
        enrichments = self._enricher.enrich_batch(c.item_id for c in candidates)

        items: list[ScoredItem] = []
        for rank, c in enumerate(candidates, start=1):
            ev = self._explainer.evidence_for(
                item_id=c.item_id,
                profile=profile,
                als_score_norm=c.als_norm,
                content_score_norm=c.content_norm,
                popularity_norm=c.popularity_norm,
            )
            explanation, reasons = self._explainer.render(ev, profile, control)

            items.append(ScoredItem(
                item_id=c.item_id,
                rank=rank,
                score=float(c.final),
                score_breakdown={
                    "als": float(c.als_norm),
                    "content": float(c.content_norm),
                    "blended": float(c.blended),
                    "popularity": float(c.popularity_norm),
                    "novelty_penalty": float(c.novelty_penalty),
                    "raw": float(c.blended - c.novelty_penalty),
                    "als_raw": float(c.als_raw),
                    "content_raw": float(c.content_raw),
                    "popularity_count": int(
                        self._popularity[self._id_to_col[c.item_id]]
                    ),
                },
                explanation=explanation,
                reasons=reasons,
                metadata=enrichments.get(c.item_id, {}) or {},
            ))
        return items
