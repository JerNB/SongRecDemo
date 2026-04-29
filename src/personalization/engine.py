"""
Scoring, blending, novelty control, and MMR diversification.

This is the core of the personalization layer. Given a
:class:`TasteProfile` and a :class:`RecommendationRequest`, produce a
ranked list of item ids with score components attached. The engine is
deliberately model-aware (it knows about ALS factors and the tag
matrix) but UI-agnostic -- wrapping it as a service, packaging it
into dataclasses, and running the explainer all happen in
``service.py``.

Pipeline at inference time
--------------------------

::

    TasteProfile
        |
        v
    ALS score s_a(i) = x . y_i       (for all items; 0 if x is None)
    Content score s_c(i) = t . m_i   (for all items; 0 if t is None)
        |
        v
    Min-max normalise each across the catalogue  -> s_a', s_c'  in [0, 1]
        |
        v
    Blend:   s_b(i) = (1 - w) * s_a' + w * s_c'
             degenerate: if only one source is present, use it alone
             regardless of ``content_weight``.
        |
        v
    Novelty demotion:
             s_f(i) = s_b(i) - lambda_nov * popularity_norm(i)
        |
        v
    Mask: drop items in (seeds U favourites U exclude_ids) and
          items that are NaN / -inf in ALS score (zero-vector items
          with no ALS exposure get a content-only contribution).
        |
        v
    Top-M candidate pool (default 500), full sort.
        |
        v
    MMR diversity rerank in tag-TF-IDF space (if lambda_div > 0):
             next_i = argmax (1 - ld) * s_f(i) - ld * max_{j selected} cos(i, j)
        |
        v
    Take first N.

Cold-start fallback
-------------------
If the TasteProfile has no ALS vector AND no tag vector, the engine
returns the popularity baseline ranking (seen-item filter applied)
and sets ``fallback_used = "popularity_cold_start"`` so the UI can
prompt the user to enter something.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

from src.personalization.profile import TasteProfile

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal candidate record
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    col: int                       # index into item_factors / tag_matrix
    item_id: str
    als_raw: float
    als_norm: float
    content_raw: float
    content_norm: float
    popularity_norm: float
    blended: float
    novelty_penalty: float
    final: float                   # pre-MMR score


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PersonalizedEngine:
    """Retrieval + rerank + diversity for one interactive request.

    Parameters
    ----------
    item_factors : np.ndarray, shape (n_items, d)
    item_tag_matrix : sp.csr_matrix, shape (n_items, n_features), L2-normalised
    item_ids : list[str]
        Row order of both matrices above (canonical catalogue).
    popularity : np.ndarray, shape (n_items,)
        Raw training-count per item, aligned to ``item_ids``. Used to
        compute the popularity penalty and fallback ranking.
    """

    def __init__(
        self,
        item_factors: np.ndarray,
        item_tag_matrix: sp.csr_matrix,
        item_ids: list[str],
        popularity: np.ndarray,
    ) -> None:
        n = item_factors.shape[0]
        if item_tag_matrix.shape[0] != n or len(item_ids) != n or popularity.shape[0] != n:
            raise ValueError(
                "item_factors, item_tag_matrix, item_ids, and popularity "
                "must all share the same leading dimension."
            )
        self._Y = item_factors.astype(np.float32, copy=False)
        self._M = item_tag_matrix.tocsr()
        self._ids = list(item_ids)
        self._id_to_col = {iid: c for c, iid in enumerate(self._ids)}

        # Popularity: normalise log-count to [0, 1] for a bounded penalty.
        # Using log1p keeps the penalty sensitive at the low end without
        # letting the one or two super-hits dominate.
        pop = np.asarray(popularity, dtype=np.float64)
        log_pop = np.log1p(np.maximum(pop, 0.0))
        denom = float(log_pop.max()) if log_pop.max() > 0 else 1.0
        self._pop_norm = (log_pop / denom).astype(np.float32)
        self._popularity_raw = pop.astype(np.int64)

    # ------------------------------------------------------------------
    # Score pipeline
    # ------------------------------------------------------------------

    def rank(
        self,
        profile: TasteProfile,
        n: int,
        content_weight: float,
        novelty: float,
        diversity: float,
        candidate_pool: int,
        exclude_ids: list[str],
    ) -> tuple[list[_Candidate], Optional[str]]:
        """Return at most ``n`` candidates, ranked. Second element is
        the fallback tag or ``None`` if the normal pipeline ran."""
        # Clip knobs into their documented range.
        cw = float(np.clip(content_weight, 0.0, 1.0))
        nv = float(np.clip(novelty, 0.0, 1.0))
        dv = float(np.clip(diversity, 0.0, 1.0))
        n = max(1, int(n))
        pool_size = max(n, int(candidate_pool))

        # ------------------------------------------------------------------
        # Raw score vectors
        # ------------------------------------------------------------------
        n_items = self._Y.shape[0]

        als_raw: Optional[np.ndarray] = None
        if profile.user_latent is not None:
            als_raw = (self._Y @ profile.user_latent.astype(np.float32)).astype(
                np.float32, copy=False
            )

        content_raw: Optional[np.ndarray] = None
        if profile.tag_profile is not None and profile.tag_profile.nnz > 0:
            content_raw = np.asarray(
                (profile.tag_profile @ self._M.T).todense()
            ).ravel().astype(np.float32)

        fallback: Optional[str] = None
        if als_raw is None and content_raw is None:
            # Cold-start: no signal at all. Return popularity ranking.
            fallback = "popularity_cold_start"
            return (
                self._popularity_fallback(n, exclude_ids, profile),
                fallback,
            )

        # ------------------------------------------------------------------
        # Min-max normalise to [0, 1] so the blend is scale-comparable.
        # We normalise over the non-masked items only, so seen items
        # don't drag the scale.
        # ------------------------------------------------------------------
        mask = self._build_mask(profile, exclude_ids)   # True = valid candidate
        if not mask.any():
            return [], "all_candidates_excluded"

        als_norm = self._minmax(als_raw, mask) if als_raw is not None else np.zeros(n_items, np.float32)
        content_norm = self._minmax(content_raw, mask) if content_raw is not None else np.zeros(n_items, np.float32)

        # ------------------------------------------------------------------
        # Blend + novelty. Degenerate cases: if only one source is
        # present, that source gets weight 1.0 regardless of ``cw``.
        # (Otherwise a user with only tags but content_weight=0 would
        # get an all-zero blended score, which is nonsense.)
        # ------------------------------------------------------------------
        if als_raw is not None and content_raw is not None:
            blended = (1.0 - cw) * als_norm + cw * content_norm
            effective_cw = cw
            if profile.tag_fold_in_item_ids and not profile.accepted_item_ids and not profile.accepted_favorite_ids:
                # Tag-only profile already influences ALS via fold-in;
                # do not double-count by pushing cw further up here.
                fallback = "tag_only_fold_in"
        elif als_raw is not None:
            blended = als_norm
            effective_cw = 0.0
        else:
            blended = content_norm
            effective_cw = 1.0

        novelty_pen = nv * self._pop_norm
        final = blended - novelty_pen

        # Apply mask as a hard -inf so masked items never surface.
        final = np.where(mask, final, -np.inf).astype(np.float32)

        # ------------------------------------------------------------------
        # Candidate pool by final score (full sort on pool size only).
        # ------------------------------------------------------------------
        pool_size = min(pool_size, int(mask.sum()))
        if pool_size <= 0:
            return [], "all_candidates_excluded"

        part = np.argpartition(-final, kth=pool_size - 1)[:pool_size]
        part_order = part[np.argsort(-final[part])]
        pool_cols = part_order.tolist()

        # Materialise _Candidate records
        candidates: list[_Candidate] = []
        for c in pool_cols:
            candidates.append(_Candidate(
                col=int(c),
                item_id=self._ids[int(c)],
                als_raw=float(als_raw[c]) if als_raw is not None else 0.0,
                als_norm=float(als_norm[c]),
                content_raw=float(content_raw[c]) if content_raw is not None else 0.0,
                content_norm=float(content_norm[c]),
                popularity_norm=float(self._pop_norm[c]),
                blended=float(blended[c]),
                novelty_penalty=float(novelty_pen[c]),
                final=float(final[c]),
            ))

        # ------------------------------------------------------------------
        # MMR diversity (optional)
        # ------------------------------------------------------------------
        if dv > 0.0 and len(candidates) > n:
            candidates = self._mmr(candidates, n=n, lam=dv)
        else:
            candidates = candidates[:n]

        return candidates, fallback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_mask(
        self,
        profile: TasteProfile,
        exclude_ids: list[str],
    ) -> np.ndarray:
        """True for items that are allowed to surface as recommendations."""
        mask = np.ones(self._Y.shape[0], dtype=bool)

        # Drop everything the user has already told us about.
        to_drop: set[str] = set()
        to_drop.update(profile.accepted_item_ids)
        to_drop.update(profile.accepted_favorite_ids)
        # Tag-fold-in pseudo-seeds: also drop, otherwise we would be
        # "recommending" items we just used as the seed set.
        to_drop.update(profile.tag_fold_in_item_ids)
        # Explicit user exclusions (thumbs-down, seen-recently, etc.)
        to_drop.update(str(i) for i in exclude_ids)

        for iid in to_drop:
            col = self._id_to_col.get(iid)
            if col is not None:
                mask[col] = False
        return mask

    @staticmethod
    def _minmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Min-max scale ``x`` to [0, 1] using only mask=True entries.

        Masked entries keep their scaled value (we compute it anyway;
        the mask application happens later). If the valid range is
        degenerate, returns zeros.
        """
        valid = x[mask]
        if valid.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        lo = float(valid.min())
        hi = float(valid.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    # ------------------------------------------------------------------
    # MMR (Carbonell & Goldstein 1998)
    # ------------------------------------------------------------------

    def _mmr(
        self,
        cand: list[_Candidate],
        n: int,
        lam: float,
    ) -> list[_Candidate]:
        """Greedy Maximal Marginal Relevance in tag-TF-IDF space.

        MMR_score(i) = (1 - lam) * final(i) - lam * max_{j in S} cos(i, j)

        The pool is already sorted by ``final`` desc on entry, so the
        first pick is trivially the top candidate.

        Note: computed on the candidate pool only (~500 rows), so the
        inner loop's cost is O(n * pool). The tag matrix is already
        L2-normalised, so cosine == dot product.
        """
        selected_cols: list[int] = []
        selected_idx_in_pool: list[int] = []
        remaining = list(range(len(cand)))

        # Precompute pool feature submatrix for fast partial sims.
        pool_cols = np.array([c.col for c in cand], dtype=np.int64)
        pool_M = self._M[pool_cols]                          # sparse

        # sim_to_selected[i] = max cosine between pool item i and any
        # already-selected item. Updated incrementally.
        sim_to_selected = np.zeros(len(cand), dtype=np.float32)
        finals = np.array([c.final for c in cand], dtype=np.float32)

        while remaining and len(selected_idx_in_pool) < n:
            if not selected_idx_in_pool:
                best = remaining[0]   # top-scored candidate
            else:
                rem_arr = np.array(remaining, dtype=np.int64)
                scores = (1.0 - lam) * finals[rem_arr] - lam * sim_to_selected[rem_arr]
                best = int(rem_arr[int(np.argmax(scores))])

            selected_idx_in_pool.append(best)
            selected_cols.append(cand[best].col)
            remaining.remove(best)

            # Update running max-sim between pool items and selected set.
            just_added_row = pool_M[best]
            new_sims = np.asarray(
                (pool_M @ just_added_row.T).todense()
            ).ravel().astype(np.float32)
            sim_to_selected = np.maximum(sim_to_selected, new_sims)

        return [cand[i] for i in selected_idx_in_pool]

    # ------------------------------------------------------------------
    # Cold-start fallback
    # ------------------------------------------------------------------

    def _popularity_fallback(
        self,
        n: int,
        exclude_ids: list[str],
        profile: TasteProfile,
    ) -> list[_Candidate]:
        """Return top-n popular items (cold-start, no profile signal)."""
        mask = self._build_mask(profile, exclude_ids)
        pop_score = self._pop_norm.copy()
        pop_score[~mask] = -np.inf
        pool_size = min(n, int(mask.sum()))
        if pool_size <= 0:
            return []
        part = np.argpartition(-pop_score, kth=pool_size - 1)[:pool_size]
        order = part[np.argsort(-pop_score[part])]
        out: list[_Candidate] = []
        for c in order:
            out.append(_Candidate(
                col=int(c),
                item_id=self._ids[int(c)],
                als_raw=0.0,
                als_norm=0.0,
                content_raw=0.0,
                content_norm=0.0,
                popularity_norm=float(self._pop_norm[c]),
                blended=float(self._pop_norm[c]),
                novelty_penalty=0.0,
                final=float(self._pop_norm[c]),
            ))
        return out
