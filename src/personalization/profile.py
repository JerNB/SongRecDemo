"""
Build an interactive user's taste profile from seed inputs.

Given a :class:`SeedInput` (item_ids, favorite_ids, tags), produce:

  1. An ALS latent vector ``x`` obtained by a single closed-form
     fold-in solve against the trained item factors -- this is the
     same per-user update HKV 2008 uses during training, so the new
     user lives in the same latent space as the 5,199 training users
     without retraining.
  2. A sparse tag-TF-IDF taste vector ``t`` (L2-normalised), obtained
     by averaging the L2-normalised tag vectors of the accepted
     seed items and/or by vectorising the free-form tag tokens the
     user typed.

Both are returned in :class:`TasteProfile`. Whichever component is
empty (e.g. "tags-only" or "items-only" users) is returned as None
and the engine gracefully falls back.

ALS fold-in derivation
----------------------
The HKV 2008 user update, for a user with observed-item set
:math:`\\Omega` and weights :math:`w_i` (all > 0):

    c_i = alpha * w_i                            (confidence shift)
    A   = Y^T Y + sum_{i in Omega} c_i * y_i y_i^T + lambda * I
    b   = sum_{i in Omega} (1 + c_i) * y_i
    x   = A^{-1} b

For uniform :math:`w_i = 1` this is exactly the normal-equation
closed form from the project's CF trainer (see
``src/recommenders/collaborative.py::_als_half_step``).

A "favourite" seed is a seed with a larger weight (default 2.0),
which raises its c_i and therefore pulls ``x`` more strongly toward
that item's direction in latent space.

Tag input handling
------------------
Free-form tag tokens are normalised with the same rules the
preprocessor uses (lower-case, strip punctuation, collapse
hyphens). Anything in the trained TF-IDF vocabulary goes through
the fitted vectoriser; unknown tokens are reported back to the
caller in :class:`TasteProfile.unknown_tags` so the UI can show a
"we don't know this tag" hint.

When the user supplies only tags (no seed items), the tag vector is
used as a query into the tag-TF-IDF item matrix; the top-K matching
items are inserted as *pseudo-seeds* for the ALS fold-in. That gives
a tag-only query a sensible ALS presence without requiring the user
to pick a song.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.data.preprocessor import _normalise_tag  # reuse exact preprocessing rules

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class TasteProfile:
    """Everything the engine needs to score candidates for one user."""

    # ALS latent vector (d,) or None if fold-in could not be performed
    # (no seeds, no tag matches).
    user_latent: Optional[np.ndarray]

    # L2-normalised tag-TF-IDF row (1, n_features) or None.
    tag_profile: Optional[sp.csr_matrix]

    # Bookkeeping for explanations + UI
    accepted_item_ids: list[str] = field(default_factory=list)
    rejected_item_ids: list[str] = field(default_factory=list)
    accepted_favorite_ids: list[str] = field(default_factory=list)
    rejected_favorite_ids: list[str] = field(default_factory=list)
    matched_tags: list[str] = field(default_factory=list)
    unknown_tags: list[str] = field(default_factory=list)
    # Items injected as pseudo-seeds from the tag query
    tag_fold_in_item_ids: list[str] = field(default_factory=list)
    # Per-accepted-item confidence weight (used by the engine when
    # building explanations that rank the influence of seeds).
    seed_weights: dict[str, float] = field(default_factory=dict)

    @property
    def has_any_signal(self) -> bool:
        return self.user_latent is not None or self.tag_profile is not None


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

class ProfileBuilder:
    """Turn user-supplied seeds / favourites / tags into a TasteProfile.

    Parameters
    ----------
    item_factors : np.ndarray, shape (n_items, d)
        The trained ALS item factor matrix in canonical column order.
    item_id_to_col : dict[str, int]
        Item-id -> column index in ``item_factors`` and the aligned
        tag TF-IDF matrix.
    tag_vectorizer : sklearn TfidfVectorizer
        The fitted vectoriser used by the preprocessor to build the
        item tag matrix. Used here to vectorise free-form user tag
        input in the same feature space.
    tag_item_matrix : sp.csr_matrix, shape (n_items, n_features)
        The item-tag TF-IDF matrix, aligned row-for-row to
        ``item_factors``. Must be L2-normalised row-wise (the engine
        expects cosine == dot).
    alpha : float
        ALS confidence scale (same as trained).
    reg : float
        ALS ridge (same as trained).
    favorite_weight : float
        Per-seed weight applied to items in ``favorite_ids``. Default
        2.0 — a favourite counts double a casual seed in both ALS
        fold-in (via alpha * weight) and tag-profile averaging.
    """

    def __init__(
        self,
        item_factors: np.ndarray,
        item_id_to_col: dict[str, int],
        tag_vectorizer,
        tag_item_matrix: sp.csr_matrix,
        alpha: float,
        reg: float,
        favorite_weight: float = 2.0,
    ) -> None:
        if item_factors.shape[0] != tag_item_matrix.shape[0]:
            raise ValueError(
                f"item_factors has {item_factors.shape[0]} rows but "
                f"tag_item_matrix has {tag_item_matrix.shape[0]} rows. "
                "They must be aligned row-for-row."
            )
        self._Y = item_factors.astype(np.float64, copy=False)
        self._item_id_to_col = dict(item_id_to_col)
        self._d = item_factors.shape[1]
        self._alpha = float(alpha)
        self._reg = float(reg)
        self._fav_weight = float(favorite_weight)

        self._tag_vectorizer = tag_vectorizer
        self._tag_vocab: dict[str, int] = dict(tag_vectorizer.vocabulary_)
        self._tag_item_matrix = tag_item_matrix.tocsr()

        # Precompute Y^T Y once; identical to the training-time cache.
        self._YtY = (self._Y.T @ self._Y)
        self._regI = self._reg * np.eye(self._d, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        item_ids: list[str],
        favorite_ids: list[str],
        tags: list[str],
        fold_in_tag_seeds: int = 25,
    ) -> TasteProfile:
        """Construct a :class:`TasteProfile` for one interactive user.

        Steps
        -----
        1. Validate seed / favourite ids against the catalogue.
        2. Normalise free-form tags; drop tokens outside the trained
           vocabulary. Build the user's tag-vector ``t_user``.
        3. If ``tags`` yielded a non-empty ``t_user`` and the seed
           pool is smaller than ``fold_in_tag_seeds``, pick the top
           matching catalogue items as pseudo-seeds and add them to
           the seed pool (weight 1.0) so the ALS fold-in has signal.
        4. ALS fold-in: closed-form normal-equation solve.
        5. Content profile: L2-normalised mean of accepted-item tag
           rows, optionally blended with ``t_user``.
        """
        prof = TasteProfile(user_latent=None, tag_profile=None)

        # -------------- 1. Validate seeds against catalogue --------------
        acc_items, rej_items = self._partition_known(item_ids)
        acc_favs, rej_favs = self._partition_known(favorite_ids)
        prof.accepted_item_ids = acc_items
        prof.rejected_item_ids = rej_items
        prof.accepted_favorite_ids = acc_favs
        prof.rejected_favorite_ids = rej_favs

        # Favourites take precedence: if the same id appears in both
        # lists, it ends up as a favourite (higher weight).
        fav_set = set(acc_favs)
        plain_seeds = [i for i in acc_items if i not in fav_set]

        # -------------- 2. Tag vectorisation ---------------------------
        matched_tags, unknown_tags, t_user = self._vectorise_tags(tags)
        prof.matched_tags = matched_tags
        prof.unknown_tags = unknown_tags

        # -------------- 3. Tag -> pseudo-seed fold-in ------------------
        pseudo_seeds: list[str] = []
        if t_user is not None and (len(plain_seeds) + len(acc_favs)) < fold_in_tag_seeds:
            k = max(0, fold_in_tag_seeds - len(plain_seeds) - len(acc_favs))
            pseudo_seeds = self._top_items_for_tag_vector(
                t_user,
                k=k,
                exclude=fav_set.union(plain_seeds),
            )
            prof.tag_fold_in_item_ids = pseudo_seeds

        # -------------- 4. Assemble weighted observation set -----------
        weighted: list[tuple[str, float]] = []
        weighted.extend((iid, self._fav_weight) for iid in acc_favs)
        weighted.extend((iid, 1.0) for iid in plain_seeds)
        weighted.extend((iid, 0.5) for iid in pseudo_seeds)  # weaker than real seeds
        prof.seed_weights = {iid: w for iid, w in weighted}

        # -------------- 5. ALS fold-in ---------------------------------
        if weighted:
            prof.user_latent = self._als_fold_in(weighted)

        # -------------- 6. Tag profile vector --------------------------
        prof.tag_profile = self._build_tag_profile(
            accepted_item_ids=[iid for iid, _ in weighted],
            user_tag_vector=t_user,
        )

        return prof

    # ------------------------------------------------------------------
    # ID validation
    # ------------------------------------------------------------------

    def _partition_known(self, ids: list[str]) -> tuple[list[str], list[str]]:
        """Split ids into (accepted-in-catalogue, rejected-unknown)."""
        accepted: list[str] = []
        rejected: list[str] = []
        seen: set[str] = set()
        for raw in ids:
            # Normalise type: callers may pass int; catalogue ids are str.
            key = str(raw)
            if key in seen:
                continue
            seen.add(key)
            if key in self._item_id_to_col:
                accepted.append(key)
            else:
                rejected.append(key)
        return accepted, rejected

    # ------------------------------------------------------------------
    # Tag -> vector
    # ------------------------------------------------------------------

    def _vectorise_tags(
        self,
        tags: list[str],
    ) -> tuple[list[str], list[str], Optional[sp.csr_matrix]]:
        """Normalise and vectorise free-form tag input.

        Returns
        -------
        matched_tags : list of tags that survived normalisation AND
            appear in the trained vocabulary.
        unknown_tags : list of user-entered tokens that either
            normalised to empty or fell outside the vocabulary.
        vec : sparse (1, n_features) L2-normalised tag vector, or
            None if no tags matched.
        """
        matched: list[str] = []
        unknown: list[str] = []
        for raw in tags:
            if raw is None:
                continue
            norm = _normalise_tag(str(raw))
            if not norm:
                unknown.append(str(raw))
                continue
            if norm in self._tag_vocab:
                matched.append(norm)
            else:
                unknown.append(str(raw))

        if not matched:
            return matched, unknown, None

        # Join with spaces so TfidfVectorizer's default tokeniser (\w+)
        # produces the same tokens we just validated. We do NOT want
        # the vectoriser to re-tokenise a multi-word tag into parts,
        # so we also build an explicit-feature override: directly
        # index into the vocabulary and set equal-weight entries, then
        # L2-normalise. This avoids surprises from multi-word tags
        # like "indie pop" that the vectoriser would have split.
        n_features = len(self._tag_vocab)
        idxs = np.array([self._tag_vocab[t] for t in matched], dtype=np.int64)
        data = np.ones(len(idxs), dtype=np.float32)
        # If a tag is repeated by the user, fold its weight additively
        # before normalisation (emulating "emphasis").
        row = sp.csr_matrix(
            (data, (np.zeros_like(idxs), idxs)),
            shape=(1, n_features),
            dtype=np.float32,
        )
        row.sum_duplicates()
        vec = normalize(row, norm="l2").tocsr()
        return matched, unknown, vec

    def _top_items_for_tag_vector(
        self,
        t: sp.csr_matrix,
        k: int,
        exclude: set[str],
    ) -> list[str]:
        """Find the k items most similar to tag vector ``t`` (cosine).

        Both sides are row-L2-normalised, so cosine == dot product.
        """
        if k <= 0:
            return []
        scores = np.asarray((t @ self._tag_item_matrix.T).todense()).ravel()

        # Mask excluded items out of consideration.
        if exclude:
            for iid in exclude:
                col = self._item_id_to_col.get(iid)
                if col is not None:
                    scores[col] = -np.inf

        # Ignore zero-score items (no overlap with user's tags)
        mask = scores > 0
        if not mask.any():
            return []

        top_k = min(k, int(mask.sum()))
        part = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        order = part[np.argsort(-scores[part])]
        col_to_id = {c: i for i, c in self._item_id_to_col.items()}
        return [col_to_id[int(c)] for c in order]

    # ------------------------------------------------------------------
    # ALS fold-in (HKV 2008 closed form)
    # ------------------------------------------------------------------

    def _als_fold_in(self, weighted: list[tuple[str, float]]) -> np.ndarray:
        """Solve for the user's latent vector given a set of weighted seeds.

            A = Y^T Y + sum_i (alpha * w_i) * y_i y_i^T + lambda * I
            b = sum_i (1 + alpha * w_i) * y_i
            x = A^{-1} b
        """
        cols = np.array(
            [self._item_id_to_col[iid] for iid, _ in weighted], dtype=np.int64
        )
        weights = np.array([w for _, w in weighted], dtype=np.float64)
        conf = self._alpha * weights                         # c_i

        Y_obs = self._Y[cols]                                # (|Omega|, d)
        # Y_Omega^T diag(c) Y_Omega = (c * Y_obs).T @ Y_obs
        A = self._YtY + (conf[:, None] * Y_obs).T @ Y_obs + self._regI
        b = ((1.0 + conf)[:, None] * Y_obs).sum(axis=0)      # (d,)

        x = np.linalg.solve(A, b)
        return x.astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # Tag taste profile
    # ------------------------------------------------------------------

    def _build_tag_profile(
        self,
        accepted_item_ids: list[str],
        user_tag_vector: Optional[sp.csr_matrix],
    ) -> Optional[sp.csr_matrix]:
        """Combine item-derived and user-typed tag evidence.

        If both sources are available, they are averaged equally,
        then L2-normalised (averaging two L2-normalised rows is not
        itself normalised).
        """
        parts: list[sp.csr_matrix] = []
        if accepted_item_ids:
            rows = [self._item_id_to_col[iid] for iid in accepted_item_ids]
            item_rows = self._tag_item_matrix[rows]           # already L2-normalised
            if item_rows.nnz > 0:
                mean_row = sp.csr_matrix(
                    item_rows.mean(axis=0)
                )
                parts.append(mean_row)
        if user_tag_vector is not None:
            parts.append(user_tag_vector)

        if not parts:
            return None

        combined = parts[0]
        for part in parts[1:]:
            combined = combined + part
        combined = normalize(combined, norm="l2").tocsr()
        if combined.nnz == 0:
            return None
        return combined
