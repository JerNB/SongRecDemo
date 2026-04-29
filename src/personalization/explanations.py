"""
Human-readable explanations for why each item was recommended.

The explanation text is the one thing an end-user actually reads, so
it has to be grounded in real evidence, not generic ("you might like
this!"). Each explanation draws on three facts we actually have:

1. The single seed (or favourite) with the highest cosine similarity
   in tag space to the recommended item -- the "because you liked X"
   anchor.
2. The overlap between the recommended item's tags and the user's
   dominant tags (from their tag profile). Up to three tags are
   reported.
3. A popularity annotation -- whether the item is a known hit or a
   long-tail pick. This is factual, not guessed.

If the user has NO tag profile (tag vector is empty) we fall back to
an ALS-flavoured explanation that is honest about being opaque
("ALS predicts a strong latent-taste match ...").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from src.personalization.profile import TasteProfile


@dataclass(frozen=True)
class ExplanationInputs:
    """All the precomputed bits the explainer needs for one item."""

    item_id: str
    als_score_norm: float
    content_score_norm: float
    popularity_norm: float
    tag_overlap: list[str]           # up to 3 shared tags
    top_seed_id: Optional[str]       # nearest-neighbour seed in tag space
    top_seed_similarity: float       # cosine in [0,1]


class Explainer:
    """Stitches grounded evidence into a one-sentence explanation.

    Parameters
    ----------
    item_tag_matrix : sp.csr_matrix
        Row-L2-normalised item tag matrix. Used to find the seed most
        similar (in tag space) to the candidate item.
    item_id_to_col : dict[str, int]
        Column lookup aligned to ``item_tag_matrix`` rows.
    item_tag_tokens : dict[str, list[str]]
        Raw normalised tag tokens per item (for surface-string
        "shared tags" display). The engine fills this in from
        ``item_features``.
    popular_threshold : float
        Items whose normalised popularity is >= this threshold are
        called out as "popular"; others as "lesser-known / long-tail".
    """

    def __init__(
        self,
        item_tag_matrix: sp.csr_matrix,
        item_id_to_col: dict[str, int],
        item_tag_tokens: dict[str, list[str]],
        popular_threshold: float = 0.25,
    ) -> None:
        self._M = item_tag_matrix
        self._col = dict(item_id_to_col)
        self._tags = item_tag_tokens
        self._pop_thresh = float(popular_threshold)

    # ------------------------------------------------------------------
    # Per-item evidence assembly
    # ------------------------------------------------------------------

    def evidence_for(
        self,
        item_id: str,
        profile: TasteProfile,
        als_score_norm: float,
        content_score_norm: float,
        popularity_norm: float,
    ) -> ExplanationInputs:
        # Shared tags with the user's aggregate taste (top-3 by user-
        # profile weight). If the user typed tags, those dominate.
        overlap = self._shared_tags(item_id, profile, max_tags=3)

        # Nearest seed in tag space (the "because you liked X" anchor).
        top_seed, top_sim = self._nearest_seed(item_id, profile)

        return ExplanationInputs(
            item_id=item_id,
            als_score_norm=float(als_score_norm),
            content_score_norm=float(content_score_norm),
            popularity_norm=float(popularity_norm),
            tag_overlap=overlap,
            top_seed_id=top_seed,
            top_seed_similarity=float(top_sim),
        )

    # ------------------------------------------------------------------
    # Render evidence into a short string + structured reasons
    # ------------------------------------------------------------------

    def render(
        self,
        ev: ExplanationInputs,
        profile: TasteProfile,
        control: dict[str, float],
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []

        if ev.top_seed_id is not None and ev.top_seed_similarity > 0.01:
            if ev.top_seed_id in profile.accepted_favorite_ids:
                anchor = f"your favourited track {ev.top_seed_id}"
            elif ev.top_seed_id in profile.tag_fold_in_item_ids:
                anchor = f"track {ev.top_seed_id} (auto-picked from your tags)"
            else:
                anchor = f"your seed track {ev.top_seed_id}"
            reasons.append(
                f"tag-space cosine {ev.top_seed_similarity:.2f} to {anchor}"
            )

        if ev.tag_overlap:
            reasons.append("shared tags: " + ", ".join(ev.tag_overlap))

        if ev.popularity_norm >= self._pop_thresh:
            reasons.append(
                f"popular pick (popularity {ev.popularity_norm:.2f} on [0,1])"
            )
        else:
            reasons.append(
                f"long-tail pick (popularity {ev.popularity_norm:.2f})"
            )

        # Expose how the blend voted so a debug UI can show it.
        reasons.append(
            f"ALS {ev.als_score_norm:.2f} / content {ev.content_score_norm:.2f} "
            f"(blend w={control.get('content_weight', 0.0):.2f})"
        )

        # One-sentence human summary.
        has_profile_signal = bool(
            profile.accepted_item_ids
            or profile.accepted_favorite_ids
            or profile.matched_tags
            or profile.tag_fold_in_item_ids
        )

        if ev.tag_overlap and ev.top_seed_id is not None:
            sentence = (
                f"Recommended because it shares the tags "
                f"[{', '.join(ev.tag_overlap)}] with your taste profile "
                f"and sits close (cos={ev.top_seed_similarity:.2f}) to "
                f"track {ev.top_seed_id} in tag space."
            )
        elif ev.top_seed_id is not None:
            sentence = (
                f"Recommended because its latent factors align with "
                f"track {ev.top_seed_id} in your seed set "
                f"(ALS match score {ev.als_score_norm:.2f})."
            )
        elif ev.tag_overlap:
            sentence = (
                f"Recommended because it shares the tags "
                f"[{', '.join(ev.tag_overlap)}] with the tags you selected."
            )
        elif not has_profile_signal:
            # Cold-start fallback: no seeds, no tags, no latent signal --
            # this pick came from the popularity ranker.
            sentence = (
                f"Shown as a popular starter pick "
                f"(popularity {ev.popularity_norm:.2f} on [0,1]) while "
                f"we have no taste signal from you yet. Seed a song or "
                f"add a tag to personalise your list."
            )
            # Swap the debug "ALS/content (blend ...)" line for the
            # same popularity story so the reasons stay honest.
            reasons = [
                f"cold-start popularity fallback "
                f"(pop={ev.popularity_norm:.2f})",
            ]
        else:
            sentence = (
                f"Recommended by ALS (latent-taste match "
                f"score {ev.als_score_norm:.2f})."
            )
        return sentence, reasons

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shared_tags(
        self,
        item_id: str,
        profile: TasteProfile,
        max_tags: int,
    ) -> list[str]:
        item_tags = self._tags.get(item_id, [])
        if not item_tags:
            return []

        # Compare against (a) the user's typed tags and
        # (b) the tags of the user's explicit seeds + favourites.
        user_tag_set: set[str] = set(profile.matched_tags)
        for iid in profile.accepted_item_ids + profile.accepted_favorite_ids:
            user_tag_set.update(self._tags.get(iid, [])[:10])

        if not user_tag_set:
            return []

        # Preserve the candidate item's own tag order (more salient
        # tags come first in our normalised lists).
        overlap = [t for t in item_tags if t in user_tag_set]
        return overlap[:max_tags]

    def _nearest_seed(
        self,
        item_id: str,
        profile: TasteProfile,
    ) -> tuple[Optional[str], float]:
        # Candidate seeds: real seeds + favourites first (prefer these
        # in the explanation), then tag-fold-in pseudo-seeds.
        seed_ids = (
            profile.accepted_favorite_ids
            + [i for i in profile.accepted_item_ids
               if i not in profile.accepted_favorite_ids]
            + profile.tag_fold_in_item_ids
        )
        if not seed_ids or item_id not in self._col:
            return None, 0.0

        item_col = self._col[item_id]
        item_row = self._M[item_col]
        if item_row.nnz == 0:
            return None, 0.0

        best_id: Optional[str] = None
        best_sim = -1.0
        for sid in seed_ids:
            col = self._col.get(sid)
            if col is None:
                continue
            s = float((item_row @ self._M[col].T).toarray().ravel()[0])
            if s > best_sim:
                best_sim = s
                best_id = sid
        return best_id, max(best_sim, 0.0)
