"""Reranker -- stage 6 of the pipeline.

Greedy MMR diversification with dynamic per-artist / per-album caps. The
crucial contract: each chosen item keeps **two** scores -- ``final_score``
(standalone per-song relevance, shown to the user) and ``rank_score`` /
``mmr_objective`` (the MMR objective that decided list *position*). The
returned list is ordered by ``rank_score``; ``final_score`` is untouched.
"""

from __future__ import annotations

import math
from typing import Any

from .text import _clip01, _cosine, _jaccard, _norm_artist, _norm_title, _source_preference_tokens
from .types import Candidate, RealSongRequest, UserProfile


class Reranker:
    """MMR rerank: ``(1 - lambda) * score - lambda * max_sim_to_picked``."""

    def rerank(
        self,
        scored: list[tuple[float, Candidate, dict[str, float]]],
        profile: UserProfile,
        req: RealSongRequest,
    ) -> list[tuple[int, Candidate, dict[str, float], dict[str, Any]]]:
        k = max(1, min(50, int(req.k)))
        lam = _clip01(req.diversity)
        if not scored:
            return []

        remaining = list(scored)
        chosen: list[tuple[int, Candidate, dict[str, float], dict[str, Any]]] = []
        artist_count: dict[str, int] = {}
        album_count: dict[str, int] = {}
        artist_cap = 1 if k <= 5 else (2 if k <= 10 else 3)
        album_cap = 2 if k <= 10 else 3

        # Track whether a card was originally below the top-k by score
        # but got pulled in by MMR; that flag drives the "diverse"
        # pick label.
        score_only_topk = {id(t[1]) for t in remaining[:k]}

        rank = 1
        while remaining and len(chosen) < k:
            best_idx = -1
            best_obj = -math.inf
            for i, (sc, cand, _bd) in enumerate(remaining):
                primary_artist = _norm_artist(cand.track.artist or
                                              (cand.track.artists[0] if cand.track.artists else ""))
                if primary_artist and artist_count.get(primary_artist, 0) >= artist_cap:
                    continue
                album_key = _norm_title(cand.track.album)
                if album_key and album_count.get(album_key, 0) >= album_cap:
                    continue
                sim = self._max_sim_to_chosen(cand, chosen)
                obj = (1.0 - lam) * sc - lam * sim
                if obj > best_obj:
                    best_obj = obj
                    best_idx = i
            if best_idx == -1:
                # Caps blocked the remaining candidates. Returning a
                # shorter list is better than filling it with repeats.
                break

            score, cand, breakdown = remaining.pop(best_idx)
            primary_artist = _norm_artist(cand.track.artist or
                                          (cand.track.artists[0] if cand.track.artists else ""))
            if primary_artist:
                artist_count[primary_artist] = artist_count.get(primary_artist, 0) + 1
            album_key = _norm_title(cand.track.album)
            if album_key:
                album_count[album_key] = album_count.get(album_key, 0) + 1

            # Two distinct scores, kept separate on purpose:
            #   final_score -- standalone per-song relevance (shown to user).
            #   rank_score  -- the MMR objective that decided list *position*.
            # The list is ordered by rank_score; the card still displays
            # final_score. Storing both keeps the explanation honest.
            breakdown["rank_score"] = float(best_obj)
            breakdown["mmr_objective"] = float(best_obj)

            mmr_meta = {
                "lambda":       float(lam),
                "promoted":     bool(id(cand) not in score_only_topk),
                "rank_score":   float(best_obj),
                "final_score":  float(breakdown.get("final_score", score)),
            }
            chosen.append((rank, cand, breakdown, mmr_meta))
            rank += 1

        return chosen

    @staticmethod
    def _max_sim_to_chosen(
        cand: Candidate,
        chosen: list[tuple[int, Candidate, dict[str, float], dict[str, Any]]],
    ) -> float:
        if not chosen:
            return 0.0
        cand_artists_norm = {_norm_artist(a) for a in (cand.track.artists or [cand.track.artist])}
        cand_artists_norm.discard("")
        cand_pref_tokens = _source_preference_tokens(cand)
        cand_album = _norm_title(cand.track.album)
        sims = []
        for _r, oc, _bd, _meta in chosen:
            other_artists = {_norm_artist(a) for a in (oc.track.artists or [oc.track.artist])}
            other_artists.discard("")
            parts: list[tuple[float, float]] = []
            parts.append((0.35, 1.0 if (cand_artists_norm & other_artists) else 0.0))
            other_pref_tokens = _source_preference_tokens(oc)
            if cand_pref_tokens and other_pref_tokens:
                parts.append((0.25, _jaccard(cand_pref_tokens, other_pref_tokens)))
            other_album = _norm_title(oc.track.album)
            if cand_album and other_album:
                parts.append((0.20, 1.0 if cand_album == other_album else 0.0))
            text_cos = _cosine(cand.text_vector, oc.text_vector)
            if cand.text_vector and oc.text_vector:
                parts.append((0.20, text_cos))
            weight = sum(w for w, _v in parts)
            sims.append(sum(w * v for w, v in parts) / weight if weight else 0.0)
        return max(sims) if sims else 0.0
