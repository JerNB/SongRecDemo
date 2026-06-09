"""Ranker -- stage 5 of the pipeline.

Applies the P0 scoring formula (unchanged) to every candidate and
returns a ``(final_score, Candidate, score_breakdown)`` tuple list. The
content/retrieval/quality split and the ``content_weight`` blend live
here; the individual sub-scores come from :mod:`scoring`.
"""

from __future__ import annotations

import math

from . import scoring
from .scoring import _RELEVANCE_GATE_DIVISOR, _W
from .text import _clip01, _cosine, _jaccard, _norm_artist, _tfidf_vectors, _token_set, _tokens
from .types import Candidate, RealSongRequest, UserProfile


class Ranker:
    """Scores candidates with the P0 hybrid relevance formula."""

    def score_all(
        self,
        candidates: dict[int, Candidate],
        profile: UserProfile,
        req: RealSongRequest,
    ) -> list[tuple[float, Candidate, dict[str, float]]]:
        out: list[tuple[float, Candidate, dict[str, float]]] = []
        nw = _clip01(req.novelty)
        cw = _clip01(req.content_weight)
        cand_list = list(candidates.values())
        profile_vec: dict[str, float] = {}
        cand_vecs: list[dict[str, float]] = []
        if cand_list:
            docs = [profile.user_profile_text] + [scoring.candidate_text(c, profile) for c in cand_list]
            vectors = _tfidf_vectors(docs)
            if len(vectors) == len(docs):
                profile_vec = vectors[0]
                cand_vecs = vectors[1:]
        for idx, cand in enumerate(cand_list):
            cand.text_vector = cand_vecs[idx] if idx < len(cand_vecs) else {}
            content_sim = _cosine(profile_vec, cand.text_vector) if profile_vec and cand.text_vector else 0.0
            breakdown = self.score_one(cand, profile, nw, cw, content_sim)
            out.append((breakdown["final_score"], cand, breakdown))
        return out

    def score_one(
        self,
        cand: Candidate,
        profile: UserProfile,
        novelty_w: float,
        content_weight: float,
        content_text_similarity: float,
    ) -> dict[str, float]:
        track = cand.track

        # Artist match: 1.0 when any candidate artist is in the user's
        # liked artists set (case- and punctuation-insensitive); else
        # token Jaccard against the union of liked artist tokens.
        cand_artist_tokens: set[str] = set()
        for a in (track.artists or [track.artist]):
            cand_artist_tokens |= _token_set(a)

        cand_artist_norms = {_norm_artist(a) for a in (track.artists or [track.artist])}
        cand_artist_norms.discard("")

        if profile.liked_artists_norm and (cand_artist_norms & profile.liked_artists_norm):
            artist_match = 1.0
        else:
            liked_tokens: set[str] = set()
            for a in profile.liked_artists_display:
                liked_tokens |= _token_set(a)
            artist_match = _jaccard(cand_artist_tokens, liked_tokens)

        # Tag match: only non-title metadata counts. A song title that
        # literally says "alternative rock" is not genre evidence.
        cand_non_title_tokens = cand_artist_tokens | _token_set(track.album)
        cand_title_tokens = _token_set(track.title)
        if profile.tag_tokens:
            hits = sum(1 for t in profile.tag_tokens if t in cand_non_title_tokens)
            direct_tag_match = float(hits) / float(len(profile.tag_tokens))
            source_tag_tokens = set()
            for hit in cand.source_hits:
                if hit.source_type in {"genre", "mood", "tag", "genre_mood", "tag_combo", "discovery", "artist_context"}:
                    source_tag_tokens.update(_tokens(hit.query))
            source_hits = sum(1 for t in profile.tag_tokens if t in source_tag_tokens)
            source_tag_match = float(source_hits) / float(len(profile.tag_tokens))
            tag_match = min(1.0, 0.70 * direct_tag_match + 0.30 * source_tag_match)
            title_hits = sum(1 for t in profile.tag_tokens if t in cand_title_tokens)
            tag_title_overlap = float(title_hits) / float(len(profile.tag_tokens))
        else:
            tag_match = 0.0
            tag_title_overlap = 0.0

        # Title match: fraction of liked-song title tokens that appear
        # in the candidate's title. Excludes generic stopwords (already
        # handled by _tokens).
        if profile.title_tokens:
            hits = sum(1 for t in _token_set(track.title) if t in profile.title_tokens)
            title_match = min(1.0, float(hits) / max(1.0, math.sqrt(len(profile.title_tokens))))
        else:
            title_match = 0.0

        metadata_quality_score = scoring.metadata_quality(track)
        retrieval_confidence_score = scoring.retrieval_confidence(cand, metadata_quality_score)
        multi_source_agreement_score = scoring.multi_source_agreement(cand)
        popularity_score = scoring.popularity_score(cand.enrichment)
        artist_authority_score = scoring.artist_authority_score(cand.enrichment)
        playable_score = scoring.playable_score(cand.enrichment)
        audio_quality_score = scoring.audio_quality_score(cand.enrichment)

        # --- Three top-level sub-scores (weights from config). ---------
        # content_score: how well the candidate matches the user's stated
        # taste (free text, artists, tags, liked-song titles).
        cw_content = _W["content"]
        content_score = min(1.0, (
            cw_content["text_similarity"] * content_text_similarity
            + cw_content["artist_match"] * artist_match
            + cw_content["tag_match"] * tag_match
            + cw_content["title_match"] * title_match
        ))

        # retrieval_score: how confident recall is, plus agreement across
        # independent search channels (NOT collaborative filtering).
        cw_retrieval = _W["retrieval"]
        retrieval_score = min(1.0, (
            cw_retrieval["retrieval_confidence"] * retrieval_confidence_score
            + cw_retrieval["multi_source_agreement"] * multi_source_agreement_score
        ))

        # quality_score: is this a real, popular, playable, well-described
        # track on the platform?
        cw_quality = _W["quality"]
        quality_score = min(1.0, (
            cw_quality["popularity"] * popularity_score
            + cw_quality["artist_authority"] * artist_authority_score
            + cw_quality["playable"] * playable_score
            + cw_quality["metadata_quality"] * metadata_quality_score
        ))

        artist_in_liked = (
            1.0 if (profile.liked_artists_norm and (cand_artist_norms & profile.liked_artists_norm))
            else 0.0
        )
        artist_affinity_score = scoring.artist_affinity(cand, profile, artist_match)
        novelty_score = scoring.novelty_score(cand, profile, artist_in_liked, tag_match)
        # trust_score kept for backward-compatible diagnostics only; it no
        # longer drives base_relevance (the three sub-scores above do).
        trust_score = scoring.trust_score(
            metadata_quality_score=metadata_quality_score,
            popularity_score=popularity_score,
            artist_authority_score=artist_authority_score,
            playable_score=playable_score,
            multi_source_agreement_score=multi_source_agreement_score,
            retrieval_confidence_score=retrieval_confidence_score,
        )

        # --- content_weight slider genuinely blends content vs retrieval.
        cw = _clip01(content_weight)
        personalized_relevance = cw * content_score + (1.0 - cw) * retrieval_score

        cw_base = _W["base"]
        base_relevance = min(1.0, (
            cw_base["personalized_relevance"] * personalized_relevance
            + cw_base["quality"] * quality_score
            + cw_base["artist_authority"] * artist_authority_score
        ))

        # Novelty still passes through the relevance gate so a novel but
        # low-relevance song cannot leapfrog relevant ones.
        relevance_gate = min(1.0, base_relevance / _RELEVANCE_GATE_DIVISOR) if _RELEVANCE_GATE_DIVISOR > 0 else 1.0
        novelty_bonus = novelty_w * novelty_score * relevance_gate
        final_score = min(1.0, max(0.0, base_relevance + novelty_bonus))

        return {
            # --- New, clearer score_breakdown structure. ---------------
            "content_score":     float(content_score),
            "retrieval_score":   float(retrieval_score),
            "multi_source_agreement": float(multi_source_agreement_score),
            "quality_score":     float(quality_score),
            "personalized_relevance": float(personalized_relevance),
            "base_relevance":    float(base_relevance),
            "content_weight":    float(cw),
            "novelty_score":     float(novelty_score),
            "novelty_bonus":     float(novelty_bonus),
            "relevance_gate":    float(relevance_gate),
            "final_score":       float(final_score),
            # rank_score is filled in by the MMR reranker; default to the
            # standalone score so a card always carries the field.
            "rank_score":        float(final_score),
            # --- Component sub-signals (useful for explanations). ------
            "content_text_similarity": float(content_text_similarity),
            "artist_match":      float(artist_match),
            "tag_match":         float(tag_match),
            "tag_title_overlap": float(tag_title_overlap),
            "title_match":       float(title_match),
            "retrieval_confidence_score": float(retrieval_confidence_score),
            "artist_affinity_score": float(artist_affinity_score),
            "popularity_score": float(popularity_score),
            "artist_authority_score": float(artist_authority_score),
            "playable_score": float(playable_score),
            "audio_quality_score": float(audio_quality_score),
            "metadata_quality_score": float(metadata_quality_score),
            "trust_score": float(trust_score),
            # --- Legacy aliases (kept so the existing frontend + smoke
            # test keep working; prefer the names above going forward). -
            "final":             float(final_score),
            "content":           float(content_score),
            "retrieval":         float(retrieval_confidence_score),
            "multi_source":      float(multi_source_agreement_score),
            "collaborative_proxy_score": float(multi_source_agreement_score),
            "novelty_term":      float(novelty_score),
        }
