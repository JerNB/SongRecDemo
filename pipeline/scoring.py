"""Shared per-candidate scoring sub-functions.

These are the building blocks of the P0 ranking formula (see
``Ranker`` and ``config.RANKING_WEIGHTS_V1``). They live here -- rather
than on the ``Ranker`` -- because the ``CandidateFilter`` also needs
some of them (popularity / authority / retrieval confidence /
multi-source agreement) for its soft / hard trust decisions, and shared
module functions avoid circular ``Filter <-> Ranker`` references.

NOTHING in this module changes the P0 maths; it is a verbatim
extraction of the previous static methods.
"""

from __future__ import annotations

import math
from typing import Optional

import config

from .text import (
    _clip01,
    _jaccard,
    _norm_artist,
    _norm_title,
    _profile_text,
    _strip_profile_tag_terms,
    _token_set,
    _tokens,
)
from .types import Candidate, CandidateEnrichment, TrackRef, UserProfile


# Centralised ranking weights, pulled into module-level names so the
# scoring code reads like the formulas in the docstrings while staying
# tunable from config.
_W = config.RANKING_WEIGHTS_V1
_RELEVANCE_GATE_DIVISOR = float(config.RANKING_RELEVANCE_GATE_DIVISOR)


def candidate_text(cand: Candidate, profile: UserProfile) -> str:
    source_queries = " ".join(h.query for h in cand.source_hits)
    inferred = []
    for phrase in profile.tag_phrases:
        ptoks = set(_tokens(phrase))
        if ptoks and ptoks <= set(_tokens(source_queries)):
            inferred.append(phrase)
    return _profile_text([
        _strip_profile_tag_terms(cand.track.title, profile.tag_tokens),
        cand.track.artist,
        " ".join(cand.track.artists or []),
        cand.track.album,
        source_queries,
        " ".join(inferred),
        cand.enrichment.lyric_excerpt if cand.enrichment else "",
        cand.enrichment.wiki_summary if cand.enrichment else "",
    ])


def metadata_quality(track: TrackRef) -> float:
    weights = [
        (track.title, 0.35),
        (track.artist, 0.30),
        (track.album, 0.15),
        (track.cover_url, 0.10),
        (track.netease_song_id, 0.10),
    ]
    return min(1.0, sum(w for val, w in weights if bool(val)))


def retrieval_confidence(cand: Candidate, metadata_quality_score: float) -> float:
    if not cand.source_hits:
        return 0.0
    vals = []
    for hit in cand.source_hits:
        rank_score = 1.0 / (1.0 + 0.20 * max(0, hit.position))
        vals.append(rank_score * _clip01(hit.reliability))
    best = max(vals)
    avg = sum(vals) / len(vals)
    return min(1.0, (0.70 * best + 0.30 * avg) * (0.80 + 0.20 * metadata_quality_score))


def multi_source_agreement(cand: Candidate) -> float:
    """Multi-source agreement / retrieval consensus signal.

    NOT collaborative filtering: the product layer has no real
    user-item interaction matrix and no learned latent factors. This
    score simply rewards a candidate that surfaced across several
    *independent* retrieval channels (artist / genre / mood / tag /
    seed_song / discovery). Showing up through many complementary
    paths is evidence the song is consistent with multiple facets of
    the user's profile, so the system is more confident it is
    relevant -- hence "retrieval consensus".
    """
    if not cand.source_hits:
        return 0.0
    source_names = {h.source_name for h in cand.source_hits}
    source_types = {h.source_type for h in cand.source_hits}
    source_count_score = min(1.0, len(source_names) / 4.0)
    type_diversity_score = min(1.0, len(source_types) / 4.0)
    has_artist = bool(source_types & {"artist", "artist_context", "seed_song", "seed_album"})
    has_taste = bool(source_types & {"genre", "mood", "tag", "genre_mood", "tag_combo", "discovery"})
    cross_signal = 1.0 if has_artist and has_taste else 0.0
    weak_title_only = source_types <= {"title"}
    score = (
        0.40 * source_count_score
        + 0.30 * type_diversity_score
        + 0.30 * cross_signal
    )
    if weak_title_only:
        score *= 0.35
    if cand.enrichment and cand.enrichment.similar_song_ids:
        source_bonus = min(0.20, len(cand.enrichment.similar_song_ids) / 100.0)
        score = min(1.0, 0.85 * score + source_bonus)
    return min(1.0, score)


def popularity_score(enr: Optional[CandidateEnrichment]) -> float:
    if enr is None or not enr.enriched:
        return 0.55
    comments = max(0, enr.comment_count or 0)
    hot = max(0, enr.hot_comment_count or 0)
    red = max(0, enr.song_red_count or 0)
    raw = comments + 3.0 * hot + 0.5 * red
    if raw <= 0:
        return 0.0
    return min(1.0, math.log1p(raw) / math.log1p(5000.0))


def artist_authority_score(enr: Optional[CandidateEnrichment]) -> float:
    if enr is None or not enr.enriched or enr.artist_follow_count is None:
        return 0.55
    followers = max(0, int(enr.artist_follow_count))
    if followers <= 0:
        return 0.0
    return min(1.0, math.log1p(followers) / math.log1p(500000.0))


def playable_score(enr: Optional[CandidateEnrichment]) -> float:
    if enr is None or not enr.enriched or enr.playable is None:
        return 0.60
    return 1.0 if enr.playable else 0.0


def audio_quality_score(enr: Optional[CandidateEnrichment]) -> float:
    if enr is None or enr.audio_quality is None:
        return 0.60
    return _clip01(enr.audio_quality)


def trust_score(
    *,
    metadata_quality_score: float,
    popularity_score: float,
    artist_authority_score: float,
    playable_score: float,
    multi_source_agreement_score: float,
    retrieval_confidence_score: float,
) -> float:
    return min(1.0, (
        0.20 * metadata_quality_score
        + 0.20 * popularity_score
        + 0.20 * artist_authority_score
        + 0.15 * playable_score
        + 0.15 * multi_source_agreement_score
        + 0.10 * retrieval_confidence_score
    ))


def artist_affinity(cand: Candidate, profile: UserProfile, artist_match: float) -> float:
    norms = {_norm_artist(a) for a in (cand.track.artists or [cand.track.artist])}
    norms.discard("")
    weighted = max((profile.seed_artist_weights.get(n, 0.0) for n in norms), default=0.0)
    return min(1.0, max(float(artist_match), weighted))


def novelty_score(
    cand: Candidate,
    profile: UserProfile,
    artist_in_liked: float,
    tag_match: float,
) -> float:
    unfamiliar_artist = 1.0 - artist_in_liked
    if cand.source_hits:
        avg_pos = sum(h.position for h in cand.source_hits) / len(cand.source_hits)
        deeper_but_not_buried = min(1.0, avg_pos / 8.0)
    else:
        deeper_but_not_buried = 0.0
    shares_taste = max(tag_match, 0.30 if profile.tag_tokens and cand.source_hits else 0.0)
    same_album_as_seed = (
        1.0 if cand.track.album and _norm_title(cand.track.album) in profile.seed_album_norms
        else 0.0
    )
    novelty = (
        0.35 * unfamiliar_artist
        + 0.25 * deeper_but_not_buried
        + 0.25 * shares_taste
        + 0.15 * (1.0 - same_album_as_seed)
    )
    return min(1.0, max(0.0, novelty))
