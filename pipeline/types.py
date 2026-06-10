"""Standard data objects + duck-typed protocols for the pipeline.

These are the *contracts* every pipeline stage speaks in:

* :class:`TrackRef` / :class:`RealSongRequest` / :class:`RealSongCard` /
  :class:`RealSongResponse` -- the public request/response surface.
* :class:`UserProfile` -- ``ProfileBuilder`` output.
* :class:`RetrievalQuery` / :class:`SourceHit` / :class:`Candidate` /
  :class:`CandidateEnrichment` -- internal retrieval/scoring records.

Keeping them in one dependency-free module lets every stage be imported
and unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .text import _maybe_float, _maybe_int


# ---------------------------------------------------------------------------
# Public request / response surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackRef:
    """A real-song reference. Either picked by the user or returned to
    them as a recommendation."""
    netease_song_id: int
    title: str
    artist: str
    artists: list[str] = field(default_factory=list)
    album: str = ""
    cover_url: str = ""

    @property
    def netease_url(self) -> str:
        return (
            f"https://music.163.com/#/song?id={int(self.netease_song_id)}"
            if self.netease_song_id else ""
        )

    def to_card_dict(self) -> dict[str, Any]:
        return {
            "netease_song_id": int(self.netease_song_id),
            "title":           self.title,
            "artist":          self.artist,
            "artists":         list(self.artists),
            "album":           self.album,
            "cover_url":       self.cover_url,
            "netease_url":     self.netease_url,
        }


@dataclass
class RealSongRequest:
    """User-friendly request: real songs in, real songs out."""
    liked_songs: list[TrackRef] = field(default_factory=list)
    liked_artists: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    excluded_song_ids: list[int] = field(default_factory=list)

    content_weight: float = 0.50
    novelty: float = 0.30
    diversity: float = 0.30
    k: int = 10

    # How many candidates per retrieval channel (capped for latency).
    candidates_per_signal: int = 12
    # How many discovery candidates to fetch on top of the focused
    # channels. 0 disables the discovery channel entirely.
    discovery_limit: int = 12
    request_id: Optional[str] = None


@dataclass
class RealSongCard:
    rank: int
    track: TrackRef
    score: float
    score_breakdown: dict[str, float]
    explanation: str
    reasons: list[str]
    matched_tags: list[str]
    sources: list[str]
    pick_type: str           # "safe" | "exploratory" | "diverse" | "balanced"
    # Distinct retrieval channel kinds that surfaced this song, e.g.
    # ["artist", "genre", "embedding"]. Additive, P2 field -- lets the
    # frontend show how a recommendation was found without breaking the
    # existing `sources` (labelled per-query names) contract.
    source_types: list[str] = field(default_factory=list)
    # P4 shadow ranking: the learned ranker's position for this card (1-indexed
    # by descending learned_score). Additive + optional -- only set when a
    # shadow model is active. The displayed order is still the rule order, so
    # this never changes ranking; it only lets us observe the model's view.
    learned_rank_position: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = self.track.to_card_dict()
        d.update({
            "rank":             int(self.rank),
            "score":            float(self.score),
            "score_breakdown":  {k: float(v) for k, v in self.score_breakdown.items()},
            "explanation":      self.explanation,
            "reasons":          list(self.reasons),
            "matched_tags":     list(self.matched_tags),
            "sources":          list(self.sources),
            "source_types":     list(self.source_types),
            "pick_type":        self.pick_type,
        })
        if self.learned_rank_position is not None:
            d["learned_rank_position"] = int(self.learned_rank_position)
        return d


@dataclass
class RealSongResponse:
    request_id: str
    items: list[RealSongCard]
    control: dict[str, Any]
    candidate_summary: dict[str, int]
    profile: dict[str, Any]
    model_info: dict[str, Any]
    fallback_used: Optional[str] = None
    # Lightweight end-to-end trace of one recommendation (new, additive
    # field; never removes anything from the existing contract).
    trace: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id":         self.request_id,
            "items":              [c.to_dict() for c in self.items],
            "control":            dict(self.control),
            "candidate_summary":  dict(self.candidate_summary),
            "profile":            dict(self.profile),
            "model_info":         dict(self.model_info),
            "fallback_used":      self.fallback_used,
            "trace":              dict(self.trace) if self.trace is not None else None,
        }


# ---------------------------------------------------------------------------
# Client + cache protocols (duck-typed for tests)
# ---------------------------------------------------------------------------

class _NeteaseClient(Protocol):
    def search_songs(self, keywords: str, limit: int = 5) -> list[dict[str, Any]]: ...
    def ping(self) -> bool: ...


class _QueryCache(Protocol):
    def get_query(self, query: str) -> Optional[list[dict[str, Any]]]: ...
    def set_query(self, query: str, candidates: list[dict[str, Any]]) -> None: ...


# ---------------------------------------------------------------------------
# Profile (ProfileBuilder output)
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    liked_track_ids: set[int]
    liked_artists_norm: set[str]    # canonical keys for "is in liked artists"
    liked_artists_display: list[str]
    tag_tokens: list[str]            # ordered, deduped tag tokens
    tag_phrases: list[str]           # original tag/mood/genre strings, for queries
    title_tokens: set[str]           # union of liked-song title tokens
    title_phrases: list[str]         # liked-song titles, for queries
    liked_title_norms: set[str]      # canonical liked-song titles to exclude covers
    user_profile_text: str
    selected_song_texts: list[str]
    seed_album_norms: set[str]
    preferred_genres: list[str]
    preferred_moods: list[str]
    preferred_tags: list[str]
    seed_artist_weights: dict[str, float]
    query_intent_terms: list[str]
    excluded_track_ids: set[int]


# Backwards-compatible alias for the pre-refactor internal name.
_Profile = UserProfile


# ---------------------------------------------------------------------------
# Retrieval / candidate records
# ---------------------------------------------------------------------------

@dataclass
class RetrievalQuery:
    source_name: str
    source_type: str
    query: str
    reliability: float
    limit: int


@dataclass
class SourceHit:
    source_name: str
    source_type: str
    query: str
    reliability: float
    position: int


@dataclass
class Candidate:
    track: TrackRef
    artist_ids: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # For each source, position is 0-indexed rank in that channel's hits.
    positions: dict[str, int] = field(default_factory=dict)
    source_hits: list[SourceHit] = field(default_factory=list)
    text_vector: dict[str, float] = field(default_factory=dict)
    enrichment: Optional["CandidateEnrichment"] = None


@dataclass
class CandidateEnrichment:
    comment_count: Optional[int] = None
    hot_comment_count: Optional[int] = None
    song_red_count: Optional[int] = None
    artist_follow_count: Optional[int] = None
    playable: Optional[bool] = None
    audio_quality: Optional[float] = None
    lyric_excerpt: str = ""
    wiki_summary: str = ""
    similar_song_ids: list[int] = field(default_factory=list)
    endpoint_errors: list[str] = field(default_factory=list)

    @property
    def enriched(self) -> bool:
        return any([
            self.comment_count is not None,
            self.hot_comment_count is not None,
            self.song_red_count is not None,
            self.artist_follow_count is not None,
            self.playable is not None,
            self.audio_quality is not None,
            self.lyric_excerpt,
            self.wiki_summary,
            self.similar_song_ids,
        ])

    def to_cache_payload(self) -> dict[str, Any]:
        return {
            "comment_count": self.comment_count,
            "hot_comment_count": self.hot_comment_count,
            "song_red_count": self.song_red_count,
            "artist_follow_count": self.artist_follow_count,
            "playable": self.playable,
            "audio_quality": self.audio_quality,
            "lyric_excerpt": self.lyric_excerpt,
            "wiki_summary": self.wiki_summary,
            "similar_song_ids": list(self.similar_song_ids),
            "endpoint_errors": list(self.endpoint_errors),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CandidateEnrichment":
        return cls(
            comment_count=_maybe_int(payload.get("comment_count")),
            hot_comment_count=_maybe_int(payload.get("hot_comment_count")),
            song_red_count=_maybe_int(payload.get("song_red_count")),
            artist_follow_count=_maybe_int(payload.get("artist_follow_count")),
            playable=payload.get("playable") if isinstance(payload.get("playable"), bool) else None,
            audio_quality=_maybe_float(payload.get("audio_quality")),
            lyric_excerpt=str(payload.get("lyric_excerpt") or ""),
            wiki_summary=str(payload.get("wiki_summary") or ""),
            similar_song_ids=[
                int(x) for x in (payload.get("similar_song_ids") or [])
                if _maybe_int(x) is not None
            ],
            endpoint_errors=[str(x) for x in (payload.get("endpoint_errors") or [])],
        )


# Backwards-compatible aliases for pre-refactor internal names.
_RetrievalQuery = RetrievalQuery
_SourceHit = SourceHit
_Candidate = Candidate
_CandidateEnrichment = CandidateEnrichment


def _merge_track(a: TrackRef, b: TrackRef) -> TrackRef:
    """Pick the more complete TrackRef when the same song shows up
    twice (e.g. once via artist channel, once via tag channel) and
    one payload has a cover_url that the other lacks."""
    return TrackRef(
        netease_song_id=int(a.netease_song_id or b.netease_song_id),
        title=a.title or b.title,
        artist=a.artist or b.artist,
        artists=list(a.artists or b.artists),
        album=a.album or b.album,
        cover_url=a.cover_url or b.cover_url,
    )


def merge_candidates_into(
    base: dict[int, Candidate],
    extra: dict[int, Candidate],
) -> int:
    """Merge ``extra`` candidates into ``base`` keyed by song_id.

    When a song already exists in ``base`` (e.g. it was recalled by both the
    NetEase search channel and the embedding channel), the two candidates'
    ``source_hits`` / ``sources`` / ``positions`` are combined onto the
    existing object and the richer :class:`TrackRef` is kept. Combining the
    source hits is what naturally raises ``multi_source_agreement`` for songs
    found through several independent channels -- without any change to the P0
    scoring formula.

    Returns the number of song_ids that were newly added to ``base``.
    """
    added = 0
    for sid, cand in extra.items():
        existing = base.get(sid)
        if existing is None:
            base[sid] = cand
            added += 1
            continue
        existing.track = _merge_track(existing.track, cand.track)
        if not existing.artist_ids and cand.artist_ids:
            existing.artist_ids = list(cand.artist_ids)
        for name in cand.sources:
            if name not in existing.sources:
                existing.sources.append(name)
        for name, pos in cand.positions.items():
            existing.positions.setdefault(name, pos)
        existing.source_hits.extend(cand.source_hits)
    return added
