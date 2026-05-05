"""
NetEase-backed real-song recommendation pipeline.

This is the *product layer* of the demo: it takes user-friendly input
(songs the user picked from search results, free-text artists, free-text
genres / moods / tags) and produces ranked recommendations that are
themselves real, identifiable songs with title, artist, album, cover
art, and a NetEase link.

How it relates to the research layer
------------------------------------
The KGRec ALS / content-based / popularity models in
:mod:`src.personalization` are the *research layer*. They were trained
on KGRec interaction data and are evaluated against KGRec test splits.
They can only score KGRec items, so they cannot directly recommend
arbitrary NetEase songs.

This module deliberately does **not** call the ALS model. Instead it
borrows the *logic* the research demonstrates:

* Build a taste profile from a handful of seeds + free-text tags.
* Retrieve candidates via multiple complementary signals.
* Blend a content-similarity score with a retrieval / popularity proxy.
* Apply a novelty term to break out of the most obvious picks.
* Apply MMR-style diversification so the result list is not all by
  one artist or one narrow tag.
* Attach human-readable explanations so the user knows *why* each
  song was suggested.

Retrieval channels
------------------
For one request we issue a small number of NetEase ``/search`` calls
(cached on disk via :class:`NeteaseCache`) and label each returned
candidate with the *channels* it came through:

* ``artist:<name>``  -- songs returned when searching for a liked or
                       explicitly-listed artist name.
* ``tag:<token>``    -- songs returned when searching for a tag /
                       genre / mood token.
* ``title:<title>``  -- songs returned when searching for a liked
                       song title (so the user gets variants and
                       neighbours of songs they already love).
* ``discovery``      -- a single "broad" query built from a couple of
                       tag tokens, used to inject novelty when the
                       user's profile is otherwise narrow.

A candidate present in *more than one* channel gets a multi-source
bonus -- a strong signal that the candidate matches several aspects
of the profile at once.

Scoring
-------
Per candidate we compute four normalised sub-scores in [0, 1]:

* ``artist_match``: how well the candidate's artist tokens match the
  profile's artist tokens (1.0 on exact match, fractional on partial).
* ``tag_match``: fraction of the user's tag tokens that appear
  somewhere in the candidate's title + artist + album.
* ``title_match``: fraction of liked-song title tokens overlapping
  the candidate's title (catches sequels / covers / variants).
* ``retrieval_score``: a position-based score (1.0 for the top result
  in a channel, decaying with rank) averaged across the channels the
  candidate appeared in, plus a multi-source bonus.

These are blended into a final score with the user's sliders:

    content   = 0.50 * artist_match + 0.30 * tag_match + 0.20 * title_match
    base      = retrieval_score
    novelty   = (1 - artist_in_liked) * (1 - retrieval_score)
    final     = (1 - content_weight) * base
              + content_weight       * content
              + novelty_slider       * novelty
              - same_artist_penalty

After scoring we run a small MMR rerank where similarity is
"same artist" (1.0) plus light tag overlap. The ``diversity`` slider
is the MMR lambda.

Pick types
----------
Each card is labelled ``safe`` / ``exploratory`` / ``diverse``:

* ``safe``         -- the candidate's artist is in the user's liked
                      set, OR the content score is high.
* ``exploratory``  -- novelty term is the dominant contribution.
* ``diverse``      -- MMR pulled this card up over a higher-scoring
                      one to keep the list varied.

The smoke test asserts that every card carries one of these labels.

Hermetic testing
----------------
The pipeline takes a duck-typed ``client`` with two methods --
``search_songs(keywords, limit)`` and ``ping()`` -- so tests can
inject :class:`FakeNeteaseClient` and run fully offline.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
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
            "pick_type":        self.pick_type,
        })
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id":         self.request_id,
            "items":              [c.to_dict() for c in self.items],
            "control":            dict(self.control),
            "candidate_summary":  dict(self.candidate_summary),
            "profile":            dict(self.profile),
            "model_info":         dict(self.model_info),
            "fallback_used":      self.fallback_used,
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
# Tokenisation helpers (kept tiny on purpose -- the pipeline lives or
# dies on whether overlap calculations agree across artist / tag /
# title fields, so we use one canonical tokeniser everywhere).
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_STOP = frozenset({
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "by",
    "with", "from", "is", "was", "are", "be", "feat", "ft", "vs",
    "remix", "version", "edit", "mix", "remastered", "remaster",
    "live", "cover", "acoustic", "instrumental", "demo", "radio",
    "karaoke", "mono", "stereo",
})
_VERSION_WORDS = frozenset({
    "live", "cover", "acoustic", "instrumental", "demo", "radio",
    "karaoke", "mono", "stereo", "remix", "version", "edit",
    "mix", "remastered", "remaster",
})
_TITLE_SUFFIX_RE = re.compile(r"[\(\[\{（【].*?[\)\]\}）】]")


def _tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        t = raw.lower()
        if not t or t in _STOP or len(t) <= 1:
            continue
        out.append(t)
    return out


def _token_set(text: Optional[str]) -> set[str]:
    return set(_tokens(text))


def _norm_artist(name: str) -> str:
    """Canonical lower-case-stripped artist key for set membership.

    NetEase and the user can disagree on punctuation / case ("bon iver"
    vs "Bon Iver"), so we match on the joined lowered tokens.
    """
    return " ".join(_tokens(name))


def _norm_title(title: str) -> str:
    """Canonical key for duplicate-title filtering.

    Search results often include covers/live/remix variants as
    ``Song (Live)`` or ``Song - Acoustic``. For recommendation purposes
    those should not be treated as fresh songs when the user already
    picked ``Song``.
    """
    raw = str(title or "")
    without_brackets = _TITLE_SUFFIX_RE.sub(" ", raw)
    # Drop common version suffixes, but keep the original title when the
    # split would remove the whole signal.
    for sep in (" - ", " -- ", " / "):
        if sep in without_brackets:
            head, tail = without_brackets.split(sep, 1)
            if _tokens(head) and _tokens(tail):
                tail_tokens = set(_tokens(tail))
                if tail_tokens <= _STOP:
                    without_brackets = head
            break
    key = " ".join(_tokens(without_brackets))
    return key or " ".join(_tokens(raw))


def _dedupe_phrases(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        s = str(raw or "").strip()
        key = " ".join(_tokens(s))
        if not s or not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _raw_tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [raw.lower() for raw in _TOKEN_RE.findall(text) if raw]


def _profile_text(parts: Iterable[str]) -> str:
    toks: list[str] = []
    for part in parts:
        toks.extend(_tokens(part))
    return " ".join(toks)


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class _Profile:
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


def _build_profile(req: RealSongRequest) -> _Profile:
    artists_norm: set[str] = set()
    artists_display: list[str] = []
    seen_disp: set[str] = set()

    def _add_artist(name: str) -> None:
        norm = _norm_artist(name)
        if not norm:
            return
        if norm not in artists_norm:
            artists_norm.add(norm)
        if name and name not in seen_disp:
            artists_display.append(name)
            seen_disp.add(name)

    for s in req.liked_songs:
        for a in (s.artists or [s.artist]):
            if a:
                _add_artist(a)

    for a in req.liked_artists:
        if a and a.strip():
            _add_artist(a.strip())

    preferred_genres = _dedupe_phrases(req.genres)
    preferred_moods = _dedupe_phrases(req.moods)
    preferred_tags = _dedupe_phrases(req.tags)

    # Keep originals for retrieval queries (multi-word genres like
    # "indie rock" should hit NetEase as one phrase) AND break them
    # down into atomic tokens for matching.
    tag_phrases = _dedupe_phrases(
        list(preferred_genres) + list(preferred_moods) + list(preferred_tags)
    )

    tag_tokens: list[str] = []
    seen_tok: set[str] = set()
    for phrase in tag_phrases:
        for tok in _tokens(phrase):
            if tok in seen_tok:
                continue
            seen_tok.add(tok)
            tag_tokens.append(tok)

    title_tokens: set[str] = set()
    title_phrases: list[str] = []
    liked_title_norms: set[str] = set()
    selected_song_texts: list[str] = []
    seed_album_norms: set[str] = set()
    for s in req.liked_songs:
        if s.title:
            title_phrases.append(s.title)
            title_tokens.update(_tokens(s.title))
            norm_title = _norm_title(s.title)
            if norm_title:
                liked_title_norms.add(norm_title)
        selected_song_texts.append(
            _profile_text([s.title, s.artist, " ".join(s.artists or []), s.album])
        )
        album_norm = _norm_title(s.album)
        if album_norm:
            seed_album_norms.add(album_norm)

    seed_artist_weights: dict[str, float] = {}
    for s in req.liked_songs:
        for a in (s.artists or [s.artist]):
            norm = _norm_artist(a)
            if norm:
                seed_artist_weights[norm] = seed_artist_weights.get(norm, 0.0) + 1.0
    for a in req.liked_artists:
        norm = _norm_artist(a)
        if norm:
            seed_artist_weights[norm] = seed_artist_weights.get(norm, 0.0) + 0.75
    max_artist_weight = max(seed_artist_weights.values(), default=1.0)
    seed_artist_weights = {
        k: min(1.0, v / max_artist_weight) for k, v in seed_artist_weights.items()
    }

    query_intent_terms = []
    seen_intent: set[str] = set()
    for phrase in tag_phrases + artists_display + title_phrases:
        for tok in _tokens(phrase):
            if tok not in seen_intent:
                seen_intent.add(tok)
                query_intent_terms.append(tok)

    user_profile_text = _profile_text(
        title_phrases
        + selected_song_texts
        + artists_display
        + preferred_genres
        + preferred_moods
        + preferred_tags
    )

    return _Profile(
        liked_track_ids={int(s.netease_song_id) for s in req.liked_songs if s.netease_song_id},
        liked_artists_norm=artists_norm,
        liked_artists_display=artists_display,
        tag_tokens=tag_tokens,
        tag_phrases=tag_phrases,
        title_tokens=title_tokens,
        title_phrases=title_phrases,
        liked_title_norms=liked_title_norms,
        user_profile_text=user_profile_text,
        selected_song_texts=selected_song_texts,
        seed_album_norms=seed_album_norms,
        preferred_genres=preferred_genres,
        preferred_moods=preferred_moods,
        preferred_tags=preferred_tags,
        seed_artist_weights=seed_artist_weights,
        query_intent_terms=query_intent_terms,
        excluded_track_ids={int(x) for x in req.excluded_song_ids if x},
    )


# ---------------------------------------------------------------------------
# Internal candidate record
# ---------------------------------------------------------------------------

@dataclass
class _RetrievalQuery:
    source_name: str
    source_type: str
    query: str
    reliability: float
    limit: int


@dataclass
class _SourceHit:
    source_name: str
    source_type: str
    query: str
    reliability: float
    position: int


@dataclass
class _Candidate:
    track: TrackRef
    sources: list[str] = field(default_factory=list)
    # For each source, position is 0-indexed rank in that channel's hits.
    positions: dict[str, int] = field(default_factory=dict)
    source_hits: list[_SourceHit] = field(default_factory=list)
    text_vector: dict[str, float] = field(default_factory=dict)
    enrichment: Optional["_CandidateEnrichment"] = None


@dataclass
class _CandidateEnrichment:
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
    def from_payload(cls, payload: dict[str, Any]) -> "_CandidateEnrichment":
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


# ---------------------------------------------------------------------------
# The recommender
# ---------------------------------------------------------------------------

class NeteaseRecommender:
    """Real-song recommendation pipeline backed by NetEase /search.

    Construction
    ------------
    >>> rec = NeteaseRecommender(client=NeteaseAPIClient(...), cache=NeteaseCache(...))
    >>> resp = rec.recommend(RealSongRequest(...))

    The optional ``cache`` parameter (any object with the small
    :class:`_QueryCache` shape) dramatically speeds up repeat calls
    -- repeated runs of the demo with the same artists / tags will
    short-circuit on the SQLite cache instead of re-hitting NetEase.
    """

    MODEL_NAME = "NetEase-Pipeline-v1"

    def __init__(
        self,
        client: _NeteaseClient,
        cache: Optional[_QueryCache] = None,
        *,
        max_per_query: int = 12,
        max_artist_queries: int = 6,
        max_tag_queries: int = 4,
        max_title_queries: int = 3,
        per_artist_cap: int = 2,
    ) -> None:
        self._client = client
        self._cache = cache
        self._max_per_query = int(max_per_query)
        self._max_artist_queries = int(max_artist_queries)
        self._max_tag_queries = int(max_tag_queries)
        self._max_title_queries = int(max_title_queries)
        self._per_artist_cap = int(per_artist_cap)
        self._enrich_top_n = 30
        self._min_comment_count = 10
        self._min_artist_follow_count = 77

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, req: RealSongRequest) -> RealSongResponse:
        profile = _build_profile(req)

        # Hard validation: a real-song demo needs at least one signal.
        # Empty request -> caller should turn this into a 400; we
        # surface it as an empty result with fallback_used set so the
        # API layer can decide.
        has_any = bool(
            profile.liked_track_ids
            or profile.liked_artists_norm
            or profile.tag_tokens
        )
        if not has_any:
            return RealSongResponse(
                request_id=req.request_id or str(uuid.uuid4()),
                items=[],
                control=self._control_echo(req),
                candidate_summary={"artist": 0, "tag": 0, "title": 0,
                                   "discovery": 0, "total_unique": 0},
                profile=self._profile_echo(profile),
                model_info=self._model_info(),
                fallback_used="no_input",
            )

        # 1. Multi-channel candidate retrieval.
        candidates_by_id, summary = self._retrieve(profile, req)

        # 2. Lightweight score, then enrich the strongest candidates.
        scored = self._score_all(candidates_by_id, profile, req)
        scored.sort(key=lambda x: (-x[0], x[1].track.title or "", x[1].track.netease_song_id))
        enriched_count = self._enrich_candidates([cand for _score, cand, _bd in scored[: self._enrich_top_n]])
        filtered_unplayable = self._filter_unplayable(candidates_by_id)
        filtered_low_trust = self._filter_low_trust(candidates_by_id)
        summary["enriched_count"] = int(enriched_count)
        summary["filtered_unplayable"] = int(filtered_unplayable)
        summary["filtered_low_trust"] = int(filtered_low_trust)
        summary["final_candidate_count"] = int(len(candidates_by_id))
        summary["total_unique"] = int(len(candidates_by_id))

        # 3. Re-score after enrichment so popularity, artist authority,
        # playability, and trust affect ranking and explanations.
        scored = self._score_all(candidates_by_id, profile, req)
        scored.sort(key=lambda x: (-x[0], x[1].track.title or "", x[1].track.netease_song_id))

        # 4. MMR rerank for diversity + per-artist cap.
        ranked = self._rerank_mmr(scored, profile, req)

        # 5. Build cards with explanations + pick type.
        cards = self._build_cards(ranked, profile)

        return RealSongResponse(
            request_id=req.request_id or str(uuid.uuid4()),
            items=cards,
            control=self._control_echo(req),
            candidate_summary=summary,
            profile=self._profile_echo(profile),
            model_info=self._model_info(),
            fallback_used=None if cards else "no_candidates",
        )

    # ------------------------------------------------------------------
    # Echo helpers
    # ------------------------------------------------------------------

    def _control_echo(self, req: RealSongRequest) -> dict[str, Any]:
        return {
            "content_weight": float(_clip01(req.content_weight)),
            "novelty":        float(_clip01(req.novelty)),
            "diversity":      float(_clip01(req.diversity)),
            "k":              int(max(1, min(50, int(req.k)))),
        }

    def _profile_echo(self, profile: _Profile) -> dict[str, Any]:
        return {
            "liked_song_ids":   sorted(profile.liked_track_ids),
            "liked_artists":    list(profile.liked_artists_display),
            "tags":             list(profile.tag_phrases),
            "preferred_genres":  list(profile.preferred_genres),
            "preferred_moods":   list(profile.preferred_moods),
            "preferred_tags":    list(profile.preferred_tags),
            "tag_tokens":       list(profile.tag_tokens),
            "title_tokens":     sorted(profile.title_tokens),
            "query_intent_terms": list(profile.query_intent_terms),
        }

    def _model_info(self) -> dict[str, Any]:
        return {
            "name":                         self.MODEL_NAME,
            "model_type":                   "real_song_hybrid_retrieval_ranking",
            "uses_netease_api":             True,
            "trained_collaborative_filtering": False,
            "collaborative_proxy_used":     True,
            "candidate_enrichment_used":    True,
            "quality_thresholds": {
                "soft_min_comment_count": self._min_comment_count,
                "soft_min_artist_follow_count": self._min_artist_follow_count,
            },
            "research_layer":               "KGRec ALS/content/popularity evaluation remains separate",
            "source":                       "NetEase /search",
        }

    # ------------------------------------------------------------------
    # 1. Candidate retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self, profile: _Profile, req: RealSongRequest,
    ) -> tuple[dict[int, _Candidate], dict[str, int]]:
        """Issue focused /search calls, dedup, filter, and label sources."""
        per_query = max(1, min(self._max_per_query, int(req.candidates_per_signal)))

        candidates: dict[int, _Candidate] = {}
        route_counts = {"artist": 0, "tag": 0, "title": 0, "discovery": 0}
        retrieved_total = 0

        for q in self._build_retrieval_queries(profile, req, per_query):
            hits = self._search_cached(q.query, q.limit)
            retrieved_total += len(hits)
            bucket = self._summary_bucket(q.source_type)
            route_counts[bucket] = route_counts.get(bucket, 0) + len(hits)
            self._merge_hits(
                candidates,
                hits,
                source_name=q.source_name,
                source_type=q.source_type,
                query=q.query,
                reliability=q.reliability,
            )

        after_dedup = len(candidates)

        filtered_liked = 0
        for sid in (profile.liked_track_ids | profile.excluded_track_ids):
            if candidates.pop(int(sid), None) is not None:
                filtered_liked += 1

        filtered_missing_metadata = self._filter_missing_metadata(candidates)
        filtered_same_title = self._filter_same_liked_title(candidates, profile)
        filtered_tag_title = self._filter_tag_title_shortcuts(candidates, profile)
        filtered_duplicate_version = self._filter_duplicate_versions(candidates, profile)

        summary = {
            "artist":                     int(route_counts.get("artist", 0)),
            "tag":                        int(route_counts.get("tag", 0)),
            "title":                      int(route_counts.get("title", 0)),
            "discovery":                  int(route_counts.get("discovery", 0)),
            "retrieved_total":            int(retrieved_total),
            "after_dedup":                int(after_dedup),
            "filtered_liked":             int(filtered_liked),
            "filtered_same_title":        int(filtered_same_title),
            "filtered_tag_title":         int(filtered_tag_title),
            "filtered_duplicate_version": int(filtered_duplicate_version),
            "filtered_missing_metadata":  int(filtered_missing_metadata),
            "final_candidate_count":      int(len(candidates)),
            "total_unique":               int(len(candidates)),
        }
        return candidates, summary

    def _enrich_candidates(self, candidates: list[_Candidate]) -> int:
        enriched = 0
        for cand in candidates:
            if cand.enrichment is not None:
                if cand.enrichment.enriched:
                    enriched += 1
                continue
            cand.enrichment = self._enrich_one(cand)
            if cand.enrichment.enriched:
                enriched += 1
        return enriched

    def _enrich_one(self, cand: _Candidate) -> _CandidateEnrichment:
        sid = int(cand.track.netease_song_id or 0)
        if not sid:
            return _CandidateEnrichment(endpoint_errors=["missing_song_id"])

        cache_key = f"songrec_demo:enrich:{sid}:v1"
        if self._cache is not None:
            try:
                hit = self._cache.get_query(cache_key)
                if hit:
                    return _CandidateEnrichment.from_payload(hit[0])
            except Exception:  # noqa: BLE001
                pass

        # Test doubles can provide deterministic enrichment without
        # emulating the full NetEase HTTP surface.
        provider = getattr(self._client, "enrich_song", None)
        if callable(provider):
            try:
                payload = provider(sid) or {}
                enr = _CandidateEnrichment.from_payload(payload)
                self._cache_enrichment(cache_key, enr)
                return enr
            except Exception as exc:  # noqa: BLE001
                return _CandidateEnrichment(endpoint_errors=[f"fake_enrich:{exc}"])

        enr = _CandidateEnrichment()

        comment = self._netease_get("/comment/music", {"id": str(sid), "limit": "1"})
        if isinstance(comment, dict):
            enr.comment_count = _maybe_int(comment.get("total"))
            hot = comment.get("hotComments")
            if isinstance(hot, list):
                enr.hot_comment_count = len(hot)
        else:
            enr.endpoint_errors.append("comment")

        dynamic = self._netease_get("/song/detail/dynamic", {"ids": str(sid)})
        if isinstance(dynamic, dict):
            enr.song_red_count = (
                _maybe_int(dynamic.get("likedCount"))
                or _maybe_int(dynamic.get("liked_count"))
                or _maybe_int(dynamic.get("subCount"))
                or _maybe_int(dynamic.get("shareCount"))
            )

        detail = self._netease_get("/song/detail", {"ids": f"[{sid}]"})
        artist_ids: list[int] = []
        if isinstance(detail, dict):
            songs = detail.get("songs")
            if isinstance(songs, list) and songs:
                row = songs[0] if isinstance(songs[0], dict) else {}
                artists = row.get("ar") or row.get("artists") or []
                if isinstance(artists, list):
                    for a in artists:
                        if isinstance(a, dict):
                            aid = _maybe_int(a.get("id"))
                            if aid:
                                artist_ids.append(aid)
                pop = _maybe_int(row.get("pop"))
                if pop is not None and enr.song_red_count is None:
                    enr.song_red_count = pop

        if artist_ids:
            artist_detail = self._netease_get("/artist/detail", {"id": str(artist_ids[0])})
            if isinstance(artist_detail, dict):
                data = artist_detail.get("data") if isinstance(artist_detail.get("data"), dict) else artist_detail
                stats = data.get("identify") if isinstance(data.get("identify"), dict) else {}
                artist = data.get("artist") if isinstance(data.get("artist"), dict) else {}
                enr.artist_follow_count = (
                    _maybe_int(data.get("followCount"))
                    or _maybe_int(data.get("fansCount"))
                    or _maybe_int(stats.get("imageDesc"))
                    or _maybe_int(artist.get("followCount"))
                )
            artist_dynamic = self._netease_get("/artist/detail/dynamic", {"id": str(artist_ids[0])})
            if isinstance(artist_dynamic, dict) and enr.artist_follow_count is None:
                enr.artist_follow_count = (
                    _maybe_int(artist_dynamic.get("followCount"))
                    or _maybe_int(artist_dynamic.get("fansCount"))
                    or _maybe_int(artist_dynamic.get("subCount"))
                )

        url_payload = self._netease_get("/song/url/v1", {"id": str(sid), "level": "standard"})
        if isinstance(url_payload, dict):
            rows = url_payload.get("data")
            if isinstance(rows, list) and rows:
                row = rows[0] if isinstance(rows[0], dict) else {}
                url = row.get("url")
                code = _maybe_int(row.get("code"))
                fee = _maybe_int(row.get("fee"))
                enr.playable = bool(url) and (code is None or code == 200) and fee != 1
                level = str(row.get("level") or "").lower()
                br = _maybe_int(row.get("br"))
                enr.audio_quality = self._audio_quality(level, br)

        lyric = self._netease_get("/lyric", {"id": str(sid)})
        if isinstance(lyric, dict):
            lrc = lyric.get("lrc") if isinstance(lyric.get("lrc"), dict) else {}
            text = str(lrc.get("lyric") or "")
            enr.lyric_excerpt = " ".join(_tokens(text)[:160])

        simi = self._netease_get("/simi/song", {"id": str(sid)})
        if isinstance(simi, dict):
            songs = simi.get("songs")
            if isinstance(songs, list):
                for row in songs[:20]:
                    if isinstance(row, dict):
                        sim_id = _maybe_int(row.get("id"))
                        if sim_id:
                            enr.similar_song_ids.append(sim_id)

        self._cache_enrichment(cache_key, enr)
        return enr

    def _cache_enrichment(self, cache_key: str, enr: _CandidateEnrichment) -> None:
        if self._cache is None:
            return
        try:
            self._cache.set_query(cache_key, [enr.to_cache_payload()])
        except Exception:  # noqa: BLE001
            pass

    def _netease_get(self, path: str, params: dict[str, str]) -> Any:
        getter = getattr(self._client, "_get", None)
        if not callable(getter):
            return None
        try:
            return getter(path, params)
        except Exception as exc:  # noqa: BLE001
            log.debug("NetEase enrichment %s failed for %s: %s", path, params, exc)
            return None

    @staticmethod
    def _audio_quality(level: str, br: Optional[int]) -> Optional[float]:
        if level:
            if level in {"hires", "jyeffect", "lossless", "exhigh"}:
                return 1.0
            if level in {"higher", "standard"}:
                return 0.8
            if level in {"medium"}:
                return 0.6
            if level in {"low"}:
                return 0.4
        if br is None:
            return None
        if br >= 999000:
            return 1.0
        if br >= 320000:
            return 0.85
        if br >= 192000:
            return 0.70
        if br >= 128000:
            return 0.55
        return 0.35

    def _filter_unplayable(self, candidates: dict[int, _Candidate]) -> int:
        dropped = 0
        for sid, cand in list(candidates.items()):
            if cand.enrichment is not None and cand.enrichment.playable is False:
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    def _filter_low_trust(self, candidates: dict[int, _Candidate]) -> int:
        dropped = 0
        for sid, cand in list(candidates.items()):
            if cand.enrichment is None or not cand.enrichment.enriched:
                continue
            pop = self._popularity_score(cand.enrichment)
            authority = self._artist_authority_score(cand.enrichment)
            route_strength = self._retrieval_confidence(cand, self._metadata_quality(cand.track))
            proxy = self._collaborative_proxy(cand)
            comments = cand.enrichment.comment_count
            followers = cand.enrichment.artist_follow_count
            weak_comments = comments is not None and comments < self._min_comment_count
            weak_artist = followers is not None and followers < self._min_artist_follow_count
            if weak_comments and weak_artist and proxy <= 0.35 and route_strength <= 0.55:
                candidates.pop(sid, None)
                dropped += 1
            elif pop <= 0.05 and authority <= 0.05 and proxy < 0.25 and route_strength < 0.45:
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    def _build_retrieval_queries(
        self,
        profile: _Profile,
        req: RealSongRequest,
        per_query: int,
    ) -> list[_RetrievalQuery]:
        queries: list[_RetrievalQuery] = []
        seen: set[tuple[str, str]] = set()

        def add(source_type: str, query: str, reliability: float, limit: int) -> None:
            q = " ".join(str(query or "").split())
            if not q:
                return
            key = (source_type, q.lower())
            if key in seen:
                return
            seen.add(key)
            queries.append(_RetrievalQuery(
                source_name=f"{source_type}:{q}",
                source_type=source_type,
                query=q,
                reliability=float(reliability),
                limit=max(1, int(limit)),
            ))

        top_genre = profile.preferred_genres[0] if profile.preferred_genres else ""
        top_mood = profile.preferred_moods[0] if profile.preferred_moods else ""
        top_tag = profile.preferred_tags[0] if profile.preferred_tags else (
            profile.tag_phrases[0] if profile.tag_phrases else ""
        )

        for artist in profile.liked_artists_display[: self._max_artist_queries]:
            add("artist", artist, 0.85, per_query)
            if top_genre:
                add("artist_context", f"{artist} {top_genre}", 0.80, per_query)
            if top_mood:
                add("artist_context", f"{artist} {top_mood}", 0.80, per_query)

        for genre in profile.preferred_genres[: self._max_tag_queries]:
            add("genre", genre, 0.75, per_query)
        for mood in profile.preferred_moods[: self._max_tag_queries]:
            add("mood", mood, 0.65, per_query)
        for tag in profile.preferred_tags[: self._max_tag_queries]:
            add("tag", tag, 0.75, per_query)
        if top_genre and top_mood:
            add("genre_mood", f"{top_genre} {top_mood}", 0.75, per_query)
        if len(profile.preferred_tags) >= 2:
            add("tag_combo", " ".join(profile.preferred_tags[:2]), 0.72, per_query)

        for song in req.liked_songs[: self._max_title_queries]:
            if song.title and song.artist:
                add("seed_song", f"{song.title} {song.artist}", 0.78, per_query)
            album_keyword = self._album_keyword(song.album)
            if song.artist and album_keyword:
                add("seed_album", f"{song.artist} {album_keyword}", 0.70, per_query)
            if song.title:
                add("title", song.title, 0.45, max(4, min(per_query, 8)))

        if int(req.discovery_limit) > 0:
            dlim = int(req.discovery_limit)
            if top_genre and top_mood:
                add("discovery", f"{top_genre} {top_mood}", 0.55, dlim)
            if top_tag:
                add("discovery", top_tag, 0.55, dlim)
            if profile.liked_artists_display and top_tag:
                add("discovery", f"{profile.liked_artists_display[0]} {top_tag}", 0.55, dlim)
            mixed = " ".join(profile.query_intent_terms[:4])
            if mixed:
                add("discovery", mixed, 0.55, dlim)

        return queries

    @staticmethod
    def _album_keyword(album: str) -> str:
        toks = _tokens(album)
        return " ".join(toks[:2])

    @staticmethod
    def _summary_bucket(source_type: str) -> str:
        if source_type in {"artist", "artist_context"}:
            return "artist"
        if source_type in {"genre", "mood", "tag", "genre_mood", "tag_combo"}:
            return "tag"
        if source_type in {"seed_song", "seed_album", "title"}:
            return "title"
        return "discovery"

    @staticmethod
    def _filter_missing_metadata(candidates: dict[int, _Candidate]) -> int:
        complete = {
            sid for sid, cand in candidates.items()
            if cand.track.title.strip() and cand.track.artist.strip()
        }
        if not complete:
            return 0
        dropped = 0
        for sid, cand in list(candidates.items()):
            if not cand.track.title.strip() or not cand.track.artist.strip():
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    @staticmethod
    def _filter_same_liked_title(candidates: dict[int, _Candidate], profile: _Profile) -> int:
        if not profile.liked_title_norms:
            return 0
        dropped = 0
        for sid, cand in list(candidates.items()):
            if _norm_title(cand.track.title) in profile.liked_title_norms:
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    def _filter_tag_title_shortcuts(
        self,
        candidates: dict[int, _Candidate],
        profile: _Profile,
    ) -> int:
        if not profile.tag_tokens:
            return 0
        dropped = 0
        for sid, cand in list(candidates.items()):
            if self._is_tag_title_shortcut(cand, profile):
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    def _filter_duplicate_versions(
        self,
        candidates: dict[int, _Candidate],
        profile: _Profile,
    ) -> int:
        raw_intent = " ".join(
            profile.title_phrases
            + profile.liked_artists_display
            + profile.preferred_genres
            + profile.preferred_moods
            + profile.preferred_tags
        )
        requested_versions = bool(set(_raw_tokens(raw_intent)) & _VERSION_WORDS)
        grouped: dict[tuple[str, str], list[tuple[int, _Candidate]]] = {}
        for sid, cand in candidates.items():
            key = (_norm_title(cand.track.title), _norm_artist(cand.track.artist))
            if key[0] and key[1]:
                grouped.setdefault(key, []).append((sid, cand))

        dropped = 0
        for _key, rows in grouped.items():
            if len(rows) <= 1:
                continue
            rows.sort(key=lambda row: self._candidate_quality_key(row[1]), reverse=True)
            keep_sid = rows[0][0]
            for sid, cand in rows[1:]:
                if requested_versions and self._has_version_word(cand.track.title):
                    continue
                if candidates.pop(sid, None) is not None:
                    dropped += 1
            if keep_sid not in candidates and rows:
                candidates[keep_sid] = rows[0][1]
        return dropped

    @staticmethod
    def _candidate_quality_key(cand: _Candidate) -> tuple[float, float, int]:
        best_rel = max((h.reliability for h in cand.source_hits), default=0.0)
        best_rank = max((1.0 / (1.0 + 0.20 * h.position) for h in cand.source_hits), default=0.0)
        meta = 0
        meta += 1 if cand.track.title else 0
        meta += 1 if cand.track.artist else 0
        meta += 1 if cand.track.album else 0
        meta += 1 if cand.track.cover_url else 0
        return (best_rel, best_rank, meta)

    @staticmethod
    def _has_version_word(text: str) -> bool:
        return bool(set(_raw_tokens(text)) & _VERSION_WORDS)

    def _search_cached(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Cache-first NetEase /search. Returns [] on empty / failure."""
        q = (query or "").strip()
        if not q:
            return []

        cache_key = f"songrec_demo:{q}::{int(limit)}"
        if self._cache is not None:
            hit = self._cache.get_query(cache_key)
            if hit is not None:
                return hit

        try:
            hits = self._client.search_songs(q, limit=int(limit)) or []
        except Exception as exc:                  # noqa: BLE001 -- never crash the demo
            log.warning("NetEase search %r failed (%s); returning no hits.", q, exc)
            hits = []

        if self._cache is not None:
            try:
                self._cache.set_query(cache_key, hits)
            except Exception as exc:              # noqa: BLE001
                log.debug("NetEase cache write failed for %r: %s", q, exc)
        return hits

    @staticmethod
    def _is_tag_title_shortcut(cand: _Candidate, profile: _Profile) -> bool:
        """True when a candidate looks like a search-term artifact.

        NetEase full-text search can return tracks literally named
        "alternative rock" for a genre query. Without genre metadata,
        that title string is not evidence that the track actually fits
        the user's taste, so tag/discovery-only candidates whose whole
        title is just the requested tag vocabulary are filtered.
        """
        source_kinds = {h.source_type for h in cand.source_hits}
        weak_tag_sources = {"genre", "mood", "tag", "genre_mood", "tag_combo", "discovery"}
        if source_kinds - weak_tag_sources:
            return False

        title_key = _norm_title(cand.track.title)
        if not title_key:
            return False

        tag_phrase_keys = {_norm_title(p) for p in profile.tag_phrases}
        tag_phrase_keys.discard("")
        if title_key in tag_phrase_keys:
            return True

        title_tokens = set(_tokens(cand.track.title))
        tag_tokens = set(profile.tag_tokens)
        return bool(title_tokens) and title_tokens <= tag_tokens

    @staticmethod
    def _merge_hits(
        candidates: dict[int, _Candidate],
        hits: list[dict[str, Any]],
        *,
        source_name: str,
        source_type: str,
        query: str,
        reliability: float,
    ) -> None:
        for pos, h in enumerate(hits):
            try:
                sid = int(h.get("netease_song_id"))
            except (TypeError, ValueError):
                continue
            if not sid:
                continue
            track = TrackRef(
                netease_song_id=sid,
                title=str(h.get("title") or "").strip(),
                artist=str(h.get("artist") or "").strip(),
                artists=[str(a).strip() for a in (h.get("artists") or []) if a],
                album=str(h.get("album") or "").strip(),
                cover_url=str(h.get("cover_url") or "").strip(),
            )
            cand = candidates.get(sid)
            if cand is None:
                cand = _Candidate(track=track)
                candidates[sid] = cand
            else:
                # Prefer the richer track payload when both exist.
                cand.track = _merge_track(cand.track, track)
            if source_name not in cand.sources:
                cand.sources.append(source_name)
            # Keep the earliest position seen for a given source label.
            cand.positions.setdefault(source_name, pos)
            cand.source_hits.append(_SourceHit(
                source_name=source_name,
                source_type=source_type,
                query=query,
                reliability=float(reliability),
                position=int(pos),
            ))

    # ------------------------------------------------------------------
    # 2. Scoring
    # ------------------------------------------------------------------

    def _score_all(
        self,
        candidates: dict[int, _Candidate],
        profile: _Profile,
        req: RealSongRequest,
    ) -> list[tuple[float, _Candidate, dict[str, float]]]:
        out: list[tuple[float, _Candidate, dict[str, float]]] = []
        nw = _clip01(req.novelty)
        cand_list = list(candidates.values())
        profile_vec: dict[str, float] = {}
        cand_vecs: list[dict[str, float]] = []
        if cand_list:
            docs = [profile.user_profile_text] + [self._candidate_text(c, profile) for c in cand_list]
            vectors = _tfidf_vectors(docs)
            if len(vectors) == len(docs):
                profile_vec = vectors[0]
                cand_vecs = vectors[1:]
        for idx, cand in enumerate(cand_list):
            cand.text_vector = cand_vecs[idx] if idx < len(cand_vecs) else {}
            content_sim = _cosine(profile_vec, cand.text_vector) if profile_vec and cand.text_vector else 0.0
            breakdown = self._score_one(cand, profile, nw, content_sim)
            out.append((breakdown["final_score"], cand, breakdown))
        return out

    def _score_one(
        self,
        cand: _Candidate,
        profile: _Profile,
        novelty_w: float,
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

        metadata_quality_score = self._metadata_quality(track)
        retrieval_confidence_score = self._retrieval_confidence(cand, metadata_quality_score)
        collaborative_proxy_score = self._collaborative_proxy(cand)
        popularity_score = self._popularity_score(cand.enrichment)
        artist_authority_score = self._artist_authority_score(cand.enrichment)
        playable_score = self._playable_score(cand.enrichment)
        audio_quality_score = self._audio_quality_score(cand.enrichment)

        content_score = (
            0.35 * content_text_similarity
            + 0.30 * artist_match
            + 0.25 * tag_match
            + 0.10 * title_match
        )

        artist_in_liked = (
            1.0 if (profile.liked_artists_norm and (cand_artist_norms & profile.liked_artists_norm))
            else 0.0
        )
        artist_affinity_score = self._artist_affinity(cand, profile, artist_match)
        novelty_score = self._novelty_score(cand, profile, artist_in_liked, tag_match)
        trust_score = self._trust_score(
            metadata_quality_score=metadata_quality_score,
            popularity_score=popularity_score,
            artist_authority_score=artist_authority_score,
            playable_score=playable_score,
            collaborative_proxy_score=collaborative_proxy_score,
            retrieval_confidence_score=retrieval_confidence_score,
        )

        base_relevance = (
            0.25 * content_score
            + 0.18 * collaborative_proxy_score
            + 0.15 * retrieval_confidence_score
            + 0.12 * artist_affinity_score
            + 0.10 * popularity_score
            + 0.10 * artist_authority_score
            + 0.05 * playable_score
            + 0.05 * metadata_quality_score
        )
        relevance_gate = min(1.0, base_relevance / 0.55)
        novelty_bonus = novelty_w * novelty_score * relevance_gate
        final_score = min(1.0, max(0.0, base_relevance + novelty_bonus))

        return {
            "final":             float(final_score),
            "final_score":       float(final_score),
            "base_relevance":    float(base_relevance),
            "content":           float(content_score),
            "content_score":     float(content_score),
            "content_text_similarity": float(content_text_similarity),
            "artist_match":      float(artist_match),
            "tag_match":         float(tag_match),
            "tag_title_overlap": float(tag_title_overlap),
            "title_match":       float(title_match),
            "retrieval":         float(retrieval_confidence_score),
            "retrieval_confidence_score": float(retrieval_confidence_score),
            "multi_source":      float(collaborative_proxy_score),
            "collaborative_proxy_score": float(collaborative_proxy_score),
            "artist_affinity_score": float(artist_affinity_score),
            "popularity_score": float(popularity_score),
            "artist_authority_score": float(artist_authority_score),
            "playable_score": float(playable_score),
            "audio_quality_score": float(audio_quality_score),
            "metadata_quality_score": float(metadata_quality_score),
            "trust_score": float(trust_score),
            "novelty_term":      float(novelty_score),
            "novelty_score":     float(novelty_score),
            "relevance_gate":    float(relevance_gate),
            "novelty_bonus":     float(novelty_bonus),
        }

    @staticmethod
    def _candidate_text(cand: _Candidate, profile: _Profile) -> str:
        source_queries = " ".join(h.query for h in cand.source_hits)
        inferred = []
        for phrase in profile.tag_phrases:
            ptoks = set(_tokens(phrase))
            if ptoks and ptoks <= set(_tokens(source_queries)):
                inferred.append(phrase)
        return _profile_text([
            cand.track.title,
            cand.track.artist,
            " ".join(cand.track.artists or []),
            cand.track.album,
            source_queries,
            " ".join(inferred),
            cand.enrichment.lyric_excerpt if cand.enrichment else "",
            cand.enrichment.wiki_summary if cand.enrichment else "",
        ])

    @staticmethod
    def _metadata_quality(track: TrackRef) -> float:
        weights = [
            (track.title, 0.35),
            (track.artist, 0.30),
            (track.album, 0.15),
            (track.cover_url, 0.10),
            (track.netease_song_id, 0.10),
        ]
        return min(1.0, sum(w for val, w in weights if bool(val)))

    @staticmethod
    def _retrieval_confidence(cand: _Candidate, metadata_quality_score: float) -> float:
        if not cand.source_hits:
            return 0.0
        vals = []
        for hit in cand.source_hits:
            rank_score = 1.0 / (1.0 + 0.20 * max(0, hit.position))
            vals.append(rank_score * _clip01(hit.reliability))
        best = max(vals)
        avg = sum(vals) / len(vals)
        return min(1.0, (0.70 * best + 0.30 * avg) * (0.80 + 0.20 * metadata_quality_score))

    @staticmethod
    def _collaborative_proxy(cand: _Candidate) -> float:
        """Multi-source co-occurrence signal, not trained CF."""
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

    @staticmethod
    def _popularity_score(enr: Optional[_CandidateEnrichment]) -> float:
        if enr is None or not enr.enriched:
            return 0.55
        comments = max(0, enr.comment_count or 0)
        hot = max(0, enr.hot_comment_count or 0)
        red = max(0, enr.song_red_count or 0)
        raw = comments + 3.0 * hot + 0.5 * red
        if raw <= 0:
            return 0.0
        return min(1.0, math.log1p(raw) / math.log1p(5000.0))

    @staticmethod
    def _artist_authority_score(enr: Optional[_CandidateEnrichment]) -> float:
        if enr is None or not enr.enriched or enr.artist_follow_count is None:
            return 0.55
        followers = max(0, int(enr.artist_follow_count))
        if followers <= 0:
            return 0.0
        return min(1.0, math.log1p(followers) / math.log1p(500000.0))

    @staticmethod
    def _playable_score(enr: Optional[_CandidateEnrichment]) -> float:
        if enr is None or not enr.enriched or enr.playable is None:
            return 0.60
        return 1.0 if enr.playable else 0.0

    @staticmethod
    def _audio_quality_score(enr: Optional[_CandidateEnrichment]) -> float:
        if enr is None or enr.audio_quality is None:
            return 0.60
        return _clip01(enr.audio_quality)

    @staticmethod
    def _trust_score(
        *,
        metadata_quality_score: float,
        popularity_score: float,
        artist_authority_score: float,
        playable_score: float,
        collaborative_proxy_score: float,
        retrieval_confidence_score: float,
    ) -> float:
        return min(1.0, (
            0.20 * metadata_quality_score
            + 0.20 * popularity_score
            + 0.20 * artist_authority_score
            + 0.15 * playable_score
            + 0.15 * collaborative_proxy_score
            + 0.10 * retrieval_confidence_score
        ))

    @staticmethod
    def _artist_affinity(cand: _Candidate, profile: _Profile, artist_match: float) -> float:
        norms = {_norm_artist(a) for a in (cand.track.artists or [cand.track.artist])}
        norms.discard("")
        weighted = max((profile.seed_artist_weights.get(n, 0.0) for n in norms), default=0.0)
        return min(1.0, max(float(artist_match), weighted))

    @staticmethod
    def _novelty_score(
        cand: _Candidate,
        profile: _Profile,
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

    # ------------------------------------------------------------------
    # 3. MMR rerank with per-artist cap
    # ------------------------------------------------------------------

    def _rerank_mmr(
        self,
        scored: list[tuple[float, _Candidate, dict[str, float]]],
        profile: _Profile,
        req: RealSongRequest,
    ) -> list[tuple[int, _Candidate, dict[str, float], dict[str, Any]]]:
        """Greedy MMR: pick the candidate maximising

            (1 - lambda) * score - lambda * max_sim_to_already_picked

        where similarity blends artist, tag/mood/genre, album, and
        text-vector overlap. A dynamic per-artist cap keeps the list
        from being dominated by one catalogue.
        """
        k = max(1, min(50, int(req.k)))
        lam = _clip01(req.diversity)
        if not scored:
            return []

        remaining = list(scored)
        chosen: list[tuple[int, _Candidate, dict[str, float], dict[str, Any]]] = []
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

            mmr_meta = {
                "lambda":     float(lam),
                "promoted":   bool(id(cand) not in score_only_topk),
            }
            chosen.append((rank, cand, breakdown, mmr_meta))
            rank += 1

        return chosen

    @staticmethod
    def _max_sim_to_chosen(
        cand: _Candidate,
        chosen: list[tuple[int, _Candidate, dict[str, float], dict[str, Any]]],
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

    # ------------------------------------------------------------------
    # 4. Cards + explanations
    # ------------------------------------------------------------------

    def _build_cards(
        self,
        ranked: list[tuple[int, _Candidate, dict[str, float], dict[str, Any]]],
        profile: _Profile,
    ) -> list[RealSongCard]:
        out: list[RealSongCard] = []
        for rank, cand, breakdown, mmr_meta in ranked:
            matched = self._matched_tags(cand, profile)
            reasons = self._reasons(cand, profile, breakdown, matched)
            pick_type = self._pick_type(breakdown, mmr_meta)
            explanation = self._explanation(cand, profile, breakdown, pick_type, matched)
            out.append(
                RealSongCard(
                    rank=rank,
                    track=cand.track,
                    score=float(breakdown["final"]),
                    score_breakdown=breakdown,
                    explanation=explanation,
                    reasons=reasons,
                    matched_tags=matched,
                    sources=list(cand.sources),
                    pick_type=pick_type,
                )
            )
        return out

    @staticmethod
    def _matched_tags(cand: _Candidate, profile: _Profile) -> list[str]:
        if not profile.tag_tokens:
            return []
        cand_text_tokens = (
            _token_set(cand.track.artist)
            | _token_set(cand.track.album)
        )
        matched: list[str] = []
        seen: set[str] = set()
        # Keep tag-phrase ordering and prefer the original (unsplit)
        # phrase when ALL of its atomic tokens are present in the
        # candidate's bag. Otherwise fall back to per-token matches.
        for phrase in profile.tag_phrases:
            parts = _tokens(phrase)
            if parts and all(p in cand_text_tokens for p in parts):
                if phrase.lower() not in seen:
                    matched.append(phrase)
                    seen.add(phrase.lower())
        for tok in profile.tag_tokens:
            if tok in cand_text_tokens and tok not in seen:
                matched.append(tok)
                seen.add(tok)
        return matched

    def _reasons(
        self,
        cand: _Candidate,
        profile: _Profile,
        breakdown: dict[str, float],
        matched_tags: list[str],
    ) -> list[str]:
        reasons: list[str] = []
        # Artist reasons.
        cand_artists_norm = {_norm_artist(a) for a in (cand.track.artists or [cand.track.artist])}
        cand_artists_norm.discard("")
        same_liked = cand_artists_norm & profile.liked_artists_norm
        if same_liked:
            # Map back to a display name.
            for disp in profile.liked_artists_display:
                if _norm_artist(disp) in same_liked:
                    reasons.append(f"Same artist as someone you like: {disp}")
                    break

        # Tag / mood reasons.
        if matched_tags:
            shown = ", ".join(matched_tags[:3])
            reasons.append(f"Matches your tags: {shown}")

        # Title-overlap reason (only when meaningfully strong).
        if breakdown.get("title_match", 0.0) >= 0.34 and profile.title_phrases:
            ref = profile.title_phrases[0]
            reasons.append(f"Title overlaps with your liked song \u201c{ref}\u201d")

        # Retrieval reason.
        chans = sorted(set(h.source_type for h in cand.source_hits))
        if "discovery" in chans and len(chans) == 1:
            reasons.append("Surfaced by the discovery channel")
        elif len(chans) >= 2:
            reasons.append(f"Found through multiple preference paths: {', '.join(chans[:4])}")

        # Novelty reason.
        if breakdown.get("novelty_score", 0.0) >= 0.45 and not same_liked:
            reasons.append("Broader discovery pick with enough relevance support")

        if breakdown.get("metadata_quality_score", 0.0) >= 0.90:
            reasons.append("Complete real-song metadata: title, artist, album, cover, and link")

        if not reasons:
            reasons.append("General match against your taste profile")
        return reasons

    @staticmethod
    def _pick_type(
        breakdown: dict[str, float],
        mmr_meta: dict[str, Any],
    ) -> str:
        if mmr_meta.get("promoted"):
            return "diverse"
        base = breakdown.get("base_relevance", 0.0)
        novelty = breakdown.get("novelty_score", 0.0)
        gate = breakdown.get("relevance_gate", 0.0)
        if base >= 0.62 and (
            breakdown.get("artist_affinity_score", 0.0) >= 0.75
            or breakdown.get("content_score", 0.0) >= 0.55
            or breakdown.get("retrieval_confidence_score", 0.0) >= 0.75
        ):
            return "safe"
        if (
            breakdown.get("tag_title_overlap", 0.0) >= 0.50
            and breakdown.get("tag_match", 0.0) <= 0.0
        ):
            return "exploratory"
        if novelty >= 0.55 and gate >= 0.65:
            return "exploratory"
        return "balanced"

    @staticmethod
    def _explanation(
        cand: _Candidate,
        profile: _Profile,
        breakdown: dict[str, float],
        pick_type: str,
        matched_tags: list[str],
    ) -> str:
        artist = cand.track.artist or "Unknown artist"
        if pick_type == "safe":
            return (
                "This is a close match because it has strong relevance support "
                "from your artists, content profile, or high-confidence retrieval paths."
            )
        if pick_type == "exploratory":
            tags = ", ".join(matched_tags[:2]) if matched_tags else "your profile"
            return (
                f"This is a broader discovery pick: it keeps {tags} in view "
                f"but comes from {artist}, giving the list a less obvious option."
            )
        if pick_type == "diverse":
            return (
                "This was added to make the list less repetitive while still "
                "matching enough of your selected mood, genre, or artist profile."
            )
        return (
            f"This is a balanced recommendation: '{cand.track.title}' by {artist} "
            "has a reasonable mix of profile similarity and retrieval support."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / float(len(a | b))


def _tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    tokenised = [_tokens(d) for d in docs]
    if len(tokenised) < 2 or not any(tokenised):
        return [{} for _ in docs]
    df: dict[str, int] = {}
    for toks in tokenised:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    n_docs = float(len(tokenised))
    vectors: list[dict[str, float]] = []
    for toks in tokenised:
        if not toks:
            vectors.append({})
            continue
        counts: dict[str, int] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
        total = float(len(toks))
        vec: dict[str, float] = {}
        for tok, count in counts.items():
            idf = math.log((1.0 + n_docs) / (1.0 + float(df.get(tok, 0)))) + 1.0
            vec[tok] = (float(count) / total) * idf
        vectors.append(vec)
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return min(1.0, max(0.0, num / (da * db)))


def _source_preference_tokens(cand: _Candidate) -> set[str]:
    pref_types = {"genre", "mood", "tag", "genre_mood", "tag_combo", "discovery", "artist_context"}
    out: set[str] = set()
    for hit in cand.source_hits:
        if hit.source_type in pref_types:
            out.update(_tokens(hit.query))
    return out


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


# ---------------------------------------------------------------------------
# Test double for hermetic smoke tests
# ---------------------------------------------------------------------------

class FakeNeteaseClient:
    """Deterministic in-memory NetEase client used by tests.

    Construct with a mapping ``query_lower -> list[song dict]`` and
    optionally a fallback list returned for any unknown query. The
    canned songs match the shape returned by
    :class:`NeteaseAPIClient.search_songs` so the recommender can be
    swapped against this without code changes.
    """

    def __init__(
        self,
        responses: Optional[dict[str, list[dict[str, Any]]]] = None,
        *,
        default: Optional[list[dict[str, Any]]] = None,
        enrichments: Optional[dict[int, dict[str, Any]]] = None,
        alive: bool = True,
    ) -> None:
        self._responses = {k.lower(): list(v) for k, v in (responses or {}).items()}
        self._default = list(default or [])
        self._enrichments = {int(k): dict(v) for k, v in (enrichments or {}).items()}
        self._alive = bool(alive)
        self.calls: list[tuple[str, int]] = []

    def search_songs(self, keywords: str, limit: int = 5) -> list[dict[str, Any]]:
        self.calls.append((keywords, int(limit)))
        if not self._alive:
            return []
        key = (keywords or "").strip().lower()
        hits = self._responses.get(key, self._default)
        return list(hits)[: int(limit)]

    def ping(self) -> bool:
        return self._alive

    def enrich_song(self, song_id: int) -> dict[str, Any]:
        return dict(self._enrichments.get(int(song_id), {
            "comment_count": 120,
            "hot_comment_count": 3,
            "song_red_count": 300,
            "artist_follow_count": 1200,
            "playable": True,
            "audio_quality": 0.8,
            "similar_song_ids": [],
        }))


__all__ = [
    "TrackRef",
    "RealSongRequest",
    "RealSongCard",
    "RealSongResponse",
    "NeteaseRecommender",
    "FakeNeteaseClient",
]
