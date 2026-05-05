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
from dataclasses import dataclass, field, asdict
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
    tags: list[str] = field(default_factory=list)        # tags + moods + genres merged
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
    pick_type: str           # "safe" | "exploratory" | "diverse"

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
})


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

    # Tags: keep originals for retrieval queries (multi-word genres
    # like "indie rock" should hit NetEase as a single phrase) AND
    # break them down into atomic tokens for matching.
    tag_phrases: list[str] = []
    seen_phrase: set[str] = set()
    for t in req.tags:
        if not t:
            continue
        s = str(t).strip()
        sl = s.lower()
        if not s or sl in seen_phrase:
            continue
        seen_phrase.add(sl)
        tag_phrases.append(s)

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
    for s in req.liked_songs:
        if s.title:
            title_phrases.append(s.title)
            title_tokens.update(_tokens(s.title))

    return _Profile(
        liked_track_ids={int(s.netease_song_id) for s in req.liked_songs if s.netease_song_id},
        liked_artists_norm=artists_norm,
        liked_artists_display=artists_display,
        tag_tokens=tag_tokens,
        tag_phrases=tag_phrases,
        title_tokens=title_tokens,
        title_phrases=title_phrases,
        excluded_track_ids={int(x) for x in req.excluded_song_ids if x},
    )


# ---------------------------------------------------------------------------
# Internal candidate record
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    track: TrackRef
    sources: list[str] = field(default_factory=list)
    # For each source, position is 0-indexed rank in that channel's hits.
    positions: dict[str, int] = field(default_factory=dict)


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

        # 2. Score, sort.
        scored = self._score_all(candidates_by_id, profile, req)
        scored.sort(key=lambda x: (-x[0], x[1].track.title or "", x[1].track.netease_song_id))

        # 3. MMR rerank for diversity + per-artist cap.
        ranked = self._rerank_mmr(scored, profile, req)

        # 4. Build cards with explanations + pick type.
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
            "tag_tokens":       list(profile.tag_tokens),
            "title_tokens":     sorted(profile.title_tokens),
        }

    def _model_info(self) -> dict[str, Any]:
        return {
            "name":           self.MODEL_NAME,
            "research_layer": "ALS-Personalized-v1 (KGRec, untouched)",
            "source":         "NetEase /search",
        }

    # ------------------------------------------------------------------
    # 1. Candidate retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self, profile: _Profile, req: RealSongRequest,
    ) -> tuple[dict[int, _Candidate], dict[str, int]]:
        """Issue a small number of /search calls, dedup, label sources."""
        per_query = max(1, min(self._max_per_query, int(req.candidates_per_signal)))

        candidates: dict[int, _Candidate] = {}
        from_artist = from_tag = from_title = from_discovery = 0

        # ---- 1a. Per-artist channels --------------------------------
        artist_queries = list(profile.liked_artists_display)[: self._max_artist_queries]
        for artist in artist_queries:
            hits = self._search_cached(artist, per_query)
            from_artist += len(hits)
            self._merge_hits(candidates, hits, source=f"artist:{artist}")

        # ---- 1b. Per-tag channels -----------------------------------
        # Use the original phrases (e.g. "indie rock") not the atomic
        # tokens, because NetEase's full-text matcher handles compound
        # terms better.
        tag_queries = list(profile.tag_phrases)[: self._max_tag_queries]
        for tag in tag_queries:
            hits = self._search_cached(tag, per_query)
            from_tag += len(hits)
            self._merge_hits(candidates, hits, source=f"tag:{tag}")

        # ---- 1c. Per-title channels ---------------------------------
        title_queries = list(profile.title_phrases)[: self._max_title_queries]
        for title in title_queries:
            hits = self._search_cached(title, per_query)
            from_title += len(hits)
            self._merge_hits(candidates, hits, source=f"title:{title}")

        # ---- 1d. Discovery channel ----------------------------------
        # One broad query built from the first two tag tokens, or
        # from the first liked artist when no tags are given. The
        # goal is to inject some variety beyond what the focused
        # channels return.
        if int(req.discovery_limit) > 0:
            disc_query = self._discovery_query(profile)
            if disc_query:
                hits = self._search_cached(disc_query, int(req.discovery_limit))
                from_discovery += len(hits)
                self._merge_hits(candidates, hits, source="discovery")

        # ---- Drop excluded / liked tracks ---------------------------
        for sid in (profile.liked_track_ids | profile.excluded_track_ids):
            candidates.pop(int(sid), None)

        summary = {
            "artist":        int(from_artist),
            "tag":           int(from_tag),
            "title":         int(from_title),
            "discovery":     int(from_discovery),
            "total_unique":  int(len(candidates)),
        }
        return candidates, summary

    def _discovery_query(self, profile: _Profile) -> str:
        if profile.tag_phrases:
            return " ".join(profile.tag_phrases[:2])
        if profile.liked_artists_display:
            return profile.liked_artists_display[0]
        return ""

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
    def _merge_hits(
        candidates: dict[int, _Candidate],
        hits: list[dict[str, Any]],
        *,
        source: str,
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
            if source not in cand.sources:
                cand.sources.append(source)
            # Keep the earliest position seen for a given source label.
            cand.positions.setdefault(source, pos)

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
        cw = _clip01(req.content_weight)
        nw = _clip01(req.novelty)
        for cand in candidates.values():
            breakdown = self._score_one(cand, profile, cw, nw)
            out.append((breakdown["final"], cand, breakdown))
        return out

    def _score_one(
        self,
        cand: _Candidate,
        profile: _Profile,
        content_weight: float,
        novelty_w: float,
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

        # Tag match: fraction of user's tag tokens appearing in the
        # candidate's metadata bag.
        cand_text_tokens = (
            _token_set(track.title) | cand_artist_tokens | _token_set(track.album)
        )
        if profile.tag_tokens:
            hits = sum(1 for t in profile.tag_tokens if t in cand_text_tokens)
            tag_match = float(hits) / float(len(profile.tag_tokens))
        else:
            tag_match = 0.0

        # Title match: fraction of liked-song title tokens that appear
        # in the candidate's title. Excludes generic stopwords (already
        # handled by _tokens).
        if profile.title_tokens:
            hits = sum(1 for t in _token_set(track.title) if t in profile.title_tokens)
            title_match = min(1.0, float(hits) / max(1.0, math.sqrt(len(profile.title_tokens))))
        else:
            title_match = 0.0

        # Retrieval score: how high did this candidate appear in each
        # channel, averaged. Channel position 0 = 1.0, position 1 =
        # ~0.83, ..., decays with rank. Multi-source bonus on top.
        if cand.positions:
            per_channel = [
                1.0 / (1.0 + 0.20 * pos) for pos in cand.positions.values()
            ]
            retrieval_avg = sum(per_channel) / len(per_channel)
        else:
            retrieval_avg = 0.0
        n_sources = len(cand.sources)
        # Multi-source bonus: 1 source -> 1.00, 2 sources -> 1.15,
        # 3 sources -> 1.25, capped at 1.30 (so a candidate present
        # in five sources doesn't overrun a near-perfect content match).
        multi_source_bonus = min(0.30, 0.15 * max(0, n_sources - 1))
        retrieval_score = min(1.0, retrieval_avg + multi_source_bonus)

        content_score = (
            0.50 * artist_match
            + 0.30 * tag_match
            + 0.20 * title_match
        )

        artist_in_liked = (
            1.0 if (profile.liked_artists_norm and (cand_artist_norms & profile.liked_artists_norm))
            else 0.0
        )
        novelty_term = (1.0 - artist_in_liked) * (1.0 - retrieval_score)

        # Same-artist penalty is applied during MMR rerank, not here,
        # because we want to keep the absolute score interpretable.
        final = (
            (1.0 - content_weight) * retrieval_score
            + content_weight * content_score
            + novelty_w * novelty_term
        )

        return {
            "final":           float(final),
            "content":         float(content_score),
            "artist_match":    float(artist_match),
            "tag_match":       float(tag_match),
            "title_match":     float(title_match),
            "retrieval":       float(retrieval_score),
            "multi_source":    float(multi_source_bonus),
            "novelty_term":    float(novelty_term),
        }

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

        where similarity is "artist matches" (1.0) plus a small bonus
        for tag overlap. Also enforces ``per_artist_cap`` so the
        result list isn't dominated by one artist's catalogue.
        """
        k = max(1, min(50, int(req.k)))
        lam = _clip01(req.diversity)
        if not scored:
            return []

        remaining = list(scored)
        chosen: list[tuple[int, _Candidate, dict[str, float], dict[str, Any]]] = []
        artist_count: dict[str, int] = {}

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
                if primary_artist and artist_count.get(primary_artist, 0) >= self._per_artist_cap:
                    continue
                sim = self._max_sim_to_chosen(cand, chosen)
                obj = (1.0 - lam) * sc - lam * sim
                if obj > best_obj:
                    best_obj = obj
                    best_idx = i
            if best_idx == -1:
                # Per-artist cap blocked everything; relax the cap to
                # finish the list cleanly.
                self._per_artist_cap += 1
                continue

            score, cand, breakdown = remaining.pop(best_idx)
            primary_artist = _norm_artist(cand.track.artist or
                                          (cand.track.artists[0] if cand.track.artists else ""))
            if primary_artist:
                artist_count[primary_artist] = artist_count.get(primary_artist, 0) + 1

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
        cand_artist = _norm_artist(cand.track.artist or "")
        cand_artists_norm = {_norm_artist(a) for a in (cand.track.artists or [cand.track.artist])}
        cand_artists_norm.discard("")
        cand_text_tokens = (
            _token_set(cand.track.title)
            | _token_set(cand.track.artist)
            | _token_set(cand.track.album)
        )
        sims = []
        for _r, oc, _bd, _meta in chosen:
            other_artists = {_norm_artist(a) for a in (oc.track.artists or [oc.track.artist])}
            other_artists.discard("")
            artist_overlap = 1.0 if (cand_artists_norm & other_artists) else 0.0
            other_tokens = (
                _token_set(oc.track.title)
                | _token_set(oc.track.artist)
                | _token_set(oc.track.album)
            )
            tok_sim = _jaccard(cand_text_tokens, other_tokens)
            sims.append(min(1.0, 0.7 * artist_overlap + 0.3 * tok_sim))
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
            _token_set(cand.track.title)
            | _token_set(cand.track.artist)
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
        chans = sorted(set(s.split(":", 1)[0] for s in cand.sources))
        if "discovery" in chans and len(chans) == 1:
            reasons.append("Surfaced by the discovery channel")
        elif len(chans) >= 2:
            reasons.append(f"Found via multiple signals: {', '.join(chans)}")

        # Novelty reason.
        if breakdown.get("novelty_term", 0.0) >= 0.40 and not same_liked:
            reasons.append("Long-tail pick (different artist, deeper search rank)")

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
        if breakdown.get("artist_match", 0.0) >= 0.95 or breakdown.get("content", 0.0) >= 0.55:
            return "safe"
        if breakdown.get("novelty_term", 0.0) >= 0.40:
            return "exploratory"
        return "safe"

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
            if breakdown.get("artist_match", 0.0) >= 0.95:
                return f"Safe pick from {artist}, who is in your liked artists."
            if matched_tags:
                return (
                    f"Safe pick: '{cand.track.title}' by {artist} matches "
                    f"{', '.join(matched_tags[:2])}."
                )
            return f"Safe pick: '{cand.track.title}' by {artist}."
        if pick_type == "exploratory":
            return (
                f"Exploratory pick: '{cand.track.title}' by {artist} -- "
                "different artist, deeper search result, but it lines up with your tags."
            )
        if pick_type == "diverse":
            return (
                f"Diverse pick: '{cand.track.title}' by {artist}, included to "
                "broaden the result list beyond the strongest matches."
            )
        return f"'{cand.track.title}' by {artist}."


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
        alive: bool = True,
    ) -> None:
        self._responses = {k.lower(): list(v) for k, v in (responses or {}).items()}
        self._default = list(default or [])
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


__all__ = [
    "TrackRef",
    "RealSongRequest",
    "RealSongCard",
    "RealSongResponse",
    "NeteaseRecommender",
    "FakeNeteaseClient",
]
