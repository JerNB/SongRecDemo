"""CandidateFilter -- stage 3 of the pipeline.

Two phases, both mutating the ``{song_id: Candidate}`` dict in place and
returning a stats dict so the trace / candidate_summary can report what
was dropped and why:

* :meth:`pre_enrichment` -- cheap, metadata-only rules run before the
  (expensive) NetEase enrichment: drop liked / excluded ids, candidates
  missing key metadata, covers of liked titles, literal tag-title
  shortcuts, and duplicate versions of the same song.
* :meth:`post_enrichment` -- rules that need enrichment signals:
  unplayable tracks (hard drop) and low-trust weak candidates.

The low-trust rule is intentionally conservative: a candidate is only
hard-dropped when *several* weak signals coincide, so most weak
candidates simply rank lower (soft penalty via the scoring stage)
rather than disappearing.
"""

from __future__ import annotations

from .scoring import (
    artist_authority_score,
    metadata_quality,
    multi_source_agreement,
    popularity_score,
    retrieval_confidence,
)
from .text import (
    _contains_tokens,
    _norm_artist,
    _norm_title,
    _raw_tokens,
    _starts_with_tokens,
    _token_set,
    _tokens,
    _STOP,
    _VERSION_WORDS,
)
from .types import Candidate, UserProfile


class CandidateFilter:
    """Filters + dedups candidates. Configurable trust thresholds keep
    the hard-drop rules tunable from one place."""

    def __init__(
        self,
        *,
        min_comment_count: int = 10,
        min_artist_follow_count: int = 77,
    ) -> None:
        self._min_comment_count = int(min_comment_count)
        self._min_artist_follow_count = int(min_artist_follow_count)

    # ------------------------------------------------------------------
    # Phase 1: pre-enrichment (metadata-only) rules.
    # ------------------------------------------------------------------

    def pre_enrichment(
        self, candidates: dict[int, Candidate], profile: UserProfile,
    ) -> dict[str, int]:
        filtered_liked = 0
        for sid in (profile.liked_track_ids | profile.excluded_track_ids):
            if candidates.pop(int(sid), None) is not None:
                filtered_liked += 1

        filtered_missing_metadata = self._filter_missing_metadata(candidates)
        filtered_same_title = self._filter_same_liked_title(candidates, profile)
        filtered_tag_title = self._filter_tag_title_shortcuts(candidates, profile)
        filtered_duplicate_version = self._filter_duplicate_versions(candidates, profile)

        return {
            "filtered_liked":             int(filtered_liked),
            "filtered_same_title":        int(filtered_same_title),
            "filtered_tag_title":         int(filtered_tag_title),
            "filtered_duplicate_version": int(filtered_duplicate_version),
            "filtered_missing_metadata":  int(filtered_missing_metadata),
        }

    # ------------------------------------------------------------------
    # Phase 2: post-enrichment (signal-dependent) rules.
    # ------------------------------------------------------------------

    def post_enrichment(self, candidates: dict[int, Candidate]) -> dict[str, int]:
        filtered_unplayable = self._filter_unplayable(candidates)
        filtered_low_trust = self._filter_low_trust(candidates)
        return {
            "filtered_unplayable": int(filtered_unplayable),
            "filtered_low_trust":  int(filtered_low_trust),
        }

    @staticmethod
    def _filter_unplayable(candidates: dict[int, Candidate]) -> int:
        dropped = 0
        for sid, cand in list(candidates.items()):
            if cand.enrichment is not None and cand.enrichment.playable is False:
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    def _filter_low_trust(self, candidates: dict[int, Candidate]) -> int:
        dropped = 0
        for sid, cand in list(candidates.items()):
            if cand.enrichment is None or not cand.enrichment.enriched:
                continue
            pop = popularity_score(cand.enrichment)
            authority = artist_authority_score(cand.enrichment)
            route_strength = retrieval_confidence(cand, metadata_quality(cand.track))
            agreement = multi_source_agreement(cand)
            comments = cand.enrichment.comment_count
            followers = cand.enrichment.artist_follow_count
            weak_comments = comments is not None and comments < self._min_comment_count
            weak_artist = followers is not None and followers < self._min_artist_follow_count
            if weak_comments and weak_artist and agreement <= 0.35 and route_strength <= 0.55:
                candidates.pop(sid, None)
                dropped += 1
            elif pop <= 0.05 and authority <= 0.05 and agreement < 0.25 and route_strength < 0.45:
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    # ------------------------------------------------------------------
    # Pre-enrichment rule implementations.
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_missing_metadata(candidates: dict[int, Candidate]) -> int:
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

    def _filter_same_liked_title(
        self, candidates: dict[int, Candidate], profile: UserProfile,
    ) -> int:
        if not profile.liked_title_norms:
            return 0
        dropped = 0
        for sid, cand in list(candidates.items()):
            if self._is_liked_title_variant(cand.track.title, profile):
                candidates.pop(sid, None)
                dropped += 1
        return dropped

    def _filter_tag_title_shortcuts(
        self, candidates: dict[int, Candidate], profile: UserProfile,
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
        self, candidates: dict[int, Candidate], profile: UserProfile,
    ) -> int:
        raw_intent = " ".join(
            profile.title_phrases
            + profile.liked_artists_display
            + profile.preferred_genres
            + profile.preferred_moods
            + profile.preferred_tags
        )
        requested_versions = bool(set(_raw_tokens(raw_intent)) & _VERSION_WORDS)
        grouped: dict[tuple[str, str], list[tuple[int, Candidate]]] = {}
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
    def _candidate_quality_key(cand: Candidate) -> tuple[float, float, int]:
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

    @staticmethod
    def _is_liked_title_variant(title: str, profile: UserProfile) -> bool:
        """Catch covers/versions of songs the user already selected."""
        if not title or not profile.liked_title_norms:
            return False
        if _norm_title(title) in profile.liked_title_norms:
            return True

        raw_tokens = _raw_tokens(title)
        has_version_hint = bool(set(raw_tokens) & _VERSION_WORDS)
        if not has_version_hint:
            return False

        semantic_tokens = [t for t in raw_tokens if t not in _STOP]
        for liked_key in profile.liked_title_norms:
            liked_tokens = liked_key.split()
            if _starts_with_tokens(semantic_tokens, liked_tokens):
                return True
            if _contains_tokens(semantic_tokens, liked_tokens):
                return True
        return False

    @staticmethod
    def _is_tag_title_shortcut(cand: Candidate, profile: UserProfile) -> bool:
        """True when a candidate looks like a search-term artifact.

        NetEase full-text search can return tracks literally named
        "alternative rock" for a genre query. Without genre metadata,
        that title string is not evidence that the track actually fits
        the user's taste, so tag/discovery-only candidates whose whole
        title is just the requested tag vocabulary are filtered.
        """
        title_key = _norm_title(cand.track.title)
        if not title_key:
            return False

        tag_phrase_keys = {_norm_title(p) for p in profile.tag_phrases}
        tag_phrase_keys.discard("")
        if title_key in tag_phrase_keys:
            return True

        title_tokens = set(_tokens(cand.track.title))
        tag_tokens = set(profile.tag_tokens)
        title_tag_hits = title_tokens & tag_tokens
        if not title_tag_hits:
            return False

        source_kinds = {h.source_type for h in cand.source_hits}
        weak_tag_sources = {"genre", "mood", "tag", "genre_mood", "tag_combo", "discovery"}
        if source_kinds - weak_tag_sources:
            return False

        non_title_tokens = _token_set(cand.track.artist) | _token_set(cand.track.album)
        return not bool(non_title_tokens & tag_tokens)
