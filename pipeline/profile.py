"""ProfileBuilder -- normalise user input into a standard UserProfile.

Stage 1 of the pipeline. Takes the user-friendly :class:`RealSongRequest`
(liked songs, free-text artists / genres / moods / tags, exclusions, and
the control sliders) and produces a single normalised
:class:`UserProfile` that every downstream stage reads from.
"""

from __future__ import annotations

from .text import (
    _dedupe_phrases,
    _norm_artist,
    _norm_title,
    _profile_text,
    _tokens,
)
from .types import RealSongRequest, UserProfile


class ProfileBuilder:
    """Builds a normalised taste profile from a request.

    Stateless and side-effect free, so it can be unit-tested directly::

        profile = ProfileBuilder().build(RealSongRequest(...))
    """

    def build(self, req: RealSongRequest) -> UserProfile:
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

        return UserProfile(
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
