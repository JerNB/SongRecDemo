"""Explainer -- stage 7 of the pipeline.

Turns ranked candidates into front-end-friendly :class:`RealSongCard`
objects: matched tags, a ``pick_type`` label, human-readable ``reasons``,
and an ``explanation`` sentence. Wording deliberately avoids any
collaborative-filtering / "similar users liked this" claims -- the
product layer has no real user-item matrix. It speaks in terms of
multi-source agreement / independent search paths instead.
"""

from __future__ import annotations

from typing import Any

from .text import _norm_artist, _token_set, _tokens
from .types import Candidate, RealSongCard, UserProfile


class Explainer:
    def build_cards(
        self,
        ranked: list[tuple[int, Candidate, dict[str, float], dict[str, Any]]],
        profile: UserProfile,
    ) -> list[RealSongCard]:
        out: list[RealSongCard] = []
        for rank, cand, breakdown, mmr_meta in ranked:
            matched = self._matched_tags(cand, profile)
            pick_type = self._pick_type(breakdown, mmr_meta)
            reasons = self._reasons(cand, profile, breakdown, matched, pick_type)
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
    def _matched_tags(cand: Candidate, profile: UserProfile) -> list[str]:
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
        cand: Candidate,
        profile: UserProfile,
        breakdown: dict[str, float],
        matched_tags: list[str],
        pick_type: str,
    ) -> list[str]:
        reasons: list[str] = []
        cand_artists_norm = {_norm_artist(a) for a in (cand.track.artists or [cand.track.artist])}
        cand_artists_norm.discard("")
        same_liked = cand_artists_norm & profile.liked_artists_norm

        # Artist signal: exact liked artist, or token / affinity overlap.
        # NOTE: never phrase this as "similar users liked this" -- there
        # is no real user-item interaction matrix behind the demo.
        if same_liked:
            disp = next((d for d in profile.liked_artists_display
                         if _norm_artist(d) in same_liked), "")
            if disp:
                reasons.append(
                    f"Matches your preferred artist or similar artist signals: {disp}")
            else:
                reasons.append("Matches your preferred artist or similar artist signals")
        elif (breakdown.get("artist_match", 0.0) >= 0.5
              or breakdown.get("artist_affinity_score", 0.0) >= 0.5):
            reasons.append("Matches your preferred artist or similar artist signals")

        # Genre / mood / tag profile.
        if matched_tags:
            shown = ", ".join(matched_tags[:3])
            reasons.append(f"Matches your selected genre / mood / tag profile: {shown}")
        elif breakdown.get("tag_match", 0.0) >= 0.34:
            reasons.append("Matches your selected genre / mood / tag profile")

        # Liked-song title overlap.
        if breakdown.get("title_match", 0.0) >= 0.34 and profile.title_phrases:
            ref = profile.title_phrases[0]
            reasons.append(f"Title overlaps with your liked song \u201c{ref}\u201d")

        # Multi-source agreement / retrieval consensus: surfaced across
        # several independent search channels.
        channels = sorted({h.source_type for h in cand.source_hits})
        if breakdown.get("multi_source_agreement", 0.0) >= 0.5 or len(channels) >= 2:
            reasons.append("Found through multiple independent search paths")

        # Quality / platform engagement signals.
        if (breakdown.get("quality_score", 0.0) >= 0.60
                or breakdown.get("metadata_quality_score", 0.0) >= 0.90):
            reasons.append("Strong metadata and platform engagement signals")

        # Pick-type framing (mirrors the explanation copy).
        if pick_type == "exploratory":
            reasons.append("Added for discovery because it is relevant but less obvious")
        elif pick_type == "diverse":
            reasons.append("Added for diversity to avoid repeating the same artist or album")

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
        cand: Candidate,
        profile: UserProfile,
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
