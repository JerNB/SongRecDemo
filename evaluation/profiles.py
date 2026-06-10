"""Seed profile loading for the offline evaluation harness.

A :class:`SeedProfile` is a fixed, named taste profile used to probe the
recommender deterministically. It maps cleanly onto a
:class:`RealSongRequest` so the harness can drive the live pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from SongRecDemo.netease_pipeline import RealSongRequest, TrackRef

_HERE = Path(__file__).resolve().parent
DEFAULT_SEED_PATH = _HERE / "seed_profiles.json"


@dataclass
class SeedProfile:
    """One fixed evaluation profile."""

    profile_id: str
    label: str = ""
    liked_songs: list[dict[str, Any]] = field(default_factory=list)
    liked_artists: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    content_weight: float = 0.5
    novelty: float = 0.3
    diversity: float = 0.3
    k: int = 10

    def to_request(
        self,
        *,
        content_weight: Optional[float] = None,
        novelty: Optional[float] = None,
        diversity: Optional[float] = None,
        k: Optional[int] = None,
    ) -> RealSongRequest:
        """Build a :class:`RealSongRequest`. The control knobs can be
        overridden so one profile can be swept across configs."""
        liked = []
        for raw in self.liked_songs:
            if not isinstance(raw, dict):
                continue
            try:
                sid = int(raw.get("netease_song_id"))
            except (TypeError, ValueError):
                continue
            liked.append(TrackRef(
                netease_song_id=sid,
                title=str(raw.get("title") or ""),
                artist=str(raw.get("artist") or ""),
                artists=[str(a) for a in (raw.get("artists") or []) if a],
                album=str(raw.get("album") or ""),
                cover_url=str(raw.get("cover_url") or ""),
            ))
        return RealSongRequest(
            liked_songs=liked,
            liked_artists=list(self.liked_artists),
            genres=list(self.genres),
            moods=list(self.moods),
            tags=list(self.tags),
            content_weight=float(self.content_weight if content_weight is None else content_weight),
            novelty=float(self.novelty if novelty is None else novelty),
            diversity=float(self.diversity if diversity is None else diversity),
            k=int(self.k if k is None else k),
            request_id=f"eval:{self.profile_id}",
        )


def load_seed_profiles(
    path: Optional[Union[str, Path]] = None,
) -> list[SeedProfile]:
    """Load the seed profiles JSON into :class:`SeedProfile` objects."""
    p = Path(path) if path is not None else DEFAULT_SEED_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    raw_profiles = data.get("profiles") if isinstance(data, dict) else data
    if not isinstance(raw_profiles, list):
        raise ValueError(f"seed profiles file {p} has no `profiles` list")

    out: list[SeedProfile] = []
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            continue
        out.append(SeedProfile(
            profile_id=str(raw.get("profile_id") or raw.get("id") or f"profile_{len(out)}"),
            label=str(raw.get("label") or ""),
            liked_songs=list(raw.get("liked_songs") or []),
            liked_artists=[str(a) for a in (raw.get("liked_artists") or [])],
            genres=[str(g) for g in (raw.get("genres") or [])],
            moods=[str(m) for m in (raw.get("moods") or [])],
            tags=[str(t) for t in (raw.get("tags") or [])],
            content_weight=float(raw.get("content_weight", 0.5)),
            novelty=float(raw.get("novelty", 0.3)),
            diversity=float(raw.get("diversity", 0.3)),
            k=int(raw.get("k", 10)),
        ))
    return out
