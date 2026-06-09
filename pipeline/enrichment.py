"""FeatureEnricher -- stage 4 of the pipeline.

Deep-enriches the strongest candidates with NetEase platform signals
(comment count, hot comments, liked/red count, artist followers,
playability, audio quality). Results are cached on disk so repeat runs
are cheap and deterministic.

Async / batch hook
------------------
:meth:`enrich` is the batch entry point and currently fetches per song
sequentially. The per-song work is isolated in :meth:`enrich_one`, so a
future async / batched implementation can subclass and override
:meth:`enrich` to fan out :meth:`enrich_one` concurrently without
touching any other stage. The standardised in/out contract is:

    input  : list[Candidate]  (cand.track.netease_song_id is the key)
    output : int              (how many were successfully enriched)
    effect : cand.enrichment is populated for every candidate
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .text import _maybe_int, _tokens
from .types import Candidate, CandidateEnrichment, RealSongRequest, _NeteaseClient, _QueryCache


log = logging.getLogger(__name__)


class FeatureEnricher:
    def __init__(
        self,
        client: _NeteaseClient,
        cache: Optional[_QueryCache] = None,
        *,
        enrich_top_n: int = 16,
        live_deep_enrichment: bool = False,
    ) -> None:
        self._client = client
        self._cache = cache
        self._enrich_top_n = int(enrich_top_n)
        self._live_deep_enrichment = bool(live_deep_enrichment)

    def enrichment_budget(self, req: RealSongRequest) -> int:
        k = max(1, min(50, int(req.k)))
        return max(6, min(self._enrich_top_n, k * 2))

    def enrich(self, candidates: list[Candidate]) -> int:
        """Batch-enrich a list of candidates (sequential today).

        Override this method to add async / batched fan-out; it is the
        single seam the rest of the pipeline depends on.
        """
        enriched = 0
        artist_follow_cache: dict[int, Optional[int]] = {}
        for cand in candidates:
            if cand.enrichment is not None:
                if cand.enrichment.enriched:
                    enriched += 1
                continue
            cand.enrichment = self.enrich_one(cand, artist_follow_cache)
            if cand.enrichment.enriched:
                enriched += 1
        return enriched

    def enrich_one(
        self,
        cand: Candidate,
        artist_follow_cache: dict[int, Optional[int]],
    ) -> CandidateEnrichment:
        sid = int(cand.track.netease_song_id or 0)
        if not sid:
            return CandidateEnrichment(endpoint_errors=["missing_song_id"])

        cache_key = f"songrec_demo:enrich:{sid}:v1"
        if self._cache is not None:
            try:
                hit = self._cache.get_query(cache_key)
                if hit:
                    return CandidateEnrichment.from_payload(hit[0])
            except Exception:  # noqa: BLE001
                pass

        # Test doubles can provide deterministic enrichment without
        # emulating the full NetEase HTTP surface.
        provider = getattr(self._client, "enrich_song", None)
        if callable(provider):
            try:
                payload = provider(sid) or {}
                enr = CandidateEnrichment.from_payload(payload)
                self._cache_enrichment(cache_key, enr)
                return enr
            except Exception as exc:  # noqa: BLE001
                return CandidateEnrichment(endpoint_errors=[f"fake_enrich:{exc}"])

        enr = CandidateEnrichment()

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

        artist_ids = list(cand.artist_ids)
        if not artist_ids or enr.song_red_count is None:
            detail = self._netease_get("/song/detail", {"ids": f"[{sid}]"})
            if isinstance(detail, dict):
                songs = detail.get("songs")
                if isinstance(songs, list) and songs:
                    row = songs[0] if isinstance(songs[0], dict) else {}
                    if not artist_ids:
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
            artist_id = artist_ids[0]
            if artist_id not in artist_follow_cache:
                artist_follow_cache[artist_id] = self._artist_follow_count(artist_id)
            enr.artist_follow_count = artist_follow_cache[artist_id]

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

        if self._live_deep_enrichment:
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

    def _artist_follow_count(self, artist_id: int) -> Optional[int]:
        artist_detail = self._netease_get("/artist/detail", {"id": str(artist_id)})
        if isinstance(artist_detail, dict):
            data = artist_detail.get("data") if isinstance(artist_detail.get("data"), dict) else artist_detail
            stats = data.get("identify") if isinstance(data.get("identify"), dict) else {}
            artist = data.get("artist") if isinstance(data.get("artist"), dict) else {}
            count = (
                _maybe_int(data.get("followCount"))
                or _maybe_int(data.get("fansCount"))
                or _maybe_int(stats.get("imageDesc"))
                or _maybe_int(artist.get("followCount"))
            )
            if count is not None:
                return count

        artist_dynamic = self._netease_get("/artist/detail/dynamic", {"id": str(artist_id)})
        if isinstance(artist_dynamic, dict):
            return (
                _maybe_int(artist_dynamic.get("followCount"))
                or _maybe_int(artist_dynamic.get("fansCount"))
                or _maybe_int(artist_dynamic.get("subCount"))
            )
        return None

    def _cache_enrichment(self, cache_key: str, enr: CandidateEnrichment) -> None:
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
