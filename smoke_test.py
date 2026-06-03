"""
Smoke test for the real-song demo backend.

Hermetic by design: a :class:`FakeNeteaseClient` is injected into the
service container before the Flask app spins up, so the test never
hits the network and never depends on a running NetEase service. The
heavy KGRec research-layer load is also disabled via ``load_kgrec=False``
because the product layer should stand on its own.

Checks
------
1. ``GET /api/health`` -- product layer ready, NetEase reported alive.
2. ``GET /api/song-search?q=...`` -- returns NetEase-shaped real-song
   hits with title / artist / album / cover / netease_song_id /
   netease_url. Empty query returns an empty list (200).
3. ``POST /api/recommend`` (real-song flow) -- selecting real songs
   and tags returns ranked cards.
4. Recommendation cards include title, artist, album, NetEase link,
   explanation, score breakdown, matched tags, and a pick_type label.
5. The main ``/api/recommend`` response NEVER exposes KGRec
   ``item_id`` as a top-level user-facing field; the only IDs in the
   payload must be NetEase song IDs.
6. Empty input -> the API responds with ``fallback_used = "no_input"``
   instead of crashing.
7. Bad body -> 400 / wrong content type -> 415.
8. ``POST /api/kgrec-recommend`` is correctly disabled when the
   server runs with ``--no-kgrec`` (the smoke-test default).

Run from the project root::

    python SongRecDemo/smoke_test.py

Exit code is ``0`` on full pass and ``1`` otherwise.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

# Make the project root + this folder importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import create_app, HOLDER                 # noqa: E402
from netease_pipeline import FakeNeteaseClient, _norm_title     # noqa: E402


# ---------------------------------------------------------------------------
# In-memory cache double for the smoke test
# ---------------------------------------------------------------------------

class _MemoryQueryCache:
    """Drop-in for :class:`NeteaseCache` that lives only in memory."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}

    def get_query(self, query: str) -> list[dict[str, Any]] | None:
        v = self._store.get(query)
        return list(v) if v is not None else None

    def set_query(self, query: str, candidates: list[dict[str, Any]]) -> None:
        self._store[query] = list(candidates)


# ---------------------------------------------------------------------------
# Canned NetEase responses used across the smoke test.
#
# The keys are lowercased queries; values mirror the shape produced by
# :class:`NeteaseAPIClient.search_songs`. Any query that isn't matched
# falls back to the ``default`` list, so every code path returns
# something sensible.
# ---------------------------------------------------------------------------

def _bon_iver() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 12345, "title": "Holocene",
         "artist": "Bon Iver", "artists": ["Bon Iver"],
         "album": "Bon Iver, Bon Iver",
         "cover_url": "https://example/holocene.jpg",
         "duration_ms": 337000},
        {"netease_song_id": 12346, "title": "Skinny Love",
         "artist": "Bon Iver", "artists": ["Bon Iver"],
         "album": "For Emma, Forever Ago",
         "cover_url": "https://example/skinny.jpg",
         "duration_ms": 240000},
        {"netease_song_id": 12347, "title": "iMi",
         "artist": "Bon Iver", "artists": ["Bon Iver"],
         "album": "i,i", "cover_url": "https://example/imi.jpg"},
        {"netease_song_id": 12348, "title": "8 (circle)",
         "artist": "Bon Iver", "artists": ["Bon Iver"],
         "album": "22, A Million", "cover_url": "https://example/8.jpg"},
    ]


def _phoebe() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 22345, "title": "Motion Sickness",
         "artist": "Phoebe Bridgers", "artists": ["Phoebe Bridgers"],
         "album": "Stranger in the Alps",
         "cover_url": "https://example/motion.jpg"},
        {"netease_song_id": 22346, "title": "Kyoto",
         "artist": "Phoebe Bridgers", "artists": ["Phoebe Bridgers"],
         "album": "Punisher", "cover_url": "https://example/kyoto.jpg"},
        {"netease_song_id": 22347, "title": "Funeral",
         "artist": "Phoebe Bridgers", "artists": ["Phoebe Bridgers"],
         "album": "Stranger in the Alps", "cover_url": "https://example/funeral.jpg"},
    ]


def _indie_mix() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 33345, "title": "Best Coast Indie",
         "artist": "Best Coast", "artists": ["Best Coast"],
         "album": "Crazy For You",
         "cover_url": "https://example/bc.jpg"},
        {"netease_song_id": 33346, "title": "Mellow Indie Anthem",
         "artist": "Indie Cindy", "artists": ["Indie Cindy"],
         "album": "Mellow Times", "cover_url": "https://example/cindy.jpg"},
        {"netease_song_id": 33347, "title": "Quiet Folk Indie",
         "artist": "Folk Roads", "artists": ["Folk Roads"],
         "album": "Roads", "cover_url": "https://example/roads.jpg"},
    ]


def _holocene_query() -> list[dict[str, Any]]:
    # When the user has picked Holocene, /api/recommend issues
    # search_songs("Holocene") -- we return the same song plus a few
    # variants so the title-channel actually contributes candidates.
    return _bon_iver() + [
        {"netease_song_id": 44345, "title": "Holocene (Live)",
         "artist": "Bon Iver", "artists": ["Bon Iver"],
         "album": "Live At Sydney Opera House",
         "cover_url": "https://example/live.jpg"},
        {"netease_song_id": 44344, "title": "Holocene",
         "artist": "Other Artist", "artists": ["Other Artist"],
         "album": "Covers Vol 0", "cover_url": "https://example/exact-cover.jpg"},
        {"netease_song_id": 44346, "title": "Holocene (Cover)",
         "artist": "Indie Cindy", "artists": ["Indie Cindy"],
         "album": "Covers Vol 1", "cover_url": "https://example/cover.jpg"},
        {"netease_song_id": 44347, "title": "Holocene Guitar Cover",
         "artist": "Bedroom Upload", "artists": ["Bedroom Upload"],
         "album": "Covers Vol 2", "cover_url": "https://example/guitar-cover.jpg"},
        {"netease_song_id": 44348, "title": "Holocene Piano Version by Jane",
         "artist": "Jane Player", "artists": ["Jane Player"],
         "album": "Piano Covers", "cover_url": "https://example/piano-cover.jpg"},
    ]


def _tag_traps() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 55345, "title": "Alternative Rock",
         "artist": "Search Term", "artists": ["Search Term"],
         "album": "Literal Matches", "cover_url": "https://example/alt-rock.jpg"},
        {"netease_song_id": 55346, "title": "Sad",
         "artist": "Mood Word", "artists": ["Mood Word"],
         "album": "Literal Matches", "cover_url": "https://example/sad.jpg"},
        {"netease_song_id": 55347, "title": "Rock",
         "artist": "Genre Word", "artists": ["Genre Word"],
         "album": "Literal Matches", "cover_url": "https://example/rock.jpg"},
        {"netease_song_id": 55348, "title": "Midnight Exit",
         "artist": "Actual Band", "artists": ["Actual Band"],
         "album": "Static Flowers", "cover_url": "https://example/midnight.jpg"},
        {"netease_song_id": 55349, "title": "Sad Song",
         "artist": "Title Matcher", "artists": ["Title Matcher"],
         "album": "Literal Matches", "cover_url": "https://example/sad-song.jpg"},
        {"netease_song_id": 55350, "title": "Alternative Rock Nights",
         "artist": "Title Matcher", "artists": ["Title Matcher"],
         "album": "Literal Matches", "cover_url": "https://example/alt-rock-nights.jpg"},
    ]


def _cap_artist() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 66341, "title": "Cap One",
         "artist": "Cap Artist", "artists": ["Cap Artist"],
         "album": "One Album", "cover_url": "https://example/cap1.jpg"},
        {"netease_song_id": 66342, "title": "Cap Two",
         "artist": "Cap Artist", "artists": ["Cap Artist"],
         "album": "One Album", "cover_url": "https://example/cap2.jpg"},
        {"netease_song_id": 66343, "title": "Cap Three",
         "artist": "Cap Artist", "artists": ["Cap Artist"],
         "album": "One Album", "cover_url": "https://example/cap3.jpg"},
        {"netease_song_id": 66344, "title": "Cap Four",
         "artist": "Cap Artist", "artists": ["Cap Artist"],
         "album": "One Album", "cover_url": "https://example/cap4.jpg"},
    ]


def _dream_pop_pool() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 77341, "title": "Soft Signal",
         "artist": "Relevant Band", "artists": ["Relevant Band"],
         "album": "Dream Pop Signals", "cover_url": "https://example/signal.jpg"},
        {"netease_song_id": 77342, "title": "Random Machine",
         "artist": "Unrelated Noise", "artists": ["Unrelated Noise"],
         "album": "Concrete Static", "cover_url": "https://example/noise.jpg"},
    ]


def _weak_title_only() -> list[dict[str, Any]]:
    return [
        {"netease_song_id": 88341, "title": "Weak B Side",
         "artist": "Tiny Upload", "artists": ["Tiny Upload"],
         "album": "Unknown Upload", "cover_url": "https://example/weak.jpg"},
    ]


CANNED = {
    "bon iver":          _bon_iver(),
    "phoebe bridgers":   _phoebe(),
    "indie folk":        _indie_mix(),
    "indie":             _indie_mix(),
    "mellow":            _indie_mix(),
    "holocene":          _holocene_query(),
    "sad":               _tag_traps(),
    "alternative rock":  _tag_traps(),
    "rock":              _tag_traps(),
    "cap artist":        _cap_artist(),
    "cap artist indie folk": _cap_artist(),
    "dream pop":         _dream_pop_pool(),
    "seed track":        _weak_title_only(),
    "indie folk mellow": _indie_mix(),
}


def _enrichments() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for rows in CANNED.values():
        for row in rows:
            sid = int(row["netease_song_id"])
            out[sid] = {
                "comment_count": 120,
                "hot_comment_count": 3,
                "song_red_count": 300,
                "artist_follow_count": 1200,
                "playable": True,
                "audio_quality": 0.8,
                "similar_song_ids": [sid + 1, sid + 2],
            }
    # A weak candidate should be filtered only when the route evidence is
    # also weak.
    out[55348] = {
        "comment_count": 2,
        "hot_comment_count": 0,
        "song_red_count": 0,
        "artist_follow_count": 20,
        "playable": True,
        "audio_quality": 0.5,
        "similar_song_ids": [],
    }
    out[77342] = {
        "comment_count": 4,
        "hot_comment_count": 0,
        "song_red_count": 0,
        "artist_follow_count": 25,
        "playable": True,
        "audio_quality": 0.5,
        "similar_song_ids": [],
    }
    out[66343] = {
        "comment_count": 200,
        "hot_comment_count": 2,
        "song_red_count": 100,
        "artist_follow_count": 5000,
        "playable": False,
        "audio_quality": 0.8,
        "similar_song_ids": [],
    }
    out[88341] = {
        "comment_count": 1,
        "hot_comment_count": 0,
        "song_red_count": 0,
        "artist_follow_count": 10,
        "playable": True,
        "audio_quality": 0.4,
        "similar_song_ids": [],
    }
    return out


# ---------------------------------------------------------------------------
# Tiny test runner (no extra dependency)
# ---------------------------------------------------------------------------

class _Result:
    __slots__ = ("name", "passed", "msg")

    def __init__(self, name: str, passed: bool, msg: str = "") -> None:
        self.name = name
        self.passed = passed
        self.msg = msg


def _run(name: str, fn: Callable[[], None]) -> _Result:
    try:
        fn()
    except AssertionError as exc:
        return _Result(name, False, f"assertion failed: {exc}")
    except Exception as exc:                  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        return _Result(name, False, f"{type(exc).__name__}: {exc}\n{tb}")
    return _Result(name, True)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_client():
    """Build the app once with a faked NetEase client.

    KGRec is disabled (``load_kgrec=False``) so the smoke test never
    waits for ALS artefacts to deserialise. The KGRec debug route is
    expected to answer 503 -- which is asserted in its own test.
    """
    fake = FakeNeteaseClient(
        responses=CANNED,
        default=_indie_mix(),
        enrichments=_enrichments(),
        alive=True,
    )
    cache = _MemoryQueryCache()
    HOLDER.inject_for_tests(client=fake, cache=cache, netease_alive=True)
    app = create_app(eager=True, load_kgrec=False)
    app.testing = True
    return app.test_client(), fake


def test_health(client) -> None:
    res = client.get("/api/health")
    _expect(res.status_code == 200, f"/api/health returned {res.status_code}")
    body = res.get_json() or {}
    _expect(body.get("ok") is True, f"/api/health ok=False: {body}")
    product = body.get("product_layer") or {}
    _expect(bool(product.get("name")),
            f"product layer name missing: {body}")
    _expect(product.get("netease_alive") is True,
            f"NetEase reported offline despite injected fake client: {product}")
    research = body.get("research_layer") or {}
    _expect(research.get("enabled") is False,
            f"research layer should be disabled in smoke test: {research}")


def test_song_search(client) -> None:
    res = client.get("/api/song-search?q=Bon%20Iver&limit=3")
    _expect(res.status_code == 200, f"song-search returned {res.status_code}")
    body = res.get_json() or {}
    _expect(body.get("ok") is True, f"song-search ok=False: {body}")
    items = body.get("items") or []
    _expect(len(items) > 0, f"no song-search hits: {body}")

    for it in items:
        for key in ("netease_song_id", "title", "artist", "artists",
                    "album", "cover_url", "netease_url"):
            _expect(key in it, f"song-search hit missing `{key}`: {it}")
        _expect(isinstance(it["netease_song_id"], int) and it["netease_song_id"] > 0,
                f"song-search hit has bad netease_song_id: {it}")
        _expect(it["netease_url"].startswith("https://music.163.com/"),
                f"song-search hit has bad netease_url: {it}")
        # The product layer must NOT be passing KGRec item_id back.
        _expect("item_id" not in it,
                f"song-search hit leaked KGRec `item_id`: {it}")

    # Empty query -> empty items, 200 OK.
    res2 = client.get("/api/song-search?q=")
    _expect(res2.status_code == 200, f"empty query returned {res2.status_code}")
    body2 = res2.get_json() or {}
    _expect(body2.get("ok") is True and body2.get("items") == [],
            f"empty query returned: {body2}")


def test_recommend_picks(client) -> None:
    """Pick a song from /api/song-search, post it back as a liked
    song with a couple of tags, and assert real-song output."""
    sres = client.get("/api/song-search?q=Bon%20Iver&limit=3")
    sbody = sres.get_json() or {}
    items = sbody.get("items") or []
    _expect(items, "no real-song hits to feed into recommend")
    pick = items[0]

    res = client.post("/api/recommend", json={
        "liked_songs":   [pick],
        "liked_artists": ["Phoebe Bridgers"],
        "genres":        ["indie folk"],
        "moods":         ["mellow"],
        "k":             8,
        "content_weight": 0.5,
        "novelty":        0.3,
        "diversity":      0.3,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    body = res.get_json() or {}
    _expect(body.get("ok") is True, f"recommend ok=False: {body}")
    data = body["data"]

    items = data.get("items") or []
    _expect(len(items) > 0, f"no recommendations: {data}")

    # The picked song itself MUST NOT show up in the recommendations.
    for it in items:
        _expect(int(it.get("netease_song_id") or 0) != int(pick["netease_song_id"]),
                f"liked song re-appeared in recommendations: {it}")

    # Profile + candidate summary populated.
    profile = data.get("profile") or {}
    _expect(int(pick["netease_song_id"]) in (profile.get("liked_song_ids") or []),
            f"profile didn't pick up liked song: {profile}")
    _expect("Phoebe Bridgers" in (profile.get("liked_artists") or []),
            f"profile missing liked artist: {profile}")
    _expect("indie folk" in (profile.get("preferred_genres") or []),
            f"profile missing preferred genre: {profile}")
    _expect("mellow" in (profile.get("preferred_moods") or []),
            f"profile missing preferred mood: {profile}")
    cs = data.get("candidate_summary") or {}
    for key in ("retrieved_total", "after_dedup", "filtered_liked",
                "filtered_same_title", "filtered_tag_title",
                "filtered_duplicate_version", "filtered_missing_metadata",
                "enriched_count", "filtered_low_trust",
                "filtered_unplayable", "final_candidate_count"):
        _expect(key in cs, f"candidate summary missing `{key}`: {cs}")
    _expect(int(cs.get("final_candidate_count") or 0) > 0,
            f"empty candidate summary: {cs}")


def test_recommend_filters_same_liked_title(client) -> None:
    """Different song ids / artists with the same liked-song title are
    covers or variants, not fresh recommendations."""
    pick = {
        "netease_song_id": 12345, "title": "Holocene",
        "artist": "Bon Iver", "artists": ["Bon Iver"],
        "album": "Bon Iver, Bon Iver",
        "cover_url": "https://example/holocene.jpg",
    }
    res = client.post("/api/recommend", json={
        "liked_songs": [pick],
        "liked_artists": ["Bon Iver"],
        "k": 8,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    data = (res.get_json() or {})["data"]
    blocked_ids = {12345, 44344, 44345, 44346, 44347, 44348}
    for it in data.get("items") or []:
        _expect(_norm_title(it.get("title") or "") != "holocene",
                f"same-title liked song leaked into recommendations: {it}")
        _expect(int(it.get("netease_song_id") or 0) not in blocked_ids,
                f"same-title cover/version leaked into recommendations: {it}")
    cs = data.get("candidate_summary") or {}
    _expect(int(cs.get("filtered_same_title") or 0) >= 4,
            f"same-title filter did not report filtered candidates: {cs}")


def test_recommend_filters_literal_tag_titles(client) -> None:
    """Genre/mood words in the title are not evidence of genre fit."""
    res = client.post("/api/recommend", json={
        "tags": ["sad", "alternative rock", "rock"],
        "k": 8,
        "content_weight": 0.8,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    data = (res.get_json() or {})["data"]
    literal_tag_titles = {"sad", "alternative rock", "rock"}
    blocked_ids = {55345, 55346, 55347, 55349, 55350}
    for it in data.get("items") or []:
        title_key = _norm_title(it.get("title") or "")
        _expect(title_key not in literal_tag_titles,
                f"literal tag-title candidate leaked into recommendations: {it}")
        _expect(int(it.get("netease_song_id") or 0) not in blocked_ids,
                f"title-only tag match leaked into recommendations: {it}")
        if title_key in literal_tag_titles:
            _expect(it.get("pick_type") != "safe",
                    f"literal tag-title candidate was marked safe: {it}")
    cs = data.get("candidate_summary") or {}
    _expect(int(cs.get("filtered_tag_title") or 0) >= 5,
            f"tag-title filter did not report filtered candidates: {cs}")


def test_recommend_filters_low_trust_title_only(client) -> None:
    pick = {
        "netease_song_id": 88000, "title": "Seed Track",
        "artist": "", "artists": [],
        "album": "", "cover_url": "",
    }
    res = client.post("/api/recommend", json={
        "liked_songs": [pick],
        "k": 5,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    data = (res.get_json() or {})["data"]
    ids = {int(it["netease_song_id"]) for it in data.get("items") or []}
    _expect(88341 not in ids, f"low-trust title-only candidate was recommended: {data}")
    cs = data.get("candidate_summary") or {}
    _expect(int(cs.get("filtered_low_trust") or 0) >= 1,
            f"low-trust filter did not report filtered candidates: {cs}")


def test_recommend_filters_unplayable(client) -> None:
    res = client.post("/api/recommend", json={
        "liked_artists": ["Cap Artist"],
        "k": 5,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    data = (res.get_json() or {})["data"]
    ids = {int(it["netease_song_id"]) for it in data.get("items") or []}
    _expect(66343 not in ids, f"known-unplayable song was recommended: {data.get('items')}")
    cs = data.get("candidate_summary") or {}
    _expect(int(cs.get("filtered_unplayable") or 0) >= 1,
            f"unplayable filter did not report filtered candidates: {cs}")


def test_recommend_card_fields(client) -> None:
    """Every card must carry real-song metadata + ranking diagnostics."""
    pick = {
        "netease_song_id": 12345, "title": "Holocene",
        "artist": "Bon Iver", "artists": ["Bon Iver"],
        "album": "Bon Iver, Bon Iver",
        "cover_url": "https://example/holocene.jpg",
    }
    res = client.post("/api/recommend", json={
        "liked_songs": [pick],
        "tags":        ["indie", "mellow"],
        "k":           6,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    items = (res.get_json() or {})["data"]["items"]
    _expect(items, "no items returned")

    pick_types = {"safe", "exploratory", "diverse", "balanced"}
    needed_breakdown = {
        "final", "content", "artist_match", "tag_match", "title_match",
        "retrieval", "multi_source", "novelty_term",
        "content_score", "content_text_similarity",
        "collaborative_proxy_score", "retrieval_confidence_score",
        "artist_affinity_score", "metadata_quality_score",
        "novelty_score", "novelty_bonus", "final_score",
        "popularity_score", "artist_authority_score", "playable_score",
        "audio_quality_score", "trust_score",
    }

    for it in items:
        for key in ("rank", "netease_song_id", "title", "artist",
                    "artists", "album", "cover_url", "netease_url",
                    "score", "score_breakdown", "explanation",
                    "reasons", "matched_tags", "pick_type", "sources"):
            _expect(key in it, f"recommendation missing `{key}`: {it}")
        _expect(isinstance(it["netease_song_id"], int) and it["netease_song_id"] > 0,
                f"recommendation has bad netease_song_id: {it}")
        _expect(it["pick_type"] in pick_types,
                f"recommendation has invalid pick_type {it['pick_type']!r}")
        # Score breakdown contract intact.
        missing = needed_breakdown - set(it["score_breakdown"] or {})
        _expect(not missing,
                f"score_breakdown missing keys: {sorted(missing)}")
        _expect(isinstance(it["explanation"], str) and it["explanation"],
                f"empty explanation on item {it.get('netease_song_id')}")
        _expect(isinstance(it["reasons"], list) and it["reasons"],
                f"empty reasons list on item {it.get('netease_song_id')}")
        # Every recommendation must carry a real title and a real
        # NetEase URL; that's the whole point of this layer.
        _expect(isinstance(it["title"], str) and it["title"],
                f"empty title on item {it.get('netease_song_id')}")
        _expect(it["netease_url"].startswith("https://music.163.com/"),
                f"item missing NetEase URL: {it}")


def test_novelty_does_not_promote_irrelevant(client) -> None:
    """Novelty is gated by relevance, so a random-looking candidate
    should not beat a candidate with real genre/profile support."""
    res = client.post("/api/recommend", json={
        "genres": ["dream pop"],
        "k": 2,
        "novelty": 1.0,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    items = (res.get_json() or {})["data"]["items"]
    titles = [it["title"] for it in items]
    _expect("Soft Signal" in titles and "Random Machine" in titles,
            f"expected both relevance-test candidates, got {titles}")
    rank = {it["title"]: it["rank"] for it in items}
    _expect(rank["Soft Signal"] < rank["Random Machine"],
            f"irrelevant candidate outranked relevant one under novelty: {items}")


def test_mmr_respects_small_k_artist_cap(client) -> None:
    res = client.post("/api/recommend", json={
        "liked_artists": ["Cap Artist"],
        "genres": ["indie folk"],
        "k": 5,
        "diversity": 0.9,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    items = (res.get_json() or {})["data"]["items"]
    cap_count = sum(1 for it in items if it.get("artist") == "Cap Artist")
    _expect(cap_count <= 1,
            f"k<=5 should keep at most one song per artist when alternatives exist: {items}")


def test_explanations_match_pick_type(client) -> None:
    res = client.post("/api/recommend", json={
        "liked_artists": ["Bon Iver"],
        "genres": ["indie folk"],
        "moods": ["mellow"],
        "k": 6,
        "diversity": 0.8,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    items = (res.get_json() or {})["data"]["items"]
    _expect(items, "no recommendations returned")
    expected_fragments = {
        "safe": "close match",
        "exploratory": "broader discovery",
        "diverse": "less repetitive",
        "balanced": "balanced recommendation",
    }
    for it in items:
        frag = expected_fragments[it["pick_type"]]
        _expect(frag in it["explanation"],
                f"explanation does not match pick_type={it['pick_type']}: {it['explanation']}")


def test_model_info_is_honest(client) -> None:
    res = client.post("/api/recommend", json={
        "liked_artists": ["Bon Iver"],
        "k": 3,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    info = ((res.get_json() or {})["data"].get("model_info") or {})
    _expect(info.get("model_type") == "real_song_hybrid_retrieval_ranking",
            f"unexpected model_type: {info}")
    _expect(info.get("uses_netease_api") is True,
            f"uses_netease_api should be true: {info}")
    _expect(info.get("trained_collaborative_filtering") is False,
            f"must not claim trained NetEase CF: {info}")
    _expect(info.get("collaborative_proxy_used") is True,
            f"collaborative proxy flag missing: {info}")
    _expect("separate" in str(info.get("research_layer", "")).lower(),
            f"research layer separation not stated: {info}")


def test_no_kgrec_ids_in_main_output(client) -> None:
    """The main /api/recommend response must never expose KGRec
    ``item_id`` as a user-facing field. The smoke test walks every
    item and every nested object and asserts it isn't there.
    """
    res = client.post("/api/recommend", json={
        "liked_artists": ["Bon Iver"],
        "genres":        ["indie folk"],
        "k":             5,
    })
    _expect(res.status_code == 200, f"recommend returned {res.status_code}")
    data = (res.get_json() or {}).get("data") or {}

    def _walk(obj: Any, path: str = "$") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "item_id":
                    raise AssertionError(
                        f"main /api/recommend response leaked KGRec "
                        f"`item_id` at {path}: {v!r}"
                    )
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    _walk(data)


def test_recommend_empty_input(client) -> None:
    """Empty body -> response with fallback_used = 'no_input', 200 OK."""
    res = client.post("/api/recommend", json={})
    _expect(res.status_code == 200,
            f"empty body returned {res.status_code}: {res.data!r}")
    body = res.get_json() or {}
    _expect(body.get("ok") is True, f"empty body ok=False: {body}")
    data = body["data"]
    _expect(data.get("fallback_used") == "no_input",
            f"expected fallback_used=no_input, got {data.get('fallback_used')!r}")
    _expect(data.get("items") == [],
            f"expected empty items on no input, got {data.get('items')}")


def test_recommend_bad_body(client) -> None:
    res = client.post(
        "/api/recommend",
        data="not-json{{",
        content_type="application/json",
    )
    _expect(res.status_code == 400,
            f"bad body returned {res.status_code}: {res.data!r}")
    body = res.get_json() or {}
    _expect(body.get("ok") is False, f"bad body ok=True: {body}")

    res2 = client.post("/api/recommend", data="{}", content_type="text/plain")
    _expect(res2.status_code == 415,
            f"non-JSON content-type returned {res2.status_code}")


def test_kgrec_debug_disabled(client) -> None:
    """In the smoke test we start with ``load_kgrec=False``; the debug
    route must answer 503 without crashing."""
    res = client.post("/api/kgrec-recommend", json={"seed_ids": ["1"]})
    _expect(res.status_code == 503,
            f"kgrec-recommend returned {res.status_code}")
    body = res.get_json() or {}
    _expect(body.get("ok") is False,
            f"kgrec-recommend should be disabled: {body}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Building Flask app with FakeNeteaseClient (no network) ...", flush=True)
    try:
        client, fake = make_client()
    except Exception as exc:                  # noqa: BLE001
        print(f"\n[FATAL] Could not build the app: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    tests = [
        ("1) GET  /api/health",                         lambda: test_health(client)),
        ("2) GET  /api/song-search returns NetEase hits", lambda: test_song_search(client)),
        ("3) Search hit -> liked-song -> /api/recommend", lambda: test_recommend_picks(client)),
        ("4) /api/recommend filters same liked-song title", lambda: test_recommend_filters_same_liked_title(client)),
        ("5) /api/recommend filters literal tag titles", lambda: test_recommend_filters_literal_tag_titles(client)),
        ("6) /api/recommend filters low-trust title-only songs", lambda: test_recommend_filters_low_trust_title_only(client)),
        ("7) /api/recommend filters unplayable songs",     lambda: test_recommend_filters_unplayable(client)),
        ("8) /api/recommend cards have real-song fields", lambda: test_recommend_card_fields(client)),
        ("9) novelty is gated by relevance",              lambda: test_novelty_does_not_promote_irrelevant(client)),
        ("10) MMR respects k<=5 artist cap",               lambda: test_mmr_respects_small_k_artist_cap(client)),
        ("11) explanations match pick_type",               lambda: test_explanations_match_pick_type(client)),
        ("12) model_info is honest about proxy scoring",   lambda: test_model_info_is_honest(client)),
        ("13) /api/recommend never leaks KGRec item_id",   lambda: test_no_kgrec_ids_in_main_output(client)),
        ("14) /api/recommend (empty body) -> no_input",    lambda: test_recommend_empty_input(client)),
        ("15) /api/recommend (bad body) -> 400/415",       lambda: test_recommend_bad_body(client)),
        ("16) /api/kgrec-recommend disabled -> 503",       lambda: test_kgrec_debug_disabled(client)),
    ]

    results: list[_Result] = []
    print()
    for name, fn in tests:
        r = _run(name, fn)
        results.append(r)
        marker = "[ OK ]" if r.passed else "[FAIL]"
        print(f"{marker} {name}")
        if not r.passed and r.msg:
            for line in r.msg.splitlines():
                print(f"        {line}")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    print()
    print(f"SUMMARY: PASS={n_pass}  FAIL={n_fail}  TOTAL={len(results)}")
    print(f"NetEase fake client received {len(fake.calls)} search call(s).")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
