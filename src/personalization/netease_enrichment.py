"""
Optional NetEase Cloud Music metadata enrichment.

Why this module exists
----------------------
The KGRec-music dataset only ships opaque item IDs, free-form Last.fm
tags, and short wiki-style description blurbs.  None of those are good
material for a polished frontend card -- a real demo wants song titles,
artist names, album art, and a clickable link.

This module adds an *optional* :class:`NeteaseMetadataEnricher` that
queries a locally-running instance of the NetEase Cloud Music API
(https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced) and
attaches those display fields when a high-confidence match is found.

The enrichment is strictly cosmetic:

* The KGRec training pipeline is **untouched**.  ALS is trained on
  KGRec interaction data only.
* Validation / test evaluation uses KGRec item IDs only -- no NetEase
  metadata leaks into metrics.
* The recommender engine still ranks items by KGRec ID and computes
  ALS, content, novelty, and explanation scores from internal data.
* If the NetEase service is unreachable, slow, or returns a low-
  confidence match, this enricher silently falls back to the
  :class:`InternalFeaturesEnricher` so the demo keeps working.

Design choices worth knowing
----------------------------
* HTTP client uses ``urllib.request`` from the standard library so we
  do not pull in a new third-party dependency for an optional feature.
* Responses are cached in a local SQLite file (``NETEASE_CACHE_PATH``)
  so repeated demo runs are fast and offline-friendly.  Negative
  results (no match / low confidence) are also cached, with a separate
  marker, so we do not retry the same hopeless query forever.
* Confidence is a token-overlap (Jaccard) score between the KGRec
  description-snippet + tags and the NetEase candidate's
  ``name + artist + album`` text, with a strong bonus when an
  artist-like KGRec tag (e.g. ``best-coast``) matches the NetEase
  artist name.
* The KGRec ``item_id`` is **never** assumed to equal a NetEase song
  ID.  The NetEase ID is stored alongside the KGRec ID in
  ``ScoredItem.metadata`` so the frontend can link out without us
  pretending the two namespaces are the same.

Public surface
--------------
* :class:`NeteaseMetadataEnricher` -- conforms to the
  :class:`MetadataEnricher` Protocol from ``enrichment.py``.
* :class:`NeteaseAPIClient` -- thin HTTP wrapper, exposed for tests
  and for code paths that want to call the API outside the enricher.
* :class:`NeteaseCache` -- on-disk SQLite cache, exposed so the smoke
  test can wipe it between runs.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from src.personalization.enrichment import (
    InternalFeaturesEnricher,
    MetadataEnricher,
    NullEnricher,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class NeteaseAPIError(Exception):
    """Raised when the NetEase API returns a non-recoverable error.

    The enricher catches this and falls back; callers wanting raw
    access (e.g. the smoke test) can use it to distinguish API
    problems from cache misses.
    """


class NeteaseAPIClient:
    """Tiny HTTP client for the NetEase Cloud Music API service.

    Only the endpoints we actually need are wrapped:

    * :meth:`search_songs` -> GET ``/search?keywords=...&type=1&limit=...``
    * :meth:`ping`         -> GET ``/`` (health check; the upstream
      project responds with a small JSON banner on the root path)

    Network failures, timeouts, non-2xx responses, malformed JSON, and
    missing fields are all surfaced as :class:`NeteaseAPIError`.  The
    enricher converts that into a graceful fallback; nothing here
    should ever raise an unexpected exception type.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        max_retries: int = 1,
        user_agent: str = "kgrec-personalized-demo/1.0",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._max_retries = max(0, int(max_retries))
        self._user_agent = user_agent

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the service answers within the configured timeout.

        Uses the search endpoint with a trivial query rather than the
        bare root, because some deployments redirect or 404 the root
        path while still serving ``/search`` correctly.
        """
        try:
            self._get("/search", {"keywords": "test", "type": "1", "limit": "1"})
            return True
        except NeteaseAPIError as exc:
            log.debug("NetEase ping failed: %s", exc)
            return False

    def search_songs(self, keywords: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for songs by free-text keywords.

        Returns a list (possibly empty) of normalised candidate dicts:

            {
              "netease_song_id": int,
              "title":           str,
              "artist":          str,   # joined "; " if multiple
              "artists":         list[str],
              "album":           str,
              "album_id":        int | None,
              "cover_url":       str | None,
              "duration_ms":     int | None,
            }

        Raises :class:`NeteaseAPIError` on transport / parse failure.
        """
        if not keywords or not keywords.strip():
            return []
        params = {
            "keywords": keywords.strip(),
            "type": "1",                   # 1 = single (song)
            "limit": str(int(limit)),
        }
        payload = self._get("/search", params)

        # The upstream API wraps results inside ``result.songs`` for
        # type=1.  Defensive parsing because field shape varies between
        # logged-in / anonymous responses.
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            return []
        songs = result.get("songs")
        if not isinstance(songs, list):
            return []

        out: list[dict[str, Any]] = []
        for s in songs:
            if not isinstance(s, dict):
                continue
            try:
                song_id = int(s.get("id"))
            except (TypeError, ValueError):
                continue
            title = str(s.get("name") or "").strip()
            artists_raw = s.get("artists") or s.get("ar") or []
            artist_names: list[str] = []
            artist_ids: list[int] = []
            for a in artists_raw:
                if isinstance(a, dict):
                    nm = a.get("name")
                    if nm:
                        artist_names.append(str(nm).strip())
                    try:
                        aid = int(a.get("id"))
                    except (TypeError, ValueError):
                        aid = 0
                    if aid:
                        artist_ids.append(aid)
            album = s.get("album") or s.get("al") or {}
            album_name = str(album.get("name") or "").strip() if isinstance(album, dict) else ""
            album_id = album.get("id") if isinstance(album, dict) else None
            try:
                album_id = int(album_id) if album_id is not None else None
            except (TypeError, ValueError):
                album_id = None
            # Album cover: the legacy ``/search`` endpoint does not
            # always include a picUrl on the album object.  If we
            # have an album id we can synthesise the standard URL,
            # which is what the NetEase web player uses.
            cover_url: Optional[str] = None
            if isinstance(album, dict):
                pic = album.get("picUrl") or album.get("blurPicUrl")
                if pic:
                    cover_url = str(pic)
            duration = s.get("duration") or s.get("dt")
            try:
                duration_ms = int(duration) if duration is not None else None
            except (TypeError, ValueError):
                duration_ms = None
            out.append({
                "netease_song_id": song_id,
                "title": title,
                "artist": "; ".join(artist_names),
                "artists": artist_names,
                "artist_ids": artist_ids,
                "album": album_name,
                "album_id": album_id,
                "cover_url": cover_url,
                "duration_ms": duration_ms,
            })
        return out

    # ------------------------------------------------------------------
    # Low-level GET with retry + timeout
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, str]) -> Any:
        url = f"{self._base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self._user_agent,
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    if resp.status >= 400:
                        raise NeteaseAPIError(
                            f"HTTP {resp.status} from {url}"
                        )
                    raw = resp.read()
                try:
                    return json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError as exc:
                    raise NeteaseAPIError(f"Invalid JSON from {url}: {exc}")
            except urllib.error.HTTPError as exc:
                # 4xx is not worth retrying (bad request); 5xx is.
                last_err = exc
                if 500 <= exc.code < 600 and attempt < self._max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise NeteaseAPIError(f"HTTP {exc.code} from {url}: {exc.reason}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                if attempt < self._max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                raise NeteaseAPIError(f"Network error contacting {url}: {exc}")
        # Loop exits only via raise; this line is for type checkers.
        raise NeteaseAPIError(f"Exhausted retries: {last_err}")


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

class NeteaseCache:
    """Tiny SQLite-backed cache for NetEase enrichment results.

    Two tables:

      ``item_cache``   -- one row per KGRec item we have already
                          attempted to enrich.  ``payload`` is the
                          JSON we will merge into the metadata dict;
                          empty payload means "no acceptable match"
                          and is also cached so we do not keep
                          retrying.
      ``query_cache``  -- per-query raw NetEase ``/search`` payloads,
                          for debugging and to enable re-scoring
                          without another HTTP hit.

    The cache is concurrency-safe within a single process via a lock
    around writes; the SQLite ``check_same_thread=False`` flag lets us
    share one connection across worker threads (the demo is single-
    threaded but FastAPI later may not be).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS item_cache (
        item_id      TEXT PRIMARY KEY,
        confidence   REAL NOT NULL,
        matched      INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        ts           REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS query_cache (
        query        TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        ts           REAL NOT NULL
    );
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(self.SCHEMA)

    # -- item cache -----------------------------------------------------

    def get_item(self, item_id: str) -> Optional[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT confidence, matched, payload_json FROM item_cache WHERE item_id = ?",
            (str(item_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        confidence, matched, payload_json = row
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError:
            payload = {}
        return {
            "confidence": float(confidence),
            "matched": bool(matched),
            "payload": payload,
        }

    def set_item(
        self,
        item_id: str,
        *,
        confidence: float,
        matched: bool,
        payload: dict[str, Any],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO item_cache "
                "(item_id, confidence, matched, payload_json, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(item_id),
                    float(confidence),
                    1 if matched else 0,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )

    # -- query cache ----------------------------------------------------

    def get_query(self, query: str) -> Optional[list[dict[str, Any]]]:
        cur = self._conn.execute(
            "SELECT payload_json FROM query_cache WHERE query = ?",
            (query,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, list) else None

    def set_query(self, query: str, candidates: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO query_cache (query, payload_json, ts) "
                "VALUES (?, ?, ?)",
                (query, json.dumps(candidates, ensure_ascii=False), time.time()),
            )

    # -- maintenance ----------------------------------------------------

    def clear(self) -> None:
        """Wipe both tables (used by the smoke test)."""
        with self._lock:
            self._conn.execute("DELETE FROM item_cache")
            self._conn.execute("DELETE FROM query_cache")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Query construction + confidence scoring
# ---------------------------------------------------------------------------

# Rough English stopword list -- keep small; we only need to strip the
# noise words that would otherwise dominate token-overlap scoring.
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "at", "by",
    "with", "from", "is", "was", "are", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "or", "but",
    "i", "you", "he", "she", "they", "we", "his", "her", "their", "our",
    "song", "track", "album", "record", "single",
})

# Tags that almost certainly describe a genre/mood/decade rather than
# an artist or song -- used to *down*-weight non-distinctive overlap.
_GENERIC_TAG_HINTS = frozenset({
    "indie", "rock", "pop", "alternative", "alt", "folk", "electronic",
    "electro", "dance", "hip", "hop", "rap", "country", "soul", "funk",
    "jazz", "classical", "punk", "metal", "ambient", "experimental",
    "acoustic", "instrumental", "vocal", "vocals", "male", "female",
    "mellow", "chill", "chillout", "happy", "sad", "energetic",
    "summer", "winter", "fall", "spring", "morning", "evening", "night",
    "favourite", "favourites", "favorite", "favorites", "fav", "love",
    "best", "good", "great", "amazing", "awesome", "favs",
    "60s", "70s", "80s", "90s", "00s", "10s", "20s",
    "twee", "garage", "psychedelic", "shoegaze", "dreampop", "synthpop",
    "lofi", "lo", "fi",
    # Tag-only emotional / curatorial labels that look like 2-word
    # "artist" tags but aren't.  Kept conservative: many real artists
    # have words like "love", "life", "girl" in their name, so we
    # only blocklist tokens that are unambiguously curatorial.
    "fuck", "yes", "stuck", "repeat", "esteem",
    "again", "would", "known", "listen",
})

# Words that often start a Capitalised phrase but aren't artists:
# sentence openers, publication names, geographic names, common nouns.
_PROPER_NOUN_BLOCKLIST = frozenset({
    "This", "That", "These", "Those", "It", "He", "She", "They", "We",
    "I", "You", "His", "Her", "Their", "Our",
    "The", "A", "An",
    "There", "Here", "Was", "Were", "Many", "Some", "All", "Most", "Few",
    "Frontwoman", "Frontman", "Singer", "Lead", "Vocalist", "Drummer",
    "Guitarist", "Bassist", "Producer", "Track", "Song", "Album",
    "Record", "Single", "Speaking", "Talking", "Saying", "Asking",
    "Following", "According", "Featuring",
    "When", "Where", "After", "Before", "During", "While", "Although",
    "Though", "However", "Then", "Now", "Today", "Yesterday", "Tomorrow",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
    "Daily", "Telegraph", "Sun", "Times", "Guardian", "Rolling", "Stone",
    "Music", "NPR", "TV", "MTV", "BBC", "CNN", "Pitchfork", "Vulture",
    "Spin", "Billboard", "Vogue", "Vanity", "Fair", "Independent",
    "America", "American", "Britain", "British", "England", "English",
    "Germany", "German", "France", "French", "Italy", "Italian",
    "Japan", "Japanese", "Spain", "Spanish",
    "London", "York", "Paris", "Tokyo", "Berlin", "Country", "Auto",
    # Common KGRec-description phrases that look proper but aren't artist:
    "Tune", "Features", "Co", "Track",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# A run of 1+ Capitalised words.  We allow short connectors ("and",
# "of", "the") between them so phrases like "Best Coast" survive but
# also "Daft Punk" or "Bon Iver".  The first word is required to be
# Capitalised and not a known sentence-opener.
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+(?:[A-Z][A-Za-z0-9]+|of|and|the|de|von))*)\b"
)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _meaningful_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) > 2 and t not in _STOPWORDS}


def _split_tag(tag: str) -> list[str]:
    """Break a hyphenated/underscored tag into atomic tokens."""
    return [
        p.lower()
        for p in re.split(r"[-_\s]+", tag)
        if p and len(p) > 1 and p.lower() not in _STOPWORDS
    ]


def _looks_like_artist_tag(tag: str) -> bool:
    """Heuristic: a multi-token tag that does not look purely generic.

    KGRec tags include real artist slugs like ``best-coast`` and
    ``bon-iver`` mixed in with ``indie-rock``, ``80s`` and ``best of
    2011``.  We reject a tag only when *all* of its tokens are
    generic (so ``"indie rock"`` and ``"garage rock"`` are out) but
    keep tags where at least one token carries content (so
    ``"best coast"`` is in -- "best" is generic but "coast" is
    not).  Final disambiguation happens in :func:`_identify_artist`,
    which cross-checks against capitalised phrases extracted from the
    description.
    """
    parts = _split_tag(tag)
    if len(parts) < 2:
        return False
    if not all(p.isalpha() for p in parts):
        return False
    if all(p in _GENERIC_TAG_HINTS for p in parts):
        return False
    return True


def _extract_proper_nouns(text: str, max_chars: int = 400) -> list[str]:
    """Pull out Capitalised phrases that look like proper nouns.

    KGRec ``desc_clean`` preserves capitalisation, so phrases like
    "Bon Iver", "Justin Vernon", "Bethany Cosentino" stand out.  This
    is the most reliable artist-name signal we have because the
    description is wiki-style prose.  We deliberately drop sentence-
    starter capitals ("This", "Frontwoman", ...) and well-known
    publication / geographic names that confuse the artist guess.
    """
    if not text:
        return []
    snippet = text[:max_chars]
    out: list[str] = []
    seen: set[str] = set()
    for m in _PROPER_NOUN_RE.finditer(snippet):
        phrase = m.group(1).strip()
        # Trim leading and trailing connectors ("Bon Iver and" ->
        # "Bon Iver"; "of the Daily Telegraph" -> "Daily Telegraph").
        words = phrase.split()
        while words and words[0].lower() in {"and", "of", "the", "de", "von"}:
            words.pop(0)
        while words and words[-1].lower() in {"and", "of", "the", "de", "von"}:
            words.pop()
        if not words:
            continue
        # Reject the phrase outright if *any* word is blocklisted -- a
        # phrase like "Lucy Jones of the Daily Telegraph" contains
        # "Daily" + "Telegraph" and is almost certainly a journalist
        # citation, not the artist we want.  This is more conservative
        # than checking only the first word but it removes the biggest
        # source of bad artist guesses.
        if any(w in _PROPER_NOUN_BLOCKLIST for w in words):
            continue
        # Single all-caps tokens (acronyms) and very short single words
        # are usually noise on their own ("NPR", "TV").  Require either
        # multi-word or a single word longer than 3 characters.
        if len(words) == 1 and (len(words[0]) <= 3 or words[0].isupper()):
            continue
        cleaned = " ".join(words)
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


@dataclass(frozen=True)
class _ItemContext:
    """Pre-computed text bag for one KGRec item, used at query time."""
    item_id: str
    desc_full: str                 # Original-case truncated description
    desc_first_sentence: str
    tags: list[str]
    desc_tokens: set[str]          # Lowercased, stop-word-pruned
    tag_tokens: set[str]
    artist_like_tags: list[str]    # Tags that pass the heuristic above
    proper_nouns: list[str]        # Capitalised phrases from description
    artist_guess: Optional[str]    # Best-effort artist name (None = unknown)


def _first_sentence(text: str, max_chars: int = 120) -> str:
    if not text:
        return ""
    # Stop at the first sentence-ending punctuation, fall back to a
    # length-trim that respects word boundaries.
    m = re.search(r"[\.!\?]\s", text)
    cut = m.start() if m else min(len(text), max_chars)
    snippet = text[:cut].strip()
    if len(snippet) > max_chars:
        space = snippet.rfind(" ", 0, max_chars)
        if space > 0:
            snippet = snippet[:space]
        else:
            snippet = snippet[:max_chars]
    return snippet


def _identify_artist(
    proper_nouns: list[str],
    artist_like_tags: list[str],
) -> Optional[str]:
    """Best-effort artist name from description + tag cross-check.

    Decision order, from highest to lowest confidence:

    1a. A multi-token artist-like tag whose tokens *exactly* match a
        proper-noun phrase from the description -- e.g. tag
        ``"best coast"`` and phrase ``"Best Coast"``.  Near-certain.
    1b. A proper-noun phrase whose tokens are a subset of a
        multi-token artist-like tag -- e.g. phrase ``"Sullivan"`` and
        tag ``"jazmine sullivan"`` (the description mentions only the
        surname but the tag carries the full name).  We return the
        more complete *tag* form, title-cased.
    1c. A tag whose tokens are a strict subset of a multi-token
        proper-noun phrase -- e.g. tag ``"bon iver"`` and phrase
        ``"Justin Vernon Bon Iver"``.  Return the phrase.
    2.  First multi-word proper-noun phrase from the description --
        e.g. ``"Justin Vernon"`` when no tag-confirmation fires.
    3.  First single-word proper noun longer than 3 characters.
    4.  ``None`` if nothing usable.  The enricher then skips the API
        call and falls back to internal metadata, which is the right
        thing to do when we cannot even guess the artist.

    Deliberately NOT in the list: "first artist-like tag without any
    confirmation."  That heuristic was the source of bad queries like
    ``"upbeat fun"`` and ``"conan tbs"`` -- there are too many
    curatorial tags whose lowercased tokens look like a band name.
    """
    if not proper_nouns and not artist_like_tags:
        return None

    proper_lower = [
        ({t.lower() for t in _tokenize(p)}, p) for p in proper_nouns
    ]

    # 1) Tag <-> proper-noun cross-confirmation.
    for tag in artist_like_tags:
        tag_parts = set(_split_tag(tag))
        if len(tag_parts) < 2:
            continue
        tag_phrase = " ".join(_split_tag(tag)).title()
        for phrase_set, phrase in proper_lower:
            if not phrase_set:
                continue
            # 1a) Exact match.
            if tag_parts == phrase_set:
                return phrase
            # 1b) Phrase fragment of a richer tag.
            if phrase_set.issubset(tag_parts):
                return tag_phrase
            # 1c) Tag fragment of a richer phrase.
            if tag_parts.issubset(phrase_set):
                return phrase

    # 2) First multi-word proper-noun phrase.
    for phrase in proper_nouns:
        if len(phrase.split()) >= 2:
            return phrase

    # 3) First single-word proper noun longer than 3 chars.
    for phrase in proper_nouns:
        if len(phrase) > 3:
            return phrase

    return None


def _build_search_query(ctx: _ItemContext, max_len: int = 50) -> str:
    """Compose a NetEase keyword query.

    NetEase's ``/search`` endpoint is essentially a title/artist
    matcher: short focused queries (1-3 words) hit, long descriptive
    sentences return zero candidates.  We therefore send only the
    artist guess and let the *scoring* stage pick which of that
    artist's songs is the right match.

    Returns ``""`` when no artist could be identified, which the
    caller treats as a "skip API, fall back" signal.
    """
    if not ctx.artist_guess:
        return ""
    query = ctx.artist_guess.strip()
    if len(query) > max_len:
        query = query[:max_len].rsplit(" ", 1)[0].strip()
    return query


def _score_candidate(ctx: _ItemContext, candidate: dict[str, Any]) -> float:
    """Confidence in [0, 1] that ``candidate`` is the right NetEase song.

    Because our query is artist-only, the dominant signal is whether
    the candidate's artist name matches the KGRec side.  Title and
    album give us a *disambiguation* signal among that artist's
    songs.

    Composition (each component is independent and additive):

    * Artist coverage: fraction of the candidate's artist tokens
      that appear in the KGRec description or tags.  >=0.8 is a
      clean match (+0.40); 0.5-0.8 is partial (+0.20); <0.5 is a
      different artist (+0.00).
    * Artist-tag confirmation (+0.10): a multi-token artist-like
      KGRec tag whose tokens are entirely contained in the
      candidate's artist name.  An additional confidence boost on
      top of the description-level artist match.
    * Title-in-description (up to +0.30): how many discriminative
      title tokens (>3 chars, not generic) appear in the KGRec
      description.  This is what disambiguates "Skinny Love" from
      "iMi" when both are by Bon Iver.
    * Album-in-description (up to +0.20): same idea for the album.

    The default ``NETEASE_MIN_CONFIDENCE = 0.40`` is calibrated so
    that a clean artist match passes on its own merits (0.40), and
    a single-token coincidence does not.  Title or album hits push
    the score firmly into the 0.50-0.80 range, which is the band a
    UI should consider "trustworthy".
    """
    title = candidate.get("title", "") or ""
    artist = candidate.get("artist", "") or ""
    album = candidate.get("album", "") or ""

    # Raw tokens: used for artist-coverage matching, since artists
    # often contain words like "best", "love", "the" that are
    # otherwise classified as generic.  The generic-word filter only
    # applies to the title/album discriminator bag below.
    item_tokens_raw = ctx.desc_tokens | ctx.tag_tokens
    if not item_tokens_raw:
        return 0.0
    item_tokens_filtered = item_tokens_raw - _GENERIC_TAG_HINTS

    artist_tokens = {
        t for t in _tokenize(artist)
        if t not in _STOPWORDS and len(t) > 1
    }
    title_tokens = {
        t for t in _tokenize(title)
        if t not in _STOPWORDS and len(t) > 3 and t not in _GENERIC_TAG_HINTS
    }
    album_tokens = {
        t for t in _tokenize(album)
        if t not in _STOPWORDS and len(t) > 3 and t not in _GENERIC_TAG_HINTS
    }

    if not artist_tokens:
        return 0.0

    artist_overlap = artist_tokens & item_tokens_raw
    artist_coverage = len(artist_overlap) / len(artist_tokens)

    score = 0.0
    if artist_coverage >= 0.8:
        score += 0.40
    elif artist_coverage >= 0.5:
        score += 0.20

    # Tag-confirmed artist: e.g. KGRec tag "bon iver" matches the
    # candidate's "Bon Iver".  Adds a confirmation bump on top of
    # the description-level artist coverage.
    for tag in ctx.artist_like_tags:
        tag_parts = set(_split_tag(tag))
        if len(tag_parts) >= 2 and tag_parts.issubset(artist_tokens):
            score += 0.10
            break

    # Title and album signals are rare but golden when present.
    title_in_desc = title_tokens & ctx.desc_tokens
    score += min(0.30, 0.10 * len(title_in_desc))

    album_in_desc = album_tokens & ctx.desc_tokens
    score += min(0.20, 0.05 * len(album_in_desc))

    return float(min(1.0, score))


# ---------------------------------------------------------------------------
# The enricher
# ---------------------------------------------------------------------------

class NeteaseMetadataEnricher:
    """Display enricher that queries NetEase Cloud Music as a side channel.

    Conforms to the :class:`MetadataEnricher` Protocol.  Instances are
    safe to share across threads (HTTP client is stateless, cache is
    locked, item context cache is built lazily under the lock).

    Parameters
    ----------
    item_features : pd.DataFrame
        Same DataFrame the :class:`InternalFeaturesEnricher` reads.
        Must be indexed by KGRec ``item_id_raw`` (str) and have
        ``tags_normalised`` and ``desc_clean`` columns.
    base_url, timeout, max_retries, search_limit, min_confidence
        See :mod:`config` defaults; these are exposed for tests and
        for callers wanting non-default behaviour.
    cache_path : Path | str | None
        Where to persist the SQLite cache; pass ``None`` to use an
        in-memory cache (useful for tests).
    fallback : MetadataEnricher | None
        Used whenever the API is unreachable or no high-confidence
        match is found.  Defaults to
        :class:`InternalFeaturesEnricher` over the same DataFrame.
    client : NeteaseAPIClient | None
        Inject a pre-built client (the smoke test uses this to swap
        in a deliberately broken endpoint).
    """

    # Fields produced specifically by NetEase enrichment.  Documented
    # so the frontend contract is explicit.
    NETEASE_FIELDS = (
        "title",
        "artist",
        "artists",
        "album",
        "cover_url",
        "netease_song_id",
        "netease_url",
        "match_confidence",
    )

    def __init__(
        self,
        item_features: pd.DataFrame,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        search_limit: Optional[int] = None,
        min_confidence: Optional[float] = None,
        cache_path: Optional[Path | str] = None,
        fallback: Optional[MetadataEnricher] = None,
        client: Optional[NeteaseAPIClient] = None,
    ) -> None:
        # Lazy-import config so this module is still importable in
        # test environments where config defaults haven't been
        # materialised (e.g. CI without artifacts).
        import config as _config

        self._features = item_features
        self._min_confidence = float(
            min_confidence if min_confidence is not None
            else _config.NETEASE_MIN_CONFIDENCE
        )
        self._search_limit = int(
            search_limit if search_limit is not None
            else _config.NETEASE_SEARCH_LIMIT
        )

        self._client = client or NeteaseAPIClient(
            base_url=base_url or _config.NETEASE_API_BASE_URL,
            timeout=(
                timeout if timeout is not None
                else _config.NETEASE_TIMEOUT_SECONDS
            ),
            max_retries=(
                max_retries if max_retries is not None
                else _config.NETEASE_MAX_RETRIES
            ),
        )

        # ``None`` cache path -> in-memory SQLite (":memory:" is not
        # used because we want sharable connection semantics, and a
        # temp file works the same for our purposes).
        if cache_path is None:
            cache_path = _config.NETEASE_CACHE_PATH
        self._cache = NeteaseCache(cache_path)

        self._fallback: MetadataEnricher = (
            fallback
            if fallback is not None
            else InternalFeaturesEnricher(item_features)
        )

        # Cached per-item context (description first sentence, tag
        # tokens, etc.).  Populated lazily because most demo runs
        # only enrich a few hundred items out of ~8 600.
        self._ctx_cache: dict[str, _ItemContext] = {}
        self._ctx_lock = threading.Lock()

        # Trip-switch: once the API has flunked too many calls in a
        # single process we stop trying until a manual reset.  This
        # bounds latency on the demo path when the Node service is
        # down.  ``_api_alive`` flips to False; ``ping`` on next
        # construction would re-enable.
        self._api_alive = True
        self._consecutive_failures = 0
        self._failure_budget = 3

    # ------------------------------------------------------------------
    # MetadataEnricher Protocol
    # ------------------------------------------------------------------

    def enrich(self, item_id: str) -> dict[str, Any]:
        try:
            return self._enrich_one(str(item_id))
        except Exception as exc:           # noqa: BLE001 -- contract: never raise.
            log.warning("NetEase enrich(%s) crashed: %s", item_id, exc)
            return self._fallback_enrich(str(item_id))

    def enrich_batch(
        self, item_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        ids = [str(i) for i in item_ids]
        out: dict[str, dict[str, Any]] = {}
        for iid in ids:
            out[iid] = self.enrich(iid)
        return out

    # ------------------------------------------------------------------
    # Implementation details
    # ------------------------------------------------------------------

    def _fallback_enrich(self, item_id: str) -> dict[str, Any]:
        try:
            md = dict(self._fallback.enrich(item_id) or {})
        except Exception as exc:           # noqa: BLE001
            log.warning("Fallback enricher crashed on %s: %s", item_id, exc)
            md = {}
        md.setdefault("source", "internal")
        return md

    def _enrich_one(self, item_id: str) -> dict[str, Any]:
        # 1) Cache hit?
        cached = self._cache.get_item(item_id)
        if cached is not None:
            base = self._fallback_enrich(item_id)
            if cached["matched"]:
                base.update(cached["payload"])
                base["source"] = "netease+internal"
                base["match_confidence"] = float(cached["confidence"])
            else:
                base.setdefault("netease_attempted", True)
            return base

        # 2) Build context from the local features table.  If the
        #    item is not in our DataFrame we cannot construct a
        #    sensible query -- internal enricher returns {} too.
        ctx = self._context_for(item_id)
        if ctx is None:
            return self._fallback_enrich(item_id)

        query = _build_search_query(ctx)
        if not query:
            self._cache.set_item(item_id, confidence=0.0, matched=False, payload={})
            return self._fallback_enrich(item_id)

        # 3) Hit NetEase (or replay from query cache).
        candidates = self._cache.get_query(query)
        if candidates is None:
            if not self._api_alive:
                return self._fallback_enrich(item_id)
            try:
                candidates = self._client.search_songs(
                    query, limit=self._search_limit
                )
                self._consecutive_failures = 0
            except NeteaseAPIError as exc:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_budget:
                    log.warning(
                        "NetEase API silenced after %d consecutive failures (%s)",
                        self._consecutive_failures, exc,
                    )
                    self._api_alive = False
                else:
                    log.info("NetEase API call failed (will fall back): %s", exc)
                return self._fallback_enrich(item_id)
            self._cache.set_query(query, candidates)

        # 4) Score, pick best, decide to accept / reject.
        best, best_score = None, 0.0
        for cand in candidates:
            score = _score_candidate(ctx, cand)
            if score > best_score:
                best, best_score = cand, score

        if best is None or best_score < self._min_confidence:
            self._cache.set_item(
                item_id, confidence=float(best_score), matched=False, payload={}
            )
            base = self._fallback_enrich(item_id)
            base["netease_attempted"] = True
            base["match_confidence"] = float(best_score)
            return base

        payload = self._netease_payload(best, best_score)
        self._cache.set_item(
            item_id, confidence=float(best_score), matched=True, payload=payload
        )
        base = self._fallback_enrich(item_id)
        base.update(payload)
        base["source"] = "netease+internal"
        return base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _context_for(self, item_id: str) -> Optional[_ItemContext]:
        with self._ctx_lock:
            cached = self._ctx_cache.get(item_id)
            if cached is not None:
                return cached
            if item_id not in self._features.index:
                return None
            row = self._features.loc[item_id]
            raw_tags = row.get("tags_normalised", None)
            tags: list[str] = list(raw_tags) if raw_tags is not None else []
            desc = str(row.get("desc_clean", "") or "")
            artist_like_tags = [t for t in tags if _looks_like_artist_tag(t)]
            proper_nouns = _extract_proper_nouns(desc)
            artist_guess = _identify_artist(proper_nouns, artist_like_tags)
            ctx = _ItemContext(
                item_id=item_id,
                desc_full=desc[:400],
                desc_first_sentence=_first_sentence(desc),
                tags=tags,
                desc_tokens=_meaningful_tokens(desc),
                tag_tokens={
                    tok
                    for tag in tags
                    for tok in _split_tag(tag)
                    if len(tok) > 2
                },
                artist_like_tags=artist_like_tags,
                proper_nouns=proper_nouns,
                artist_guess=artist_guess,
            )
            self._ctx_cache[item_id] = ctx
            return ctx

    @staticmethod
    def _netease_payload(candidate: dict[str, Any], score: float) -> dict[str, Any]:
        sid = candidate.get("netease_song_id")
        payload: dict[str, Any] = {
            "title": candidate.get("title") or None,
            "artist": candidate.get("artist") or None,
            "artists": list(candidate.get("artists") or []),
            "album": candidate.get("album") or None,
            "cover_url": candidate.get("cover_url") or None,
            "netease_song_id": sid,
            "netease_url": (
                f"https://music.163.com/#/song?id={sid}" if sid else None
            ),
            "match_confidence": float(score),
        }
        # Drop empty fields so the frontend only sees what we actually
        # have, instead of a bag of nulls.
        return {k: v for k, v in payload.items() if v not in (None, "", [])}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset_health(self) -> None:
        """Re-enable the API after the failure trip-switch fired."""
        self._api_alive = True
        self._consecutive_failures = 0

    def clear_cache(self) -> None:
        """Wipe the on-disk cache (mainly for the smoke test)."""
        self._cache.clear()

    def close(self) -> None:
        """Release the SQLite connection.

        Important on Windows, where an open SQLite file blocks the
        enclosing directory from being removed. The smoke test calls
        this in a ``finally`` so a temp directory tears down cleanly.
        """
        try:
            self._cache.close()
        except Exception:                    # noqa: BLE001
            pass

    def __enter__(self) -> "NeteaseMetadataEnricher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "NeteaseMetadataEnricher",
    "NeteaseAPIClient",
    "NeteaseAPIError",
    "NeteaseCache",
]
