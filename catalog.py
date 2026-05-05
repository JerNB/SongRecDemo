"""
In-memory access to the enriched KGRec catalog.

Three small things live here:

* :class:`EnrichedCatalog`  -- loads the JSON catalog produced by
  ``scripts/build_enriched_catalog.py`` and provides a cheap, in-memory
  full-text search ranked across title / artist / album / tags /
  description.
* :class:`CatalogMetadataEnricher` -- conforms to
  :class:`src.personalization.enrichment.MetadataEnricher` so the
  recommender's ``ScoredItem.metadata`` is filled from the same source
  the search index uses.  This guarantees that whatever the user
  searched and clicked is rendered the same way when the recommender
  surfaces it again.
* :func:`confidence_level` -- the canonical implementation of the
  metadata-confidence policy used by the website (and asserted by the
  smoke tests). The frontend mirrors the same logic in
  ``static/app.js`` so the frontend and tests agree.

None of this code touches the model, the training pipeline, the
preprocessing outputs, or the saved validation results.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence policy (single source of truth on the Python side)
# ---------------------------------------------------------------------------

# The thresholds here MUST stay in sync with the frontend mirror in
# ``SongRecDemo/static/app.js::confidenceLevel``.  Both are tested in
# ``smoke_test.py``.
CONF_FULL_THRESHOLD       = 0.60
CONF_BEST_GUESS_THRESHOLD = 0.50
CONF_UNCERTAIN_THRESHOLD  = 0.40

DISPLAY_FULL       = "full"
DISPLAY_BEST_GUESS = "best-guess"
DISPLAY_UNCERTAIN  = "uncertain"
DISPLAY_INTERNAL   = "internal"

ALL_DISPLAY_MODES = (
    DISPLAY_FULL, DISPLAY_BEST_GUESS, DISPLAY_UNCERTAIN, DISPLAY_INTERNAL,
)


def confidence_level(metadata: Optional[dict[str, Any]]) -> str:
    """Map a metadata dict to one of the four documented display modes.

    Policy
    ------
    - ``match_confidence >= 0.60`` and we have a NetEase song id        -> ``"full"``
    - ``0.50 <= match_confidence < 0.60`` and we have a NetEase song id -> ``"best-guess"``
    - ``0.40 <= match_confidence < 0.50`` and we have a NetEase song id -> ``"uncertain"``
    - everything else (no NetEase match, no confidence, internal source)
      -> ``"internal"``

    Parameters
    ----------
    metadata : dict | None
        The dict the recommender (or catalog) returned. May be empty.
    """
    if not metadata:
        return DISPLAY_INTERNAL
    src = metadata.get("source") or ""
    has_netease = bool(metadata.get("netease_song_id"))
    conf = metadata.get("match_confidence")

    # Without a NetEase song id we cannot claim a real-song display,
    # regardless of the confidence number that may have been set when
    # the API was attempted but didn't clear the threshold.
    if "netease" not in src or not has_netease:
        return DISPLAY_INTERNAL
    if conf is None:
        return DISPLAY_INTERNAL

    try:
        c = float(conf)
    except (TypeError, ValueError):
        return DISPLAY_INTERNAL

    if c >= CONF_FULL_THRESHOLD:
        return DISPLAY_FULL
    if c >= CONF_BEST_GUESS_THRESHOLD:
        return DISPLAY_BEST_GUESS
    if c >= CONF_UNCERTAIN_THRESHOLD:
        return DISPLAY_UNCERTAIN
    return DISPLAY_INTERNAL


# ---------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SEARCH_STOP = frozenset({
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for",
    "by", "with", "from", "is", "was", "are", "be",
})


def _tokenise(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _meaningful(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t and t not in _SEARCH_STOP]


# Per-field weights for the search ranker. Tuned for "song-search vibe":
# artist matches dominate, then title, then everything else. Tags get a
# meaningful score because users often type a vibe ("indie", "80s")
# rather than a name.
_FIELD_WEIGHTS = {
    "artist":      5.0,
    "title":       4.0,
    "album":       2.0,
    "tag":         2.0,
    "description": 0.6,
}

# Bonus when the entire query string (case-insensitive, normalised
# whitespace) appears verbatim in artist or title.
_EXACT_BONUS = 6.0
# Bonus when every query token shows up across artist+title combined.
_ALL_TOKENS_IN_NAME_BONUS = 2.5
# Multiplier applied to the final score from the catalog match
# confidence so real-song catalog entries float to the top when query
# matches are otherwise tied (a tag match for an internal-only entry
# shouldn't out-rank an artist match for a real song).
_CONFIDENCE_BOOST = 1.5


class _IndexedItem:
    """Pre-tokenised view of a catalog row, kept tiny on purpose."""

    __slots__ = (
        "item_id",
        "title", "artist", "album",
        "title_tokens", "artist_tokens", "album_tokens",
        "tag_tokens", "desc_tokens",
        "match_confidence", "raw",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.item_id = str(raw["item_id"])
        self.title = raw.get("title")
        self.artist = raw.get("artist")
        self.album = raw.get("album")
        self.title_tokens = set(_meaningful(_tokenise(self.title)))
        self.artist_tokens = set(_meaningful(_tokenise(self.artist)))
        self.album_tokens = set(_meaningful(_tokenise(self.album)))
        # Tag tokens: split each tag on non-alphanumerics so multi-word
        # tags like "indie rock" produce {indie, rock}.
        self.tag_tokens = set()
        for tag in raw.get("tags") or []:
            self.tag_tokens.update(_meaningful(_tokenise(tag)))
        self.desc_tokens = set(_meaningful(_tokenise(raw.get("description"))))
        self.match_confidence = raw.get("match_confidence")
        self.raw = raw


class EnrichedCatalog:
    """In-memory loader + searcher for the enriched catalog.

    The catalog is a small JSON file (~few MB at 8 640 items) so we
    eagerly parse it and keep everything in memory. Search is a linear
    scan with per-field weighting -- fast enough that we don't need an
    inverted index, and easy to reason about.
    """

    def __init__(self, items: list[dict[str, Any]], summary: Optional[dict[str, Any]] = None) -> None:
        self._items: list[_IndexedItem] = [_IndexedItem(it) for it in items]
        self._by_id: dict[str, _IndexedItem] = {it.item_id: it for it in self._items}
        # Token -> set of item indexes; used to short-circuit the linear
        # scan to candidate items that contain at least one query token
        # in any indexed field.
        self._postings: dict[str, set[int]] = defaultdict(set)
        for i, it in enumerate(self._items):
            for tok in (it.title_tokens | it.artist_tokens | it.album_tokens
                        | it.tag_tokens | it.desc_tokens):
                self._postings[tok].add(i)
        self._summary = summary or {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "EnrichedCatalog":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Enriched catalog not found at {path}. "
                f"Build it first:\n  python scripts/build_enriched_catalog.py"
            )
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        items = blob.get("items") or []
        if not isinstance(items, list):
            raise ValueError(f"Catalog at {path} has no 'items' list.")
        return cls(items=items, summary=blob.get("summary") or {})

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self._summary)

    def has(self, item_id: str) -> bool:
        return str(item_id) in self._by_id

    def get(self, item_id: str) -> Optional[dict[str, Any]]:
        it = self._by_id.get(str(item_id))
        return dict(it.raw) if it is not None else None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Rank items against ``query`` and return up to ``limit`` results.

        Each result is a fresh dict containing the catalog row plus a
        synthetic ``_score`` and ``display`` field so callers don't need
        to recompute the confidence policy.
        """
        if not query or not query.strip():
            return []
        q = query.strip()
        q_lower = q.lower()
        tokens = _meaningful(_tokenise(q_lower))
        if not tokens:
            return []

        # Candidate set = union of postings for query tokens. For 8 640
        # items the union may cover most of the catalog for vague
        # queries ("indie") but the per-item scoring is cheap.
        candidate_idx: set[int] = set()
        for tok in tokens:
            candidate_idx.update(self._postings.get(tok, set()))
        if not candidate_idx:
            return []

        results: list[tuple[float, _IndexedItem]] = []
        for idx in candidate_idx:
            it = self._items[idx]
            score = self._score_item(tokens, q_lower, it)
            if score > 0:
                results.append((score, it))

        results.sort(key=lambda x: (-x[0], x[1].item_id))
        out: list[dict[str, Any]] = []
        for score, it in results[: max(1, int(limit))]:
            row = dict(it.raw)
            row["_score"] = round(float(score), 4)
            row["display"] = confidence_level(row)
            out.append(row)
        return out

    @staticmethod
    def _score_item(tokens: list[str], q_lower: str, it: _IndexedItem) -> float:
        score = 0.0
        for tok in tokens:
            if tok in it.artist_tokens:
                score += _FIELD_WEIGHTS["artist"]
            if tok in it.title_tokens:
                score += _FIELD_WEIGHTS["title"]
            if tok in it.album_tokens:
                score += _FIELD_WEIGHTS["album"]
            if tok in it.tag_tokens:
                score += _FIELD_WEIGHTS["tag"]
            if tok in it.desc_tokens:
                score += _FIELD_WEIGHTS["description"]

        # Whole-query exact-substring bonus on artist/title.
        for field in (it.artist, it.title):
            if field and q_lower in field.lower():
                score += _EXACT_BONUS
                break

        # Bonus when every query token appears somewhere in name (artist
        # OR title) -- protects "best coast" from being beaten by a
        # description that mentions the word "best" twice.
        name_tokens = it.artist_tokens | it.title_tokens
        if name_tokens and all(t in name_tokens for t in tokens):
            score += _ALL_TOKENS_IN_NAME_BONUS

        # Confidence boost: real-song matches outrank pure-internal hits.
        conf = it.match_confidence
        if score > 0 and conf is not None:
            try:
                score += _CONFIDENCE_BOOST * float(conf)
            except (TypeError, ValueError):
                pass
        return score


# ---------------------------------------------------------------------------
# Catalog-backed enricher
# ---------------------------------------------------------------------------

class CatalogMetadataEnricher:
    """Returns the same metadata that the search endpoint sees.

    Conforms to :class:`MetadataEnricher`. Used by the demo server so
    every recommendation is annotated with the catalog's display
    fields without making any network calls.

    If an item is missing from the catalog (shouldn't happen for the
    normal build, but safe to guard against), the optional
    ``fallback`` enricher is consulted.
    """

    # Catalog rows include some plumbing we don't want to leak into the
    # ScoredItem.metadata (the search index doesn't need them at runtime).
    _DROP_FROM_METADATA = ("search_text", "_score", "display")

    def __init__(
        self,
        catalog: EnrichedCatalog,
        *,
        fallback: Optional[Any] = None,
    ) -> None:
        self._catalog = catalog
        self._fallback = fallback

    def enrich(self, item_id: str) -> dict[str, Any]:
        row = self._catalog.get(str(item_id))
        if row is None:
            if self._fallback is not None:
                try:
                    return dict(self._fallback.enrich(item_id) or {})
                except Exception as exc:        # noqa: BLE001
                    log.warning("Catalog fallback crashed on %s: %s", item_id, exc)
            return {}
        # Strip search-only / synthetic fields and drop empties so the
        # ScoredItem.metadata stays clean.
        out: dict[str, Any] = {}
        for k, v in row.items():
            if k in self._DROP_FROM_METADATA:
                continue
            if v in (None, "", []):
                continue
            out[k] = v
        # Always re-derive the display mode so it agrees with the
        # canonical confidence_level() implementation, even if the
        # catalog file was hand-edited.
        out.setdefault("source", "internal")
        out["display"] = confidence_level(out)
        return out

    def enrich_batch(self, item_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {iid: self.enrich(iid) for iid in (str(i) for i in item_ids)}
