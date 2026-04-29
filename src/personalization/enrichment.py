"""
Metadata enrichment layer (stubs + interface).

Why this exists
---------------
KGRec-music item IDs are opaque integers. The recommendation engine
returns them as-is because the engine has no display obligation. A
polished frontend demo, however, will want human-readable titles,
artist names, album art, etc. Those fields are not in KGRec-music and
would come from an external source (MusicBrainz, Last.fm, Spotify API).

To keep the interface future-proof without coupling the engine to any
specific external API, the engine calls a :class:`MetadataEnricher`
after ranking and writes whatever it returns into
``ScoredItem.metadata``. Swapping enrichers (null -> internal ->
MusicBrainz) is a one-line change in the service wiring.

Three implementations are shipped:

* :class:`NullEnricher`              -- returns ``{}``, the default.
* :class:`InternalFeaturesEnricher`  -- pulls tags and description
  snippets from the pre-computed ``item_features`` DataFrame. Good
  enough to prototype a UI without external API calls.
* ``ExternalAPIEnricher``            -- not implemented. The Protocol
  below documents the contract so a future hire can drop one in.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MetadataEnricher(Protocol):
    """Strategy interface for display-field enrichment.

    A conforming implementation must return a JSON-serialisable dict
    per item. Any subset of the recommended keys is fine -- the
    frontend code must treat every key as optional.

    Recommended keys (not enforced)
    -------------------------------
      ``title``        : str   -- track title, ideally cleaned
      ``artist``       : str   -- artist name
      ``album``        : str   -- album name
      ``cover_url``    : str   -- URL of album art
      ``preview_url``  : str   -- URL of a short audio preview
      ``tags``         : list[str]  -- human-readable tags
      ``description``  : str   -- short description / blurb
      ``external_ids`` : dict  -- e.g. ``{"mbid": ..., "spotify": ...}``

    The enricher must be cheap to call at recommendation time; cache
    aggressively. Raising for a single item must not kill the whole
    response -- return ``{}`` for that item instead.
    """

    def enrich(self, item_id: str) -> dict[str, Any]:
        """Return display fields for one item. Must never raise."""
        ...

    def enrich_batch(
        self, item_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        """Return display fields keyed by item_id. Missing items map to ``{}``."""
        ...


# ---------------------------------------------------------------------------
# Default implementation: do nothing.
# ---------------------------------------------------------------------------

class NullEnricher:
    """Zero-effort enricher. Every call returns an empty dict.

    The engine uses this by default so recommendations still work with
    zero external-metadata configuration. The UI, if present, sees no
    extra fields and must fall back to ``item_id`` for display.
    """

    def enrich(self, item_id: str) -> dict[str, Any]:
        return {}

    def enrich_batch(
        self, item_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        return {iid: {} for iid in item_ids}


# ---------------------------------------------------------------------------
# Internal enricher: uses the data we already have.
# ---------------------------------------------------------------------------

class InternalFeaturesEnricher:
    """Minimal enricher that reads from the existing ``item_features``.

    Fields returned
    ---------------
      ``tags``        : up to ``max_tags`` normalised tag tokens
      ``description`` : first ``desc_chars`` characters of cleaned
                         description (trimmed at a word boundary)
      ``has_tags``    : bool -- useful for a UI that wants to hide
                        items with no tag evidence

    It does NOT invent titles or artists; those are not in the
    dataset. A production demo should layer a real external enricher
    on top (or beside) this one. The :class:`MetadataEnricher`
    Protocol supports stacking via a thin composite wrapper -- not
    implemented here, but intentionally easy to add.
    """

    def __init__(
        self,
        item_features: pd.DataFrame,
        max_tags: int = 8,
        desc_chars: int = 240,
    ) -> None:
        self._features = item_features
        self._max_tags = int(max_tags)
        self._desc_chars = int(desc_chars)

    def enrich(self, item_id: str) -> dict[str, Any]:
        if item_id not in self._features.index:
            return {}
        row = self._features.loc[item_id]
        raw_tags = row.get("tags_normalised", None)
        if raw_tags is None:
            tags: list[str] = []
        else:
            tags = list(raw_tags)[: self._max_tags]
        desc_val = row.get("desc_clean", "")
        desc = str(desc_val) if desc_val is not None else ""
        if len(desc) > self._desc_chars:
            # Trim to a clean word boundary to avoid mid-word cut-offs
            # that look ugly in a UI card.
            cut = desc.rfind(" ", 0, self._desc_chars)
            if cut <= 0:
                cut = self._desc_chars
            desc = desc[:cut].rstrip(",. ") + "..."
        return {
            "tags": tags,
            "description": desc,
            "has_tags": bool(row.get("has_tags", False)) or len(tags) > 0,
        }

    def enrich_batch(
        self, item_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        return {iid: self.enrich(iid) for iid in item_ids}
