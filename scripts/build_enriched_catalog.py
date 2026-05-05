"""
Offline builder for the enriched KGRec catalog used by the demo website.

What this script does
---------------------
For every KGRec item that the preprocessing stage produced, look up the
best display metadata we can attach and save a single JSON document at
``artifacts/models/enriched_catalog.json``.

The website's search endpoint (``GET /api/search``) and the demo's
runtime metadata enricher (``SongRecDemo/catalog.py::CatalogMetadataEnricher``)
both read from this file. Building it offline once gives the demo:

* Fast, deterministic startup (no API calls during browsing).
* Real-song search by title/artist/album/tag/description.
* Stable display fields whether or not the NetEase API is running
  later.

Why a separate offline step
---------------------------
The recommender training pipeline, ALS model, content-based model,
evaluation pipeline, and saved validation results are intentionally
NOT modified by this script.  Enrichment is a *display-side* concern,
and we keep it strictly outside the modelling code.

How enrichment works
--------------------
For each item we call :class:`NeteaseMetadataEnricher` (which already
caches results in ``artifacts/netease_cache.sqlite`` and falls back to
internal KGRec metadata on low-confidence / API failure). That means:

* Re-running the script never re-hits the NetEase API for items it
  already attempted (the SQLite cache short-circuits).
* The script is safe to interrupt and resume.
* If the API is down, every item gets the internal-features
  (tags + description) row only -- the catalog is still complete and
  the demo still works.

CLI
---
``--no-netease``   skip the NetEase API entirely; use internal-features
                    enricher only.  Useful for offline development.
``--limit N``      enrich only the first N items (smoke / debug runs).
``--output PATH``  override the output path.
``--no-cache``     wipe the NetEase SQLite cache before starting.
``--quiet``        less console chatter.

Run::

    python scripts/build_enriched_catalog.py
    python scripts/build_enriched_catalog.py --no-netease
    python scripts/build_enriched_catalog.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when launched as ``python scripts/...``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config                                              # noqa: E402
from src.data.artifacts import load_processed_artifacts    # noqa: E402
from src.personalization import (                          # noqa: E402
    InternalFeaturesEnricher,
    NeteaseMetadataEnricher,
)
from src.personalization.enrichment import MetadataEnricher  # noqa: E402

log = logging.getLogger("build_enriched_catalog")

DEFAULT_OUTPUT = config.MODELS_DIR / "enriched_catalog.json"

# Whitelisted display fields we copy out of the enricher payload.
# Anything else stays internal to keep the catalog tight.
NETEASE_DISPLAY_FIELDS = (
    "title",
    "artist",
    "artists",
    "album",
    "cover_url",
    "netease_song_id",
    "netease_url",
    "match_confidence",
    "source",
    "netease_attempted",
)

INTERNAL_DISPLAY_FIELDS = (
    "tags",
    "description",
    "has_tags",
)

# Lowercased tokens that are pure noise for a search index. Kept short
# on purpose -- the search ranker already weights field-matches, so an
# aggressive stoplist would only hurt recall.
_SEARCH_STOP = frozenset({
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for",
    "by", "with", "from", "is", "was", "are", "be",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _build_search_text(
    *,
    title: Optional[str],
    artist: Optional[str],
    album: Optional[str],
    tags: list[str],
    description: Optional[str],
) -> str:
    """Compose a single, normalised string used by the cheap full-text scan.

    Each field's tokens are kept separately in the search ranker so the
    catalog also stores the raw fields. ``search_text`` is just a
    redundant convenience field for substring fallbacks and for users
    who want to pre-filter the catalog with simple tools.
    """
    parts: list[str] = []
    for s in (title, artist, album, description):
        if s:
            parts.extend(t for t in _tokenise(str(s)) if t not in _SEARCH_STOP)
    for tag in tags or []:
        parts.extend(t for t in _tokenise(str(tag)) if t not in _SEARCH_STOP)
    # Deduplicate but preserve order so the field is human-readable.
    seen: set[str] = set()
    out: list[str] = []
    for tok in parts:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return " ".join(out)


def _description_snippet(text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    return text[:cut].rstrip(",. ") + "..."


# ---------------------------------------------------------------------------
# Enricher selection
# ---------------------------------------------------------------------------

def _make_enricher(*, use_netease: bool, item_features) -> MetadataEnricher:
    if not use_netease:
        log.info("NetEase enrichment disabled; using InternalFeaturesEnricher only.")
        return InternalFeaturesEnricher(item_features)

    log.info("NetEase enrichment enabled (cache: %s)", config.NETEASE_CACHE_PATH)
    log.info("API base URL: %s", config.NETEASE_API_BASE_URL)
    enricher = NeteaseMetadataEnricher(
        item_features=item_features,
        fallback=InternalFeaturesEnricher(item_features),
    )

    # Probe the live API once. If it doesn't answer, pre-trip the
    # enricher's "API alive" flag so the rest of the run is cache-only
    # and fast -- we still get the value of cached matches without
    # paying a 5s timeout per uncached item.
    from src.personalization.netease_enrichment import NeteaseAPIClient
    probe = NeteaseAPIClient(
        base_url=config.NETEASE_API_BASE_URL,
        timeout=min(2.0, float(config.NETEASE_TIMEOUT_SECONDS)),
        max_retries=0,
    )
    if probe.ping():
        log.info("NetEase API responded; live enrichment enabled.")
    else:
        log.warning(
            "NetEase API at %s is not responding; running in CACHE-ONLY "
            "mode (previously cached matches still surface, but no new "
            "API calls will be made).",
            config.NETEASE_API_BASE_URL,
        )
        # NeteaseMetadataEnricher uses this private flag as its
        # trip-switch; setting it pre-emptively avoids the per-item
        # timeout overhead.
        enricher._api_alive = False           # noqa: SLF001
    return enricher


# ---------------------------------------------------------------------------
# Per-item record assembly
# ---------------------------------------------------------------------------

def _record_for_item(
    item_id: str,
    enriched: dict[str, Any],
) -> dict[str, Any]:
    """Project the enricher's free-form dict to the catalog schema."""
    title = enriched.get("title")
    artist = enriched.get("artist")
    artists = enriched.get("artists") or ([artist] if artist else [])
    album = enriched.get("album")
    cover_url = enriched.get("cover_url")
    netease_url = enriched.get("netease_url")
    netease_song_id = enriched.get("netease_song_id")
    match_confidence = enriched.get("match_confidence")
    source = enriched.get("source") or "internal"

    # Cap tags at 12 to keep the catalog file small; the demo card only
    # renders the first 8 anyway.
    raw_tags = enriched.get("tags") or []
    tags = [str(t) for t in raw_tags[:12]]

    description = _description_snippet(str(enriched.get("description") or ""))

    record: dict[str, Any] = {
        "item_id": str(item_id),
        # Display fields (any may be absent -> frontend falls back).
        "title": str(title) if title else None,
        "artist": str(artist) if artist else None,
        "artists": [str(a) for a in artists if a],
        "album": str(album) if album else None,
        "cover_url": str(cover_url) if cover_url else None,
        "netease_song_id": int(netease_song_id) if netease_song_id else None,
        "netease_url": str(netease_url) if netease_url else None,
        "match_confidence": (
            float(match_confidence) if match_confidence is not None else None
        ),
        "source": source,
        # Internal context, kept for the "uncertain"/"internal" display
        # branches the frontend renders when confidence is low.
        "tags": tags,
        "description": description,
        "has_tags": bool(enriched.get("has_tags") or len(tags) > 0),
        # Convenience field for cheap search.
        "search_text": _build_search_text(
            title=title, artist=artist, album=album,
            tags=tags, description=description,
        ),
    }
    return record


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_catalog(
    *,
    use_netease: bool,
    limit: Optional[int] = None,
    output_path: Path = DEFAULT_OUTPUT,
    wipe_cache: bool = False,
    progress_every: int = 200,
) -> dict[str, Any]:
    """Enrich every KGRec item and write the catalog to disk.

    Returns the catalog dict (also persisted to ``output_path``).
    """
    log.info("Loading processed artefacts ...")
    arts = load_processed_artifacts()
    item_features = arts.item_features
    item_ids = list(map(str, item_features.index))
    if limit is not None:
        item_ids = item_ids[: int(limit)]
    log.info("Will enrich %d items.", len(item_ids))

    enricher = _make_enricher(use_netease=use_netease, item_features=item_features)
    if wipe_cache and isinstance(enricher, NeteaseMetadataEnricher):
        log.info("Wiping NetEase cache before run.")
        enricher.clear_cache()
        enricher.reset_health()

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    last_log = started
    for i, item_id in enumerate(item_ids, start=1):
        try:
            md = enricher.enrich(item_id) or {}
        except Exception as exc:                # noqa: BLE001 -- never crash the build.
            log.warning("Enricher crashed on item %s: %s", item_id, exc)
            md = {}
        records.append(_record_for_item(item_id, md))

        if i % progress_every == 0 or i == len(item_ids):
            now = time.monotonic()
            rate = i / max(now - started, 1e-6)
            log.info(
                "  %d/%d  (%.1f items/s, +%.2fs since last log)",
                i, len(item_ids), rate, now - last_log,
            )
            last_log = now

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    by_source: Counter[str] = Counter(r["source"] for r in records)
    by_band: Counter[str] = Counter()
    for r in records:
        c = r.get("match_confidence")
        if c is None or r["source"] == "internal":
            by_band["internal"] += 1
        elif c >= 0.60:
            by_band["full (>=0.60)"] += 1
        elif c >= 0.50:
            by_band["best-guess (0.50-0.60)"] += 1
        elif c >= 0.40:
            by_band["uncertain (0.40-0.50)"] += 1
        else:
            by_band["below-threshold"] += 1

    summary = {
        "n_items": len(records),
        "by_source": dict(by_source),
        "by_confidence_band": dict(by_band),
        "use_netease": bool(use_netease),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    catalog = {
        "version": 1,
        "summary": summary,
        "items": records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing catalog -> %s", output_path)
    output_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Pretty summary on stderr so it survives stdout redirection.
    log.info("Done. Summary:")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)

    # Close the SQLite cache cleanly (matters on Windows).
    close = getattr(enricher, "close", None)
    if callable(close):
        try:
            close()
        except Exception:                       # noqa: BLE001
            pass

    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-netease", action="store_true",
                   help="Skip the NetEase API; use internal-features enricher only.")
    p.add_argument("--limit", type=int, default=None,
                   help="Enrich only the first N items (debug / smoke runs).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--no-cache", dest="wipe_cache", action="store_true",
                   help="Wipe the NetEase SQLite cache before starting.")
    p.add_argument("--quiet", action="store_true",
                   help="Less console chatter.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    build_catalog(
        use_netease=not args.no_netease,
        limit=args.limit,
        output_path=args.output,
        wipe_cache=args.wipe_cache,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
