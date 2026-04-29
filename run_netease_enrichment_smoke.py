"""
Smoke test for the NetEase metadata enrichment side-channel.

The test exercises the four behaviours we promise about
:class:`NeteaseMetadataEnricher`:

  1. *API available*           -- a live ping to ``NETEASE_API_BASE_URL``
                                  succeeds, and the search endpoint
                                  returns at least one well-formed
                                  candidate for a known query.
  2. *API unavailable*         -- when pointed at a deliberately wrong
                                  URL, the enricher falls back to the
                                  internal enricher silently.
  3. *Low-confidence fallback* -- when the API answers but no candidate
                                  scores above ``min_confidence``, the
                                  enricher returns internal metadata
                                  with ``netease_attempted=True`` and
                                  no NetEase fields.
  4. *Successful enrichment*   -- a real KGRec item picks up at least
                                  ``netease_song_id`` and ``title``
                                  from the API and surfaces them in
                                  the merged metadata dict.

Run::

    python run_netease_enrichment_smoke.py
    python run_netease_enrichment_smoke.py --base-url http://localhost:3000

By default the script picks two KGRec item IDs whose tags clearly hint
at well-known artists ("best-coast", "bon-iver") so the live test
has a fair chance even with a small ``search_limit``.  Override with
``--items 1234,5678`` if you want to point it at your own picks.

Exit code is 0 if every reachable assertion passed and 1 otherwise.
Test 1 (live API) is reported as SKIPPED -- not failed -- when the
service is not running, so the script doubles as a useful "is the
NetEase server up?" probe.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import config
from src.data.artifacts import load_processed_artifacts
from src.personalization import (
    NeteaseAPIClient,
    NeteaseAPIError,
    NeteaseMetadataEnricher,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("netease_smoke")


# ---------------------------------------------------------------------------
# Tiny test framework (no pytest dependency for a smoke script)
# ---------------------------------------------------------------------------

class _Result:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


def _print_result(name: str, status: str, detail: str = "") -> None:
    bar = "-" * 78
    print(bar)
    flag = {
        _Result.PASS: "[ OK ]",
        _Result.FAIL: "[FAIL]",
        _Result.SKIP: "[SKIP]",
    }[status]
    print(f"{flag}  {name}")
    if detail:
        for line in detail.rstrip().splitlines():
            print(f"        {line}")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _pick_demo_items(item_features) -> list[str]:
    """Choose a couple of KGRec items whose tags scream a clear artist.

    The preprocessor stores tags in a space-normalised form (hyphens
    are replaced with spaces; see ``_normalise_tag`` in
    ``src/data/preprocessor.py``), so we match against that form here.

    Falls back to the first two rows if nothing matches.
    """
    preferred = [
        "best coast",
        "bon iver",
        "daft punk",
        "the strokes",
        "radiohead",
        "arcade fire",
        "vampire weekend",
    ]
    picked: list[str] = []
    seen: set[str] = set()
    # Materialise the tag column once -- ``row.get`` on a pandas row
    # returns numpy-backed arrays for list-like cells, which break
    # ``or []`` short-circuiting because of ambiguous truth.
    tag_col = item_features["tags_normalised"]
    for needle in preferred:
        for iid, raw_tags in tag_col.items():
            sid = str(iid)
            if sid in seen or raw_tags is None:
                continue
            tags_lower = {str(t).lower() for t in list(raw_tags)}
            if needle in tags_lower:
                picked.append(sid)
                seen.add(sid)
                break
        if len(picked) >= 2:
            break
    if len(picked) < 2:
        for iid in item_features.index:
            sid = str(iid)
            if sid in seen:
                continue
            picked.append(sid)
            seen.add(sid)
            if len(picked) >= 2:
                break
    return picked[:2]


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_api_available(base_url: str, timeout: float) -> tuple[str, str]:
    """Live ping + a real search round trip."""
    client = NeteaseAPIClient(base_url=base_url, timeout=timeout, max_retries=0)
    if not client.ping():
        return _Result.SKIP, (
            f"NetEase API at {base_url} is not reachable.\n"
            f"Start it with: cd api-enhanced && node app.js\n"
            f"Or run the official Docker image on port 3000."
        )
    try:
        results = client.search_songs("hello adele", limit=3)
    except NeteaseAPIError as exc:
        return _Result.FAIL, f"/search call raised: {exc}"
    if not results:
        return _Result.FAIL, "Live /search returned zero results for a known query."
    sample = results[0]
    missing = [k for k in ("netease_song_id", "title", "artist") if not sample.get(k)]
    if missing:
        return _Result.FAIL, f"Candidate is missing required fields: {missing}\n{sample}"
    return _Result.PASS, (
        f"GET /search returned {len(results)} candidate(s); "
        f"top match: {sample['artist']} -- {sample['title']} "
        f"(id={sample['netease_song_id']})"
    )


def test_api_unavailable_fallback(item_features, item_id: str) -> tuple[str, str]:
    """Point the enricher at a dead URL; it must return internal metadata."""
    # ``ignore_cleanup_errors=True`` smooths over a Windows quirk where
    # the SQLite file may briefly stay locked even after .close(); on
    # Linux/macOS this is a no-op.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = Path(tmp) / "cache.sqlite"
        enr = NeteaseMetadataEnricher(
            item_features=item_features,
            base_url="http://127.0.0.1:1",   # nothing listens on port 1
            timeout=0.5,
            max_retries=0,
            cache_path=cache,
        )
        try:
            md = enr.enrich(item_id)
        finally:
            enr.close()
    if not isinstance(md, dict):
        return _Result.FAIL, f"enrich() returned {type(md).__name__}, expected dict"
    leaked = [k for k in NeteaseMetadataEnricher.NETEASE_FIELDS if k in md]
    if leaked:
        return _Result.FAIL, f"NetEase fields leaked despite dead API: {leaked}"
    if not (md.get("tags") or md.get("description")):
        return _Result.FAIL, (
            f"Fallback metadata is empty for item {item_id}; "
            f"InternalFeaturesEnricher should at least return tags/description.\n"
            f"Got: {md}"
        )
    return _Result.PASS, (
        f"Dead API -> fallback metadata kept "
        f"(tags={len(md.get('tags', []))}, has_desc={bool(md.get('description'))}, "
        f"source={md.get('source')!r})"
    )


def test_low_confidence_fallback(item_features, item_id: str) -> tuple[str, str]:
    """Force a deliberately bad query; the score must fall below threshold."""

    class _AlwaysIrrelevantClient:
        """Stub client that returns a single junk candidate.

        This guarantees the confidence scorer rejects the match, so we
        can verify the fallback path without depending on the live API.
        """

        def search_songs(self, keywords, limit=5):  # noqa: D401, ARG002
            return [{
                "netease_song_id": 123456789,
                "title": "Zzzz Unrelated Synthwave Demo",
                "artist": "Stub Artist Of Nowhere",
                "artists": ["Stub Artist Of Nowhere"],
                "album": "Random Compilation 9999",
                "album_id": None,
                "cover_url": None,
                "duration_ms": None,
            }]

        def ping(self):  # pragma: no cover - not used here
            return True

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = Path(tmp) / "cache.sqlite"
        enr = NeteaseMetadataEnricher(
            item_features=item_features,
            min_confidence=0.99,            # nothing should pass this
            cache_path=cache,
            client=_AlwaysIrrelevantClient(),
        )
        try:
            md = enr.enrich(item_id)
        finally:
            enr.close()
    if md.get("netease_song_id"):
        return _Result.FAIL, (
            f"Low-confidence match was accepted: {md.get('netease_song_id')}"
        )
    if not md.get("netease_attempted"):
        return _Result.FAIL, (
            "Expected metadata to record netease_attempted=True even on rejection."
        )
    if md.get("match_confidence") is None:
        return _Result.FAIL, "Expected match_confidence to be reported on rejection."
    return _Result.PASS, (
        f"Low-confidence match rejected as designed "
        f"(score={md['match_confidence']:.3f} < threshold)."
    )


def test_successful_enrichment(
    item_features,
    item_ids: list[str],
    base_url: str,
    timeout: float,
) -> tuple[str, str]:
    """End-to-end: live API + real KGRec items, expect at least one match."""
    client = NeteaseAPIClient(base_url=base_url, timeout=timeout, max_retries=0)
    if not client.ping():
        return _Result.SKIP, (
            f"NetEase API at {base_url} is not reachable; "
            f"successful-enrichment test cannot run live."
        )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = Path(tmp) / "cache.sqlite"
        enr = NeteaseMetadataEnricher(
            item_features=item_features,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            min_confidence=0.20,            # generous: we just want signs of life
            cache_path=cache,
        )
        matched: list[tuple[str, dict[str, Any]]] = []
        scores: list[float] = []
        try:
            for iid in item_ids:
                md = enr.enrich(iid)
                score = float(md.get("match_confidence") or 0.0)
                scores.append(score)
                if md.get("netease_song_id"):
                    matched.append((iid, md))
        finally:
            enr.close()

    if not matched:
        return _Result.FAIL, (
            "No KGRec item passed the confidence threshold against the live API.\n"
            f"Tried items: {item_ids}\n"
            f"Best confidences: {[f'{s:.3f}' for s in scores]}\n"
            f"This may indicate a tag/description mismatch -- try other items."
        )
    iid, md = matched[0]
    return _Result.PASS, (
        f"KGRec item {iid} -> NetEase id={md['netease_song_id']} "
        f"({md.get('artist')} -- {md.get('title')}) "
        f"confidence={md['match_confidence']:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=config.NETEASE_API_BASE_URL,
        help=f"NetEase API base URL (default: {config.NETEASE_API_BASE_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.NETEASE_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {config.NETEASE_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--items",
        type=str,
        default=None,
        help=(
            "Comma-separated KGRec item IDs to use for the live-match test. "
            "Defaults to two items auto-picked from the catalogue."
        ),
    )
    args = parser.parse_args()

    log.info("Loading KGRec processed artefacts ...")
    arts = load_processed_artifacts()

    if args.items:
        chosen = [t.strip() for t in args.items.split(",") if t.strip()]
    else:
        chosen = _pick_demo_items(arts.item_features)
    log.info("Using KGRec item ids for tests: %s", chosen)

    print()
    print("=" * 78)
    print(" NetEase enrichment smoke test")
    print(f"   base_url       : {args.base_url}")
    print(f"   timeout        : {args.timeout}s")
    print(f"   demo items     : {chosen}")
    print(f"   min_confidence : {config.NETEASE_MIN_CONFIDENCE}")
    print("=" * 78)

    cases: list[tuple[str, tuple[str, str]]] = []

    cases.append((
        "1) API available + valid /search response",
        test_api_available(args.base_url, args.timeout),
    ))
    cases.append((
        "2) API unavailable -> fallback to internal metadata",
        test_api_unavailable_fallback(arts.item_features, chosen[0]),
    ))
    cases.append((
        "3) Low-confidence match -> fallback to internal metadata",
        test_low_confidence_fallback(arts.item_features, chosen[0]),
    ))
    cases.append((
        "4) Successful end-to-end NetEase enrichment",
        test_successful_enrichment(
            arts.item_features, chosen, args.base_url, args.timeout
        ),
    ))

    failed = 0
    skipped = 0
    for name, (status, detail) in cases:
        _print_result(name, status, detail)
        if status == _Result.FAIL:
            failed += 1
        elif status == _Result.SKIP:
            skipped += 1

    print("-" * 78)
    summary = (
        f"PASS={len(cases) - failed - skipped}  "
        f"FAIL={failed}  SKIP={skipped}  TOTAL={len(cases)}"
    )
    print(f" SUMMARY: {summary}")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
