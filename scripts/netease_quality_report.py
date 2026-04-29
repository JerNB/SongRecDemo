"""
Quality report for the NetEase metadata enricher.

Runs a fan of recommendation requests against the live NetEase API
service and aggregates how often the enricher attached external
metadata, how confident the matches were, and what kinds of items
fell back to internal-only metadata.

This script is purely an *evaluation* helper -- it does NOT modify
the recommender model or the evaluation pipeline. It only inspects
``ScoredItem.metadata`` after enrichment, which is the same field
the website demo will surface.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import config
from src.data.artifacts import load_processed_artifacts
from src.personalization import (
    InternalFeaturesEnricher,
    NeteaseMetadataEnricher,
    RecommendationRequest,
    RecommendationService,
    SeedInput,
)


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


def _build_scenarios(svc: RecommendationService) -> list[tuple[str, RecommendationRequest]]:
    """A spread of seeds covering different KGRec items."""
    catalogue = list(svc._item_ids)               # type: ignore[attr-defined]
    # Pick a few interesting items we already know about.
    known_good = [s for s in ["1970", "1022", "1679", "2076", "1", "500"] if s in catalogue]
    extra = [catalogue[i] for i in (5, 50, 250, 750, 1500, 3000, 5000) if i < len(catalogue)]
    return [
        (
            "S1: known-good seeds (Bon Iver, Jazmine Sullivan, Best Coast)",
            RecommendationRequest(
                seeds=SeedInput(item_ids=known_good[:3]),
                n=10, novelty=0.2, content_weight=0.4, diversity=0.3,
            ),
        ),
        (
            "S2: tag-driven (indie + folk)",
            RecommendationRequest(
                seeds=SeedInput(tags=["indie", "folk", "acoustic"]),
                n=10, novelty=0.3, content_weight=0.5,
            ),
        ),
        (
            "S3: tag-driven (electronic + dance)",
            RecommendationRequest(
                seeds=SeedInput(tags=["electronic", "dance", "house"]),
                n=10, novelty=0.4, content_weight=0.5,
            ),
        ),
        (
            "S4: rock-ish seeds + alternative tag",
            RecommendationRequest(
                seeds=SeedInput(item_ids=extra[:2], tags=["alternative", "rock"]),
                n=10, novelty=0.2, content_weight=0.4, diversity=0.3,
            ),
        ),
        (
            "S5: cold start (popularity)",
            RecommendationRequest(seeds=SeedInput(), n=10),
        ),
        (
            "S6: pop / chart-y",
            RecommendationRequest(
                seeds=SeedInput(tags=["pop", "chart"]),
                n=10, novelty=0.1, content_weight=0.5,
            ),
        ),
        (
            "S7: high novelty long-tail",
            RecommendationRequest(
                seeds=SeedInput(item_ids=known_good[:2]),
                n=10, novelty=1.0, content_weight=0.3,
            ),
        ),
        (
            "S8: large-fanout sweep",
            RecommendationRequest(
                seeds=SeedInput(item_ids=extra[:3]),
                n=15, novelty=0.4, content_weight=0.4, diversity=0.4,
            ),
        ),
    ]


def _bucket(conf: float) -> str:
    if conf <= 0.0:
        return "0.00 (fallback)"
    if conf < 0.40:
        return "<0.40 (rejected)"
    if conf < 0.50:
        return "0.40-0.50 (artist-only)"
    if conf < 0.60:
        return "0.50-0.60 (artist+tag/title)"
    if conf < 0.70:
        return "0.60-0.70 (strong)"
    return ">=0.70 (excellent)"


def main() -> None:
    print("Loading KGRec processed artefacts ...", file=sys.stderr)
    arts = load_processed_artifacts()
    fallback = InternalFeaturesEnricher(arts.item_features)
    enricher = NeteaseMetadataEnricher(
        item_features=arts.item_features,
        base_url=config.NETEASE_API_BASE_URL,
        fallback=fallback,
    )

    print("Building RecommendationService ...", file=sys.stderr)
    svc = RecommendationService.from_artifacts(enricher=enricher)
    print(f"  catalogue size = {svc.catalogue_size()}", file=sys.stderr)

    scenarios = _build_scenarios(svc)

    all_items: list[dict[str, Any]] = []
    per_scenario: list[dict[str, Any]] = []

    for title, req in scenarios:
        resp = svc.recommend(req)
        records = []
        for it in resp.items:
            md = it.metadata or {}
            source = md.get("source") or "internal"
            is_netease = "netease" in source
            records.append({
                "scenario": title,
                "kgrec_id": it.item_id,
                "score": it.score,
                "metadata_source": source,
                "match_confidence": md.get("match_confidence") if is_netease else None,
                "netease_id": md.get("netease_song_id"),
                "title": md.get("title"),
                "artist": md.get("artist"),
                "album": md.get("album"),
                "page_url": md.get("netease_url"),
                "cover_url": md.get("cover_url"),
                "tags": (md.get("tags") or [])[:5],
                "desc": (md.get("description") or "")[:120],
            })
        all_items.extend(records)
        per_scenario.append({"title": title, "n_items": len(records), "items": records})

    # ----- aggregate stats -----
    n_total = len(all_items)
    n_enriched = sum(1 for r in all_items if r["metadata_source"] and "netease" in r["metadata_source"])
    n_fallback = n_total - n_enriched
    bucket_counts = collections.Counter(_bucket(r["match_confidence"] or 0.0) for r in all_items)
    confs = [r["match_confidence"] for r in all_items if r["match_confidence"]]

    print()
    print("=" * 78)
    print(f"NetEase enrichment quality report  ({n_total} items across "
          f"{len(scenarios)} scenarios)")
    print("=" * 78)
    print(f"Enriched from NetEase   : {n_enriched:4d}  ({n_enriched / n_total:.0%})")
    print(f"Fell back to internal   : {n_fallback:4d}  ({n_fallback / n_total:.0%})")
    if confs:
        confs_sorted = sorted(confs)
        print(f"Confidence (n={len(confs)}): "
              f"min={min(confs):.2f}  median={confs_sorted[len(confs)//2]:.2f}  "
              f"max={max(confs):.2f}  mean={sum(confs)/len(confs):.2f}")
    print()
    print("Confidence buckets:")
    for k in [
        ">=0.70 (excellent)", "0.60-0.70 (strong)", "0.50-0.60 (artist+tag/title)",
        "0.40-0.50 (artist-only)", "<0.40 (rejected)", "0.00 (fallback)",
    ]:
        if bucket_counts.get(k, 0):
            print(f"  {k:<32s} {bucket_counts[k]}")

    # ----- best matches -----
    print()
    print("-" * 78)
    print("Top-confidence enriched examples:")
    print("-" * 78)
    enriched = [r for r in all_items if r["metadata_source"] and "netease" in r["metadata_source"]]
    enriched.sort(key=lambda r: -(r["match_confidence"] or 0.0))
    seen = set()
    shown = 0
    for r in enriched:
        key = (r["artist"], r["title"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  KGRec {r['kgrec_id']:>6}  conf={r['match_confidence']:.2f}  "
              f"-> {r['artist']} -- {r['title']}  (album={r['album']!r})")
        print(f"           desc: {r['desc']!r}")
        shown += 1
        if shown >= 8:
            break

    # ----- low-confidence-but-accepted -----
    print()
    print("-" * 78)
    print("Lowest-confidence enriched examples (just-passed-threshold):")
    print("-" * 78)
    enriched.sort(key=lambda r: (r["match_confidence"] or 0.0))
    seen.clear()
    shown = 0
    for r in enriched:
        key = (r["artist"], r["title"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  KGRec {r['kgrec_id']:>6}  conf={r['match_confidence']:.2f}  "
              f"-> {r['artist']} -- {r['title']}  (album={r['album']!r})")
        print(f"           desc: {r['desc']!r}")
        shown += 1
        if shown >= 6:
            break

    # ----- fallback examples -----
    print()
    print("-" * 78)
    print("Fallback (internal-only) examples:")
    print("-" * 78)
    fb = [r for r in all_items if not r["metadata_source"] or "netease" not in r["metadata_source"]]
    seen_ids = set()
    shown = 0
    for r in fb:
        if r["kgrec_id"] in seen_ids:
            continue
        seen_ids.add(r["kgrec_id"])
        print(f"  KGRec {r['kgrec_id']:>6}  tags={r['tags']}")
        print(f"           desc: {r['desc']!r}")
        shown += 1
        if shown >= 6:
            break

    # ----- write JSON dump -----
    out = Path("artifacts/netease_quality_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(per_scenario, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print()
    print(f"Wrote per-scenario dump to {out}")
    enricher.close()


if __name__ == "__main__":
    main()
