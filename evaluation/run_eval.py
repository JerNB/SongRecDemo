"""Offline evaluation runner (P3).

Drives the recommender against the fixed seed profiles and emits a single
JSON *diagnostic report*: per-profile metrics plus an aggregate summary.

Usage
-----
From the project root::

    python -m SongRecDemo.evaluation.run_eval                 # real NetEase
    python -m SongRecDemo.evaluation.run_eval --out report.json
    python -m SongRecDemo.evaluation.run_eval --offline       # canned client

The ``--offline`` flag swaps in a deterministic fake NetEase client so the
harness runs without any network -- handy for CI and for the smoke test,
which calls :func:`run_evaluation` directly with its own fake client.

This harness reports DIAGNOSTIC metrics only (coverage / diversity / novelty
/ source mix / embedding share / duplicate rate / latency / score
distribution). It does not compute accuracy metrics because there are no
human relevance labels yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when run as a script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

from SongRecDemo.netease_pipeline import NeteaseRecommender  # noqa: E402

from .metrics import aggregate_metrics, compute_all_metrics  # noqa: E402
from .profiles import SeedProfile, load_seed_profiles  # noqa: E402

log = logging.getLogger("songrec_eval")


def run_evaluation(
    recommender: NeteaseRecommender,
    profiles: list[SeedProfile],
    *,
    k: Optional[int] = None,
) -> dict[str, Any]:
    """Run every profile through ``recommender`` and assemble a report."""
    per_profile: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for profile in profiles:
        req = profile.to_request(k=k)
        try:
            resp = recommender.recommend(req)
            data = resp.to_dict()
            items = data.get("items") or []
            trace = data.get("trace") or {}
            metrics = compute_all_metrics(items, trace, k=req.k)
            per_profile.append({
                "profile_id": profile.profile_id,
                "label": profile.label,
                "request_id": data.get("request_id"),
                "num_items": len(items),
                "fallback_used": data.get("fallback_used"),
                "metrics": metrics,
            })
        except Exception as exc:  # noqa: BLE001 -- one bad profile shouldn't kill the run
            log.warning("profile %s failed: %s", profile.profile_id, exc)
            per_profile.append({
                "profile_id": profile.profile_id,
                "label": profile.label,
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": None,
            })

    return {
        "report_type": "diagnostic",
        "note": (
            "Diagnostic metrics only (no human relevance labels). "
            "Not accuracy metrics; no Precision/Recall/NDCG."
        ),
        "pipeline_version": config.PIPELINE_VERSION,
        "ranking_config_version": config.RANKING_CONFIG_VERSION,
        "k": k,
        "num_profiles": len(profiles),
        "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        "profiles": per_profile,
        "aggregate": aggregate_metrics(
            [p for p in per_profile if p.get("metrics")]
        ),
    }


def build_default_recommender(*, offline: bool = False) -> NeteaseRecommender:
    """Construct a recommender for a standalone run."""
    if offline:
        from SongRecDemo.netease_pipeline import FakeNeteaseClient, SongFeatureStore
        client = FakeNeteaseClient(
            responses={},
            default=[
                {"netease_song_id": 1001, "title": "Offline Sample One",
                 "artist": "Sample Artist A", "artists": ["Sample Artist A"],
                 "album": "Sample Album A", "cover_url": "https://example/1001.jpg"},
                {"netease_song_id": 1002, "title": "Offline Sample Two",
                 "artist": "Sample Artist B", "artists": ["Sample Artist B"],
                 "album": "Sample Album B", "cover_url": "https://example/1002.jpg"},
                {"netease_song_id": 1003, "title": "Offline Sample Three",
                 "artist": "Sample Artist C", "artists": ["Sample Artist C"],
                 "album": "Sample Album C", "cover_url": "https://example/1003.jpg"},
            ],
            alive=True,
        )
        return NeteaseRecommender(
            client=client,
            cache=None,
            feature_store=SongFeatureStore(":memory:"),
            feedback_store=None,
            feedback_logging_enabled=False,
        )

    # Real run: live NetEase client + on-disk caches/stores.
    from src.personalization.netease_enrichment import NeteaseAPIClient, NeteaseCache
    client = NeteaseAPIClient(
        base_url=config.NETEASE_API_BASE_URL,
        timeout=float(config.NETEASE_TIMEOUT_SECONDS),
        max_retries=int(config.NETEASE_MAX_RETRIES),
    )
    cache = NeteaseCache(config.NETEASE_CACHE_PATH)
    return NeteaseRecommender(client=client, cache=cache)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", default=None, help="Path to seed_profiles.json.")
    p.add_argument("--out", default=None, help="Write the JSON report to this path.")
    p.add_argument("--k", type=int, default=None, help="Override list length k.")
    p.add_argument("--offline", action="store_true",
                   help="Use a deterministic fake NetEase client (no network).")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    profiles = load_seed_profiles(args.profiles)
    recommender = build_default_recommender(offline=args.offline)
    report = run_evaluation(recommender, profiles, k=args.k)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        log.info("Wrote diagnostic report for %d profiles to %s",
                 len(profiles), out_path)
    else:
        print(text)

    agg = report.get("aggregate") or {}
    log.info(
        "Aggregate: diversity=%.3f embedding_share=%.3f novelty=%.3f "
        "dup_artist=%.3f mean_final=%.3f latency_ms=%.1f",
        agg.get("mean_diversity", 0.0),
        agg.get("mean_embedding_share", 0.0),
        agg.get("mean_novelty_score", 0.0),
        agg.get("mean_duplicate_artist_rate", 0.0),
        agg.get("mean_final_score", 0.0),
        agg.get("mean_latency_ms", 0.0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
