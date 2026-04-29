"""
Interactive CLI demo for the personalized recommender.

This is the text-only stand-in for the future website demo. It
exercises exactly the same :class:`RecommendationService` interface
the website will call, so anything you can do here will be reachable
from the web UI. The goal is to prove that the logic + contract are
complete before the frontend is written.

Three modes are supported:

1. ``--interactive`` (default)
    Prompts you for seeds, favourites, tags, and control knobs;
    prints ranked results + explanations.

2. ``--request path/to/request.json``
    Reads a :class:`RecommendationRequest` in JSON form and writes a
    :class:`RecommendationResponse` to stdout (pretty JSON). This is
    exactly how a REST handler would look on the wire.

3. ``--example``
    Runs three canned scenarios -- seed-only, tags-only, and a
    cold-start -- so you can sanity-check the system without typing.

All output respects the dataclass contract defined in
``src/personalization/interface.py``.
"""

from __future__ import annotations

import argparse
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("personalized_demo")


# ---------------------------------------------------------------------------
# Helpers: parse CLI/JSON input, render output
# ---------------------------------------------------------------------------

def _parse_csv(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_float(raw: str, default: float) -> float:
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(raw: str, default: int) -> int:
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _request_from_dict(data: dict[str, Any]) -> RecommendationRequest:
    """Inflate a RecommendationRequest from a plain dict (e.g. JSON)."""
    seeds_raw = data.get("seeds") or {}
    seeds = SeedInput(
        item_ids=[str(x) for x in (seeds_raw.get("item_ids") or [])],
        favorite_ids=[str(x) for x in (seeds_raw.get("favorite_ids") or [])],
        tags=[str(x) for x in (seeds_raw.get("tags") or [])],
        exclude_ids=[str(x) for x in (seeds_raw.get("exclude_ids") or [])],
    )
    return RecommendationRequest(
        seeds=seeds,
        n=int(data.get("n", 20)),
        novelty=float(data.get("novelty", 0.2)),
        content_weight=float(data.get("content_weight", 0.25)),
        diversity=float(data.get("diversity", 0.0)),
        candidate_pool=int(data.get("candidate_pool", 500)),
        fold_in_tag_seeds=int(data.get("fold_in_tag_seeds", 25)),
        request_id=data.get("request_id"),
    )


def _print_response_pretty(resp_dict: dict[str, Any]) -> None:
    """Human-readable rendering (not JSON) for the CLI demo."""
    bar = "=" * 78
    dash = "-" * 78
    print()
    print(bar)
    print(f" request_id   : {resp_dict['request_id']}")
    print(f" fallback     : {resp_dict['fallback_used']}")
    ctrl = resp_dict["control"]
    print(
        f" controls     : novelty={ctrl['novelty']:.2f}  "
        f"content_weight={ctrl['content_weight']:.2f}  "
        f"diversity={ctrl['diversity']:.2f}  n={ctrl['n']}"
    )
    info = resp_dict["model_info"]
    print(
        f" model        : {info['name']} "
        f"(d={info['factors']}, alpha={info['alpha']}, "
        f"lambda={info['reg']}, iters={info['iterations']})"
    )
    ss = resp_dict["seed_summary"]
    print(f" seeds kept   : {ss['accepted_item_ids']}")
    print(f" favs kept    : {ss['accepted_favorite_ids']}")
    print(f" tags matched : {ss['matched_tags']}")
    if ss["rejected_item_ids"] or ss["rejected_favorite_ids"] or ss["unknown_tags"]:
        print(f" rejected     : items={ss['rejected_item_ids']}  "
              f"favs={ss['rejected_favorite_ids']}  tags={ss['unknown_tags']}")
    if ss["tag_fold_in_item_ids"]:
        print(f" tag->seeds   : {ss['tag_fold_in_item_ids'][:10]}"
              + (" ..." if len(ss["tag_fold_in_item_ids"]) > 10 else ""))
    print(dash)

    for it in resp_dict["items"]:
        sb = it["score_breakdown"]
        meta = it.get("metadata") or {}
        title = meta.get("title")
        artist = meta.get("artist")
        header = f" #{it['rank']:>2}  id={it['item_id']}"
        if title or artist:
            header += f"   {artist or '?'} -- {title or '?'}"
        header += f"   score={it['score']:+.4f}"
        print(header)
        print(f"       why: {it['explanation']}")
        if meta.get("tags"):
            print(f"       meta tags: {meta['tags']}")
        if meta.get("description"):
            snippet = meta["description"]
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            print(f"       meta desc: {snippet}")
        print(
            f"       breakdown: ALS={sb['als']:.3f}  content={sb['content']:.3f}  "
            f"blended={sb['blended']:.3f}  nov_penalty={sb['novelty_penalty']:.3f}  "
            f"pop_count={sb['popularity_count']}"
        )
    print(bar)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(svc: RecommendationService) -> None:
    print()
    print("=" * 78)
    print(" Personalized music recommender -- interactive demo")
    print(f" catalogue   : {svc.catalogue_size()} items")
    print(f" tag vocab   : {svc.vocabulary_size()} tokens")
    print("=" * 78)
    print(
        " Type seed item IDs, favourite item IDs, and/or free-form tags.\n"
        " Leave a field blank to skip it.\n"
        " Item IDs are the KGRec-music raw ids (strings like '1679').\n"
    )

    while True:
        print("-" * 78)
        seeds_raw = input("Seed item IDs (comma-separated, blank to skip): ").strip()
        favs_raw = input("Favourite item IDs (comma-separated, blank to skip): ").strip()
        tags_raw = input("Tags (comma-separated, e.g. indie,mellow,80s): ").strip()

        n = _parse_int(input("How many recommendations [20]: "), 20)
        novelty = _parse_float(input("Novelty (0 = popular, 1 = long-tail) [0.2]: "), 0.2)
        cw = _parse_float(input("Content weight (0 = pure ALS, 1 = pure tags) [0.25]: "), 0.25)
        diversity = _parse_float(input("Diversity (0 = off, 1 = MMR max) [0.0]: "), 0.0)

        req = RecommendationRequest(
            seeds=SeedInput(
                item_ids=_parse_csv(seeds_raw),
                favorite_ids=_parse_csv(favs_raw),
                tags=_parse_csv(tags_raw),
            ),
            n=n,
            novelty=novelty,
            content_weight=cw,
            diversity=diversity,
        )

        resp = svc.recommend(req)
        _print_response_pretty(dataclasses.asdict(resp))

        again = input("\nAnother request? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            print("Bye.")
            return


# ---------------------------------------------------------------------------
# Canned examples
# ---------------------------------------------------------------------------

def run_examples(svc: RecommendationService) -> None:
    """Three canned scenarios that exercise the main branches.

    1. Seed-only: the classic "I like these three songs; what else?"
    2. Tags-only: cold-ish start driven by free-form tag entry.
    3. Cold-start: no input at all; should fall back to popularity.
    """
    # Pick three real catalogue ids for the seed example.
    any_three = list(svc._item_ids[:3])   # type: ignore[attr-defined]

    scenarios = [
        (
            "Scenario 1 -- Seed-only (ALS-heavy)",
            RecommendationRequest(
                seeds=SeedInput(item_ids=any_three),
                n=5,
                novelty=0.0,
                content_weight=0.1,
                diversity=0.0,
            ),
        ),
        (
            "Scenario 2 -- Tags-only (tag fold-in)",
            RecommendationRequest(
                seeds=SeedInput(tags=["indie", "mellow", "80s"]),
                n=5,
                novelty=0.5,
                content_weight=0.4,
                diversity=0.2,
            ),
        ),
        (
            "Scenario 3 -- Cold-start (no input at all)",
            RecommendationRequest(
                seeds=SeedInput(),
                n=5,
            ),
        ),
        (
            "Scenario 4 -- Novelty pushed to max (long-tail)",
            RecommendationRequest(
                seeds=SeedInput(item_ids=any_three[:2]),
                n=5,
                novelty=1.0,
                content_weight=0.2,
            ),
        ),
        (
            "Scenario 5 -- Diversity on (MMR rerank)",
            RecommendationRequest(
                seeds=SeedInput(item_ids=any_three[:2], tags=["indie"]),
                n=5,
                novelty=0.2,
                content_weight=0.3,
                diversity=0.8,
            ),
        ),
    ]

    for title, req in scenarios:
        print()
        print("#" * 78)
        print("# " + title)
        print("#" * 78)
        resp = svc.recommend(req)
        _print_response_pretty(dataclasses.asdict(resp))


# ---------------------------------------------------------------------------
# JSON request/response mode
# ---------------------------------------------------------------------------

def run_request_file(svc: RecommendationService, path: Path, output: Path | None) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    req = _request_from_dict(data)
    resp = svc.recommend(req)
    out_blob = json.dumps(dataclasses.asdict(resp), indent=2, ensure_ascii=False, default=float)
    if output is None:
        print(out_blob)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(out_blob, encoding="utf-8")
        print(f"Wrote response to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive", action="store_true",
        help="Prompt for seeds / tags / controls (default if no other mode).",
    )
    mode.add_argument(
        "--example", action="store_true",
        help="Run a suite of canned scenarios and exit.",
    )
    mode.add_argument(
        "--request", type=Path, default=None,
        help="Path to a JSON RecommendationRequest; writes JSON response to stdout.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="When used with --request, write the JSON response here instead of stdout.",
    )
    parser.add_argument(
        "--netease", action="store_true",
        help=(
            "Attach the optional NetEase Cloud Music metadata enricher "
            "(http://localhost:3000 by default; see docs/netease_api_setup.md). "
            "Falls back to internal-features metadata if the API is unreachable "
            "or no high-confidence match is found -- the recommender pipeline "
            "itself is unaffected."
        ),
    )
    parser.add_argument(
        "--netease-base-url", default=None,
        help=(
            "Override NETEASE_API_BASE_URL for this run "
            f"(default: {config.NETEASE_API_BASE_URL})."
        ),
    )
    args = parser.parse_args()

    enricher = None
    if args.netease:
        log.info("Attaching NetEase metadata enricher (base_url=%s)",
                 args.netease_base_url or config.NETEASE_API_BASE_URL)
        # Build the NetEase enricher over the same item_features the
        # internal enricher uses, with InternalFeaturesEnricher as its
        # graceful fallback.  We load artefacts twice (once here, once
        # inside from_artifacts) but only the cheap pandas read; this
        # keeps the enricher wiring localised to this CLI flag.
        arts = load_processed_artifacts()
        enricher = NeteaseMetadataEnricher(
            item_features=arts.item_features,
            base_url=args.netease_base_url,
            fallback=InternalFeaturesEnricher(arts.item_features),
        )

    log.info("Loading RecommendationService ...")
    svc = RecommendationService.from_artifacts(enricher=enricher)
    log.info("Service ready  (catalogue=%d items)", svc.catalogue_size())

    if args.example:
        run_examples(svc)
        return 0
    if args.request is not None:
        run_request_file(svc, args.request, args.output)
        return 0
    run_interactive(svc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
