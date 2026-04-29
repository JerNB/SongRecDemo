"""
Train the ALS backbone and persist it for the personalized recommender.

This is a one-shot script: run it once after preprocessing, and the
result (item factors + hyperparameters) is cached to
``config.ALS_STATE_FILE``. The future website / CLI demo loads this
file at startup via :meth:`RecommendationService.from_artifacts`
instead of retraining on every boot.

The training is bit-for-bit identical to the one used in the main
evaluation (same data, same hyperparameters, same RNG seed). The
only difference is what we save: the personalized recommender needs
only the *item* side, because an interactive user will get their
own latent vector via fold-in at request time.

Usage
-----
::

    python run_train_personalized.py                 # trains + saves
    python run_train_personalized.py --force         # retrain even if cached
    python run_train_personalized.py --smoke-test    # run a quick recommendation after saving
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import config
from src.data.artifacts import load_processed_artifacts
from src.personalization import (
    RecommendationRequest,
    RecommendationService,
    SeedInput,
)
from src.recommenders.collaborative import CollaborativeFilteringRecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_personalized")


def train_and_save(force: bool = False) -> None:
    if config.ALS_STATE_FILE.exists() and not force:
        log.info(
            "ALS state already exists at %s. Use --force to retrain.",
            config.ALS_STATE_FILE,
        )
        return

    arts = load_processed_artifacts()
    log.info(
        "Loaded artefacts: train=%d interactions, %d users, %d items",
        len(arts.train),
        arts.train["user_id_raw"].nunique(),
        arts.train["item_id_raw"].nunique(),
    )

    model = CollaborativeFilteringRecommender(
        factors=config.CF_FACTORS,
        regularization=config.CF_REGULARIZATION,
        iterations=config.CF_ITERATIONS,
        alpha=config.CF_ALPHA,
        random_state=config.SPLIT_SEED,
    )
    t0 = time.perf_counter()
    log.info("Fitting ALS (%s) ...", model.name)
    model.fit(arts.train, arts.item_features)
    log.info("Training finished in %.1fs", time.perf_counter() - t0)

    model.save_state(config.ALS_STATE_FILE)


def smoke_test() -> None:
    """Load the service and fire one representative request."""
    log.info("Running smoke test via RecommendationService ...")
    svc = RecommendationService.from_artifacts()

    # Pick an item from the catalogue and use it as a seed.
    any_item = next(iter(svc._item_ids))   # type: ignore[attr-defined]
    req = RecommendationRequest(
        seeds=SeedInput(
            item_ids=[any_item],
            tags=["indie", "mellow"],
        ),
        n=5,
        novelty=0.3,
        content_weight=0.25,
        diversity=0.0,
    )
    resp = svc.recommend(req)
    print()
    print("=" * 78)
    print(f"Request id: {resp.request_id}  | fallback: {resp.fallback_used}")
    print(f"Accepted seeds: {resp.seed_summary['accepted_item_ids']}")
    print(f"Matched tags  : {resp.seed_summary['matched_tags']}")
    print(f"Unknown tags  : {resp.seed_summary['unknown_tags']}")
    print("-" * 78)
    for item in resp.items:
        print(f" #{item.rank}  id={item.item_id}   score={item.score:+.4f}")
        print(f"      why: {item.explanation}")
        print(f"      reasons: {item.reasons}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if the state file exists.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one recommendation request after saving.")
    args = parser.parse_args()

    train_and_save(force=args.force)

    if args.smoke_test:
        smoke_test()

    return 0


if __name__ == "__main__":
    sys.exit(main())
