"""
End-to-end pipeline for the KGRec-music recommendation experiment.

Run this file to execute the full experiment in sequence:
  1. Preprocessing  (can be skipped if artefacts already exist)
  2. Model training
  3. Validation evaluation (used for hyperparameter reporting)
  4. Test evaluation  (run ONCE at the very end; do not tune on test)
  5. Results saved to artifacts/results/

Usage
-----
    python pipeline.py                  # full run
    python pipeline.py --skip-preproc   # use existing split artefacts
    python pipeline.py --split val      # evaluate on validation only

Design notes
------------
- The pipeline loads split artefacts written by the preprocessor.
  If they do not exist, preprocessing is run automatically.
- All three models are fitted on the SAME training split.
- The evaluator receives the SAME val/test ground truth for all models.
- Test evaluation is gated by a flag so it cannot be run accidentally
  during hyperparameter tuning (use --split val for that).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys

import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import config
from src.data.preprocessor import run_preprocessing
from src.evaluation.evaluator import Evaluator
from src.recommenders.collaborative import CollaborativeFilteringRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_artefacts(skip_preproc: bool) -> None:
    """Run preprocessing if split files are missing or skip_preproc is False."""
    files_exist = all(
        p.exists()
        for p in [
            config.TRAIN_FILE, config.VAL_FILE,
            config.TEST_FILE, config.ITEM_FEATURES_FILE, config.ID_MAPS_FILE,
        ]
    )
    if not files_exist or not skip_preproc:
        if not files_exist:
            log.info("Split artefacts not found – running preprocessing.")
        else:
            log.info("--skip-preproc not set – re-running preprocessing.")
        run_preprocessing()
    else:
        log.info("Using existing split artefacts in %s", config.SPLITS_DIR)


def _load_artefacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test interactions and item features from disk."""
    train = pd.read_parquet(config.TRAIN_FILE)
    val = pd.read_parquet(config.VAL_FILE)
    test = pd.read_parquet(config.TEST_FILE)
    item_features = pd.read_parquet(config.ITEM_FEATURES_FILE)
    return train, val, test, item_features


def _build_content_vectors(item_features: pd.DataFrame):
    """Build the item feature matrix used for diversity metric and CB model.

    This re-uses the ContentBasedRecommender's internal vectoriser but
    exposes the matrix for the Evaluator's diversity computation.

    Returns (item_vectors: csr_matrix, item_index: list[str])
    """
    cb_for_vectors = ContentBasedRecommender(feature_mode=config.CB_FEATURE_MODE)

    # Fake a minimal train_df just to satisfy the fit() signature
    # (we only need the vectoriser to produce item vectors; user seen is not used here)
    dummy_train = pd.DataFrame({
        "user_id_raw": ["0"],
        "item_id_raw": [item_features.index[0]],
        "user_idx": [0],
        "item_idx": [0],
    })
    cb_for_vectors.fit(dummy_train, item_features)
    return cb_for_vectors._item_vectors, cb_for_vectors._item_index


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(skip_preproc: bool = True, eval_split: str = "val") -> pd.DataFrame:
    # ------------------------------------------------------------------
    # Stage 1: Preprocessing
    # ------------------------------------------------------------------
    _ensure_artefacts(skip_preproc)
    train, val, test, item_features = _load_artefacts()

    log.info(
        "Artefacts loaded → train=%d | val=%d | test=%d | items=%d",
        len(train), len(val), len(test), len(item_features),
    )

    # ------------------------------------------------------------------
    # Stage 2: Build content vectors (needed for diversity metric)
    # ------------------------------------------------------------------
    log.info("Building item content vectors for diversity evaluation …")
    item_vectors, item_index = _build_content_vectors(item_features)

    # ------------------------------------------------------------------
    # Stage 3: Instantiate models
    # ------------------------------------------------------------------
    models = [
        PopularityRecommender(score_mode=config.POPULARITY_SCORE),
        CollaborativeFilteringRecommender(
            factors=config.CF_FACTORS,
            regularization=config.CF_REGULARIZATION,
            iterations=config.CF_ITERATIONS,
            alpha=config.CF_ALPHA,
        ),
        ContentBasedRecommender(
            feature_mode=config.CB_FEATURE_MODE,
            top_k_candidates=config.CB_TOP_K_CANDIDATES,
        ),
    ]

    # ------------------------------------------------------------------
    # Stage 4: Fit all models on training data
    # ------------------------------------------------------------------
    for model in models:
        log.info("Fitting %s …", model.name)
        model.fit(train, item_features)

    # ------------------------------------------------------------------
    # Stage 5: Evaluate
    # ------------------------------------------------------------------
    evaluator = Evaluator(
        val_df=val,
        test_df=test,
        train_df=train,
        item_features_df=item_features,
        item_vectors=item_vectors,
        item_index=item_index,
        k_values=config.EVAL_K_VALUES,
    )

    results = evaluator.run_all(models, split=eval_split)
    evaluator.print_summary(results)
    evaluator.save_results(results, filename=f"results_{eval_split}.csv")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KGRec-music recommendation pipeline"
    )
    parser.add_argument(
        "--skip-preproc",
        action="store_true",
        default=False,
        help="Use existing split artefacts instead of re-running preprocessing.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help=(
            "Which held-out split to evaluate on. "
            "Use 'val' during development; run 'test' only once at the end."
        ),
    )
    args = parser.parse_args()

    if args.split == "test":
        confirm = input(
            "\n[WARNING] You are about to evaluate on the TEST split.\n"
            "This should only be done ONCE after all hyperparameters are fixed.\n"
            "Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    main(skip_preproc=args.skip_preproc, eval_split=args.split)
