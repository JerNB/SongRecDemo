"""
Shared evaluation harness.

Every model is evaluated through this single class so that:
- All models see identical ground-truth splits.
- The same K values, catalogue, and beyond-accuracy inputs are used for all.
- Results land in a comparable DataFrame with a model-name column.
- No model-specific code lives here.

Usage
-----
    evaluator = Evaluator(val_df, item_features_df, training_df)
    results_df = evaluator.run_all(models, split="val")

Or for a single model:
    row = evaluator.evaluate(model, split="val")
"""

from __future__ import annotations

import logging
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config
from src.evaluation.metrics import compute_all_metrics
from src.recommenders.base import BaseRecommender

log = logging.getLogger(__name__)

Split = Literal["val", "test"]


class Evaluator:
    """Runs all models through the same offline evaluation protocol.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation split (output of preprocessor).
    test_df : pd.DataFrame
        Test split.
    train_df : pd.DataFrame
        Training split; used to:
        - determine the evaluation user set
        - compute popularity scores (for Novelty metric)
        - identify the training catalogue (for Coverage metric)
    item_features_df : pd.DataFrame
        Item content features (index = item_id_raw).
    item_vectors : sp.csr_matrix, optional
        L2-normalised content vectors for Intra-List Diversity.
        If None, diversity metric is skipped.
    item_index : list[str], optional
        item_id_raw ordering matching rows of item_vectors.
    k_values : list[int]
        Cutoff values for @K metrics.
    """

    def __init__(
        self,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        item_vectors: Optional[sp.csr_matrix] = None,
        item_index: Optional[list[str]] = None,
        k_values: Optional[list[int]] = None,
    ) -> None:
        self.val_df = val_df
        self.test_df = test_df
        self.train_df = train_df
        self.item_features_df = item_features_df
        self.item_vectors = item_vectors
        self.item_index = item_index
        self.k_values = k_values or config.EVAL_K_VALUES

        # Precompute ground-truth sets
        self._val_ground_truth = self._build_ground_truth(val_df)
        self._test_ground_truth = self._build_ground_truth(test_df)

        # Training catalogue = all items that appear in training
        self._catalogue: set[str] = set(train_df["item_id_raw"].unique())

        # Item popularity from training (used for Novelty)
        self._item_popularity: dict[str, int] = (
            train_df["item_id_raw"].value_counts().to_dict()
        )

        # Evaluation users = users present in training with ≥1 ground-truth item
        train_users = set(train_df["user_id_raw"].unique())
        self._val_users = sorted(
            train_users & set(self._val_ground_truth.keys())
        )
        self._test_users = sorted(
            train_users & set(self._test_ground_truth.keys())
        )

        log.info(
            "Evaluator ready: val_users=%d | test_users=%d | catalogue=%d",
            len(self._val_users),
            len(self._test_users),
            len(self._catalogue),
        )

    @staticmethod
    def _build_ground_truth(df: pd.DataFrame) -> dict[str, set[str]]:
        """Group held-out interactions into per-user relevance sets."""
        gt: dict[str, set[str]] = {}
        for uid, group in df.groupby("user_id_raw"):
            gt[uid] = set(group["item_id_raw"].tolist())
        return gt

    # ------------------------------------------------------------------
    # Core evaluate method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: BaseRecommender,
        split: Split = "val",
    ) -> dict[str, object]:
        """Evaluate one model on the specified split.

        Returns
        -------
        dict with keys: "model", "split", and one key per metric (e.g.
        "precision@10", "recall@10", ...).
        """
        ground_truth, users = (
            (self._val_ground_truth, self._val_users)
            if split == "val"
            else (self._test_ground_truth, self._test_users)
        )

        # Candidate generation protocol:
        # - Request n=max(K_values) items from the model.
        # - The model internally scores all catalogue items, removes the
        #   user's training interactions (exclude_seen=True), and returns
        #   the top-n from the remainder.
        # - Ties broken by item_id ascending (see each model's recommend()).
        # - At |catalogue|=8640, scoring is exhaustive (no ANN pre-filter).
        # - Users that raise (e.g. ALS KeyError for unseen users) get an
        #   empty list; the evaluator counts and logs them as failures.
        k_max = max(self.k_values)
        recommendations: dict[str, list[str]] = {}

        t0 = time.perf_counter()
        n_failed = 0
        for uid in users:
            try:
                recs = model.recommend(uid, n=k_max, exclude_seen=True)
                recommendations[uid] = [item for item, _ in recs]
            except Exception as exc:
                log.debug("recommend() failed for user %s: %s", uid, exc)
                n_failed += 1
                recommendations[uid] = []

        elapsed = time.perf_counter() - t0
        if n_failed:
            log.warning("%d users failed recommend(); got empty lists.", n_failed)
        log.info(
            "Recommendations generated for %d users in %.2fs (model=%s, split=%s)",
            len(users), elapsed, model.name, split,
        )

        metrics = compute_all_metrics(
            recommendations=recommendations,
            ground_truth=ground_truth,
            catalogue=self._catalogue,
            k_values=self.k_values,
            item_popularity=self._item_popularity,
            item_vectors=self.item_vectors,
            item_index=self.item_index,
            n_train_users=len(set(self.train_df["user_id_raw"])),
        )

        row: dict[str, object] = {"model": model.name, "split": split}
        row.update(metrics)
        return row

    # ------------------------------------------------------------------
    # Batch: evaluate all models
    # ------------------------------------------------------------------

    def run_all(
        self,
        models: list[BaseRecommender],
        split: Split = "val",
    ) -> pd.DataFrame:
        """Evaluate every model and return a results DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per model.  Columns: "model", "split", and one column
            per metric × K combination.  Sort key is config.PRIMARY_K.
        """
        rows = []
        for model in models:
            log.info("Evaluating model: %s on split=%s", model.name, split)
            row = self.evaluate(model, split=split)
            rows.append(row)
            # Log primary K metrics immediately so progress is visible
            pk = config.PRIMARY_K
            log.info(
                "  precision@%d=%.4f  recall@%d=%.4f  ndcg@%d=%.4f  coverage@%d=%.4f",
                pk, row.get(f"precision@{pk}", float("nan")),
                pk, row.get(f"recall@{pk}", float("nan")),
                pk, row.get(f"ndcg@{pk}", float("nan")),
                pk, row.get(f"coverage@{pk}", float("nan")),
            )

        results_df = pd.DataFrame(rows)
        return results_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, results_df: pd.DataFrame, filename: str = "results.csv") -> None:
        """Write results DataFrame to RESULTS_DIR."""
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = config.RESULTS_DIR / filename
        results_df.to_csv(out, index=False)
        log.info("Results saved to %s", out)

    # ------------------------------------------------------------------
    # Sanity report
    # ------------------------------------------------------------------

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Pretty-print a comparison table for the primary K."""
        pk = config.PRIMARY_K
        cols = ["model"] + [
            c for c in results_df.columns
            if str(pk) in c and c != "split"
        ]
        available = [c for c in cols if c in results_df.columns]
        print(f"\n=== Offline Evaluation Summary (K={pk}) ===")
        print(results_df[available].to_string(index=False))
        print()
