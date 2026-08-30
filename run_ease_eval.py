"""Tune EASE on an inner training holdout, then evaluate once on validation."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time

import pandas as pd
from sklearn.preprocessing import normalize

import config
from run_retrieve_rank_experiments import inner_per_user_split
from src.data.artifacts import load_processed_artifacts
from src.evaluation.evaluator import Evaluator
from src.recommenders.ease import EASERecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ease_eval")


def _evaluator(arts, train, val) -> Evaluator:
    return Evaluator(
        val_df=val,
        test_df=arts.test,
        train_df=train,
        item_features_df=arts.item_features,
        item_vectors=normalize(arts.tfidf_matrix, norm="l2").tocsr(),
        item_index=list(arts.tfidf_item_index),
        k_values=config.EVAL_K_VALUES,
    )


def main(lambdas: list[float]) -> pd.DataFrame:
    started = time.perf_counter()
    arts = load_processed_artifacts()
    base_train, inner_holdout = inner_per_user_split(arts.train)
    inner_eval = _evaluator(arts, base_train, inner_holdout)
    tuning_rows = []

    for value in lambdas:
        model = EASERecommender(value).fit(base_train, arts.item_features)
        row = inner_eval.evaluate(model, split="val")
        row["regularization"] = value
        tuning_rows.append(row)
        log.info(
            "EASE lambda=%g inner NDCG@10=%.6f NDCG@20=%.6f",
            value, row["ndcg@10"], row["ndcg@20"],
        )
        del model
        gc.collect()

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["ndcg@10", "ndcg@20"], ascending=False
    ).reset_index(drop=True)
    selected_lambda = float(tuning.iloc[0]["regularization"])
    log.info("Selected EASE lambda=%g", selected_lambda)

    model = EASERecommender(selected_lambda).fit(arts.train, arts.item_features)
    row = _evaluator(arts, arts.train, arts.val).evaluate(model, split="val")
    row["regularization"] = selected_lambda
    results = pd.DataFrame([row])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(config.RESULTS_DIR / "ease_inner_sweep.csv", index=False)
    results.to_csv(config.RESULTS_DIR / "ease_val.csv", index=False)
    with open(config.RESULTS_DIR / "ease_val.json", "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, default=float)

    print("\n=== EASE validation result ===")
    print(results[[
        "model", "precision@10", "recall@10", "ndcg@10",
        "precision@20", "recall@20", "ndcg@20",
        "coverage@20", "novelty@20",
    ]].to_string(index=False))
    log.info("Completed in %.2fs", time.perf_counter() - started)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambdas", default="100,300,500,1000",
        help="Comma-separated EASE L2 regularization values.",
    )
    args = parser.parse_args()
    main([float(value) for value in args.lambdas.split(",")])
