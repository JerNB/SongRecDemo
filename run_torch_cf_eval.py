"""Inner-tune and validation-evaluate BPR-MF and LightGCN."""

from __future__ import annotations

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
from src.recommenders.torch_cf import BPRMFRecommender, LightGCNRecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("torch_cf_eval")


def evaluator(arts, train, val) -> Evaluator:
    return Evaluator(
        val_df=val,
        test_df=arts.test,
        train_df=train,
        item_features_df=arts.item_features,
        item_vectors=normalize(arts.tfidf_matrix, norm="l2").tocsr(),
        item_index=list(arts.tfidf_item_index),
        k_values=config.EVAL_K_VALUES,
    )


def model_factory(kind: str, epochs: int):
    common = dict(
        factors=64,
        epochs=epochs,
        learning_rate=0.01,
        regularization=1e-4,
        random_state=42,
    )
    if kind == "bpr":
        return BPRMFRecommender(batch_size=8192, **common)
    if kind == "lightgcn":
        return LightGCNRecommender(layers=3, **common)
    raise ValueError(kind)


def main() -> pd.DataFrame:
    started = time.perf_counter()
    arts = load_processed_artifacts()
    base_train, inner_holdout = inner_per_user_split(arts.train)
    inner_eval = evaluator(arts, base_train, inner_holdout)
    tuning_rows = []

    epoch_grid = {"bpr": (10, 30), "lightgcn": (30, 100)}
    for kind in ("bpr", "lightgcn"):
        for epochs in epoch_grid[kind]:
            model = model_factory(kind, epochs).fit(base_train, arts.item_features)
            row = inner_eval.evaluate(model, split="val")
            row.update({"kind": kind, "epochs": epochs})
            tuning_rows.append(row)
            log.info(
                "%s epochs=%d inner NDCG@10=%.6f NDCG@20=%.6f",
                kind, epochs, row["ndcg@10"], row["ndcg@20"],
            )
            del model
            gc.collect()

    tuning = pd.DataFrame(tuning_rows)
    selected = (
        tuning.sort_values(["ndcg@10", "ndcg@20"], ascending=False)
        .groupby("kind", as_index=False)
        .first()
    )
    outer_eval = evaluator(arts, arts.train, arts.val)
    rows = []
    for record in selected.to_dict("records"):
        kind = str(record["kind"])
        epochs = int(record["epochs"])
        log.info("Refit selected %s epochs=%d on full train", kind, epochs)
        model = model_factory(kind, epochs).fit(arts.train, arts.item_features)
        row = outer_eval.evaluate(model, split="val")
        row.update({"kind": kind, "epochs": epochs})
        rows.append(row)
        del model
        gc.collect()

    results = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(config.RESULTS_DIR / "torch_cf_inner_sweep.csv", index=False)
    results.to_csv(config.RESULTS_DIR / "torch_cf_val.csv", index=False)
    with open(config.RESULTS_DIR / "torch_cf_val.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=float)

    print("\n=== BPR-MF / LightGCN validation results ===")
    print(results[[
        "model", "precision@10", "recall@10", "ndcg@10",
        "precision@20", "recall@20", "ndcg@20",
        "coverage@20", "novelty@20",
    ]].to_string(index=False))
    log.info("Completed in %.2fs", time.perf_counter() - started)
    return results


if __name__ == "__main__":
    main()
