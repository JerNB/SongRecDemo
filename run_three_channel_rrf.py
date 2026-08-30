"""Tune and evaluate EASE + ALS + TF-IDF retrieval with weighted RRF."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import config
from run_retrieve_rank_experiments import (
    build_retrievers,
    exact_ranks,
    inner_per_user_split,
)
from src.data.artifacts import ProcessedArtifacts, load_processed_artifacts
from src.evaluation.metrics import compute_all_metrics
from src.recommenders.ease import EASERecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("three_channel_rrf")


@dataclass
class FusionDataset:
    users: list[str]
    groups: np.ndarray
    items: np.ndarray
    rank_distances: np.ndarray  # columns: EASE, ALS, TF-IDF
    labels: np.ndarray
    relevant_totals: np.ndarray


def build_fusion_dataset(
    hybrid,
    ease: EASERecommender,
    relevance: dict[str, set[str]],
    ease_k: int = 500,
    als_k: int = 500,
    tfidf_k: int = 100,
) -> tuple[FusionDataset, dict]:
    ease_cols = np.asarray(
        [ease._item_id_to_col[iid] for iid in hybrid._item_index], dtype=np.int32
    )
    groups: list[int] = []
    item_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    users: list[str] = []
    relevant_totals: list[int] = []
    source_recalls = {"ease": [], "als": [], "tfidf": [], "union": []}
    union_sizes: list[int] = []

    for user_no, uid in enumerate(hybrid._user_ids, start=1):
        relevant = relevance.get(uid, set())
        if not relevant:
            continue
        als_scores, tfidf_scores = hybrid.component_scores(uid)
        ease_row = ease._user_id_to_row[uid]
        ease_scores = ease._scores[ease_row, ease_cols].copy()
        seen = hybrid._seen_positions[hybrid._user_pos[uid]]
        ease_scores[seen] = -np.inf

        ease_ranks, ease_order = exact_ranks(ease_scores)
        als_ranks, als_order = exact_ranks(als_scores)
        tfidf_ranks, tfidf_order = exact_ranks(tfidf_scores)
        candidates = np.union1d(
            np.union1d(ease_order[:ease_k], als_order[:als_k]),
            tfidf_order[:tfidf_k],
        ).astype(np.int32, copy=False)
        n_unseen = max(len(ease_order), 2)
        distances = np.column_stack([
            (ease_ranks[candidates] - 1) / (n_unseen - 1),
            (als_ranks[candidates] - 1) / (n_unseen - 1),
            (tfidf_ranks[candidates] - 1) / (n_unseen - 1),
        ]).astype(np.float32)
        labels = np.fromiter(
            (hybrid._item_index[int(pos)] in relevant for pos in candidates),
            dtype=np.int8,
            count=len(candidates),
        )

        def recall(order: np.ndarray, k: int) -> float:
            ids = {hybrid._item_index[int(pos)] for pos in order[:k]}
            return len(ids & relevant) / len(relevant)

        source_recalls["ease"].append(recall(ease_order, ease_k))
        source_recalls["als"].append(recall(als_order, als_k))
        source_recalls["tfidf"].append(recall(tfidf_order, tfidf_k))
        source_recalls["union"].append(float(labels.sum()) / len(relevant))
        union_sizes.append(len(candidates))
        groups.append(len(candidates))
        item_chunks.append(candidates)
        distance_chunks.append(distances)
        label_chunks.append(labels)
        users.append(uid)
        relevant_totals.append(len(relevant))
        if user_no % 1000 == 0:
            log.info("  fusion candidates %d/%d users", user_no, len(hybrid._user_ids))

    analysis = {
        "ease_recall": float(np.mean(source_recalls["ease"])),
        "als_recall": float(np.mean(source_recalls["als"])),
        "tfidf_recall": float(np.mean(source_recalls["tfidf"])),
        "union_recall": float(np.mean(source_recalls["union"])),
        "mean_union_size": float(np.mean(union_sizes)),
        "median_union_size": float(np.median(union_sizes)),
        "ease_k": ease_k,
        "als_k": als_k,
        "tfidf_k": tfidf_k,
    }
    return FusionDataset(
        users=users,
        groups=np.asarray(groups, dtype=np.int32),
        items=np.concatenate(item_chunks),
        rank_distances=np.vstack(distance_chunks),
        labels=np.concatenate(label_chunks),
        relevant_totals=np.asarray(relevant_totals, dtype=np.int32),
    ), analysis


def fusion_scores(dataset: FusionDataset, params: dict) -> np.ndarray:
    weights = np.asarray(
        [params["ease_weight"], params["als_weight"], params["tfidf_weight"]],
        dtype=np.float32,
    )
    return np.sum(
        weights[np.newaxis, :] / (float(params["constant"]) + dataset.rank_distances),
        axis=1,
    )


def mean_ndcg(scores: np.ndarray, dataset: FusionDataset, k: int) -> float:
    values = []
    offset = 0
    for group, n_relevant in zip(dataset.groups, dataset.relevant_totals):
        end = offset + int(group)
        order = np.argsort(-scores[offset:end], kind="stable")[:k]
        labels = dataset.labels[offset:end]
        dcg = sum(
            float(labels[pos]) / math.log2(rank + 1)
            for rank, pos in enumerate(order, start=1)
        )
        ideal = min(int(n_relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
        values.append(dcg / idcg if idcg else 0.0)
        offset = end
    return float(np.mean(values))


def tune(dataset: FusionDataset) -> tuple[dict, pd.DataFrame]:
    rows = []
    # EASE is already much stronger, so search EASE-dominant mixtures plus
    # the pure-EASE endpoint. Remaining mass is split between ALS/content.
    for constant in (0.005, 0.01, 0.025, 0.05):
        for ease_weight in (0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
            remainder = 1.0 - ease_weight
            for als_share in (0.0, 0.50, 0.75, 1.0):
                als_weight = remainder * als_share
                tfidf_weight = remainder - als_weight
                params = {
                    "constant": constant,
                    "ease_weight": ease_weight,
                    "als_weight": als_weight,
                    "tfidf_weight": tfidf_weight,
                }
                scores = fusion_scores(dataset, params)
                rows.append({
                    **params,
                    "ndcg@10": mean_ndcg(scores, dataset, 10),
                    "ndcg@20": mean_ndcg(scores, dataset, 20),
                })
    sweep = pd.DataFrame(rows).drop_duplicates(
        ["constant", "ease_weight", "als_weight", "tfidf_weight"]
    ).sort_values(["ndcg@10", "ndcg@20"], ascending=False).reset_index(drop=True)
    selected = sweep.iloc[0].to_dict()
    params = {
        key: float(selected[key])
        for key in ("constant", "ease_weight", "als_weight", "tfidf_weight")
    }
    params["inner_ndcg@10"] = float(selected["ndcg@10"])
    params["inner_ndcg@20"] = float(selected["ndcg@20"])
    return params, sweep


def recommendations(
    dataset: FusionDataset,
    scores: np.ndarray,
    item_index: list[str],
    n: int = 20,
) -> dict[str, list[str]]:
    output = {}
    offset = 0
    for uid, group in zip(dataset.users, dataset.groups):
        end = offset + int(group)
        local_items = dataset.items[offset:end]
        local_scores = scores[offset:end]
        order = np.lexsort((local_items, -local_scores.astype(np.float64)))[:n]
        output[uid] = [item_index[int(local_items[pos])] for pos in order]
        offset = end
    return output


def evaluate(name: str, recs: dict[str, list[str]], arts: ProcessedArtifacts) -> dict:
    ground_truth = arts.val.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()
    metrics = compute_all_metrics(
        recommendations=recs,
        ground_truth=ground_truth,
        catalogue=set(arts.train["item_id_raw"].unique()),
        k_values=config.EVAL_K_VALUES,
        item_popularity=arts.train["item_id_raw"].value_counts().to_dict(),
        item_vectors=normalize(arts.tfidf_matrix, norm="l2").tocsr(),
        item_index=list(arts.tfidf_item_index),
        n_train_users=arts.train["user_id_raw"].nunique(),
    )
    return {"model": name, "split": "val", **metrics}


def main() -> pd.DataFrame:
    started = time.perf_counter()
    arts = load_processed_artifacts()
    base_train, inner_holdout = inner_per_user_split(arts.train)
    inner_relevance = inner_holdout.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()

    log.info("Fit inner ALS/TF-IDF and EASE")
    inner_hybrid = build_retrievers(base_train, arts)
    inner_ease = EASERecommender(100).fit(base_train, arts.item_features)
    inner_data, inner_analysis = build_fusion_dataset(
        inner_hybrid, inner_ease, inner_relevance
    )
    params, sweep = tune(inner_data)
    log.info("Selected three-channel RRF: %s", params)
    del inner_hybrid, inner_ease, inner_data

    log.info("Refit all retrieval channels on full train")
    full_hybrid = build_retrievers(arts.train, arts)
    full_ease = EASERecommender(100).fit(arts.train, arts.item_features)
    outer_relevance = arts.val.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()
    outer_data, outer_analysis = build_fusion_dataset(
        full_hybrid, full_ease, outer_relevance
    )
    scores = fusion_scores(outer_data, params)
    recs = recommendations(outer_data, scores, full_hybrid._item_index)
    row = evaluate("ThreeChannel-RRF(EASE+ALS+TFIDF)", recs, arts)
    row.update({key: float(value) for key, value in params.items()})
    results = pd.DataFrame([row])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(config.RESULTS_DIR / "three_channel_rrf_inner_sweep.csv", index=False)
    results.to_csv(config.RESULTS_DIR / "three_channel_rrf_val.csv", index=False)
    with open(config.RESULTS_DIR / "three_channel_rrf_val.json", "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, default=float)
    with open(config.RESULTS_DIR / "three_channel_candidate_analysis.json", "w", encoding="utf-8") as handle:
        json.dump({"inner": inner_analysis, "outer_val": outer_analysis}, handle, indent=2)

    print("\n=== Three-channel RRF validation result ===")
    print(results[[
        "model", "precision@10", "recall@10", "ndcg@10",
        "precision@20", "recall@20", "ndcg@20",
        "coverage@20", "novelty@20",
    ]].to_string(index=False))
    print("\nSelected parameters:", json.dumps(params, indent=2))
    print("Candidate analysis:", json.dumps({"inner": inner_analysis, "outer_val": outer_analysis}, indent=2))
    log.info("Completed in %.2fs", time.perf_counter() - started)
    return results


if __name__ == "__main__":
    main()
