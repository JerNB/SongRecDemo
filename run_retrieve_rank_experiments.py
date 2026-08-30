"""Leakage-safe multi-retrieval and learning-to-rank experiments for KGRec.

Pipeline
--------
1. Split the existing training interactions per user into base-train and an
   inner ranker holdout.  Original validation/test remain untouched.
2. Fit ALS and tag-TF-IDF retrievers on base-train.
3. Measure candidate recall and the oracle ranking ceiling.
4. Train a scalable logistic ranker and XGBoost LambdaMART on candidates.
5. Refit the retrievers on full train and evaluate frozen rankers on the
   original validation split through the project's shared metric functions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from xgboost import XGBRanker

import config
from src.data.artifacts import ProcessedArtifacts, load_processed_artifacts
from src.evaluation.metrics import compute_all_metrics
from src.recommenders.collaborative import CollaborativeFilteringRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrieve_rank")

FEATURE_NAMES = [
    "als_score",
    "tfidf_score",
    "als_rank_score",
    "tfidf_rank_score",
    "score_min",
    "score_max",
    "score_product",
    "score_abs_gap",
    "retrieved_by_als",
    "retrieved_by_tfidf",
    "retrieved_by_both",
    "als_top20",
    "tfidf_top20",
    "log_item_popularity",
    "item_has_tags",
    "log_item_tag_count",
    "log_user_history_size",
    "user_history_tag_coverage",
    "tfidf_missing",
    "tfidf_high_als_bottom_half",
    "als_high_tfidf_bottom_half",
]


@dataclass
class CandidateDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    users: list[str]
    item_positions: np.ndarray
    relevant_totals: np.ndarray


def inner_per_user_split(
    train_df: pd.DataFrame,
    holdout_fraction: float = 0.15,
    random_state: int = 31415,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split every user's original training interactions."""
    rng = np.random.default_rng(random_state)
    base_indices: list[int] = []
    holdout_indices: list[int] = []
    for _, group in train_df.groupby("user_id_raw", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n_holdout = max(1, int(round(len(indices) * holdout_fraction)))
        n_holdout = min(n_holdout, len(indices) - 1)
        holdout_indices.extend(indices[:n_holdout].tolist())
        base_indices.extend(indices[n_holdout:].tolist())
    base = train_df.loc[base_indices].sort_index().reset_index(drop=True)
    holdout = train_df.loc[holdout_indices].sort_index().reset_index(drop=True)
    return base, holdout


def build_retrievers(
    train_df: pd.DataFrame,
    arts: ProcessedArtifacts,
) -> HybridRecommender:
    item_vectors = normalize(arts.tfidf_matrix, norm="l2").tocsr()
    model = HybridRecommender(
        als_model=CollaborativeFilteringRecommender(
            factors=config.CF_FACTORS,
            regularization=config.CF_REGULARIZATION,
            iterations=config.CF_ITERATIONS,
            alpha=config.CF_ALPHA,
            random_state=config.SPLIT_SEED,
        ),
        content_model=ContentBasedRecommender(
            item_vectors=item_vectors,
            item_index=list(arts.tfidf_item_index),
            feature_mode=config.CB_FEATURE_MODE,
        ),
        als_weight=0.65,
        missing_content_policy="penalize",
    )
    model.fit(train_df, arts.item_features)
    return model


def exact_ranks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact 1-based ranks and score-sorted valid item positions."""
    valid = np.flatnonzero(np.isfinite(scores))
    order_local = np.lexsort((valid, -scores[valid].astype(np.float64)))
    ordered = valid[order_local]
    ranks = np.zeros(scores.size, dtype=np.int32)
    ranks[ordered] = np.arange(1, ordered.size + 1, dtype=np.int32)
    return ranks, ordered.astype(np.int32, copy=False)


def _oracle_ndcg(n_candidate_hits: int, n_relevant: int, k: int) -> float:
    actual = min(n_candidate_hits, k)
    ideal = min(n_relevant, k)
    if ideal == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, actual + 1))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg


def candidate_analysis(
    model: HybridRecommender,
    relevance: dict[str, set[str]],
    als_cutoffs: tuple[int, ...] = (100, 300, 500),
    tfidf_cutoffs: tuple[int, ...] = (50, 100, 200),
    union_als_k: int = 500,
    union_tfidf_k: int = 100,
) -> dict:
    accum: dict[str, list[float]] = {}
    for k in als_cutoffs:
        accum[f"als_recall@{k}"] = []
    for k in tfidf_cutoffs:
        accum[f"tfidf_recall@{k}"] = []
    accum["union_recall"] = []
    for k in (10, 20):
        accum[f"union_oracle_recall@{k}"] = []
        accum[f"union_oracle_ndcg@{k}"] = []
    candidate_sizes: list[int] = []

    for user_no, uid in enumerate(model._user_ids, start=1):
        relevant = relevance.get(uid, set())
        if not relevant:
            continue
        als, tfidf = model.component_scores(uid)
        _, als_order = exact_ranks(als)
        _, tfidf_order = exact_ranks(tfidf)
        for k in als_cutoffs:
            ids = {model._item_index[int(pos)] for pos in als_order[:k]}
            accum[f"als_recall@{k}"].append(len(ids & relevant) / len(relevant))
        for k in tfidf_cutoffs:
            ids = {model._item_index[int(pos)] for pos in tfidf_order[:k]}
            accum[f"tfidf_recall@{k}"].append(len(ids & relevant) / len(relevant))

        candidates = np.union1d(
            als_order[:union_als_k], tfidf_order[:union_tfidf_k]
        )
        candidate_ids = {model._item_index[int(pos)] for pos in candidates}
        hits = len(candidate_ids & relevant)
        candidate_sizes.append(int(candidates.size))
        accum["union_recall"].append(hits / len(relevant))
        for k in (10, 20):
            accum[f"union_oracle_recall@{k}"].append(
                min(hits, k) / len(relevant)
            )
            accum[f"union_oracle_ndcg@{k}"].append(
                _oracle_ndcg(hits, len(relevant), k)
            )
        if user_no % 1000 == 0:
            log.info("  candidate analysis %d/%d users", user_no, len(model._user_ids))

    output = {key: float(np.mean(values)) for key, values in accum.items()}
    output.update({
        "users": len(accum["union_recall"]),
        "union_als_k": union_als_k,
        "union_tfidf_k": union_tfidf_k,
        "mean_union_size": float(np.mean(candidate_sizes)),
        "median_union_size": float(np.median(candidate_sizes)),
    })
    return output


class FeatureBuilder:
    """Build aligned candidate features from fitted retrieval score rows."""

    def __init__(
        self,
        model: HybridRecommender,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame,
        als_k: int = 500,
        tfidf_k: int = 100,
    ) -> None:
        self.model = model
        self.als_k = als_k
        self.tfidf_k = tfidf_k
        item_index = model._item_index
        popularity = train_df["item_id_raw"].value_counts().to_dict()
        self.item_pop = np.asarray(
            [math.log1p(popularity.get(iid, 0)) for iid in item_index],
            dtype=np.float32,
        )
        self.has_tags = np.asarray(
            [bool(item_features.loc[iid, "has_tags"]) for iid in item_index],
            dtype=np.float32,
        )
        self.tag_count = np.asarray(
            [math.log1p(len(item_features.loc[iid, "tags_normalised"])) for iid in item_index],
            dtype=np.float32,
        )
        history = train_df.groupby("user_id_raw")["item_id_raw"].apply(list).to_dict()
        self.user_history_size: dict[str, float] = {}
        self.user_tag_coverage: dict[str, float] = {}
        for uid, items in history.items():
            self.user_history_size[uid] = math.log1p(len(items))
            tagged = sum(
                bool(item_features.loc[iid, "has_tags"])
                for iid in items
                if iid in item_features.index
            )
            self.user_tag_coverage[uid] = tagged / max(len(items), 1)

    def user_candidates(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        als, tfidf = self.model.component_scores(uid)
        als_ranks, als_order = exact_ranks(als)
        tfidf_ranks, tfidf_order = exact_ranks(tfidf)
        candidates = np.union1d(
            als_order[: self.als_k], tfidf_order[: self.tfidf_k]
        ).astype(np.int32, copy=False)
        n_unseen = max(int(np.isfinite(als).sum()), 2)

        a = als[candidates].astype(np.float32, copy=False)
        c = tfidf[candidates].astype(np.float32, copy=False)
        ar = als_ranks[candidates]
        cr = tfidf_ranks[candidates]
        ar_score = 1.0 - (ar.astype(np.float32) - 1.0) / (n_unseen - 1.0)
        cr_score = 1.0 - (cr.astype(np.float32) - 1.0) / (n_unseen - 1.0)
        in_als = (ar <= self.als_k).astype(np.float32)
        in_content = (cr <= self.tfidf_k).astype(np.float32)
        both = in_als * in_content
        missing = (self.has_tags[candidates] == 0).astype(np.float32)

        columns = [
            a,
            c,
            ar_score,
            cr_score,
            np.minimum(a, c),
            np.maximum(a, c),
            a * c,
            np.abs(a - c),
            in_als,
            in_content,
            both,
            (ar <= 20).astype(np.float32),
            (cr <= 20).astype(np.float32),
            self.item_pop[candidates],
            self.has_tags[candidates],
            self.tag_count[candidates],
            np.full(candidates.size, self.user_history_size[uid], dtype=np.float32),
            np.full(candidates.size, self.user_tag_coverage[uid], dtype=np.float32),
            missing,
            ((cr <= 20) & (ar > n_unseen / 2)).astype(np.float32),
            ((ar <= 20) & (cr > n_unseen / 2)).astype(np.float32),
        ]
        X = np.column_stack(columns).astype(np.float32, copy=False)
        return candidates, X


def build_candidate_dataset(
    builder: FeatureBuilder,
    users: list[str],
    relevance: dict[str, set[str]],
    max_train_negatives: int | None = None,
    random_state: int = 42,
    require_positive: bool = True,
) -> CandidateDataset:
    rng = np.random.default_rng(random_state)
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    item_chunks: list[np.ndarray] = []
    kept_users: list[str] = []
    group_sizes: list[int] = []
    relevant_totals: list[int] = []

    for user_no, uid in enumerate(users, start=1):
        candidates, X = builder.user_candidates(uid)
        relevant = relevance.get(uid, set())
        y = np.fromiter(
            (builder.model._item_index[int(pos)] in relevant for pos in candidates),
            dtype=np.int8,
            count=candidates.size,
        )
        # All-zero groups do not contribute useful pairs to a supervised
        # ranker, so omit them from meta-training/evaluation.
        if require_positive and not y.any():
            continue

        if max_train_negatives is not None:
            positives = np.flatnonzero(y == 1)
            negatives = np.flatnonzero(y == 0)
            if negatives.size > max_train_negatives:
                # Half hard negatives (highest max component score), half
                # random negatives for broader calibration.
                n_hard = max_train_negatives // 2
                hard_score = X[negatives, FEATURE_NAMES.index("score_max")]
                hard_local = np.argpartition(-hard_score, kth=n_hard - 1)[:n_hard]
                hard = negatives[hard_local]
                remaining = np.setdiff1d(negatives, hard, assume_unique=False)
                n_random = min(max_train_negatives - n_hard, remaining.size)
                random_neg = rng.choice(remaining, size=n_random, replace=False)
                keep = np.sort(np.concatenate([positives, hard, random_neg]))
                candidates, X, y = candidates[keep], X[keep], y[keep]

        X_chunks.append(X)
        y_chunks.append(y)
        item_chunks.append(candidates)
        kept_users.append(uid)
        group_sizes.append(len(y))
        relevant_totals.append(len(relevant))
        if user_no % 500 == 0:
            log.info("  rank dataset %d/%d users", user_no, len(users))

    return CandidateDataset(
        X=np.vstack(X_chunks),
        y=np.concatenate(y_chunks),
        groups=np.asarray(group_sizes, dtype=np.int32),
        users=kept_users,
        item_positions=np.concatenate(item_chunks),
        relevant_totals=np.asarray(relevant_totals, dtype=np.int32),
    )


def mean_group_ndcg(
    scores: np.ndarray,
    dataset: CandidateDataset,
    k: int = 10,
) -> float:
    values: list[float] = []
    offset = 0
    for group_size, n_relevant in zip(dataset.groups, dataset.relevant_totals):
        end = offset + int(group_size)
        local_scores = scores[offset:end]
        local_y = dataset.y[offset:end]
        order = np.argsort(-local_scores, kind="stable")[:k]
        dcg = sum(
            float(local_y[pos]) / math.log2(rank + 1)
            for rank, pos in enumerate(order, start=1)
        )
        ideal = min(int(n_relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
        values.append(dcg / idcg if idcg else 0.0)
        offset = end
    return float(np.mean(values))


def tune_rrf(dataset: CandidateDataset) -> tuple[dict, pd.DataFrame]:
    als_rank_score = dataset.X[:, FEATURE_NAMES.index("als_rank_score")]
    tfidf_rank_score = dataset.X[:, FEATURE_NAMES.index("tfidf_rank_score")]
    # Recover approximate ordinal rank from the stored percentile score.
    # Only relative order matters inside 1/(constant + rank).
    als_order_proxy = 1.0 - als_rank_score
    tfidf_order_proxy = 1.0 - tfidf_rank_score
    rows = []
    for constant in (0.001, 0.005, 0.01, 0.025):
        for als_weight in (0.60, 0.70, 0.80, 0.90, 0.95):
            scores = (
                als_weight / (constant + als_order_proxy)
                + (1.0 - als_weight) / (constant + tfidf_order_proxy)
            )
            rows.append({
                "constant": constant,
                "als_weight": als_weight,
                "content_weight": 1.0 - als_weight,
                "ndcg@10": mean_group_ndcg(scores, dataset, 10),
                "ndcg@20": mean_group_ndcg(scores, dataset, 20),
            })
    frame = pd.DataFrame(rows).sort_values(
        ["ndcg@10", "ndcg@20"], ascending=False
    ).reset_index(drop=True)
    return frame.iloc[0].to_dict(), frame


def rrf_scores(X: np.ndarray, params: dict) -> np.ndarray:
    a = 1.0 - X[:, FEATURE_NAMES.index("als_rank_score")]
    c = 1.0 - X[:, FEATURE_NAMES.index("tfidf_rank_score")]
    weight = float(params["als_weight"])
    constant = float(params["constant"])
    return weight / (constant + a) + (1.0 - weight) / (constant + c)


def recommendations_from_scores(
    dataset: CandidateDataset,
    scores: np.ndarray,
    item_index: list[str],
    n: int = 20,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    offset = 0
    for uid, group_size in zip(dataset.users, dataset.groups):
        end = offset + int(group_size)
        local_items = dataset.item_positions[offset:end]
        local_scores = scores[offset:end]
        order = np.lexsort((local_items, -local_scores.astype(np.float64)))[:n]
        output[uid] = [item_index[int(local_items[pos])] for pos in order]
        offset = end
    return output


def evaluate_recommendations(
    name: str,
    recommendations: dict[str, list[str]],
    arts: ProcessedArtifacts,
) -> dict[str, object]:
    ground_truth = arts.val.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()
    # Users omitted from ranker data (no retrieved inner positive is possible
    # during meta-training, but outer validation inference includes everyone).
    for uid in ground_truth:
        recommendations.setdefault(uid, [])
    metrics = compute_all_metrics(
        recommendations=recommendations,
        ground_truth=ground_truth,
        catalogue=set(arts.train["item_id_raw"].unique()),
        k_values=config.EVAL_K_VALUES,
        item_popularity=arts.train["item_id_raw"].value_counts().to_dict(),
        item_vectors=normalize(arts.tfidf_matrix, norm="l2").tocsr(),
        item_index=list(arts.tfidf_item_index),
        n_train_users=arts.train["user_id_raw"].nunique(),
    )
    return {"model": name, "split": "val", **metrics}


def main(
    inner_holdout_fraction: float = 0.15,
    ranker_eval_user_fraction: float = 0.20,
    als_candidates: int = 500,
    tfidf_candidates: int = 100,
) -> pd.DataFrame:
    started = time.perf_counter()
    arts = load_processed_artifacts()
    results_dir = config.RESULTS_DIR
    models_dir = config.MODELS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    base_train, ranker_holdout = inner_per_user_split(
        arts.train, holdout_fraction=inner_holdout_fraction
    )
    log.info(
        "Inner split: base_train=%d ranker_holdout=%d users=%d",
        len(base_train), len(ranker_holdout), base_train["user_id_raw"].nunique(),
    )
    inner_relevance = (
        ranker_holdout.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()
    )

    log.info("Stage 1/5: fit inner retrievers")
    inner_model = build_retrievers(base_train, arts)
    inner_analysis = candidate_analysis(
        inner_model,
        inner_relevance,
        union_als_k=als_candidates,
        union_tfidf_k=tfidf_candidates,
    )

    all_users = list(inner_model._user_ids)
    rng = np.random.default_rng(2718)
    shuffled = np.asarray(all_users, dtype=object)
    rng.shuffle(shuffled)
    n_eval_users = int(round(len(shuffled) * ranker_eval_user_fraction))
    eval_user_set = set(shuffled[:n_eval_users].tolist())
    meta_train_users = [uid for uid in all_users if uid not in eval_user_set]
    meta_eval_users = [uid for uid in all_users if uid in eval_user_set]

    inner_builder = FeatureBuilder(
        inner_model, base_train, arts.item_features,
        als_k=als_candidates, tfidf_k=tfidf_candidates,
    )
    log.info("Stage 2/5: build meta-training candidates/features")
    meta_train = build_candidate_dataset(
        inner_builder, meta_train_users, inner_relevance,
        max_train_negatives=250, random_state=42,
    )
    meta_eval = build_candidate_dataset(
        inner_builder, meta_eval_users, inner_relevance,
        max_train_negatives=None, random_state=43, require_positive=False,
    )
    log.info(
        "Meta data: train rows=%d groups=%d positives=%d | eval rows=%d groups=%d positives=%d",
        len(meta_train.y), len(meta_train.groups), int(meta_train.y.sum()),
        len(meta_eval.y), len(meta_eval.groups), int(meta_eval.y.sum()),
    )

    log.info("Stage 3/5: tune RRF and train rankers")
    rrf_params, rrf_sweep = tune_rrf(meta_eval)
    log.info("Selected RRF params: %s", rrf_params)

    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=2000,
            tol=1e-5,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])
    logistic.fit(meta_train.X, meta_train.y)
    logistic_eval_score = logistic.decision_function(meta_eval.X)

    lambdamart = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=5.0,
        tree_method="hist",
        n_jobs=12,
        random_state=42,
        early_stopping_rounds=40,
    )
    lambdamart.fit(
        meta_train.X,
        meta_train.y,
        group=meta_train.groups,
        eval_set=[(meta_eval.X, meta_eval.y)],
        eval_group=[meta_eval.groups],
        verbose=25,
    )
    lambda_eval_score = lambdamart.predict(meta_eval.X)
    rrf_eval_score = rrf_scores(meta_eval.X, rrf_params)
    meta_metrics = {
        "rrf": {
            "ndcg@10": mean_group_ndcg(rrf_eval_score, meta_eval, 10),
            "ndcg@20": mean_group_ndcg(rrf_eval_score, meta_eval, 20),
            "params": rrf_params,
        },
        "logistic": {
            "ndcg@10": mean_group_ndcg(logistic_eval_score, meta_eval, 10),
            "ndcg@20": mean_group_ndcg(logistic_eval_score, meta_eval, 20),
        },
        "lambdamart": {
            "ndcg@10": mean_group_ndcg(lambda_eval_score, meta_eval, 10),
            "ndcg@20": mean_group_ndcg(lambda_eval_score, meta_eval, 20),
            "best_iteration": int(lambdamart.best_iteration),
        },
    }

    with open(models_dir / "logistic_ranker.pkl", "wb") as handle:
        pickle.dump({"model": logistic, "features": FEATURE_NAMES}, handle)
    lambdamart.save_model(models_dir / "lambdamart_ranker.json")

    # Free the inner score cache before allocating the full-train cache.
    del meta_train, meta_eval, inner_builder, inner_model

    log.info("Stage 4/5: refit retrievers on full original train")
    full_model = build_retrievers(arts.train, arts)
    outer_relevance = arts.val.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()
    outer_analysis = candidate_analysis(
        full_model,
        outer_relevance,
        union_als_k=als_candidates,
        union_tfidf_k=tfidf_candidates,
    )
    full_builder = FeatureBuilder(
        full_model, arts.train, arts.item_features,
        als_k=als_candidates, tfidf_k=tfidf_candidates,
    )
    outer = build_candidate_dataset(
        full_builder,
        list(full_model._user_ids),
        outer_relevance,
        max_train_negatives=None,
        random_state=44,
        require_positive=False,
    )
    # Unlike meta-training, every outer user must be represented.  The union
    # normally contains at least one validation positive for almost all users;
    # report the exact count in candidate analysis.
    log.info("Outer candidate rows=%d groups=%d", len(outer.y), len(outer.groups))

    log.info("Stage 5/5: frozen validation evaluation")
    score_sets = {
        "RetrieveRank-RRF": rrf_scores(outer.X, rrf_params),
        "RetrieveRank-Logistic": logistic.decision_function(outer.X),
        "RetrieveRank-LambdaMART": lambdamart.predict(outer.X),
    }
    rows = []
    for name, scores in score_sets.items():
        recs = recommendations_from_scores(
            outer, scores, full_model._item_index, n=max(config.EVAL_K_VALUES)
        )
        rows.append(evaluate_recommendations(name, recs, arts))
    results = pd.DataFrame(rows)

    results.to_csv(results_dir / "retrieve_rank_val.csv", index=False)
    with open(results_dir / "retrieve_rank_val.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=float)
    rrf_sweep.to_csv(results_dir / "rrf_inner_sweep.csv", index=False)
    with open(results_dir / "retrieve_rank_candidate_analysis.json", "w", encoding="utf-8") as handle:
        json.dump({"inner": inner_analysis, "outer_val": outer_analysis}, handle, indent=2)
    with open(results_dir / "retrieve_rank_meta_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(meta_metrics, handle, indent=2, default=float)
    pd.DataFrame({
        "feature": FEATURE_NAMES,
        "logistic_coefficient": logistic.named_steps["model"].coef_[0],
        "lambdamart_importance": lambdamart.feature_importances_,
    }).sort_values("lambdamart_importance", ascending=False).to_csv(
        results_dir / "retrieve_rank_feature_importance.csv", index=False
    )

    print("\n=== Retrieve + rank validation results ===")
    print(results[[
        "model", "precision@10", "recall@10", "ndcg@10",
        "precision@20", "recall@20", "ndcg@20",
    ]].to_string(index=False))
    print("\nCandidate analysis:")
    print(json.dumps({"inner": inner_analysis, "outer_val": outer_analysis}, indent=2))
    print("\nInner ranker metrics:")
    print(json.dumps(meta_metrics, indent=2, default=float))
    log.info("Completed in %.2fs", time.perf_counter() - started)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-holdout-fraction", type=float, default=0.15)
    parser.add_argument("--ranker-eval-user-fraction", type=float, default=0.20)
    parser.add_argument("--als-candidates", type=int, default=500)
    parser.add_argument("--tfidf-candidates", type=int, default=100)
    args = parser.parse_args()
    main(
        inner_holdout_fraction=args.inner_holdout_fraction,
        ranker_eval_user_fraction=args.ranker_eval_user_fraction,
        als_candidates=args.als_candidates,
        tfidf_candidates=args.tfidf_candidates,
    )
