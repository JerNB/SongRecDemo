"""Train, tune, evaluate, and diagnose an ALS + tag-TF-IDF hybrid.

The validation split is used for an exploratory weight sweep.  The test split
is loaded only because it is part of the shared artifact/evaluator contract;
it is never evaluated here.  A future final report should freeze the selected
weight and evaluate it once on test.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import config
from src.data.artifacts import ProcessedArtifacts, load_processed_artifacts
from src.evaluation.evaluator import Evaluator
from src.recommenders.collaborative import CollaborativeFilteringRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hybrid_eval")

DEFAULT_WEIGHTS = [
    0.0, 0.25, 0.50, 0.60, 0.625, 0.65, 0.675, 0.70,
    0.725, 0.75, 0.775, 0.80, 0.85, 0.90, 0.95, 1.0,
]
DEFAULT_MISSING_POLICIES = ["penalize", "neutral"]


def _parse_weights(value: str) -> list[float]:
    weights = sorted({float(v.strip()) for v in value.split(",") if v.strip()})
    if not weights or any(not 0.0 <= w <= 1.0 for w in weights):
        raise argparse.ArgumentTypeError("weights must be comma-separated values in [0, 1]")
    return weights


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _rank_all(scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Return exact 1-based ranks; non-finite seen items receive rank 0."""
    valid = np.flatnonzero(np.isfinite(scores))
    order_local = np.lexsort((valid, -scores[valid].astype(np.float64)))
    ordered = valid[order_local]
    ranks = np.zeros(scores.size, dtype=np.int32)
    ranks[ordered] = np.arange(1, ordered.size + 1, dtype=np.int32)
    return ranks, int(ordered.size)


def _reason(
    direction: str,
    has_tags: bool,
    tag_count: int,
    popularity: int,
    low_pop_threshold: float,
) -> str:
    if direction == "TFIDF_high_ALS_low":
        if popularity <= low_pop_threshold:
            return "Strong tag match but weak co-listening support; the item is long-tail."
        return "Tags align with the user profile, but ALS finds little latent-neighbour evidence."
    if not has_tags:
        return "ALS finds co-listening evidence, while TF-IDF has no usable tags for the item."
    if tag_count <= 3:
        return "ALS finds co-listening evidence, while the item's tag evidence is sparse."
    return "Latent co-listening is strong even though explicit tags weakly match the profile."


def analyse_disagreements(
    model: HybridRecommender,
    arts: ProcessedArtifacts,
    top_k: int = 20,
    sample_per_direction: int = 250,
) -> tuple[pd.DataFrame, dict]:
    """Summarise cases where one component's top-K is in the other's bottom half."""
    ground_truth = (
        arts.val.groupby("user_id_raw")["item_id_raw"].apply(set).to_dict()
    )
    history_sizes = arts.train.groupby("user_id_raw").size().to_dict()
    popularity = arts.train["item_id_raw"].value_counts().to_dict()
    pop_values = np.asarray(list(popularity.values()), dtype=np.float64)
    low_pop_threshold = float(np.quantile(pop_values, 0.25))

    stats = {
        "ALS_high_TFIDF_low": {
            "top_candidates": 0, "top_hits": 0, "strong_cases": 0,
            "strong_hits": 0, "missing_tags": 0, "popularity": [],
            "tag_counts": [], "opposite_ranks": [],
        },
        "TFIDF_high_ALS_low": {
            "top_candidates": 0, "top_hits": 0, "strong_cases": 0,
            "strong_hits": 0, "missing_tags": 0, "popularity": [],
            "tag_counts": [], "opposite_ranks": [],
        },
    }
    heaps: dict[str, list[tuple[float, int, dict]]] = {
        key: [] for key in stats
    }
    counter = 0

    for user_no, uid in enumerate(model._user_ids, start=1):
        als, content = model.component_scores(uid)
        als_ranks, n_unseen = _rank_all(als)
        content_ranks, _ = _rank_all(content)
        relevant = ground_truth.get(uid, set())

        als_top = np.flatnonzero((als_ranks > 0) & (als_ranks <= top_k))
        als_top = als_top[np.argsort(als_ranks[als_top])]
        content_top = np.flatnonzero((content_ranks > 0) & (content_ranks <= top_k))
        content_top = content_top[np.argsort(content_ranks[content_top])]
        directions = [
            ("ALS_high_TFIDF_low", als_top, als_ranks, content_ranks, als, content),
            ("TFIDF_high_ALS_low", content_top, content_ranks, als_ranks, content, als),
        ]
        for direction, top_items, high_ranks, other_ranks, high_scores, other_scores in directions:
            bucket = stats[direction]
            for pos in top_items:
                iid = model._item_index[int(pos)]
                hit = iid in relevant
                bucket["top_candidates"] += 1
                bucket["top_hits"] += int(hit)
                other_rank = int(other_ranks[pos])
                # "Low" is deliberately strict and interpretable: bottom
                # half of the unseen catalogue for the opposite model.
                if other_rank <= n_unseen / 2:
                    continue

                features = arts.item_features.loc[iid]
                tags = list(features["tags_normalised"])
                pop = int(popularity.get(iid, 0))
                tag_count = len(tags)
                has_tags = bool(features["has_tags"]) and tag_count > 0
                bucket["strong_cases"] += 1
                bucket["strong_hits"] += int(hit)
                bucket["missing_tags"] += int(not has_tags)
                bucket["popularity"].append(pop)
                bucket["tag_counts"].append(tag_count)
                bucket["opposite_ranks"].append(other_rank)

                severity = other_rank / max(n_unseen, 1)
                record = {
                    "direction": direction,
                    "user_id_raw": uid,
                    "item_id_raw": iid,
                    "high_model_rank": int(high_ranks[pos]),
                    "opposite_model_rank": other_rank,
                    "opposite_rank_percentile": severity,
                    "als_score_norm": float(als[pos]),
                    "tfidf_score_norm": float(content[pos]),
                    "is_validation_relevant": bool(hit),
                    "user_train_history_size": int(history_sizes.get(uid, 0)),
                    "item_train_user_count": pop,
                    "has_tags": has_tags,
                    "tag_count": tag_count,
                    "sample_tags": " | ".join(str(tag) for tag in tags[:12]),
                    "likely_reason": _reason(
                        direction, has_tags, tag_count, pop, low_pop_threshold
                    ),
                }
                counter += 1
                entry = (severity, counter, record)
                heap = heaps[direction]
                if len(heap) < sample_per_direction:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)

        if user_no % 1000 == 0:
            log.info("  disagreement analysis: %d/%d users", user_no, len(model._user_ids))

    records = []
    for heap in heaps.values():
        records.extend(entry[2] for entry in sorted(heap, reverse=True))
    samples = pd.DataFrame(records)
    if not samples.empty:
        samples = samples.sort_values(
            ["direction", "opposite_rank_percentile", "high_model_rank"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    summary: dict[str, object] = {
        "definition": (
            f"High = component top-{top_k}; low = opposite component bottom 50% "
            "among unseen catalogue items."
        ),
        "validation_note": "Hits are membership in the frozen validation relevance set.",
        "directions": {},
    }
    for direction, bucket in stats.items():
        n_top = int(bucket["top_candidates"])
        n_strong = int(bucket["strong_cases"])
        pops = np.asarray(bucket["popularity"], dtype=float)
        tags = np.asarray(bucket["tag_counts"], dtype=float)
        opp = np.asarray(bucket["opposite_ranks"], dtype=float)
        summary["directions"][direction] = {
            "top_candidates": n_top,
            "top_candidate_hit_rate": float(bucket["top_hits"] / n_top) if n_top else 0.0,
            "strong_disagreement_cases": n_strong,
            "strong_share_of_top_candidates": float(n_strong / n_top) if n_top else 0.0,
            "strong_disagreement_hit_rate": float(bucket["strong_hits"] / n_strong) if n_strong else 0.0,
            "missing_tag_rate": float(bucket["missing_tags"] / n_strong) if n_strong else 0.0,
            "median_item_train_user_count": float(np.median(pops)) if pops.size else None,
            "median_tag_count": float(np.median(tags)) if tags.size else None,
            "median_opposite_rank": float(np.median(opp)) if opp.size else None,
        }
    return samples, summary


def _metric_table(rows: list[dict], baseline_rows: list[dict | None]) -> pd.DataFrame:
    output = []
    for label, row in [
        ("Popularity", baseline_rows[0]),
        ("ALS", baseline_rows[1]),
        ("TF-IDF", baseline_rows[2]),
        ("Hybrid", rows[0]),
    ]:
        if row is None:
            continue
        record: dict[str, object] = {"model": label, "internal_name": row["model"]}
        for k in (10, 20):
            for metric in ("precision", "recall", "ndcg", "hit_rate", "coverage", "novelty", "diversity"):
                key = f"{metric}@{k}"
                record[key] = float(row[key])
        output.append(record)
    return pd.DataFrame(output)


def _validate_pure_endpoints(
    sweep: pd.DataFrame,
    als_baseline: dict | None,
    content_baseline: dict | None,
    tolerance: float = 1e-12,
) -> None:
    """Ensure the blend endpoints reproduce the historical pure rankers."""
    checks = [(1.0, als_baseline, "ALS"), (0.0, content_baseline, "TF-IDF")]
    for weight, baseline, label in checks:
        if baseline is None:
            log.warning("Cannot validate %s endpoint: historical result is missing", label)
            continue
        endpoint = sweep[
            (sweep["missing_content_policy"] == "penalize")
            & np.isclose(sweep["als_weight"], weight)
        ]
        if endpoint.empty:
            log.warning("Cannot validate %s endpoint: weight %.1f was not swept", label, weight)
            continue
        row = endpoint.iloc[0]
        metric_keys = [key for key in baseline if "@" in key and key in row.index]
        largest_delta = max(
            abs(float(row[key]) - float(baseline[key])) for key in metric_keys
        )
        if largest_delta > tolerance:
            raise AssertionError(
                f"{label} endpoint does not reproduce historical metrics; "
                f"largest absolute delta={largest_delta:.3e}."
            )
        log.info("Verified %s endpoint against history (max delta %.3e)", label, largest_delta)


def _to_markdown(df: pd.DataFrame, digits: int = 6) -> str:
    """Render a small DataFrame as Markdown without the optional tabulate dep."""
    columns = [str(column) for column in df.columns]
    rendered_rows: list[list[str]] = []
    for row in df.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{digits}f}")
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered_rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    selected: dict,
    comparison: pd.DataFrame,
    sweep: pd.DataFrame,
    disagreement: dict,
) -> None:
    als_row = comparison.loc[comparison["model"] == "ALS"]
    hybrid_row = comparison.loc[comparison["model"] == "Hybrid"]

    lines = [
        "# KGRec ALS + TF-IDF hybrid experiment",
        "",
        "## Protocol",
        "",
        "Both components were retrained on the frozen training split. Scores were min-max "
        "normalised independently for every user over all unseen catalogue items, then "
        "blended. The weight sweep and model selection use validation only; test remains "
        "untouched for a later final, one-shot evaluation.",
        "",
        f"Selected genuine hybrid: **ALS weight {selected['als_weight']:.3f} / "
        f"TF-IDF weight {selected['content_weight']:.3f}; missing-content policy "
        f"`{selected['missing_content_policy']}`**, selected by NDCG@10 "
        "(Recall@20 as tie-break).",
        "",
        "## K@10 and K@20 comparison",
        "",
        _to_markdown(comparison),
        "",
    ]
    if not als_row.empty and not hybrid_row.empty:
        als = als_row.iloc[0]
        hybrid = hybrid_row.iloc[0]
        lines.extend([
            "## Hybrid delta versus historical ALS",
            "",
            f"- Precision@10: {hybrid['precision@10'] - als['precision@10']:+.6f}",
            f"- Recall@10: {hybrid['recall@10'] - als['recall@10']:+.6f}",
            f"- NDCG@10: {hybrid['ndcg@10'] - als['ndcg@10']:+.6f}",
            f"- Precision@20: {hybrid['precision@20'] - als['precision@20']:+.6f}",
            f"- Recall@20: {hybrid['recall@20'] - als['recall@20']:+.6f}",
            f"- NDCG@20: {hybrid['ndcg@20'] - als['ndcg@20']:+.6f}",
            "",
        ])

    lines.extend(["## Weight sweep", "", _to_markdown(sweep), ""])
    policy_best = (
        sweep.sort_values(["ndcg@10", "ndcg@20"], ascending=False)
        .groupby("missing_content_policy", as_index=False)
        .first()[["missing_content_policy", "als_weight", "ndcg@10", "ndcg@20"]]
    )
    lines.extend([
        "## Missing-content policy check",
        "",
        _to_markdown(policy_best),
        "",
    ])
    lines.extend(["## Strong component disagreements", "", disagreement["definition"], ""])
    for direction, values in disagreement["directions"].items():
        lines.extend([
            f"### {direction}",
            "",
            f"- Strong cases: {values['strong_disagreement_cases']} "
            f"({values['strong_share_of_top_candidates']:.2%} of that component's top-20)",
            f"- Validation hit rate among strong cases: "
            f"{values['strong_disagreement_hit_rate']:.4%}",
            f"- Missing-tag rate: {values['missing_tag_rate']:.2%}",
            f"- Median training-user popularity: {values['median_item_train_user_count']}",
            f"- Median tag count: {values['median_tag_count']}",
            "",
        ])

    a = disagreement["directions"]["ALS_high_TFIDF_low"]
    c = disagreement["directions"]["TFIDF_high_ALS_low"]
    lines.extend(["## Interpretation and next adjustment", ""])
    if c["strong_disagreement_hit_rate"] > a["strong_disagreement_hit_rate"]:
        lines.append(
            "TF-IDF-high/ALS-low disagreements hit validation more often than the reverse. "
            "That supports a larger content contribution or a conditional content boost."
        )
    else:
        lines.append(
            "Extreme TF-IDF-high/ALS-low cases are much less reliable than the reverse. "
            "TF-IDF helps mainly as a re-ranking signal when it agrees at least moderately "
            "with ALS; isolated content-only outliers should not be rescued."
        )
    if selected["missing_content_policy"] == "penalize":
        lines.append(
            "Treating missing TF-IDF evidence as neutral was tested directly and scored worse "
            "than the original penalty policy. Although many ALS-high/TF-IDF-low cases lack "
            "tags, that disagreement subset also hits less often than ALS's overall top-20; "
            "restoring those items therefore adds more false positives than true positives."
        )
    lines.append(
        "The selected rule should now be frozen before a one-shot test evaluation. Further "
        "model changes would require a new inner validation split or cross-validation to avoid "
        "reusing this validation set indefinitely."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    weights: list[float],
    sample_per_direction: int = 250,
    missing_policies: list[str] | None = None,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    arts = load_processed_artifacts()
    item_vectors = normalize(arts.tfidf_matrix, norm="l2").tocsr()

    hybrid = HybridRecommender(
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
        als_weight=weights[0],
    )
    log.info("Fitting hybrid components on %d training interactions", len(arts.train))
    hybrid.fit(arts.train, arts.item_features)

    evaluator = Evaluator(
        val_df=arts.val,
        test_df=arts.test,
        train_df=arts.train,
        item_features_df=arts.item_features,
        item_vectors=item_vectors,
        item_index=list(arts.tfidf_item_index),
        k_values=config.EVAL_K_VALUES,
    )

    rows = []
    policies = missing_policies or DEFAULT_MISSING_POLICIES
    for policy in policies:
        hybrid.missing_content_policy = policy
        for weight in weights:
            hybrid.als_weight = weight
            log.info(
                "Evaluating policy=%s ALS weight %.3f / TF-IDF weight %.3f",
                policy, weight, 1.0 - weight,
            )
            row = evaluator.evaluate(hybrid, split="val")
            row["als_weight"] = weight
            row["content_weight"] = 1.0 - weight
            row["missing_content_policy"] = policy
            rows.append(row)
    sweep = pd.DataFrame(rows)

    baseline_rows = [
        _load_json(config.RESULTS_DIR / "popularity_val.json"),
        _load_json(config.RESULTS_DIR / "collaborative_val.json"),
        _load_json(config.RESULTS_DIR / "content_based_val.json"),
    ]
    _validate_pure_endpoints(sweep, baseline_rows[1], baseline_rows[2])

    genuine = sweep[(sweep["als_weight"] > 0.0) & (sweep["als_weight"] < 1.0)]
    if genuine.empty:
        raise ValueError("Weight grid must contain at least one genuine mixed weight.")
    selected_idx = genuine.sort_values(
        ["ndcg@10", "recall@20", "precision@10", "als_weight"],
        ascending=[False, False, False, True],
    ).index[0]
    selected = sweep.loc[selected_idx].to_dict()
    hybrid.als_weight = float(selected["als_weight"])
    hybrid.missing_content_policy = str(selected["missing_content_policy"])
    log.info("Selected %s", hybrid.name)

    samples, disagreement = analyse_disagreements(
        hybrid, arts, top_k=20, sample_per_direction=sample_per_direction
    )

    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(results_dir / "hybrid_weight_sweep_val.csv", index=False)
    pd.DataFrame([selected]).to_csv(results_dir / "hybrid_val.csv", index=False)
    with open(results_dir / "hybrid_val.json", "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2, default=float)
    samples.to_csv(results_dir / "hybrid_disagreement_samples.csv", index=False)
    with open(results_dir / "hybrid_disagreement_summary.json", "w", encoding="utf-8") as handle:
        json.dump(disagreement, handle, indent=2, default=float)

    comparison = _metric_table([selected], baseline_rows)
    comparison.to_csv(results_dir / "hybrid_comparison_val.csv", index=False)
    report_sweep = sweep[[
        "missing_content_policy", "als_weight", "content_weight",
        "precision@10", "recall@10", "ndcg@10",
        "precision@20", "recall@20", "ndcg@20", "coverage@20", "novelty@20",
    ]]
    _write_report(
        results_dir / "hybrid_experiment_report.md",
        selected,
        comparison,
        report_sweep,
        disagreement,
    )

    print("\n=== Selected hybrid ===")
    print(
        f"ALS weight={selected['als_weight']:.3f}, "
        f"TF-IDF weight={selected['content_weight']:.3f}, "
        f"missing={selected['missing_content_policy']}"
    )
    print(comparison.to_string(index=False))
    print(f"\nArtifacts saved under {results_dir}")
    log.info("Done in %.2fs", time.perf_counter() - t0)
    return sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=_parse_weights,
        default=DEFAULT_WEIGHTS,
        help="Comma-separated ALS weights in [0,1].",
    )
    parser.add_argument(
        "--missing-policies",
        nargs="+",
        choices=["penalize", "neutral"],
        default=DEFAULT_MISSING_POLICIES,
        help="How zero-tag items contribute to the blend.",
    )
    parser.add_argument(
        "--samples-per-direction",
        type=int,
        default=250,
        help="Maximum saved examples for each strong disagreement direction.",
    )
    args = parser.parse_args()
    main(
        args.weights,
        sample_per_direction=args.samples_per_direction,
        missing_policies=args.missing_policies,
    )
