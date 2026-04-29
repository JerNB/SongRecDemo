"""
Collaborative-Filtering (ALS) evaluation runner.

Scope
-----
Runs ONLY the ALS collaborative-filtering model through the shared
evaluation harness on the validation split. Uses the same
``ProcessedArtifacts`` contract and the same ``Evaluator`` instance the
popularity baseline uses, so every metric is measured identically.

What it does
------------
1.  Load preprocessed artefacts via ``src.data.artifacts`` (no raw data).
2.  L2-normalise the training-only tag TF-IDF matrix so the Intra-List
    Diversity metric is measured in the same feature space the content
    model will use. This is the ONLY shared resource between CF and the
    diversity metric — the CF model itself is content-agnostic.
3.  Fit ``CollaborativeFilteringRecommender`` on training interactions.
4.  Evaluate on the VALIDATION split at K = 5, 10, 20 using:
        Precision@K, Recall@K, F1@K, NDCG@K, Hit-Rate@K,
        Coverage@K, Novelty@K, Intra-List Diversity@K.
5.  Save results to ``artifacts/results/collaborative_val.csv`` and
    ``artifacts/results/collaborative_val.json``.
6.  Load the popularity baseline's saved results, compute head-to-head
    deltas at the primary K, and print a concise interpretation of what
    ALS buys (or costs) relative to popularity on accuracy and
    beyond-accuracy metrics.

Reproducibility
---------------
- All inputs are frozen by preprocessing.
- ALS factor initialisation uses ``config.SPLIT_SEED`` (42) as its RNG
  seed, so retraining is bit-for-bit reproducible on the same NumPy /
  BLAS build.
- Test split is NOT touched. Only the full pipeline may access it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import normalize

import config
from src.data.artifacts import load_processed_artifacts
from src.evaluation.evaluator import Evaluator
from src.recommenders.collaborative import CollaborativeFilteringRecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cf_eval")


POP_RESULTS_FILE = config.RESULTS_DIR / "popularity_val.json"


def _load_baseline() -> dict | None:
    """Load the saved popularity-baseline metrics if present."""
    if not POP_RESULTS_FILE.exists():
        log.warning(
            "Popularity baseline results not found at %s — skipping "
            "head-to-head comparison. Run run_popularity_eval.py first.",
            POP_RESULTS_FILE,
        )
        return None
    with open(POP_RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_delta(a: float, b: float) -> str:
    """Pretty-print (ALS - Popularity) as an absolute delta and pct change."""
    d = a - b
    if b == 0:
        pct = "n/a"
    else:
        pct = f"{(d / b) * 100:+.1f}%"
    return f"{d:+.4f} ({pct})"


def _interpret(als_row: dict, pop_row: dict | None,
               n_train_users: int, n_catalogue: int) -> str:
    """Build the ALS-vs-Popularity interpretation block."""
    pk = config.PRIMARY_K

    def m(row: dict, key: str) -> float:
        return float(row.get(key, float("nan")))

    prec = m(als_row, f"precision@{pk}")
    rec = m(als_row, f"recall@{pk}")
    ndcg = m(als_row, f"ndcg@{pk}")
    hr = m(als_row, f"hit_rate@{pk}")
    cov = m(als_row, f"coverage@{pk}")
    div = m(als_row, f"diversity@{pk}")
    nov = m(als_row, f"novelty@{pk}")

    n_items_shown = int(round(cov * n_catalogue)) if cov == cov else 0

    lines = [
        "",
        "=== ALS collaborative filtering — interpretation ===",
        f"Evaluated on the VALIDATION split at K = {pk} "
        f"({n_train_users} users, {n_catalogue} training-catalogue items).",
        "",
        "ALS on its own",
        f"  Precision@{pk} = {prec:.4f} | Recall@{pk} = {rec:.4f} | "
        f"NDCG@{pk} = {ndcg:.4f} | Hit-Rate@{pk} = {hr:.4f}",
        f"  Coverage@{pk}  = {cov:.4f}  (~{n_items_shown} distinct items "
        f"ever recommended across all {n_train_users} users)",
        f"  Diversity@{pk} = {div:.4f}  "
        "(mean pairwise cosine distance in tag-TF-IDF space)",
        f"  Novelty@{pk}   = {nov:.4f}  (mean -log2(pop / n_users))",
        "",
    ]

    if pop_row is None:
        lines.append(
            "Popularity baseline not available for comparison — run "
            "run_popularity_eval.py and re-run this script to get the "
            "head-to-head block."
        )
        lines.append("")
        return "\n".join(lines)

    p_prec = m(pop_row, f"precision@{pk}")
    p_rec = m(pop_row, f"recall@{pk}")
    p_ndcg = m(pop_row, f"ndcg@{pk}")
    p_hr = m(pop_row, f"hit_rate@{pk}")
    p_cov = m(pop_row, f"coverage@{pk}")
    p_div = m(pop_row, f"diversity@{pk}")
    p_nov = m(pop_row, f"novelty@{pk}")

    lines.extend([
        "Head-to-head vs popularity baseline (ALS - Popularity)",
        f"  Precision@{pk}:  ALS={prec:.4f}   Pop={p_prec:.4f}   "
        f"delta = {_fmt_delta(prec, p_prec)}",
        f"  Recall@{pk}:     ALS={rec:.4f}   Pop={p_rec:.4f}   "
        f"delta = {_fmt_delta(rec, p_rec)}",
        f"  NDCG@{pk}:       ALS={ndcg:.4f}   Pop={p_ndcg:.4f}   "
        f"delta = {_fmt_delta(ndcg, p_ndcg)}",
        f"  Hit-Rate@{pk}:   ALS={hr:.4f}   Pop={p_hr:.4f}   "
        f"delta = {_fmt_delta(hr, p_hr)}",
        f"  Coverage@{pk}:   ALS={cov:.4f}   Pop={p_cov:.4f}   "
        f"delta = {_fmt_delta(cov, p_cov)}",
        f"  Novelty@{pk}:    ALS={nov:.4f}   Pop={p_nov:.4f}   "
        f"delta = {_fmt_delta(nov, p_nov)}",
        f"  Diversity@{pk}:  ALS={div:.4f}   Pop={p_div:.4f}   "
        f"delta = {_fmt_delta(div, p_div)}",
        "",
    ])

    # Qualitative read of the head-to-head numbers.
    # The framing is deliberately conditional on the observed signs of
    # the deltas so the interpretation stays honest if retraining shifts
    # the numbers slightly. The thresholds are chosen to flag meaningful
    # (not floating-point-noise) differences.
    def _sign(delta: float, thresh: float = 1e-4) -> str:
        if delta > thresh:
            return "higher"
        if delta < -thresh:
            return "lower"
        return "effectively tied with"

    lines.extend([
        "What the comparison shows",
        f"  Accuracy: ALS is {_sign(prec - p_prec)} on Precision@{pk}, "
        f"{_sign(rec - p_rec)} on Recall@{pk}, and "
        f"{_sign(hr - p_hr)} on Hit-Rate@{pk}. Any accuracy gain ALS posts "
        "over popularity is evidence that users' histories contain "
        "structure beyond \"everyone likes the hits\" -- i.e. the latent "
        "factors are picking up real preference signal. An accuracy loss "
        "would mean the confidence/regularisation mix is over-smoothing "
        "toward popularity (the factors collapse onto the popularity "
        "direction when alpha or lambda are set badly).",
        "",
        f"  Catalogue exposure: Coverage@{pk} is {_sign(cov - p_cov)} than "
        f"popularity ({cov:.4f} vs {p_cov:.4f}). Because ALS personalises, "
        "different users see different top-K lists, so many more distinct "
        "items reach at least one user's list. This is the first concrete "
        "discovery gain over a non-personalised ranker: the catalogue "
        f"exposure rises from ~{int(round(p_cov * n_catalogue))} items to "
        f"~{n_items_shown} items across the {n_train_users}-user population.",
        "",
        f"  Novelty: ALS novelty is {_sign(nov - p_nov)} than popularity "
        f"({nov:.4f} vs {p_nov:.4f} bits). Popularity by construction "
        "serves the lowest-surprise items first, so anything that ranks "
        "at least some mid-tail items above the head will post higher "
        "novelty. How much higher is the headline discovery metric for "
        "this comparison.",
        "",
        f"  Diversity: Intra-list diversity in tag-TF-IDF space is "
        f"{_sign(div - p_div)} for ALS ({div:.4f} vs {p_div:.4f}). "
        "Popularity's diversity was already high because the top-10 "
        "hits span multiple genres; ALS can move this in either "
        "direction. It tends to DROP intra-list diversity when the "
        "latent factors concentrate recommendations within one user's "
        "dominant genre -- a known failure mode worth reporting rather "
        "than hiding.",
        "",
        "Take-away for the three-model comparison",
        "  Popularity still anchors the accuracy floor and the coverage/"
        "novelty floors. ALS is the first model in the study that exploits "
        "per-user structure, so its accuracy gain (if any) measures how "
        "much personalisation helps on this dataset, and its coverage/"
        "novelty gain measures how much of that personalisation spends "
        "itself on long-tail exposure. The content-based model, coming "
        "next, will be asked to match or beat ALS on both axes using a "
        "different information source (tags) -- that is the real "
        "accuracy-vs-discovery trade-off the project set out to study.",
        "",
    ])
    return "\n".join(lines)


def main() -> pd.DataFrame:
    t0 = time.perf_counter()

    # --------------------------------------------------------------
    # 1. Load processed artefacts.
    # --------------------------------------------------------------
    arts = load_processed_artifacts()
    log.info(
        "Loaded artefacts: train=%d rows | val=%d rows | test=%d rows | "
        "items=%d | tag_tfidf=%s",
        len(arts.train), len(arts.val), len(arts.test),
        len(arts.item_features), arts.tfidf_matrix.shape,
    )

    # --------------------------------------------------------------
    # 2. Item-feature matrix for the diversity metric.
    # --------------------------------------------------------------
    item_vectors = normalize(arts.tfidf_matrix, norm="l2")
    item_index = list(arts.tfidf_item_index)
    log.info(
        "Diversity feature space: shape=%s, nnz=%d, zero-vector items=%d",
        item_vectors.shape, item_vectors.nnz,
        int((item_vectors.getnnz(axis=1) == 0).sum()),
    )

    # --------------------------------------------------------------
    # 3. Fit ALS on training only.
    # --------------------------------------------------------------
    model = CollaborativeFilteringRecommender(
        factors=config.CF_FACTORS,
        regularization=config.CF_REGULARIZATION,
        iterations=config.CF_ITERATIONS,
        alpha=config.CF_ALPHA,
        random_state=config.SPLIT_SEED,
    )
    log.info("Fitting %s on %d training interactions ...",
             model.name, len(arts.train))
    model.fit(arts.train, arts.item_features)

    # --------------------------------------------------------------
    # 4. Evaluate on validation.
    # --------------------------------------------------------------
    evaluator = Evaluator(
        val_df=arts.val,
        test_df=arts.test,
        train_df=arts.train,
        item_features_df=arts.item_features,
        item_vectors=item_vectors,
        item_index=item_index,
        k_values=config.EVAL_K_VALUES,
    )
    row = evaluator.evaluate(model, split="val")
    results_df = pd.DataFrame([row])

    # --------------------------------------------------------------
    # 5. Persist results.
    # --------------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / "collaborative_val.csv"
    json_path = config.RESULTS_DIR / "collaborative_val.json"
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, default=float)
    log.info("Saved results -> %s", csv_path)
    log.info("Saved results -> %s", json_path)

    # --------------------------------------------------------------
    # 6. Summary + interpretation (head-to-head vs popularity).
    # --------------------------------------------------------------
    evaluator.print_summary(results_df)

    print("=== Metrics across all K values ===")
    k_cols = sorted(
        [c for c in results_df.columns if "@" in c],
        key=lambda c: (c.split("@")[0], int(c.split("@")[1])),
    )
    for c in k_cols:
        print(f"  {c:<18s} = {results_df.iloc[0][c]:.4f}")
    print()

    pop_row = _load_baseline()
    n_train_users = arts.train["user_id_raw"].nunique()
    n_catalogue = arts.train["item_id_raw"].nunique()
    print(_interpret(row, pop_row, n_train_users, n_catalogue))

    log.info("Done in %.2fs", time.perf_counter() - t0)
    return results_df


if __name__ == "__main__":
    main()
