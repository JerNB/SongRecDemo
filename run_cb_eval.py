"""
Content-Based (tag TF-IDF) evaluation runner.

Scope
-----
Runs ONLY the content-based recommender through the shared evaluation
harness on the validation split. Uses the same ``ProcessedArtifacts``
contract and the same ``Evaluator`` instance the other two models use,
so every metric is measured identically.

Key design choice: the **same L2-normalised tag TF-IDF matrix** that
feeds the Intra-List Diversity metric is what the content-based model
scores in. That is intentional -- it means CB's diversity number
reflects the feature space the model actually uses, and it removes a
second, inconsistent TF-IDF fit that the previous version of the
recommender was doing internally.

What it does
------------
1. Load preprocessed artefacts via ``src.data.artifacts``.
2. L2-normalise the training-only tag TF-IDF matrix.
3. Fit ``ContentBasedRecommender`` on training interactions.
4. Evaluate on VALIDATION at K = 5, 10, 20 with:
       Precision@K, Recall@K, F1@K, NDCG@K, Hit-Rate@K,
       Coverage@K, Novelty@K, Intra-List Diversity@K.
5. Save ``artifacts/results/content_based_val.csv`` and ``.json``.
6. Load the popularity AND CF saved results and print a three-way
   head-to-head at the primary K.

Reproducibility
---------------
- All inputs come from the frozen preprocessing artefacts.
- No randomness: profiles are deterministic averages, ranking is
  deterministic top-K with lexsort tie-break.
- Test split is NOT touched.
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
from sklearn.preprocessing import normalize

import config
from src.data.artifacts import load_processed_artifacts
from src.evaluation.evaluator import Evaluator
from src.recommenders.content_based import ContentBasedRecommender


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cb_eval")


POP_RESULTS_FILE = config.RESULTS_DIR / "popularity_val.json"
CF_RESULTS_FILE = config.RESULTS_DIR / "collaborative_val.json"


def _load_json(path) -> dict | None:
    if not path.exists():
        log.warning("Results file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_delta(a: float, b: float) -> str:
    d = a - b
    if b == 0:
        pct = "n/a"
    else:
        pct = f"{(d / b) * 100:+.1f}%"
    return f"{d:+.4f} ({pct})"


def _interpret(
    cb_row: dict,
    pop_row: dict | None,
    cf_row: dict | None,
    n_train_users: int,
    n_catalogue: int,
) -> str:
    pk = config.PRIMARY_K

    def m(row: dict | None, key: str) -> float:
        if row is None:
            return float("nan")
        return float(row.get(key, float("nan")))

    prec = m(cb_row, f"precision@{pk}")
    rec = m(cb_row, f"recall@{pk}")
    ndcg = m(cb_row, f"ndcg@{pk}")
    hr = m(cb_row, f"hit_rate@{pk}")
    cov = m(cb_row, f"coverage@{pk}")
    div = m(cb_row, f"diversity@{pk}")
    nov = m(cb_row, f"novelty@{pk}")

    n_items_shown = int(round(cov * n_catalogue)) if cov == cov else 0

    lines = [
        "",
        "=== Content-based (tag TF-IDF) -- interpretation ===",
        f"Evaluated on the VALIDATION split at K = {pk} "
        f"({n_train_users} users, {n_catalogue} training-catalogue items).",
        "",
        "Content-based on its own",
        f"  Precision@{pk} = {prec:.4f} | Recall@{pk} = {rec:.4f} | "
        f"NDCG@{pk} = {ndcg:.4f} | Hit-Rate@{pk} = {hr:.4f}",
        f"  Coverage@{pk}  = {cov:.4f}  (~{n_items_shown} distinct items "
        f"ever recommended across all {n_train_users} users)",
        f"  Diversity@{pk} = {div:.4f}  "
        "(mean pairwise cosine distance in tag-TF-IDF space -- "
        "the SAME space the model scores in)",
        f"  Novelty@{pk}   = {nov:.4f}  (mean -log2(pop / n_users))",
        "",
    ]

    def _sign(delta: float, thresh: float = 1e-4) -> str:
        if delta > thresh:
            return "higher"
        if delta < -thresh:
            return "lower"
        return "effectively tied with"

    # Three-way head-to-head table -----------------------------------------
    def _row(name: str, p_val: float, c_val: float, cb_val: float) -> str:
        return (
            f"  {name:<14s}  Pop={p_val:.4f}   CF={c_val:.4f}   "
            f"CB={cb_val:.4f}"
        )

    if pop_row is not None and cf_row is not None:
        p = {
            "precision": m(pop_row, f"precision@{pk}"),
            "recall": m(pop_row, f"recall@{pk}"),
            "ndcg": m(pop_row, f"ndcg@{pk}"),
            "hit_rate": m(pop_row, f"hit_rate@{pk}"),
            "coverage": m(pop_row, f"coverage@{pk}"),
            "novelty": m(pop_row, f"novelty@{pk}"),
            "diversity": m(pop_row, f"diversity@{pk}"),
        }
        c = {
            "precision": m(cf_row, f"precision@{pk}"),
            "recall": m(cf_row, f"recall@{pk}"),
            "ndcg": m(cf_row, f"ndcg@{pk}"),
            "hit_rate": m(cf_row, f"hit_rate@{pk}"),
            "coverage": m(cf_row, f"coverage@{pk}"),
            "novelty": m(cf_row, f"novelty@{pk}"),
            "diversity": m(cf_row, f"diversity@{pk}"),
        }
        cb = {
            "precision": prec, "recall": rec, "ndcg": ndcg,
            "hit_rate": hr, "coverage": cov,
            "novelty": nov, "diversity": div,
        }
        lines.extend([
            "Three-way head-to-head at K=%d  (Pop | CF | CB)" % pk,
            _row(f"Precision@{pk}", p["precision"], c["precision"], cb["precision"]),
            _row(f"Recall@{pk}",    p["recall"],    c["recall"],    cb["recall"]),
            _row(f"NDCG@{pk}",      p["ndcg"],      c["ndcg"],      cb["ndcg"]),
            _row(f"Hit-Rate@{pk}",  p["hit_rate"],  c["hit_rate"],  cb["hit_rate"]),
            _row(f"Coverage@{pk}",  p["coverage"],  c["coverage"],  cb["coverage"]),
            _row(f"Novelty@{pk}",   p["novelty"],   c["novelty"],   cb["novelty"]),
            _row(f"Diversity@{pk}", p["diversity"], c["diversity"], cb["diversity"]),
            "",
            "Deltas vs popularity baseline",
            f"  Precision  CB - Pop = {_fmt_delta(cb['precision'], p['precision'])}",
            f"  Recall     CB - Pop = {_fmt_delta(cb['recall'],    p['recall'])}",
            f"  Hit-Rate   CB - Pop = {_fmt_delta(cb['hit_rate'],  p['hit_rate'])}",
            f"  Coverage   CB - Pop = {_fmt_delta(cb['coverage'],  p['coverage'])}",
            f"  Novelty    CB - Pop = {_fmt_delta(cb['novelty'],   p['novelty'])}",
            f"  Diversity  CB - Pop = {_fmt_delta(cb['diversity'], p['diversity'])}",
            "",
            "Deltas vs CF (ALS)",
            f"  Precision  CB - CF  = {_fmt_delta(cb['precision'], c['precision'])}",
            f"  Recall     CB - CF  = {_fmt_delta(cb['recall'],    c['recall'])}",
            f"  Hit-Rate   CB - CF  = {_fmt_delta(cb['hit_rate'],  c['hit_rate'])}",
            f"  Coverage   CB - CF  = {_fmt_delta(cb['coverage'],  c['coverage'])}",
            f"  Novelty    CB - CF  = {_fmt_delta(cb['novelty'],   c['novelty'])}",
            f"  Diversity  CB - CF  = {_fmt_delta(cb['diversity'], c['diversity'])}",
            "",
            "What the three-way comparison shows",
            f"  Accuracy: CB is {_sign(cb['precision'] - p['precision'])} than "
            f"popularity on Precision@{pk} and "
            f"{_sign(cb['precision'] - c['precision'])} than CF. The content "
            "signal captures genre / mood clusters that tend to match "
            "what users already listen to, but it does not know about "
            "co-listening structure -- so whether it beats CF on accuracy "
            "is a measurement about this particular dataset, not a "
            "general property.",
            "",
            f"  Catalogue exposure: Coverage@{pk} is "
            f"{_sign(cb['coverage'] - p['coverage'])} than popularity and "
            f"{_sign(cb['coverage'] - c['coverage'])} than CF. CB "
            "personalises so coverage is almost always much higher than "
            "popularity; whether it is higher or lower than CF tells us "
            "how much of each user's neighbourhood the tag space actually "
            "spans.",
            "",
            f"  Novelty: CB novelty is "
            f"{_sign(cb['novelty'] - p['novelty'])} than popularity and "
            f"{_sign(cb['novelty'] - c['novelty'])} than CF. Because tag "
            "similarity has no built-in bias toward popular items, CB "
            "can surface long-tail items that share the user's tag "
            "profile even when almost nobody else has listened to them. "
            "This is the mechanism through which CB typically buys "
            "novelty at an accuracy cost.",
            "",
            f"  Diversity: Intra-list diversity in tag space is "
            f"{_sign(cb['diversity'] - p['diversity'])} than popularity "
            f"and {_sign(cb['diversity'] - c['diversity'])} than CF. This "
            "is the most important pattern of the three-way comparison: "
            "CB scores items by proximity in tag space, so its top-K "
            "lists concentrate inside a small tag neighbourhood and "
            "intra-list diversity DROPS by construction. Reporting this "
            "is the point -- it is exactly the discovery trade-off the "
            "project set out to quantify.",
            "",
            "Take-away for the three-model comparison",
            "  The three models span a clean trade-off:",
            "    Popularity -- high diversity of a sort (cross-genre top "
            "hits) but near-zero coverage and the lowest novelty.",
            "    CF (ALS)   -- leverages co-listening to personalise "
            "broadly; typically the best accuracy, with strong coverage "
            "and novelty gains. Diversity can move either way.",
            "    Content    -- the natural \"long-tail tag neighbourhood\" "
            "model. Usually the highest novelty but often the LOWEST "
            "intra-list diversity (the lists are by design similar to "
            "each other in tag space). Whether it is a useful accuracy "
            "model depends on how well tags align with latent user "
            "taste on this dataset.",
            "",
            "The numeric direction of each delta above tells us which of "
            "these roles each model actually plays on KGRec-music at our "
            "chosen hyperparameters.",
            "",
        ])
    elif pop_row is not None or cf_row is not None:
        lines.append(
            "Only a partial baseline set is available. Run "
            "run_popularity_eval.py AND run_cf_eval.py first to get the "
            "full three-way comparison."
        )
        lines.append("")
    else:
        lines.append(
            "No baseline results available for comparison. Run "
            "run_popularity_eval.py and run_cf_eval.py first."
        )
        lines.append("")

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
    # 2. L2-normalise the tag TF-IDF matrix. This is the feature
    #    space CB scores in AND the space the diversity metric
    #    measures in.
    # --------------------------------------------------------------
    item_vectors = normalize(arts.tfidf_matrix, norm="l2")
    item_index = list(arts.tfidf_item_index)
    log.info(
        "Content feature space: shape=%s, nnz=%d, zero-vector items=%d",
        item_vectors.shape, item_vectors.nnz,
        int((item_vectors.getnnz(axis=1) == 0).sum()),
    )

    # --------------------------------------------------------------
    # 3. Fit the content-based recommender.
    # --------------------------------------------------------------
    model = ContentBasedRecommender(
        item_vectors=item_vectors,
        item_index=item_index,
        feature_mode=config.CB_FEATURE_MODE,
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
    csv_path = config.RESULTS_DIR / "content_based_val.csv"
    json_path = config.RESULTS_DIR / "content_based_val.json"
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, default=float)
    log.info("Saved results -> %s", csv_path)
    log.info("Saved results -> %s", json_path)

    # --------------------------------------------------------------
    # 6. Summary + three-way interpretation.
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

    pop_row = _load_json(POP_RESULTS_FILE)
    cf_row = _load_json(CF_RESULTS_FILE)
    n_train_users = arts.train["user_id_raw"].nunique()
    n_catalogue = arts.train["item_id_raw"].nunique()
    print(_interpret(row, pop_row, cf_row, n_train_users, n_catalogue))

    log.info("Done in %.2fs", time.perf_counter() - t0)
    return results_df


if __name__ == "__main__":
    main()
