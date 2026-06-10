"""Train the shadow learned ranker from the feedback logs (P4).

Usage
-----
From the project root::

    python -m SongRecDemo.learning.train_ranker
    python -m SongRecDemo.learning.train_ranker --feedback-db data/feedback.sqlite
    python -m SongRecDemo.learning.train_ranker --model-type logistic

What it does
------------
1. Reads ``feedback.sqlite`` and builds (X, y, sample_weight) via
   :mod:`SongRecDemo.learning.dataset`.
2. Fails *soft* when there is not enough data: if fewer than
   ``config.LEARNED_RANKER_MIN_SAMPLES`` samples (or only one class is
   present) it prints an explanation and exits 0 -- it never raises.
3. Trains a LogisticRegression (default) and prints a training summary
   (sample counts, feature names, train AUC if feasible, coefficients).
4. Saves the model to ``data/learned_ranker.joblib`` and the feature schema
   to ``data/learned_ranker_schema.json``.

The trained model is consumed by the recommender in *shadow mode* only -- it
does not change the live P0 ordering.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when run as a script / module.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

from SongRecDemo.pipeline.feedback import FeedbackStore  # noqa: E402

from .dataset import TrainingData, build_training_data  # noqa: E402
from .ranker import LearnedRanker  # noqa: E402

log = logging.getLogger("songrec_train_ranker")


def _train_auc(ranker: LearnedRanker, data: TrainingData) -> Optional[float]:
    """In-sample AUC -- a sanity check only (NOT a generalisation estimate)."""
    try:
        from sklearn.metrics import roc_auc_score

        scores = ranker.predict_scores(data.X)
        return float(roc_auc_score(data.y, scores))
    except Exception as exc:  # noqa: BLE001
        log.debug("train AUC not computed: %s", exc)
        return None


def train_from_store(
    store: FeedbackStore,
    *,
    model_type: str = "logistic",
    min_samples: Optional[int] = None,
    model_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    save: bool = True,
) -> dict[str, Any]:
    """Build data, train, optionally save. Returns a JSON-friendly summary.

    Never raises on too-little data: the returned dict has
    ``trained=False`` and a ``reason`` instead.
    """
    min_samples = int(config.LEARNED_RANKER_MIN_SAMPLES if min_samples is None else min_samples)
    model_path = Path(model_path or config.LEARNED_RANKER_MODEL_PATH)
    schema_path = Path(schema_path or config.LEARNED_RANKER_SCHEMA_PATH)

    data = build_training_data(store)
    summary: dict[str, Any] = dict(data.summary())
    summary["model_type"] = model_type
    summary["trained"] = False

    if data.num_samples < min_samples:
        summary["reason"] = (
            f"not enough training samples: have {data.num_samples}, "
            f"need >= {min_samples}. Collect more feedback first."
        )
        return summary

    num_classes = len(set(data.y))
    if num_classes < 2:
        summary["reason"] = (
            "only one label class present "
            f"({'all positive' if (data.y and data.y[0] == 1) else 'all negative'}); "
            "need both engaged and not-engaged samples to train."
        )
        return summary

    ranker = LearnedRanker(model_type=model_type, feature_names=data.feature_names)
    ranker.fit(data.X, data.y, sample_weight=data.sample_weight)

    summary["trained"] = True
    summary["train_auc"] = _train_auc(ranker, data)
    summary["feature_importance"] = ranker.feature_importance()

    if save:
        ranker.save(model_path)
        schema = {
            "model_type": model_type,
            "feature_names": data.feature_names,
            "num_samples": data.num_samples,
            "positive_count": data.positive_count,
            "negative_count": data.negative_count,
            "weak_negative_count": data.weak_negative_count,
            "pipeline_version": config.PIPELINE_VERSION,
            "ranking_config_version": config.RANKING_CONFIG_VERSION,
        }
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["model_path"] = str(model_path)
        summary["schema_path"] = str(schema_path)

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 64)
    print("Shadow learned ranker -- training summary")
    print("=" * 64)
    print(f"  model_type          : {summary.get('model_type')}")
    print(f"  num_samples         : {summary.get('num_samples')}")
    print(f"  positive_count      : {summary.get('positive_count')}")
    print(f"  negative_count      : {summary.get('negative_count')}")
    print(f"  weak_negative_count : {summary.get('weak_negative_count')}")
    print(f"  num_features        : {summary.get('num_features')}")
    print(f"  feature_names       : {summary.get('feature_names')}")

    if not summary.get("trained"):
        print()
        print(f"  [SKIPPED] {summary.get('reason')}")
        print("  Nothing was saved; this is expected, not an error.")
        return

    auc = summary.get("train_auc")
    print(f"  train_auc           : {auc:.4f}" if isinstance(auc, float) else
          f"  train_auc           : {auc}")
    print("  feature_importance (coef / importance):")
    importance = summary.get("feature_importance") or {}
    for name, val in sorted(importance.items(), key=lambda kv: -abs(kv[1])):
        print(f"      {name:<28} {val:+.4f}")
    print()
    print(f"  saved model  -> {summary.get('model_path')}")
    print(f"  saved schema -> {summary.get('schema_path')}")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feedback-db", default=None,
                   help="Path to feedback.sqlite (default: config.FEEDBACK_STORE_PATH).")
    p.add_argument("--model-type", default="logistic",
                   help="logistic (default) | hgb | rf.")
    p.add_argument("--min-samples", type=int, default=None,
                   help="Override the minimum sample count before training.")
    p.add_argument("--model-out", default=None, help="Override model output path.")
    p.add_argument("--schema-out", default=None, help="Override schema output path.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    db_path = args.feedback_db or config.FEEDBACK_STORE_PATH
    try:
        store = FeedbackStore(db_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIPPED] could not open feedback store at {db_path}: {exc}")
        return 0

    summary = train_from_store(
        store,
        model_type=args.model_type,
        min_samples=args.min_samples,
        model_path=Path(args.model_out) if args.model_out else None,
        schema_path=Path(args.schema_out) if args.schema_out else None,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
