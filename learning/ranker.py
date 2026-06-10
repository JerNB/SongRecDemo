"""LearnedRanker -- the lightweight P4 shadow ranking model.

First version is a plain, explainable classifier (LogisticRegression by
default) wrapped in a StandardScaler pipeline. It is trained on the weak
labels from :mod:`SongRecDemo.learning.dataset` and outputs a ``learned_score``
in [0, 1] -- the model's estimated probability that a user would engage with a
card.

Why logistic regression first
------------------------------
* Interpretable: per-feature coefficients show what the model learned.
* Stable + fast on the small feedback logs we start with.
* ``predict_proba`` already lives in [0, 1], so the learned_score is naturally
  normalised without any extra calibration.

Tree backends (HistGradientBoosting / RandomForest) are supported for later
experiments but are NOT the recommended first model.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union

import joblib

from .dataset import feature_vector

log = logging.getLogger(__name__)

_MODEL_FORMAT_VERSION = 1


class LearnedRanker:
    """A trained (or trainable) learned ranker that emits a learned_score."""

    def __init__(
        self,
        *,
        model_type: str = "logistic",
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self.model_type = str(model_type or "logistic").strip().lower()
        self.feature_names: list[str] = list(feature_names or [])
        self._pipeline: Any = None
        self._positive_label: int = 1
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Construction of the underlying estimator
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Any:
        from sklearn.pipeline import Pipeline

        if self.model_type in {"logistic", "logistic_regression", "lr"}:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight=None)),
            ])
        if self.model_type in {"hgb", "hist_gradient_boosting",
                               "histgradientboosting"}:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return Pipeline([
                ("clf", HistGradientBoostingClassifier()),
            ])
        if self.model_type in {"rf", "random_forest", "randomforest"}:
            from sklearn.ensemble import RandomForestClassifier
            return Pipeline([
                ("clf", RandomForestClassifier(n_estimators=200, random_state=0)),
            ])
        raise ValueError(f"unknown model_type {self.model_type!r}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "LearnedRanker":
        """Fit the model. Requires both classes to be present in ``y``."""
        import numpy as np

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        classes = set(int(v) for v in np.unique(y_arr))
        if len(classes) < 2:
            raise ValueError(
                f"need at least two classes to train, got {sorted(classes)}"
            )

        self._pipeline = self._build_pipeline()
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = np.asarray(sample_weight, dtype=float)
        self._pipeline.fit(X_arr, y_arr, **fit_kwargs)

        clf = self._pipeline.named_steps["clf"]
        # Index of the positive class (label 1) inside predict_proba columns.
        cls_list = list(int(c) for c in clf.classes_)
        self._positive_label = cls_list.index(1) if 1 in cls_list else len(cls_list) - 1
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted and self._pipeline is not None

    def _vectorize(self, features: Union[dict, Sequence[float]]) -> list[float]:
        if isinstance(features, dict):
            if not self.feature_names:
                raise ValueError("feature_names unknown; cannot vectorise a dict")
            return feature_vector(features, self.feature_names)
        return [float(v) for v in features]

    def predict_score(self, features: Union[dict, Sequence[float]]) -> float:
        """Return the learned_score (probability of engagement) in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("LearnedRanker is not fitted")
        import numpy as np

        vec = np.asarray([self._vectorize(features)], dtype=float)
        proba = self._pipeline.predict_proba(vec)[0]
        score = float(proba[self._positive_label])
        return max(0.0, min(1.0, score))

    def predict_scores(
        self, rows: Sequence[Union[dict, Sequence[float]]]
    ) -> list[float]:
        if not self.is_fitted:
            raise RuntimeError("LearnedRanker is not fitted")
        import numpy as np

        if not rows:
            return []
        mat = np.asarray([self._vectorize(r) for r in rows], dtype=float)
        proba = self._pipeline.predict_proba(mat)[:, self._positive_label]
        return [max(0.0, min(1.0, float(p))) for p in proba]

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[str, float]:
        """Per-feature coefficients (logistic) or importances (trees)."""
        if not self.is_fitted:
            return {}
        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coefs = list(clf.coef_[0])
            return {n: float(c) for n, c in zip(self.feature_names, coefs)}
        if hasattr(clf, "feature_importances_"):
            imps = list(clf.feature_importances_)
            return {n: float(c) for n, c in zip(self.feature_names, imps)}
        return {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Any) -> None:
        from pathlib import Path as _Path

        p = _Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("learned_ranker.save: could not create dir for %s: %s", p, exc)
        joblib.dump({
            "format_version": _MODEL_FORMAT_VERSION,
            "model_type": self.model_type,
            "feature_names": list(self.feature_names),
            "positive_label": int(self._positive_label),
            "pipeline": self._pipeline,
        }, str(p))

    @classmethod
    def load(cls, path: Any) -> "LearnedRanker":
        blob = joblib.load(str(path))
        ranker = cls(
            model_type=str(blob.get("model_type") or "logistic"),
            feature_names=list(blob.get("feature_names") or []),
        )
        ranker._pipeline = blob.get("pipeline")
        ranker._positive_label = int(blob.get("positive_label", 1))
        ranker._fitted = ranker._pipeline is not None
        return ranker
