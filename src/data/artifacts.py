"""
Loader for preprocessing artefacts.

Single entry point (``load_processed_artifacts``) that the recommender
and evaluator modules call.  Centralising the loading logic here means
downstream code never touches the raw file paths directly, so the
output contract of the preprocessor is enforced by one module.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass

import pandas as pd
import scipy.sparse as sp

import config
from src.data.preprocessor import IDMapper   # noqa: F401  (pickle re-import)

log = logging.getLogger(__name__)


@dataclass
class ProcessedArtifacts:
    """Container for everything the downstream pipeline needs."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    item_features: pd.DataFrame

    user_mapper: IDMapper
    item_mapper: IDMapper

    tfidf_matrix: sp.csr_matrix
    tfidf_item_index: list[str]
    tfidf_vectorizer: object   # sklearn TfidfVectorizer; typed as object to
                                # avoid a hard import dependency here.

    summary: dict


def load_processed_artifacts() -> ProcessedArtifacts:
    """Load all preprocessing outputs from ``config.SPLITS_DIR``.

    Raises
    ------
    FileNotFoundError
        If any expected artefact is missing.  Run the preprocessor first:
        ``python -m src.data.preprocessor``.
    """
    required = [
        config.TRAIN_FILE,
        config.VAL_FILE,
        config.TEST_FILE,
        config.ITEM_FEATURES_FILE,
        config.ID_MAPS_FILE,
        config.TFIDF_MATRIX_FILE,
        config.TFIDF_VECTORIZER_FILE,
        config.TFIDF_ITEM_INDEX_FILE,
        config.PREPROCESSING_SUMMARY_FILE,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing artefacts: "
            + ", ".join(str(p) for p in missing)
            + "\nRun `python -m src.data.preprocessor` to generate them."
        )

    log.info("Loading processed artefacts from %s", config.SPLITS_DIR)

    train = pd.read_parquet(config.TRAIN_FILE)
    val = pd.read_parquet(config.VAL_FILE)
    test = pd.read_parquet(config.TEST_FILE)
    item_features = pd.read_parquet(config.ITEM_FEATURES_FILE)

    with open(config.ID_MAPS_FILE, "rb") as f:
        maps = pickle.load(f)

    tfidf_matrix = sp.load_npz(config.TFIDF_MATRIX_FILE)
    with open(config.TFIDF_VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    with open(config.TFIDF_ITEM_INDEX_FILE, "r", encoding="utf-8") as f:
        tfidf_item_index = json.load(f)

    with open(config.PREPROCESSING_SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return ProcessedArtifacts(
        train=train,
        val=val,
        test=test,
        item_features=item_features,
        user_mapper=maps["user_mapper"],
        item_mapper=maps["item_mapper"],
        tfidf_matrix=tfidf_matrix,
        tfidf_item_index=tfidf_item_index,
        tfidf_vectorizer=vectorizer,
        summary=summary,
    )
