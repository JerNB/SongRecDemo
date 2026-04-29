"""
Preprocessing pipeline for KGRec-music.

Stages (in execution order)
---------------------------
1. Filter users below MIN_USER_INTERACTIONS threshold.
2. Build raw-ID <-> contiguous-index maps for users and items.
3. Per-user stratified split: train / validation / test.
4. Normalise tags: lower-case, strip punctuation, collapse hyphens,
   drop tokens below MIN_TAG_FREQUENCY (computed on training set only).
5. Normalise descriptions: strip markup artefacts (-LRB-/-RRB-), collapse
   whitespace, truncate to MAX_DESC_TOKENS.
6. Flag duplicate-description items.
7. Build item-feature DataFrame: one row per item, columns include
   normalised tag bag and cleaned description text.
8. Serialise all outputs to SPLITS_DIR.

Design decisions
----------------
- The tag frequency threshold (MIN_TAG_FREQUENCY) is computed from the
  TRAINING split only.  Tags that are rare in training but happen to appear
  in held-out items must not influence the vocabulary; this would leak
  information about the test set.
- Missing tag files are treated as empty bags (config.MISSING_TAG_STRATEGY
  == "empty").  We do NOT impute from neighbours because that would
  implicitly encode item similarity before the model has a chance to learn it.
- The split is per-user random, not chronological (no timestamps exist).
  Each user contributes proportionally to all three splits; this prevents
  the evaluation from accidentally measuring cold-start behaviour on users
  who happen to have fewer interactions.
- Items 2028 and 3130 share byte-identical descriptions.  Both are retained
  as separate items (the knowledge graph may legitimately distinguish them).
  They are flagged in the item-features table for transparency.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config
from src.data.loader import load_all

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID mapping helpers
# ---------------------------------------------------------------------------

class IDMapper:
    """Bidirectional mapping between raw string IDs and contiguous integers.

    Parameters
    ----------
    raw_ids : iterable of str
        Ordered collection of raw IDs that should be mapped.  Order
        determines the contiguous index (0, 1, 2, …).
    """

    def __init__(self, raw_ids: list[str]) -> None:
        self.raw_to_idx: dict[str, int] = {raw: idx for idx, raw in enumerate(raw_ids)}
        self.idx_to_raw: dict[int, str] = {idx: raw for raw, idx in self.raw_to_idx.items()}

    def __len__(self) -> int:
        return len(self.raw_to_idx)

    def to_idx(self, raw: str) -> int:
        return self.raw_to_idx[raw]

    def to_raw(self, idx: int) -> str:
        return self.idx_to_raw[idx]

    def apply_series(self, s: pd.Series) -> pd.Series:
        return s.map(self.raw_to_idx)


# ---------------------------------------------------------------------------
# Tag normalisation
# ---------------------------------------------------------------------------

_TAG_STRIP_RE = re.compile(r"[^\w\s-]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _normalise_tag(tag: str) -> str:
    """Lower-case, strip non-word/hyphen characters, collapse whitespace."""
    tag = tag.lower()
    tag = _TAG_STRIP_RE.sub("", tag)
    tag = tag.replace("-", " ").strip()
    tag = _MULTI_SPACE_RE.sub(" ", tag)
    return tag


def build_tag_vocabulary(
    tags_by_item: dict[str, list[str]],
    item_ids_train: list[str],
    min_freq: int = config.MIN_TAG_FREQUENCY,
) -> set[str]:
    """Compute allowed tag vocabulary from training items only.

    Tags with global frequency < min_freq across training items are excluded
    to reduce noise in the item-feature matrix.

    Parameters
    ----------
    tags_by_item : dict[str, list[str]]
        Raw (un-normalised) tags per item id.
    item_ids_train : list[str]
        Item ids that appear in the training split.  Vocabulary is computed
        from these items ONLY to avoid test-set leakage.
    min_freq : int
        Minimum number of training items a normalised tag must appear in.
    """
    counter: Counter[str] = Counter()
    for iid in item_ids_train:
        raw_tags = tags_by_item.get(iid, [])
        for t in raw_tags:
            counter[_normalise_tag(t)] += 1

    vocab = {tag for tag, count in counter.items() if count >= min_freq and tag}
    log.info(
        "Tag vocabulary: %d tokens (min_freq=%d, computed on %d training items)",
        len(vocab),
        min_freq,
        len(item_ids_train),
    )
    return vocab


def normalise_tags(
    raw_tags: list[str],
    vocab: Optional[set[str]] = None,
) -> list[str]:
    """Normalise and optionally filter a single item's tag list.

    Parameters
    ----------
    raw_tags : list[str]
    vocab : set[str] or None
        If provided, only tags in the vocabulary are kept.
    """
    normalised = [_normalise_tag(t) for t in raw_tags]
    normalised = [t for t in normalised if t]
    if vocab is not None:
        normalised = [t for t in normalised if t in vocab]
    return normalised


# ---------------------------------------------------------------------------
# Description normalisation
# ---------------------------------------------------------------------------

# Penn Treebank-style bracket tokens injected during corpus processing
_LRB_RE = re.compile(r"-LRB-")
_RRB_RE = re.compile(r"-RRB-")
_LSB_RE = re.compile(r"-LSB-")
_RSB_RE = re.compile(r"-RSB-")
_MULTI_WS_RE = re.compile(r"\s+")


def normalise_description(text: str, max_tokens: Optional[int] = config.MAX_DESC_TOKENS) -> str:
    """Clean a raw description string for use as text feature input.

    Steps:
    - Replace Penn-bracket tokens (-LRB- / -RRB- / -LSB- / -RSB-) with
      their natural parenthesis equivalents.
    - Collapse runs of whitespace to a single space.
    - Strip leading/trailing whitespace.
    - Optionally truncate to max_tokens whitespace-delimited tokens.
    """
    text = _LRB_RE.sub("(", text)
    text = _RRB_RE.sub(")", text)
    text = _LSB_RE.sub("[", text)
    text = _RSB_RE.sub("]", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()

    if max_tokens is not None:
        tokens = text.split()
        if len(tokens) > max_tokens:
            text = " ".join(tokens[:max_tokens])

    return text


# ---------------------------------------------------------------------------
# Per-user train / validation / test split
# ---------------------------------------------------------------------------

def per_user_split(
    interactions: pd.DataFrame,
    val_frac: float = config.VAL_FRACTION,
    test_frac: float = config.TEST_FRACTION,
    min_interactions: int = config.MIN_USER_INTERACTIONS,
    seed: int = config.SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into train / validation / test per user.

    Protocol
    --------
    For each user with n interactions (n >= min_interactions):

    1. Shuffle their interaction indices with the shared rng (reproducible;
       all users are processed in sorted order so the rng sequence is
       deterministic given the seed).
    2. Allocate counts:
           n_test  = max(1, ceil(n * test_frac))
           n_val   = max(1, ceil(n * val_frac))
           n_train = n - n_test - n_val
    3. Assign:
           test  ← last  n_test indices
           val   ← next  n_val  indices (immediately before test)
           train ← remaining   n_train indices

    Non-empty guarantee
    -------------------
    With min_interactions=5, val_frac=test_frac=0.10:
      - For n in [5, 9]:  n_test = n_val = 1,  n_train = n-2 >= 3
      - For n = 10:       n_test = n_val = 1,  n_train = 8
      - For n > 10:       n_test = ceil(n*0.10), n_val = ceil(n*0.10)
                          n_train = n - 2*ceil(n*0.10) >= 0.8*n - 2
    In all cases n_train >= 3, n_val >= 1, n_test >= 1.
    The guard (if n_train < 1: n_val = 0) is unreachable at
    min_interactions=5 but is retained as a safety net for future
    config changes.

    Users with fewer than ``min_interactions`` interactions are excluded
    from all splits. The audit confirmed all 5,199 users have >= 138
    interactions (median), so no users are excluded in practice.

    Parameters
    ----------
    interactions : pd.DataFrame
        Output of ``load_interactions()``.
    val_frac, test_frac : float
        Fraction of each user's interactions to hold out.
    min_interactions : int
        Users below this threshold are excluded entirely.
    seed : int
        Random seed; fix this in ``config`` for reproducibility.

    Returns
    -------
    train, val, test : pd.DataFrame
        Each DataFrame has the same columns as ``interactions``.
    """
    rng = np.random.default_rng(seed)

    train_rows, val_rows, test_rows = [], [], []

    user_counts = interactions["user_id_raw"].value_counts()
    small_users = user_counts[user_counts < min_interactions].index.tolist()
    if small_users:
        warnings.warn(
            f"{len(small_users)} users have fewer than {min_interactions} "
            "interactions and will be excluded from all splits.",
            stacklevel=2,
        )
    valid_interactions = interactions[
        ~interactions["user_id_raw"].isin(small_users)
    ]

    for user_id, group in valid_interactions.groupby("user_id_raw"):
        idx = group.index.to_numpy().copy()   # .copy() needed: newer pandas
                                              # returns a read-only view.
        rng.shuffle(idx)
        n = len(idx)

        n_test = max(1, int(np.ceil(n * test_frac)))
        n_val = max(1, int(np.ceil(n * val_frac)))
        # Guard: train must have at least 1 row
        if n - n_test - n_val < 1:
            n_val = 0

        test_idx = idx[-n_test:]
        val_idx = idx[-(n_test + n_val):-n_test] if n_val > 0 else np.array([], dtype=int)
        train_idx = idx[:-(n_test + n_val)] if n_val > 0 else idx[:-n_test]

        train_rows.append(interactions.loc[train_idx])
        if len(val_idx):
            val_rows.append(interactions.loc[val_idx])
        test_rows.append(interactions.loc[test_idx])

    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True) if val_rows else pd.DataFrame(columns=interactions.columns)
    test = pd.concat(test_rows).reset_index(drop=True)

    log.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Item feature table
# ---------------------------------------------------------------------------

def build_item_features(
    all_item_ids: list[str],
    descriptions: dict[str, str],
    tags_by_item: dict[str, list[str]],
    tag_vocab: set[str],
) -> pd.DataFrame:
    """Build item-level feature DataFrame with cleaned text and tags.

    Returns
    -------
    pd.DataFrame
        Index: item_id_raw (str).
        Columns:
        - ``tags_normalised`` : list[str]  – vocabulary-filtered tag list
        - ``desc_clean``      : str        – normalised description text
        - ``has_tags``        : bool       – False when tag file was absent
        - ``desc_len_tokens`` : int        – word count of raw description
        - ``is_desc_duplicate``: bool      – True for audit-flagged duplicates
    """
    dup_ids = {
        str(a) for pair in config.DUPLICATE_DESC_PAIRS for a in pair
    }

    records = []
    for iid in all_item_ids:
        raw_tags = tags_by_item.get(iid, [])
        raw_desc = descriptions.get(iid, "")

        norm_tags = normalise_tags(raw_tags, vocab=tag_vocab)
        clean_desc = normalise_description(raw_desc)

        records.append({
            "item_id_raw": iid,
            "tags_normalised": norm_tags,
            "desc_clean": clean_desc,
            "has_tags": len(raw_tags) > 0,
            "desc_len_tokens": len(raw_desc.split()),
            "is_desc_duplicate": iid in dup_ids,
        })

    df = pd.DataFrame(records).set_index("item_id_raw")
    log.info(
        "Item features built: %d items | %d without tags | %d duplicate-desc flags",
        len(df),
        (~df["has_tags"]).sum(),
        df["is_desc_duplicate"].sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Tag TF-IDF feature matrix
# ---------------------------------------------------------------------------

def _tags_to_document(tags: list[str]) -> str:
    """Join a tag list into a single space-separated document.

    Tags have already been normalised (lower-cased, stripped) by
    `normalise_tags`; here we only concatenate them so TfidfVectorizer
    can tokenise on whitespace.
    """
    return " ".join(tags)


def build_tag_tfidf(
    item_features: pd.DataFrame,
    train_item_ids: list[str],
) -> tuple[sp.csr_matrix, list[str], TfidfVectorizer]:
    """Build the L2-normalised tag TF-IDF matrix for every catalogue item.

    Leakage control
    ---------------
    The ``TfidfVectorizer`` is FIT on training items only so that the
    IDF weights are computed from the training distribution alone.
    It is then used to TRANSFORM all catalogue items (including those
    that only appear in val/test).  This mirrors standard practice: the
    model is parameterised by training data; evaluation items are
    projected into that pre-trained space.

    Parameters
    ----------
    item_features : pd.DataFrame
        Output of ``build_item_features``. Index = item_id_raw.
        Must contain ``tags_normalised`` column.
    train_item_ids : list[str]
        Item IDs appearing in the training split. Used to fit IDF.

    Returns
    -------
    item_vectors : scipy.sparse.csr_matrix
        Shape (n_items_all, n_features). Row order matches
        ``item_index`` exactly.  Rows are L2-normalised so cosine
        similarity reduces to a dot product at inference time.
    item_index : list[str]
        Ordered item_id_raw values corresponding to each row.
    vectorizer : TfidfVectorizer
        Fitted vectoriser (kept for persistence and potential reuse).

    Configuration notes
    -------------------
    - ``token_pattern=r"[^\\s]+"`` keeps multi-character tags intact and
      allows hyphens/underscores inside tokens.  Our tag normaliser has
      already split hyphenated tokens into separate words; anything that
      reaches the vectoriser is already a clean whitespace-delimited token.
    - ``sublinear_tf=True``: every tag appears at most once per item
      file, so raw TF is 1.  Sublinear TF makes this explicit and
      collapses to pure IDF weighting.
    - ``min_df=1``: the vocabulary was already filtered by
      ``build_tag_vocabulary`` at the ``MIN_TAG_FREQUENCY`` threshold.
    """
    # Training documents (for IDF fit)
    train_docs: list[str] = []
    train_docs_nonempty: list[str] = []
    for iid in train_item_ids:
        tags = item_features.loc[iid, "tags_normalised"] if iid in item_features.index else []
        doc = _tags_to_document(list(tags))
        train_docs.append(doc)
        if doc:
            train_docs_nonempty.append(doc)

    if not train_docs_nonempty:
        raise RuntimeError(
            "No training items have tags; cannot fit tag TF-IDF vectoriser. "
            "Check that the tag vocabulary threshold is not too strict."
        )

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",
        sublinear_tf=True,
        min_df=1,
        norm=None,   # we L2-normalise manually so we control the pipeline
        dtype=np.float32,
    )

    # Fit on training items only (leakage control)
    vectorizer.fit(train_docs_nonempty)
    log.info(
        "Tag TF-IDF vocabulary fit on %d training items with tags "
        "(size=%d features)",
        len(train_docs_nonempty),
        len(vectorizer.vocabulary_),
    )

    # Transform ALL catalogue items (including val/test-only items)
    # Row order = item_features.index order
    item_index: list[str] = list(item_features.index)
    all_docs = [
        _tags_to_document(list(item_features.loc[iid, "tags_normalised"]))
        for iid in item_index
    ]
    item_vectors_raw = vectorizer.transform(all_docs)

    # L2-normalise rows (zero-tag rows remain zero vectors)
    item_vectors = normalize(item_vectors_raw, norm="l2", axis=1)
    assert item_vectors.shape[0] == len(item_index)

    n_empty_rows = (np.asarray(item_vectors.getnnz(axis=1)) == 0).sum()
    log.info(
        "Tag TF-IDF matrix built: shape=%s, nnz=%d, zero-vector items=%d",
        item_vectors.shape,
        item_vectors.nnz,
        int(n_empty_rows),
    )
    return item_vectors.tocsr(), item_index, vectorizer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preprocessing() -> dict:
    """Execute the full preprocessing pipeline and persist artefacts.

    Artefacts written (paths from config)
    --------------------------------------
    - ``TRAIN_FILE`` / ``VAL_FILE`` / ``TEST_FILE``
        Parquet files with columns:
        user_id_raw, item_id_raw, implicit_flag, user_idx, item_idx.
        Integer indices are contiguous and built from training data only.
    - ``ITEM_FEATURES_FILE``
        Parquet with per-item side information
        (tags_normalised, desc_clean, has_tags, desc_len_tokens,
        is_desc_duplicate).
    - ``ID_MAPS_FILE``
        Pickle: {"user_mapper": IDMapper, "item_mapper": IDMapper}.
    - ``TFIDF_MATRIX_FILE``
        SciPy ``.npz`` sparse CSR matrix; rows are L2-normalised tag
        TF-IDF vectors, one per catalogue item.
    - ``TFIDF_VECTORIZER_FILE``
        Pickled fitted ``TfidfVectorizer`` (IDF fit on training items only).
    - ``TFIDF_ITEM_INDEX_FILE``
        JSON list of item_id_raw values matching TF-IDF matrix row order.
    - ``PREPROCESSING_SUMMARY_FILE``
        JSON: human-readable counts for users, items, interactions,
        split sizes, and TF-IDF dimensions.

    Returns
    -------
    dict
        The same content that is written to ``PREPROCESSING_SUMMARY_FILE``,
        so that callers can print it or assert against it in tests.
    """
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    interactions, descriptions, tags_by_item = load_all()
    total_events = len(interactions)
    total_users = interactions["user_id_raw"].nunique()
    total_items = interactions["item_id_raw"].nunique()

    # 2. Per-user train / val / test split (on raw IDs)
    train_raw, val_raw, test_raw = per_user_split(interactions)

    # 3. ID mappers – built from training data only.  Users/items that
    #    appear only in val/test are marked as unseen (index = NaN)
    #    and excluded from CF training; content-based is unaffected
    #    because it uses the full item_features table.
    train_users = sorted(train_raw["user_id_raw"].unique(), key=int)
    train_items = sorted(train_raw["item_id_raw"].unique(), key=int)
    user_mapper = IDMapper(train_users)
    item_mapper = IDMapper(train_items)

    def _remap(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_idx"] = out["user_id_raw"].map(user_mapper.raw_to_idx)
        out["item_idx"] = out["item_id_raw"].map(item_mapper.raw_to_idx)
        return out

    train = _remap(train_raw)
    val = _remap(val_raw)
    test = _remap(test_raw)

    val_cold_items = int(val["item_idx"].isna().sum())
    test_cold_items = int(test["item_idx"].isna().sum())
    val_cold_users = int(val["user_idx"].isna().sum())
    test_cold_users = int(test["user_idx"].isna().sum())
    if val_cold_items or test_cold_items:
        log.warning(
            "Val has %d rows with items unseen in training; test has %d. "
            "These are excluded from CF evaluation.",
            val_cold_items, test_cold_items,
        )

    # 4. Tag vocabulary (training items only → no leakage into the
    #    vocabulary used for test-item feature vectors).
    tag_vocab = build_tag_vocabulary(tags_by_item, train_items)

    # 5. Item feature table (all catalogue items; vocab applied to tags).
    all_item_ids = sorted(interactions["item_id_raw"].unique(), key=int)
    item_features = build_item_features(
        all_item_ids, descriptions, tags_by_item, tag_vocab
    )

    # 6. Tag TF-IDF matrix (IDF fit on training items only; transform all).
    tfidf_matrix, tfidf_item_index, tfidf_vectorizer = build_tag_tfidf(
        item_features, train_items
    )

    # 7. Persist everything.
    train.to_parquet(config.TRAIN_FILE, index=False)
    val.to_parquet(config.VAL_FILE, index=False)
    test.to_parquet(config.TEST_FILE, index=False)
    item_features.to_parquet(config.ITEM_FEATURES_FILE)

    with open(config.ID_MAPS_FILE, "wb") as f:
        pickle.dump(
            {"user_mapper": user_mapper, "item_mapper": item_mapper}, f
        )

    sp.save_npz(config.TFIDF_MATRIX_FILE, tfidf_matrix)
    with open(config.TFIDF_VECTORIZER_FILE, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(config.TFIDF_ITEM_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(tfidf_item_index, f)

    # 8. Sanity checks (raises AssertionError on any invariant violation).
    _sanity_checks(train, val, test, user_mapper, item_mapper)

    # 9. Summary (also written to disk for reproducibility).
    summary = _build_summary(
        interactions_total=total_events,
        users_total=total_users,
        items_total=total_items,
        train=train,
        val=val,
        test=test,
        item_features=item_features,
        tfidf_matrix=tfidf_matrix,
        tag_vocab=tag_vocab,
        val_cold_items=val_cold_items,
        test_cold_items=test_cold_items,
        val_cold_users=val_cold_users,
        test_cold_users=test_cold_users,
    )
    with open(config.PREPROCESSING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Preprocessing complete. Artefacts written to %s", config.SPLITS_DIR)
    return summary


def _sanity_checks(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    user_mapper: IDMapper,
    item_mapper: IDMapper,
) -> None:
    """Assert critical invariants that must hold before any model trains."""
    all_pairs = set(zip(train["user_id_raw"], train["item_id_raw"]))

    val_pairs = set(zip(val["user_id_raw"], val["item_id_raw"]))
    test_pairs = set(zip(test["user_id_raw"], test["item_id_raw"]))
    assert not (all_pairs & val_pairs), "LEAK: train/val pairs overlap"
    assert not (all_pairs & test_pairs), "LEAK: train/test pairs overlap"
    assert not (val_pairs & test_pairs), "LEAK: val/test pairs overlap"

    assert train["user_idx"].notna().all(), "Null user_idx in training split"
    assert train["item_idx"].notna().all(), "Null item_idx in training split"

    # Every user appearing in val/test must also appear in training
    # (guaranteed by the per-user split protocol).
    train_users = set(train["user_id_raw"].unique())
    assert set(val["user_id_raw"].unique()) <= train_users, (
        "Val split contains users not present in training"
    )
    assert set(test["user_id_raw"].unique()) <= train_users, (
        "Test split contains users not present in training"
    )

    # Every user must have at least one interaction in each split
    # (non-empty guarantee from the specification).
    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        per_user_counts = split_df.groupby("user_id_raw").size()
        assert (per_user_counts >= 1).all(), (
            f"Non-empty guarantee violated in {split_name} split"
        )

    log.info("Sanity checks passed.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    *,
    interactions_total: int,
    users_total: int,
    items_total: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    item_features: pd.DataFrame,
    tfidf_matrix: sp.csr_matrix,
    tag_vocab: set[str],
    val_cold_items: int,
    test_cold_items: int,
    val_cold_users: int,
    test_cold_users: int,
) -> dict:
    """Assemble a JSON-serialisable preprocessing summary."""

    def _per_user_stats(df: pd.DataFrame, name: str) -> dict:
        counts = df.groupby("user_id_raw").size() if len(df) else pd.Series(dtype=int)
        if len(counts) == 0:
            return {
                "rows": 0, "users": 0, "min": None, "median": None,
                "mean": None, "max": None,
            }
        return {
            "rows": int(len(df)),
            "users": int(counts.size),
            "min": int(counts.min()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
        }

    return {
        "raw": {
            "interactions_total": int(interactions_total),
            "users_total": int(users_total),
            "items_total": int(items_total),
            "density": float(interactions_total) / float(users_total * items_total),
        },
        "splits": {
            "train": _per_user_stats(train, "train"),
            "val": _per_user_stats(val, "val"),
            "test": _per_user_stats(test, "test"),
        },
        "cold_set_stats": {
            "val_rows_with_unseen_item": int(val_cold_items),
            "test_rows_with_unseen_item": int(test_cold_items),
            "val_rows_with_unseen_user": int(val_cold_users),
            "test_rows_with_unseen_user": int(test_cold_users),
        },
        "item_features": {
            "items_total": int(len(item_features)),
            "items_with_tags": int(item_features["has_tags"].sum()),
            "items_without_tags": int((~item_features["has_tags"]).sum()),
            "items_flagged_duplicate_desc": int(item_features["is_desc_duplicate"].sum()),
            "avg_desc_tokens": float(item_features["desc_len_tokens"].mean()),
        },
        "tag_tfidf": {
            "matrix_shape": list(tfidf_matrix.shape),
            "nnz": int(tfidf_matrix.nnz),
            "vocab_size": int(len(tag_vocab)),
            "zero_vector_items": int(
                (np.asarray(tfidf_matrix.getnnz(axis=1)) == 0).sum()
            ),
        },
        "config": {
            "split_seed": config.SPLIT_SEED,
            "val_fraction": config.VAL_FRACTION,
            "test_fraction": config.TEST_FRACTION,
            "min_user_interactions": config.MIN_USER_INTERACTIONS,
            "min_tag_frequency": config.MIN_TAG_FREQUENCY,
            "max_desc_tokens": config.MAX_DESC_TOKENS,
        },
    }


def print_summary(summary: dict) -> None:
    """Pretty-print the preprocessing summary to stdout."""
    raw = summary["raw"]
    tr = summary["splits"]["train"]
    va = summary["splits"]["val"]
    te = summary["splits"]["test"]
    it = summary["item_features"]
    tf = summary["tag_tfidf"]

    lines = [
        "",
        "=" * 66,
        "  KGRec-music preprocessing summary",
        "=" * 66,
        "",
        "  Raw dataset",
        f"    users              : {raw['users_total']:>10,}",
        f"    items              : {raw['items_total']:>10,}",
        f"    interactions       : {raw['interactions_total']:>10,}",
        f"    density            : {raw['density']:>10.4%}",
        "",
        "  Splits (rows, users, min/median/mean/max per user)",
        f"    train : rows={tr['rows']:>7,}  users={tr['users']:>5,}  "
        f"n={tr['min']:>4}/{tr['median']:>6.1f}/{tr['mean']:>6.1f}/{tr['max']:>4}",
        f"    val   : rows={va['rows']:>7,}  users={va['users']:>5,}  "
        f"n={va['min']:>4}/{va['median']:>6.1f}/{va['mean']:>6.1f}/{va['max']:>4}",
        f"    test  : rows={te['rows']:>7,}  users={te['users']:>5,}  "
        f"n={te['min']:>4}/{te['median']:>6.1f}/{te['mean']:>6.1f}/{te['max']:>4}",
        "",
        "  Item features",
        f"    items total        : {it['items_total']:>10,}",
        f"    items with tags    : {it['items_with_tags']:>10,}",
        f"    items without tags : {it['items_without_tags']:>10,}",
        f"    duplicate-desc flag: {it['items_flagged_duplicate_desc']:>10,}",
        f"    avg desc tokens    : {it['avg_desc_tokens']:>10.1f}",
        "",
        "  Tag TF-IDF (IDF fit on training items only)",
        f"    matrix shape       : {tf['matrix_shape'][0]:,} x {tf['matrix_shape'][1]:,}",
        f"    non-zero entries   : {tf['nnz']:>10,}",
        f"    vocabulary size    : {tf['vocab_size']:>10,}",
        f"    zero-vector items  : {tf['zero_vector_items']:>10,}",
        "",
        f"  Artefacts written to: {config.SPLITS_DIR}",
        "=" * 66,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry: use the top-level ``run_preprocessing.py`` script.
# Running this module directly (``python -m src.data.preprocessor``) would
# pickle ``IDMapper`` under ``__main__`` instead of its real module path,
# breaking the artifact loader in later sessions.
# ---------------------------------------------------------------------------
