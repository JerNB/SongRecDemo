"""
Raw data loader for KGRec-music.

Responsibilities
----------------
- Read implicit_lf_dataset.csv and return a validated DataFrame.
- Read descriptions/{item_id}.txt files into a dict keyed by raw item id.
- Read tags/{item_id}.txt files into a dict keyed by raw item id.
- Emit clear, actionable warnings for any data-quality issues found during
  audit (duplicate descriptions, missing tag files) so that downstream
  stages do not silently swallow them.

This module is intentionally read-only: it never modifies the raw files.
All returned objects use raw dataset IDs (strings), not remapped indices.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

_INTERACTIONS_COLUMNS = ["user_id_raw", "item_id_raw", "implicit_flag"]


def load_interactions(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the implicit feedback TSV and return a validated DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id_raw`` (str), ``item_id_raw`` (str),
        ``implicit_flag`` (int8).  All values are guaranteed to be
        numeric strings; malformed rows are dropped and counted.

    Notes
    -----
    - No header in the raw file; columns are positional.
    - All 751,531 rows in the audited dataset have exactly 3 tab-separated
      fields and flag value ``1``.  The loader validates this assumption
      and warns loudly if it changes.
    - Duplicate (user, item) pairs are dropped (keep first) with a warning
      because the audit found 0 such pairs; any future reappearance signals
      a data integrity issue rather than repeated listens.
    """
    csv_path = path or config.INTERACTIONS_CSV
    log.info("Loading interactions from %s", csv_path)

    df = pd.read_csv(
        csv_path,
        sep="\t",
        header=None,
        names=_INTERACTIONS_COLUMNS,
        dtype={"user_id_raw": str, "item_id_raw": str, "implicit_flag": "Int8"},
    )

    original_len = len(df)

    # Drop rows with non-numeric IDs
    numeric_mask = (
        df["user_id_raw"].str.isdigit() & df["item_id_raw"].str.isdigit()
    )
    bad = (~numeric_mask).sum()
    if bad:
        warnings.warn(
            f"{bad} rows have non-numeric user_id or item_id and will be dropped.",
            stacklevel=2,
        )
        df = df[numeric_mask].copy()

    # Validate implicit flag – must always be 1 in this dataset
    unexpected_flags = df["implicit_flag"].dropna().unique()
    unexpected_flags = [f for f in unexpected_flags if f != 1]
    if unexpected_flags:
        warnings.warn(
            f"Unexpected implicit_flag values found: {unexpected_flags}. "
            "This dataset is expected to contain only 1s.",
            stacklevel=2,
        )

    # Drop duplicate (user, item) pairs
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["user_id_raw", "item_id_raw"], keep="first")
    dup_count = before_dedup - len(df)
    if dup_count:
        warnings.warn(
            f"{dup_count} duplicate (user, item) pairs found and removed. "
            "The audit expected 0; investigate the source file.",
            stacklevel=2,
        )

    df = df.reset_index(drop=True)
    log.info(
        "Loaded %d valid interactions (%d dropped) | %d users | %d items",
        len(df),
        original_len - len(df),
        df["user_id_raw"].nunique(),
        df["item_id_raw"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------

def load_descriptions(
    item_ids: Optional[list[str]] = None,
    desc_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Load description text for each item, keyed by raw item_id string.

    Parameters
    ----------
    item_ids:
        If provided, load only these item ids.  If None, load all *.txt
        files in the description directory.
    desc_dir:
        Override default description directory (mainly for tests).

    Returns
    -------
    dict[str, str]
        ``{item_id_raw: description_text}``.  Items with no file receive an
        empty string (logged at DEBUG level).

    Notes
    -----
    - The audit found exactly one pair of item ids with byte-identical
      description text (2028, 3130).  Both are loaded normally; the
      preprocessor records this pair in the item-features table's
      ``is_desc_duplicate`` column so downstream models can inspect it.
    """
    base = desc_dir or config.DESC_DIR
    log.info("Loading descriptions from %s", base)

    if item_ids is None:
        paths = list(base.glob("*.txt"))
        ids_to_load = [p.stem for p in paths]
    else:
        ids_to_load = item_ids

    descriptions: dict[str, str] = {}
    missing: list[str] = []
    hash_map: dict[str, list[str]] = {}  # sha256 -> list of item ids with that body

    for iid in ids_to_load:
        p = base / f"{iid}.txt"
        if not p.exists():
            missing.append(iid)
            descriptions[iid] = ""
            continue
        body = p.read_text(encoding="utf-8", errors="replace").strip()
        descriptions[iid] = body

        h = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
        hash_map.setdefault(h, []).append(iid)

    if missing:
        log.debug("%d items had no description file: %s …", len(missing), missing[:5])

    # Warn about byte-identical descriptions across distinct item IDs
    dup_groups = {h: ids for h, ids in hash_map.items() if len(ids) > 1}
    if dup_groups:
        example = next(iter(dup_groups.values()))
        warnings.warn(
            f"{len(dup_groups)} groups of items share byte-identical descriptions "
            f"(e.g. item_ids {sorted(example, key=int)[:6]}). "
            "Verify whether these are truly distinct tracks.",
            stacklevel=2,
        )

    log.info("Descriptions loaded: %d items", len(descriptions))
    return descriptions


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def load_tags(
    item_ids: Optional[list[str]] = None,
    tag_dir: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Load space-separated Last.fm tags for each item.

    Returns
    -------
    dict[str, list[str]]
        ``{item_id_raw: [tag1, tag2, ...]}``.  Items with no tag file
        receive an empty list.  The caller (preprocessor) decides how
        to handle the empty-list case; this loader does not impute.

    Notes
    -----
    - The audit found 401 items with no tag file.  These will appear as
      empty lists here.  The ``MISSING_TAG_STRATEGY`` in config controls
      how the feature builder handles them.
    - Raw tags are NOT normalised here; normalisation (lower-casing,
      punctuation stripping, hyphen collapsing) is done in the preprocessor
      to keep the loader side-effect-free.
    """
    base = tag_dir or config.TAG_DIR
    log.info("Loading tags from %s", base)

    if item_ids is None:
        paths = list(base.glob("*.txt"))
        ids_to_load = [p.stem for p in paths]
    else:
        ids_to_load = item_ids

    tags: dict[str, list[str]] = {}
    missing: list[str] = []

    for iid in ids_to_load:
        p = base / f"{iid}.txt"
        if not p.exists():
            missing.append(iid)
            tags[iid] = []
            continue
        raw = p.read_text(encoding="utf-8", errors="replace").strip()
        tags[iid] = raw.split() if raw else []

    if missing:
        log.info(
            "%d / %d items have no tag file (will use empty tag list).",
            len(missing),
            len(ids_to_load),
        )

    return tags


# ---------------------------------------------------------------------------
# Convenience: load all three at once
# ---------------------------------------------------------------------------

def load_all(
    interactions_path: Optional[Path] = None,
    desc_dir: Optional[Path] = None,
    tag_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    """Load interactions, descriptions, and tags in one call.

    Returns
    -------
    interactions : pd.DataFrame
    descriptions : dict[str, str]
    tags         : dict[str, list[str]]
    """
    interactions = load_interactions(interactions_path)
    item_ids = interactions["item_id_raw"].unique().tolist()
    descriptions = load_descriptions(item_ids, desc_dir)
    tags = load_tags(item_ids, tag_dir)
    return interactions, descriptions, tags
