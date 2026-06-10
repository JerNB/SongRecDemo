"""Training data builder for the shadow learned ranker (P4).

Reads the P3 :class:`~SongRecDemo.pipeline.feedback.FeedbackStore` logs and
turns them into supervised samples -- one per *exposed* recommendation card
(``recommendation_item``). The label is a first-version weak-supervision rule
derived from the ``user_feedback`` events recorded against that card.

Why one sample per recommendation_item
--------------------------------------
Returning a card to the client already counts as an exposure (an impression).
So every ``recommendation_item`` row is a candidate the user saw; the feedback
events tell us what they did with it. A card with no feedback is an
``impression-only`` sample: a *weak* negative (low sample_weight), because the
user not clicking does not strongly mean dislike.

Label rule (v1, weak supervision)
----------------------------------
For each (request_id, song_id) we look at all feedback events:

* any ``dislike`` / ``not_interested`` / ``skip``  -> label  -1.0  (negative,
  overrides everything else -- an explicit reject wins)
* else ``like`` / ``add_to_playlist``              -> label  +1.0
* else ``open_netease_url`` / ``play_preview`` /
  ``click``                                        -> label  +0.7
* else ``why_clicked``                             -> label  +0.5
* else (impression only / no event)                -> label   0.0  (weak
  negative, sample_weight = LEARNED_RANKER_WEAK_NEGATIVE_WEIGHT)

The graded positive values pick the *strongest* positive behaviour seen.

For a binary classifier the float label is binarised: ``label > 0`` is the
positive class (engaged), ``label <= 0`` is the negative class (ignored /
rejected). The graded value and the weak-negative weight are carried in
``sample_weight`` so a strong like counts more than a passing click and a mere
impression barely counts at all.

Feature parity
--------------
:func:`extract_feature_dict` is the single source of truth for features. The
same function is reused at *inference* time via :func:`feature_dict_from_card`
so the shadow scorer sees exactly the features the model was trained on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label rules (weak supervision, v1)
# ---------------------------------------------------------------------------

# event_type -> positive intensity in (0, 1]. Higher = stronger positive.
LABEL_RULES: dict[str, float] = {
    "like": 1.0,
    "add_to_playlist": 1.0,
    "open_netease_url": 0.7,
    "play_preview": 0.7,
    "click": 0.7,
    "why_clicked": 0.5,
}

# Events that explicitly reject a card; these override any positive event.
NEGATIVE_EVENTS: frozenset = frozenset({"dislike", "not_interested", "skip"})

# Events that carry no preference signal on their own.
NEUTRAL_EVENTS: frozenset = frozenset({"impression"})


@dataclass
class LabelOutcome:
    label: float            # graded label in [-1.0, 1.0]
    binary: int             # 1 if engaged (label > 0) else 0
    sample_weight: float    # confidence weight
    kind: str               # "positive" | "negative" | "weak_negative"


def label_for_events(event_types: Iterable[str]) -> LabelOutcome:
    """Apply the v1 weak-supervision rule to a bag of event types."""
    types = {str(e or "").strip().lower() for e in event_types}
    types.discard("")

    weak_negative_weight = float(config.LEARNED_RANKER_WEAK_NEGATIVE_WEIGHT)

    if types & NEGATIVE_EVENTS:
        # An explicit reject overrides everything, even a prior like.
        return LabelOutcome(label=-1.0, binary=0, sample_weight=1.0,
                            kind="negative")

    positives = [LABEL_RULES[t] for t in types if t in LABEL_RULES]
    if positives:
        return LabelOutcome(label=max(positives), binary=1, sample_weight=1.0,
                            kind="positive")

    # Impression-only (or only neutral events): a weak negative.
    return LabelOutcome(label=0.0, binary=0, sample_weight=weak_negative_weight,
                        kind="weak_negative")


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Numeric features read straight off the logged recommendation_item row.
_ITEM_NUMERIC_FEATURES: list[str] = [
    "content_score",
    "retrieval_score",
    "multi_source_agreement",
    "quality_score",
    "novelty_score",
    "final_score",
    "rank_score",
    "rank_position",
]

# Numeric features read off the joined recommendation_request row.
_REQUEST_NUMERIC_FEATURES: list[str] = [
    "content_weight",
    "novelty",
    "diversity",
]

# Source-type derived binary flags: feature name -> source_type values that
# set it to 1.0 when present in the card's source_types list.
FEATURE_SOURCE_FLAGS: dict[str, frozenset] = {
    "has_embedding_source": frozenset({"embedding"}),
    "has_artist_source": frozenset({"artist", "artist_context"}),
    "has_genre_source": frozenset({"genre", "genre_mood"}),
    "has_mood_source": frozenset({"mood", "genre_mood"}),
    "has_tag_source": frozenset({"tag", "tag_combo"}),
    "has_seed_song_source": frozenset({"seed_song", "seed_album"}),
    "has_discovery_source": frozenset({"discovery"}),
}

# pick_type is one-hot encoded across the known labels.
PICK_TYPES: list[str] = ["safe", "exploratory", "diverse", "balanced"]

# Optional numeric features included only when at least one sample carries a
# real (non-null) value -- they are not in the P3 item schema yet but may be
# logged later. Keeping them optional keeps the schema honest about what the
# data actually contains.
_OPTIONAL_NUMERIC_FEATURES: list[str] = [
    "popularity_score",
    "artist_authority_score",
    "metadata_quality_score",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _source_types_from_item(item: dict[str, Any]) -> list[str]:
    """Read the source_types list from an item row / card-derived dict."""
    if "source_types" in item and item["source_types"] is not None:
        raw = item["source_types"]
    else:
        raw = item.get("source_types_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except (TypeError, ValueError):
            raw = []
    if not isinstance(raw, (list, tuple, set)):
        return []
    return [str(s).strip().lower() for s in raw if s]


def extract_feature_dict(
    item: dict[str, Any],
    request: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """Compute the full feature dict for one exposed card.

    ``item`` is a ``recommendation_item`` row (or a card-derived dict);
    ``request`` is the joined ``recommendation_request`` row (or None).
    Every always-present feature is emitted; optional features are emitted
    only when the source dict actually carries a non-null value for them.
    """
    request = request or {}
    feats: dict[str, float] = {}

    for name in _ITEM_NUMERIC_FEATURES:
        feats[name] = _as_float(item.get(name))

    for name in _REQUEST_NUMERIC_FEATURES:
        # Fall back to the item dict so an inference-time card (which carries
        # the controls directly) still resolves these.
        val = request.get(name)
        if val is None:
            val = item.get(name)
        feats[name] = _as_float(val)

    source_types = set(_source_types_from_item(item))
    feats["num_source_types"] = float(len(source_types))
    for flag_name, trigger in FEATURE_SOURCE_FLAGS.items():
        feats[flag_name] = 1.0 if (source_types & trigger) else 0.0

    pick_type = str(item.get("pick_type") or "").strip().lower()
    for pt in PICK_TYPES:
        feats[f"pick_type_{pt}"] = 1.0 if pick_type == pt else 0.0

    # Optional features only when genuinely present (non-null).
    for name in _OPTIONAL_NUMERIC_FEATURES:
        if item.get(name) is not None:
            feats[name] = _as_float(item.get(name))

    return feats


def feature_names_for(samples: list[dict[str, float]]) -> list[str]:
    """Deterministic ordered feature list given the extracted sample dicts.

    Always-present features come first in a fixed order; optional features are
    appended (sorted) only if any sample carries them.
    """
    base: list[str] = []
    base.extend(_ITEM_NUMERIC_FEATURES)
    base.extend(_REQUEST_NUMERIC_FEATURES)
    base.append("num_source_types")
    base.extend(FEATURE_SOURCE_FLAGS.keys())
    base.extend(f"pick_type_{pt}" for pt in PICK_TYPES)

    present_optional: set[str] = set()
    for s in samples:
        for name in _OPTIONAL_NUMERIC_FEATURES:
            if name in s:
                present_optional.add(name)
    base.extend(n for n in _OPTIONAL_NUMERIC_FEATURES if n in present_optional)
    return base


def feature_vector(feats: dict[str, float], feature_names: list[str]) -> list[float]:
    """Project a feature dict onto an ordered feature-name list."""
    return [float(feats.get(name, 0.0)) for name in feature_names]


# ---------------------------------------------------------------------------
# Inference-time helper: build a feature dict from a live RealSongCard
# ---------------------------------------------------------------------------

def feature_dict_from_card(card: Any, req: Any = None) -> dict[str, float]:
    """Build the same feature dict from a live :class:`RealSongCard`.

    This guarantees train/inference feature parity: the shadow scorer feeds the
    model exactly the features it was trained on, derived from the card's
    score_breakdown, source_types, pick_type and rank plus the request controls.
    """
    bd = dict(getattr(card, "score_breakdown", None) or {})
    item: dict[str, Any] = {
        "content_score": bd.get("content_score"),
        "retrieval_score": bd.get("retrieval_score"),
        "multi_source_agreement": bd.get("multi_source_agreement"),
        "quality_score": bd.get("quality_score"),
        "novelty_score": bd.get("novelty_score"),
        "final_score": bd.get("final_score"),
        "rank_score": bd.get("rank_score"),
        "rank_position": getattr(card, "rank", None),
        "source_types": list(getattr(card, "source_types", None) or []),
        "pick_type": getattr(card, "pick_type", ""),
    }
    request: dict[str, Any] = {}
    if req is not None:
        request = {
            "content_weight": getattr(req, "content_weight", None),
            "novelty": getattr(req, "novelty", None),
            "diversity": getattr(req, "diversity", None),
        }
    return extract_feature_dict(item, request)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """The materialised training set + a human-readable summary."""
    feature_names: list[str] = field(default_factory=list)
    X: list[list[float]] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    sample_weight: list[float] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)          # graded labels
    num_samples: int = 0
    positive_count: int = 0
    negative_count: int = 0
    weak_negative_count: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": int(self.num_samples),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "weak_negative_count": int(self.weak_negative_count),
            "num_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
        }


def build_training_data(store: Any) -> TrainingData:
    """Build a :class:`TrainingData` from a FeedbackStore.

    One sample per ``recommendation_item`` row. The label comes from the
    ``user_feedback`` events sharing the same (request_id, song_id). Robust to
    an empty store -- returns an empty TrainingData rather than raising.
    """
    items = store.get_all_items()
    requests = store.get_all_requests()
    feedback = store.get_all_feedback()

    # Index feedback by (request_id, song_id) -> set of event types.
    events_by_key: dict[tuple[str, int], set[str]] = {}
    for ev in feedback:
        rid = str(ev.get("request_id") or "")
        sid = ev.get("song_id")
        et = str(ev.get("event_type") or "").strip().lower()
        if sid is None or not et:
            continue
        events_by_key.setdefault((rid, int(sid)), set()).add(et)

    # First pass: extract feature dicts + outcomes so the schema can be
    # computed from the optional features actually present.
    feat_dicts: list[dict[str, float]] = []
    outcomes: list[LabelOutcome] = []
    for item in items:
        rid = str(item.get("request_id") or "")
        sid = item.get("song_id")
        if sid is None:
            continue
        request_row = requests.get(rid)
        feats = extract_feature_dict(item, request_row)
        types = events_by_key.get((rid, int(sid)), set())
        outcome = label_for_events(types)
        feat_dicts.append(feats)
        outcomes.append(outcome)

    feature_names = feature_names_for(feat_dicts)

    data = TrainingData(feature_names=feature_names)
    for feats, outcome in zip(feat_dicts, outcomes):
        data.X.append(feature_vector(feats, feature_names))
        data.y.append(int(outcome.binary))
        data.sample_weight.append(float(outcome.sample_weight))
        data.labels.append(float(outcome.label))
        if outcome.kind == "positive":
            data.positive_count += 1
        elif outcome.kind == "negative":
            data.negative_count += 1
        else:
            data.weak_negative_count += 1
    data.num_samples = len(data.X)
    return data
