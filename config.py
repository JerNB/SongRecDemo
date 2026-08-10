"""
Central configuration for the KGRec-music recommendation project.

All file paths, preprocessing constants, split parameters, model
hyperparameters, and evaluation settings live here. Importing this
module in every other module ensures one source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# Raw dataset tree (do not modify anything under here)
DATA_RAW = ROOT / "KGRec-dataset" / "KGRec-dataset" / "KGRec-music"
INTERACTIONS_CSV = DATA_RAW / "implicit_lf_dataset.csv"
DESC_DIR = DATA_RAW / "descriptions"
TAG_DIR = DATA_RAW / "tags"

# Processed artifacts written by the preprocessing stage
ARTIFACTS = ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS / "splits"
MODELS_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"

# Persisted ALS backbone for the personalized recommender.
# Written by run_train_personalized.py; loaded by the RecommendationService.
ALS_STATE_FILE = MODELS_DIR / "als_state.pkl"

# Filenames for serialised split artefacts
TRAIN_FILE = SPLITS_DIR / "train.parquet"
VAL_FILE = SPLITS_DIR / "val.parquet"
TEST_FILE = SPLITS_DIR / "test.parquet"
ITEM_FEATURES_FILE = SPLITS_DIR / "item_features.parquet"   # tags + text per item
ID_MAPS_FILE = SPLITS_DIR / "id_maps.pkl"                   # raw_id <-> contiguous index

# Tag TF-IDF artefacts (produced by preprocessing; consumed by the
# content-based recommender AND the intra-list diversity metric).
TFIDF_MATRIX_FILE = SPLITS_DIR / "tag_tfidf_matrix.npz"     # scipy csr sparse
TFIDF_VECTORIZER_FILE = SPLITS_DIR / "tag_tfidf_vectorizer.pkl"
TFIDF_ITEM_INDEX_FILE = SPLITS_DIR / "tag_tfidf_item_index.json"  # ordered item_id_raw

# Human-readable preprocessing summary (recomputed on every run)
PREPROCESSING_SUMMARY_FILE = SPLITS_DIR / "preprocessing_summary.json"

# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------

# Fraction of each user's interactions held back for evaluation.
# Rationale: with ~145 events/user on average, holding 20 % (≈29 items)
# for val+test leaves ≈116 for training, which is ample for CF and
# popularity baselines.  The remaining 20 % is split equally so each
# evaluation split has ~15 held-out items per user – enough to compute
# meaningful Precision/Recall @K for K up to 10.
SPLIT_SEED = 42
VAL_FRACTION = 0.10    # 10 % of each user's interactions → validation
TEST_FRACTION = 0.10   # 10 % of each user's interactions → test
# Minimum interactions a user must have to be included.
# Users with fewer than MIN_USER_INTERACTIONS cannot be split reliably.
MIN_USER_INTERACTIONS = 5

# ---------------------------------------------------------------------------
# Known data-quality flags (discovered during audit)
# ---------------------------------------------------------------------------

# Item IDs whose description text is byte-identical (possible data error).
# During preprocessing these are logged as a warning; both are kept with
# their own IDs because they might represent distinct tracks in the KG.
DUPLICATE_DESC_PAIRS: list[tuple[int, int]] = [(2028, 3130)]

# ---------------------------------------------------------------------------
# Tag / text preprocessing
# ---------------------------------------------------------------------------

# Tags that appear below this global frequency across all items are dropped
# before building the item-feature matrix.  Avoids extremely rare tags
# polluting TF-IDF or bag-of-tags vectors.
MIN_TAG_FREQUENCY = 3

# When an item has no tags/*.txt file, its tag representation is treated as
# an empty bag.  Do NOT impute neighbour tags; that would inject information
# that is not in the dataset.
MISSING_TAG_STRATEGY = "empty"   # alternative: "mean_vector" (document if changed)

# Maximum number of description tokens kept per item (truncation guard for
# long wiki-style blurbs).  Set to None to keep full text.
MAX_DESC_TOKENS = 512

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Recommendation list length evaluated at each K
EVAL_K_VALUES: list[int] = [5, 10, 20]

# Primary K used in tables / plots throughout the paper
PRIMARY_K = 10

# Diversity feature source is defined alongside the content-based model
# below (DIVERSITY_FEATURE_SOURCE); it must match CB_FEATURE_MODE so that
# diversity is measured in the same feature space the CB model uses.

# ---------------------------------------------------------------------------
# Popularity baseline
# ---------------------------------------------------------------------------

# Popularity scores are derived exclusively from TRAINING interactions.
# Items unseen in training receive score 0 (cold items in the catalogue).
POPULARITY_SCORE = "count"   # "count" | "log_count" (document choice in report)

# ---------------------------------------------------------------------------
# Collaborative filtering (implicit matrix factorisation)
# ---------------------------------------------------------------------------

CF_FACTORS = 64          # latent dimension
CF_REGULARIZATION = 0.01
CF_ITERATIONS = 20
CF_ALPHA = 40            # confidence scaling: c_ui = 1 + alpha * r_ui (r_ui = 1)
CF_NUM_THREADS = 4       # parallelism for implicit library

# ---------------------------------------------------------------------------
# Content-based recommender
# ---------------------------------------------------------------------------

# Feature representation used by the main content-based model.
#
# Decision: "tags_bow" (TF-IDF on normalised Last.fm tag tokens).
#
# Rationale:
#   1. Tags ("indie", "mellow", "psychedelic", "80s") are structured
#      semantic labels that directly encode genre, mood, and era — exactly
#      the dimensions of music similarity we want to capture.
#   2. Descriptions are narrative prose (artist trivia, album context).
#      High-frequency story words ("song", "released", "guitar") act as
#      noise in a similarity space; artist-name tokens create spurious
#      similarity between different tracks by the same artist.
#   3. A combined representation requires an α weight to balance the
#      tag sub-matrix (~500-1500 columns) against the desc sub-matrix
#      (~10,000 columns after max_features truncation). Tuning α would
#      consume validation budget and introduce a hyperparameter that
#      muddies the model comparison. Tags alone are the cleaner choice.
#   4. With sublinear_tf=True, TF-IDF on tags reduces to IDF weighting
#      (each tag appears once per item), which correctly penalises common
#      genre labels and up-weights discriminative niche tags.
#   5. 401 items lack tag files; these receive zero-vector rows and will
#      not be recommended by the content model. This is documented, not
#      papered over with imputation.
#
# Alternatives available but not used in the main experiment:
#   "tfidf_desc" : TF-IDF on cleaned description text (see rationale above)
#   "combined"   : weighted horizontal stack (requires tuning α; excluded)
CB_FEATURE_MODE: str = "tags_bow"

# Candidate retrieval at inference:
# At 8640 items the full catalogue is scored exhaustively (O(n_items * d)
# per user), no ANN index needed.  CB_TOP_K_CANDIDATES acts as a
# pre-filter for the re-rank step; set to n_items to disable pre-filtering.
CB_TOP_K_CANDIDATES: int = 500   # > max(EVAL_K_VALUES)=20; effectively exhaustive

# The item-feature space used by the Intra-List Diversity metric.
# Must match CB_FEATURE_MODE so that diversity scores are measured in the
# same space as the content model's similarity scores.
DIVERSITY_FEATURE_SOURCE: str = "tags"   # "tags" | "tfidf_desc"

# ---------------------------------------------------------------------------
# NetEase Cloud Music API (optional metadata enrichment)
# ---------------------------------------------------------------------------
#
# This block configures an OPTIONAL display-side enricher only.  It is
# never read by the training pipeline, the ALS model, the evaluator, or
# the personalized scoring engine -- those operate exclusively on KGRec
# item IDs and KGRec item features.  The NetEase enricher only attaches
# prettier frontend metadata (title / artist / album / cover image /
# NetEase song ID + page URL) to the already-ranked recommendations,
# and it falls back to the internal-features enricher whenever the API
# is unreachable or no high-confidence match is found.
#
# Reference Node.js service:
#   https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced
#
# Run it locally on port 3000 (see docs/netease_api_setup.md):
#   git clone https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced
#   cd api-enhanced && pnpm install && node app.js
#
# All values can be overridden via environment variables of the same
# name -- handy for swapping between localhost, a Docker container, or
# disabling the enricher entirely without code changes.
import os as _os

# Base URL of the running NetEase API service.  The enricher prepends
# this to its endpoint paths (e.g. ``/search``, ``/song/detail``).
NETEASE_API_BASE_URL: str = _os.environ.get(
    "NETEASE_API_BASE_URL", "http://localhost:3000"
).rstrip("/")

# Local on-disk cache for NetEase API responses.  Repeated demo runs
# replay from this cache instead of re-hitting the API, which makes the
# demo deterministic and friendly to flaky network conditions.
NETEASE_CACHE_PATH: Path = ARTIFACTS / "netease_cache.sqlite"

# HTTP request timeout (seconds) for a single call to the NetEase API.
# Kept short on purpose: the recommender response is interactive, and
# we would rather fall back to internal metadata than block the UI.
NETEASE_TIMEOUT_SECONDS: float = float(
    _os.environ.get("NETEASE_TIMEOUT_SECONDS", "5.0")
)

# Max number of retries for transient HTTP failures (timeouts, 5xx).
# After the budget is exhausted the enricher falls back gracefully.
NETEASE_MAX_RETRIES: int = int(_os.environ.get("NETEASE_MAX_RETRIES", "1"))

# How many search candidates to fetch per query before scoring.  Larger
# = more chances to find a good match; smaller = lower API load.
NETEASE_SEARCH_LIMIT: int = int(_os.environ.get("NETEASE_SEARCH_LIMIT", "5"))

# Minimum match-confidence (in [0, 1]) required to accept a NetEase
# candidate.  Below this the enricher falls back to internal metadata
# only, so we never display a wrong title just because the API
# returned *some* result.  Tuned empirically against KGRec-music: the
# scorer in netease_enrichment.py awards 0.40 for full artist-token
# coverage and an extra +0.10 when a KGRec tag confirms the artist.
# 0.40 therefore lets a clean description-level artist match through
# (e.g. "Yeah Yeah Yeahs" mentioned in the desc but with no matching
# tag), while a single-token coincidence -- typical for unrelated
# candidates -- never reaches it.
NETEASE_MIN_CONFIDENCE: float = float(
    _os.environ.get("NETEASE_MIN_CONFIDENCE", "0.40")
)

# ---------------------------------------------------------------------------
# Product-layer ranking weights (NetEase real-song pipeline)
# ---------------------------------------------------------------------------
#
# These weights drive the Ranker inside SongRecDemo/netease_pipeline.py.
# They are deliberately centralised here so the scoring blend can be tuned
# in one place instead of being scattered as magic numbers across scoring
# functions.
#
# How the pieces fit together
# ---------------------------
# Per candidate the Ranker computes three normalised sub-scores in [0, 1]:
#
#   content_score   -- text similarity + artist / tag / title overlap with
#                      the user's stated taste. "Do the words match?"
#   retrieval_score -- retrieval confidence + multi-source agreement across
#                      independent search channels. "How sure is recall?"
#   quality_score   -- popularity, artist authority, playability, metadata
#                      completeness. "Is this a real, healthy track?"
#
# The per-request ``content_weight`` slider blends the first two:
#
#   personalized_relevance = content_weight * content_score
#                          + (1 - content_weight) * retrieval_score
#
# A high content_weight makes recommendations lean on the user's text /
# liked songs / artists / tags / genres; a low content_weight makes them
# lean on retrieval confidence, multi-channel agreement, and platform
# signals. The ``base`` block below then mixes personalized_relevance with
# overall quality and a small standalone artist-authority term.
RANKING_WEIGHTS_V1: dict[str, dict[str, float]] = {
    "base": {
        "personalized_relevance": 0.70,
        "quality": 0.20,
        "artist_authority": 0.10,
    },
    "content": {
        "text_similarity": 0.40,
        "artist_match": 0.25,
        "tag_match": 0.25,
        "title_match": 0.10,
    },
    "retrieval": {
        "retrieval_confidence": 0.60,
        "multi_source_agreement": 0.40,
    },
    "quality": {
        "popularity": 0.35,
        "artist_authority": 0.30,
        "playable": 0.20,
        "metadata_quality": 0.15,
    },
}

# Relevance gate for the novelty bonus. novelty_bonus is multiplied by
# ``min(1, base_relevance / RANKING_RELEVANCE_GATE_DIVISOR)`` so a novel but
# low-relevance candidate cannot leapfrog genuinely relevant ones. Lower the
# divisor to let novelty kick in earlier; raise it to demand more relevance
# before novelty counts.
RANKING_RELEVANCE_GATE_DIVISOR: float = 0.55

# ---------------------------------------------------------------------------
# P2: local song_feature_store + embedding recall channel
# ---------------------------------------------------------------------------
#
# The product layer keeps a small local feature store of every NetEase song
# it has ever seen (title / artists / album / platform signals + a stable
# content text per song). On top of that store sits an OPTIONAL second
# retrieval channel -- ``embedding_recall`` -- that finds songs whose content
# text is semantically close to the user's profile text. This makes the
# recommender progressively less dependent on live NetEase /search as the
# local catalogue grows.
#
# Crucially, this is an ADDITIVE recall channel: it never replaces the
# NetEase search channel and never touches the P0 ranking formula. Its hits
# are merged into the same candidate pool by song_id, so a song surfaced by
# both channels simply gains extra source hits (raising multi_source_agreement
# naturally). Final ordering is still decided by the unchanged P0 Ranker.

# Master switch for the embedding recall channel.
EMBEDDING_RECALL_ENABLED: bool = (
    _os.environ.get("EMBEDDING_RECALL_ENABLED", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

# How many semantically-similar songs the embedding channel recalls.
EMBEDDING_RECALL_TOP_K: int = int(_os.environ.get("EMBEDDING_RECALL_TOP_K", "30"))

# Reliability assigned to embedding-channel SourceHits. Deliberately in the
# 0.60-0.75 band: a semantic match is decent evidence but weaker than an exact
# artist hit (0.85). Used by the P0 retrieval_confidence sub-score unchanged.
EMBEDDING_RECALL_RELIABILITY: float = float(
    _os.environ.get("EMBEDDING_RECALL_RELIABILITY", "0.68")
)

# Minimum number of songs the local store must hold before the embedding
# index is built / queried. Below this we skip embedding recall to avoid
# small-sample mis-recall (and to avoid the cold-start case where the store
# is empty -- the system then runs on NetEase search alone, no errors).
EMBEDDING_MIN_CORPUS_SIZE: int = int(
    _os.environ.get("EMBEDDING_MIN_CORPUS_SIZE", "20")
)

# Embedding backend. v1 is a pure-local TF-IDF + TruncatedSVD pipeline
# (fast, dependency-light, deterministic). The Embedder interface leaves
# room for a "sentence_transformers" backend later without touching callers.
EMBEDDING_MODEL_TYPE: str = _os.environ.get("EMBEDDING_MODEL_TYPE", "tfidf_svd")

# Latent dimensionality of the TruncatedSVD projection over the TF-IDF space.
EMBEDDING_SVD_DIM: int = int(_os.environ.get("EMBEDDING_SVD_DIM", "64"))

# On-disk SQLite path for the local song feature store. Relative paths are
# resolved against the project root so the demo can be launched from anywhere.
_FEATURE_STORE_PATH_RAW: str = _os.environ.get(
    "FEATURE_STORE_PATH", "data/song_feature_store.sqlite"
)
FEATURE_STORE_PATH: Path = (
    Path(_FEATURE_STORE_PATH_RAW)
    if Path(_FEATURE_STORE_PATH_RAW).is_absolute()
    else ROOT / _FEATURE_STORE_PATH_RAW
)

# ---------------------------------------------------------------------------
# P3: feedback logging + offline evaluation
# ---------------------------------------------------------------------------
#
# The product layer logs every recommendation exposure (the request and each
# returned card) plus any user feedback events (click / like / dislike / ...)
# into a local SQLite store. This builds the raw training data a learned
# ranker would later consume -- WITHOUT changing the P0 ranking formula now.
# Logging is strictly fire-and-forget: any failure is swallowed so it can
# never break the recommendation path.

# Master switch for feedback logging.
FEEDBACK_LOGGING_ENABLED: bool = (
    _os.environ.get("FEEDBACK_LOGGING_ENABLED", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

# On-disk SQLite path for the feedback store. Relative paths resolve against
# the project root so the demo can be launched from anywhere.
_FEEDBACK_STORE_PATH_RAW: str = _os.environ.get(
    "FEEDBACK_STORE_PATH", "data/feedback.sqlite"
)
FEEDBACK_STORE_PATH: Path = (
    Path(_FEEDBACK_STORE_PATH_RAW)
    if Path(_FEEDBACK_STORE_PATH_RAW).is_absolute()
    else ROOT / _FEEDBACK_STORE_PATH_RAW
)

# Allowed user_feedback event types. The /api/feedback endpoint validates
# against this whitelist. Names are deliberately honest: there is no real
# listen-completion signal yet, so we only record explicit UI interactions.
FEEDBACK_EVENT_TYPES: frozenset = frozenset({
    "impression",
    "click",
    "play_preview",
    "like",
    "dislike",
    "skip",
    "add_to_playlist",
    "open_netease_url",
    "why_clicked",
})

# Version stamps written onto every recommendation_request row so logs from
# different algorithm revisions never get silently mixed together.
PIPELINE_VERSION: str = "p3-feedback-eval"
RANKING_CONFIG_VERSION: str = "ranking_weights_v1"

# ---------------------------------------------------------------------------
# P4: shadow learned ranker
# ---------------------------------------------------------------------------
#
# A lightweight learned ranker trained offline on the P3 feedback logs. In
# this first stage it NEVER drives the live ordering -- the P0 rule ranker
# still decides final_score / rank_score. When shadow mode is on and a
# trained model is present, the recommender additionally computes a
# ``learned_score`` per candidate purely for analysis / evaluation, so we can
# observe how a learned model would rank versus the rules before ever letting
# it take over.

# Master switch. When True the learned ranker would be allowed to influence
# ranking -- DELIBERATELY left False for P4: the learned score is observed,
# never acted on.
LEARNED_RANKER_ENABLED: bool = (
    _os.environ.get("LEARNED_RANKER_ENABLED", "0").strip().lower()
    not in {"0", "false", "no", "off"}
)

# Shadow mode: compute and attach learned_score (and a learned_rank_position)
# without changing the rule ordering. On by default so the model is observable
# as soon as one is trained, but harmless when no model file exists.
LEARNED_RANKER_SHADOW_MODE: bool = (
    _os.environ.get("LEARNED_RANKER_SHADOW_MODE", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

# On-disk path of the trained joblib model. Relative paths resolve against the
# project root so the demo can be launched from anywhere. The companion
# feature schema lives next to it as ``learned_ranker_schema.json``.
_LEARNED_RANKER_MODEL_PATH_RAW: str = _os.environ.get(
    "LEARNED_RANKER_MODEL_PATH", "data/learned_ranker.joblib"
)
LEARNED_RANKER_MODEL_PATH: Path = (
    Path(_LEARNED_RANKER_MODEL_PATH_RAW)
    if Path(_LEARNED_RANKER_MODEL_PATH_RAW).is_absolute()
    else ROOT / _LEARNED_RANKER_MODEL_PATH_RAW
)

# Companion feature schema written by the training CLI.
LEARNED_RANKER_SCHEMA_PATH: Path = LEARNED_RANKER_MODEL_PATH.with_name(
    "learned_ranker_schema.json"
)

# Minimum number of training samples the CLI requires before it will train.
# Below this it prints an explanation and exits 0 (fail-soft, not an error).
LEARNED_RANKER_MIN_SAMPLES: int = int(
    _os.environ.get("LEARNED_RANKER_MIN_SAMPLES", "50")
)

# Sample weight assigned to impression-only (weak negative) training rows.
# Deliberately low: a user not clicking does not strongly mean dislike.
LEARNED_RANKER_WEAK_NEGATIVE_WEIGHT: float = float(
    _os.environ.get("LEARNED_RANKER_WEAK_NEGATIVE_WEIGHT", "0.2")
)
