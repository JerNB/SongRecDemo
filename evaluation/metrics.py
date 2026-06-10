"""Diagnostic metrics for the offline evaluation harness (P3).

Every function here takes the list of recommendation *cards* (the dicts from
:meth:`RealSongCard.to_dict`) plus the recommendation ``trace`` dict, and
returns a JSON-friendly metric value.

These are DIAGNOSTIC metrics, not accuracy metrics: with no human relevance
labels we cannot honestly report Precision / Recall / NDCG. They describe the
*shape* of a recommendation list (how varied, how novel, where it came from)
so different configs can be compared.
"""

from __future__ import annotations

import math
from typing import Any

from SongRecDemo.netease_pipeline import _norm_artist, _norm_title, _tokens


# ---------------------------------------------------------------------------
# Small helpers to read fields off a card dict
# ---------------------------------------------------------------------------

def _bd(card: dict[str, Any]) -> dict[str, Any]:
    return card.get("score_breakdown") or {}


def _artist_norms(card: dict[str, Any]) -> set[str]:
    artists = card.get("artists") or ([card.get("artist")] if card.get("artist") else [])
    norms = {_norm_artist(a) for a in artists if a}
    norms.discard("")
    return norms


def _source_types(card: dict[str, Any]) -> list[str]:
    return [str(s) for s in (card.get("source_types") or []) if s]


def _topk(items: list[dict[str, Any]], k: int | None) -> list[dict[str, Any]]:
    if k is None or k <= 0:
        return list(items)
    return list(items[:k])


# ---------------------------------------------------------------------------
# 1. coverage@k
# ---------------------------------------------------------------------------

def coverage_at_k(items: list[dict[str, Any]], k: int | None = None) -> dict[str, float]:
    """Unique artists / albums / source-types relative to the list size."""
    top = _topk(items, k)
    n = len(top)
    if n == 0:
        return {"unique_artists": 0.0, "unique_albums": 0.0,
                "unique_source_types": 0.0, "n": 0.0}
    artists: set[str] = set()
    albums: set[str] = set()
    stypes: set[str] = set()
    for c in top:
        artists |= _artist_norms(c)
        alb = _norm_title(c.get("album") or "")
        if alb:
            albums.add(alb)
        stypes.update(_source_types(c))
    return {
        "unique_artists": round(len(artists) / n, 4),
        "unique_albums": round(len(albums) / n, 4),
        "unique_source_types": round(float(len(stypes)), 4),
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# 2. diversity@k  (mean pairwise dissimilarity)
# ---------------------------------------------------------------------------

def _card_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Lightweight content similarity between two cards, mirroring the
    components the Reranker uses (artist / album / source-type / title)
    but operating on the serialised card payload."""
    parts: list[tuple[float, float]] = []

    a_art, b_art = _artist_norms(a), _artist_norms(b)
    parts.append((0.40, 1.0 if (a_art & b_art) else 0.0))

    a_alb, b_alb = _norm_title(a.get("album") or ""), _norm_title(b.get("album") or "")
    if a_alb and b_alb:
        parts.append((0.20, 1.0 if a_alb == b_alb else 0.0))

    a_st, b_st = set(_source_types(a)), set(_source_types(b))
    if a_st and b_st:
        parts.append((0.20, len(a_st & b_st) / len(a_st | b_st)))

    a_tt, b_tt = set(_tokens(a.get("title") or "")), set(_tokens(b.get("title") or ""))
    if a_tt and b_tt:
        parts.append((0.20, len(a_tt & b_tt) / len(a_tt | b_tt)))

    weight = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / weight if weight else 0.0


def diversity_at_k(items: list[dict[str, Any]], k: int | None = None) -> float:
    """Mean pairwise dissimilarity (1 - similarity) across the list."""
    top = _topk(items, k)
    n = len(top)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - _card_similarity(top[i], top[j])
            pairs += 1
    return round(total / pairs, 4) if pairs else 0.0


# ---------------------------------------------------------------------------
# 3. novelty@k
# ---------------------------------------------------------------------------

def novelty_at_k(items: list[dict[str, Any]], k: int | None = None) -> dict[str, float]:
    """Mean novelty_score and a popularity-based novelty (1 - popularity)."""
    top = _topk(items, k)
    if not top:
        return {"mean_novelty_score": 0.0, "mean_inverse_popularity": 0.0}
    nov = [float(_bd(c).get("novelty_score") or 0.0) for c in top]
    pop = [float(_bd(c).get("popularity_score") or 0.0) for c in top]
    inv_pop = [1.0 - p for p in pop]
    return {
        "mean_novelty_score": round(sum(nov) / len(nov), 4),
        "mean_inverse_popularity": round(sum(inv_pop) / len(inv_pop), 4),
    }


# ---------------------------------------------------------------------------
# 4. source_mix@k
# ---------------------------------------------------------------------------

def source_mix_at_k(items: list[dict[str, Any]], k: int | None = None) -> dict[str, float]:
    """Share of each source type across the (multi-labelled) list.

    A card can carry several source types; shares are normalised by the
    total number of source-type tags so they sum to ~1.0.
    """
    top = _topk(items, k)
    counts: dict[str, int] = {}
    total = 0
    for c in top:
        for st in _source_types(c):
            counts[st] = counts.get(st, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {st: round(n / total, 4) for st, n in sorted(counts.items())}


# ---------------------------------------------------------------------------
# 5. score_distribution
# ---------------------------------------------------------------------------

def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "var": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "mean": round(mean, 4),
        "var": round(var, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def score_distribution(items: list[dict[str, Any]], k: int | None = None) -> dict[str, Any]:
    """Mean / variance / min / max of final_score and rank_score."""
    top = _topk(items, k)
    final = [float(_bd(c).get("final_score") or c.get("score") or 0.0) for c in top]
    rank = [float(_bd(c).get("rank_score") or 0.0) for c in top]
    return {"final_score": _dist(final), "rank_score": _dist(rank)}


# ---------------------------------------------------------------------------
# 6. embedding_share@k
# ---------------------------------------------------------------------------

def embedding_share_at_k(items: list[dict[str, Any]], k: int | None = None) -> float:
    """Fraction of cards whose source_types include the embedding channel."""
    top = _topk(items, k)
    if not top:
        return 0.0
    hits = sum(1 for c in top if "embedding" in _source_types(c))
    return round(hits / len(top), 4)


# ---------------------------------------------------------------------------
# 7. duplicate_rate@k
# ---------------------------------------------------------------------------

def duplicate_rate_at_k(items: list[dict[str, Any]], k: int | None = None) -> dict[str, float]:
    """Repetition by artist / canonical title / album.

    Each rate is ``1 - unique/total`` so 0.0 means a perfectly varied list
    and higher means more repeats.
    """
    top = _topk(items, k)
    n = len(top)
    if n == 0:
        return {"artist": 0.0, "title": 0.0, "album": 0.0}

    artist_keys: list[str] = []
    title_keys: list[str] = []
    album_keys: list[str] = []
    for c in top:
        norms = _artist_norms(c)
        artist_keys.append(sorted(norms)[0] if norms else "")
        title_keys.append(_norm_title(c.get("title") or ""))
        album_keys.append(_norm_title(c.get("album") or ""))

    def _rate(keys: list[str]) -> float:
        present = [x for x in keys if x]
        if not present:
            return 0.0
        return round(1.0 - len(set(present)) / len(present), 4)

    return {
        "artist": _rate(artist_keys),
        "title": _rate(title_keys),
        "album": _rate(album_keys),
    }


# ---------------------------------------------------------------------------
# 8b. learned shadow ranking comparison (P4)
# ---------------------------------------------------------------------------

def _learned_score(card: dict[str, Any]) -> float | None:
    val = _bd(card).get("learned_score")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _spearman(rank_a: list[float], rank_b: list[float]) -> float:
    """Spearman rank correlation between two equal-length rank vectors.

    Inputs are 0-indexed positions (lower = better). Returns rho in [-1, 1];
    0.0 when undefined (n < 2 or no variance).
    """
    n = len(rank_a)
    if n < 2 or len(rank_b) != n:
        return 0.0
    mean_a = sum(rank_a) / n
    mean_b = sum(rank_b) / n
    cov = sum((rank_a[i] - mean_a) * (rank_b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in rank_a)
    var_b = sum((x - mean_b) ** 2 for x in rank_b)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return round(cov / math.sqrt(var_a * var_b), 4)


def learned_shadow_metrics(
    items: list[dict[str, Any]], k: int | None = None
) -> dict[str, Any] | None:
    """Compare the rule ranking with the shadow learned ranking.

    Returns ``None`` when no card carries a ``learned_score`` (i.e. no shadow
    model is active), so the caller can cleanly skip these metrics.
    """
    top = _topk(items, k)
    learned = [(c, _learned_score(c)) for c in top]
    scored = [(c, s) for c, s in learned if s is not None]
    if not scored:
        return None

    scores = [s for _c, s in scored]
    n = len(scored)

    # Rule order = displayed order; learned order = by descending learned_score.
    rule_rank = list(range(n))
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    learned_rank = [0] * n
    for pos, idx in enumerate(order):
        learned_rank[idx] = pos

    # top-k overlap by song id.
    kk = n if (k is None or k <= 0) else min(k, n)
    rule_top = {int(scored[i][0].get("netease_song_id") or 0) for i in range(kk)}
    learned_top = {int(scored[order[i]][0].get("netease_song_id") or 0) for i in range(kk)}
    overlap = (len(rule_top & learned_top) / kk) if kk else 0.0

    # Disagreements: cards whose rule vs learned position differs by >= 2.
    disagreements: list[dict[str, Any]] = []
    for i in range(n):
        delta = abs(rule_rank[i] - learned_rank[i])
        if delta >= 2:
            disagreements.append({
                "netease_song_id": int(scored[i][0].get("netease_song_id") or 0),
                "title": str(scored[i][0].get("title") or ""),
                "rule_position": rule_rank[i] + 1,
                "learned_position": learned_rank[i] + 1,
                "learned_score": round(scores[i], 4),
            })

    return {
        "num_scored": n,
        "learned_score_distribution": _dist(scores),
        "rank_correlation_between_rule_and_learned": _spearman(
            [float(r) for r in rule_rank], [float(r) for r in learned_rank]
        ),
        "top_k_overlap_between_rule_and_learned": round(overlap, 4),
        "num_cases_where_model_disagrees": len(disagreements),
        "cases_where_model_disagrees": disagreements,
    }


# ---------------------------------------------------------------------------
# 8. latency  (from the trace)
# ---------------------------------------------------------------------------

def latency_ms(trace: dict[str, Any] | None) -> float:
    if not trace:
        return 0.0
    return round(float(trace.get("latency_ms") or 0.0), 2)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(
    items: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Compute every diagnostic metric for one recommendation list."""
    trace = trace or {}
    metrics: dict[str, Any] = {
        "k": int(k) if k else len(items),
        "num_items": len(_topk(items, k)),
        "coverage": coverage_at_k(items, k),
        "diversity": diversity_at_k(items, k),
        "novelty": novelty_at_k(items, k),
        "source_mix": source_mix_at_k(items, k),
        "embedding_share": embedding_share_at_k(items, k),
        "duplicate_rate": duplicate_rate_at_k(items, k),
        "score_distribution": score_distribution(items, k),
        "latency_ms": latency_ms(trace),
        "trace_summary": {
            "num_raw_candidates": int(trace.get("num_raw_candidates") or 0),
            "num_final_candidates": int(trace.get("num_final_candidates") or 0),
            "embedding_recall_enabled": bool(trace.get("embedding_recall_enabled")),
            "num_embedding_candidates": int(trace.get("num_embedding_candidates") or 0),
            "model_version": str(trace.get("model_version") or ""),
            "ranking_config_version": str(trace.get("ranking_config_version") or ""),
        },
    }
    # P4: shadow learned-ranking comparison -- only when a model produced
    # learned scores. Skipped entirely (key absent) otherwise.
    learned = learned_shadow_metrics(items, k)
    if learned is not None:
        metrics["learned_shadow"] = learned
    return metrics


def aggregate_metrics(per_profile: list[dict[str, Any]]) -> dict[str, Any]:
    """Average the scalar diagnostic metrics across profiles for a summary."""
    if not per_profile:
        return {}

    def _mean(getter) -> float:
        vals = []
        for row in per_profile:
            try:
                vals.append(float(getter(row["metrics"])))
            except (KeyError, TypeError, ValueError):
                continue
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    agg = {
        "num_profiles": len(per_profile),
        "mean_diversity": _mean(lambda m: m["diversity"]),
        "mean_embedding_share": _mean(lambda m: m["embedding_share"]),
        "mean_novelty_score": _mean(lambda m: m["novelty"]["mean_novelty_score"]),
        "mean_coverage_unique_artists": _mean(lambda m: m["coverage"]["unique_artists"]),
        "mean_duplicate_artist_rate": _mean(lambda m: m["duplicate_rate"]["artist"]),
        "mean_final_score": _mean(lambda m: m["score_distribution"]["final_score"]["mean"]),
        "mean_latency_ms": _mean(lambda m: m["latency_ms"]),
    }

    # P4: aggregate the shadow comparison only across profiles that have it.
    learned_rows = [p for p in per_profile if (p.get("metrics") or {}).get("learned_shadow")]
    if learned_rows:
        def _lmean(getter) -> float:
            vals = []
            for row in learned_rows:
                try:
                    vals.append(float(getter(row["metrics"]["learned_shadow"])))
                except (KeyError, TypeError, ValueError):
                    continue
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        agg["learned_shadow"] = {
            "num_profiles_with_learned": len(learned_rows),
            "mean_rank_correlation": _lmean(
                lambda ls: ls["rank_correlation_between_rule_and_learned"]),
            "mean_top_k_overlap": _lmean(
                lambda ls: ls["top_k_overlap_between_rule_and_learned"]),
            "mean_num_disagreements": _lmean(
                lambda ls: ls["num_cases_where_model_disagrees"]),
        }
    return agg
