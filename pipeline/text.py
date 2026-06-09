"""Pure text / tokenisation / vector helpers for the NetEase pipeline.

These have **no** dependency on any other pipeline module, so every
stage (profile building, retrieval, filtering, ranking, reranking,
explanation) can share one canonical tokeniser. The pipeline lives or
dies on whether overlap calculations agree across artist / tag / title
fields, so keeping these in one place avoids subtle drift.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_STOP = frozenset({
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "by",
    "with", "from", "is", "was", "are", "be", "feat", "ft", "vs",
    "remix", "version", "edit", "mix", "remastered", "remaster",
    "live", "cover", "acoustic", "instrumental", "demo", "radio",
    "karaoke", "mono", "stereo",
})
_VERSION_WORDS = frozenset({
    "live", "cover", "acoustic", "instrumental", "demo", "radio",
    "karaoke", "mono", "stereo", "remix", "version", "edit",
    "mix", "remastered", "remaster",
})
_TITLE_SUFFIX_RE = re.compile("[\\(\\[\\{\uff08\u3010].*?[\\)\\]\\}\uff09\u3011]")


def _tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        t = raw.lower()
        if not t or t in _STOP or len(t) <= 1:
            continue
        out.append(t)
    return out


def _token_set(text: Optional[str]) -> set[str]:
    return set(_tokens(text))


def _norm_artist(name: str) -> str:
    """Canonical lower-case-stripped artist key for set membership.

    NetEase and the user can disagree on punctuation / case ("bon iver"
    vs "Bon Iver"), so we match on the joined lowered tokens.
    """
    return " ".join(_tokens(name))


def _norm_title(title: str) -> str:
    """Canonical key for duplicate-title filtering.

    Search results often include covers/live/remix variants as
    ``Song (Live)`` or ``Song - Acoustic``. For recommendation purposes
    those should not be treated as fresh songs when the user already
    picked ``Song``.
    """
    raw = str(title or "")
    without_brackets = _TITLE_SUFFIX_RE.sub(" ", raw)
    # Drop common version suffixes, but keep the original title when the
    # split would remove the whole signal.
    for sep in (" - ", " -- ", " / "):
        if sep in without_brackets:
            head, tail = without_brackets.split(sep, 1)
            if _tokens(head) and _tokens(tail):
                tail_tokens = set(_tokens(tail))
                if tail_tokens <= _STOP:
                    without_brackets = head
            break
    key = " ".join(_tokens(without_brackets))
    return key or " ".join(_tokens(raw))


def _starts_with_tokens(values: list[str], prefix: list[str]) -> bool:
    return bool(prefix) and len(values) >= len(prefix) and values[: len(prefix)] == prefix


def _contains_tokens(values: list[str], phrase: list[str]) -> bool:
    if not phrase or len(values) < len(phrase):
        return False
    limit = len(values) - len(phrase) + 1
    return any(values[i:i + len(phrase)] == phrase for i in range(limit))


def _strip_profile_tag_terms(title: str, profile_tag_tokens: Iterable[str]) -> str:
    """Remove user tag words from a title before text-sim scoring."""
    tag_terms = set(profile_tag_tokens)
    if not tag_terms:
        return title or ""
    return " ".join(t for t in _tokens(title) if t not in tag_terms)


def _dedupe_phrases(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        s = str(raw or "").strip()
        key = " ".join(_tokens(s))
        if not s or not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _raw_tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [raw.lower() for raw in _TOKEN_RE.findall(text) if raw]


def _profile_text(parts: Iterable[str]) -> str:
    toks: list[str] = []
    for part in parts:
        toks.extend(_tokens(part))
    return " ".join(toks)


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / float(len(a | b))


def _tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    tokenised = [_tokens(d) for d in docs]
    if len(tokenised) < 2 or not any(tokenised):
        return [{} for _ in docs]
    df: dict[str, int] = {}
    for toks in tokenised:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    n_docs = float(len(tokenised))
    vectors: list[dict[str, float]] = []
    for toks in tokenised:
        if not toks:
            vectors.append({})
            continue
        counts: dict[str, int] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
        total = float(len(toks))
        vec: dict[str, float] = {}
        for tok, count in counts.items():
            idf = math.log((1.0 + n_docs) / (1.0 + float(df.get(tok, 0)))) + 1.0
            vec[tok] = (float(count) / total) * idf
        vectors.append(vec)
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return min(1.0, max(0.0, num / (da * db)))


def _source_preference_tokens(cand: "Any") -> set[str]:
    pref_types = {"genre", "mood", "tag", "genre_mood", "tag_combo", "discovery", "artist_context"}
    out: set[str] = set()
    for hit in cand.source_hits:
        if hit.source_type in pref_types:
            out.update(_tokens(hit.query))
    return out
