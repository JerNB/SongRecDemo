"""Embedder -- P2 local text-embedding + similarity search.

Turns song content texts and the user's profile text into vectors and
finds the nearest songs by cosine similarity. The v1 backend is a pure
local TF-IDF + TruncatedSVD pipeline: no network, no model download,
deterministic, and good enough to beat raw token overlap for a demo.

The index lives in memory: the caller fits it from the feature store on
startup (and re-fits as the store grows). FAISS / sentence-transformers
can slot in later behind the same interface without touching callers.

Interface
---------
    embedder.fit(corpus, ids)            -> None   (build the index)
    embedder.encode(texts)               -> np.ndarray
    embedder.search(query_text, top_k)   -> list[EmbeddingMatch]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingMatch:
    """One semantic-search hit."""

    song_id: int
    score: float          # cosine similarity in [0, 1]
    rank: int             # 0-indexed position in the result list


class Embedder:
    """Local TF-IDF + TruncatedSVD embedder with cosine search.

    Robust by construction: if the corpus is too small / too sparse for an
    SVD projection it transparently falls back to L2-normalised TF-IDF
    vectors, and if even that is impossible it reports ``ready == False`` and
    returns no matches rather than raising.
    """

    def __init__(
        self,
        *,
        model_type: str = "tfidf_svd",
        svd_dim: int = 64,
    ) -> None:
        self._model_type = str(model_type or "tfidf_svd")
        self._svd_dim = max(2, int(svd_dim))

        self._vectorizer = None   # sklearn TfidfVectorizer
        self._svd = None          # sklearn TruncatedSVD | None (None => raw tf-idf)
        self._ids: list[int] = []
        self._doc_matrix: Optional[np.ndarray] = None   # (n_docs, dim), L2-normed
        self._ready = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def size(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, corpus: Sequence[str], ids: Optional[Sequence[int]] = None) -> None:
        """Build the index from ``corpus`` (aligned with ``ids``)."""
        self._ready = False
        self._doc_matrix = None
        self._ids = []

        docs = [str(t or "") for t in corpus]
        if ids is None:
            ids = list(range(len(docs)))
        ids = [int(i) for i in ids]

        # Drop empty docs (keep ids aligned).
        paired = [(i, d) for i, d in zip(ids, docs) if d.strip()]
        if len(paired) < 2:
            log.info("Embedder.fit: corpus too small (%d usable docs); index off.", len(paired))
            return

        self._ids = [i for i, _ in paired]
        keep_docs = [d for _, d in paired]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedder.fit: scikit-learn unavailable (%s); index off.", exc)
            return

        try:
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b\w+\b",
                min_df=1,
            )
            tfidf = self._vectorizer.fit_transform(keep_docs)
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedder.fit: vectoriser failed (%s); index off.", exc)
            self._vectorizer = None
            return

        n_docs, n_features = tfidf.shape
        if n_features == 0:
            log.info("Embedder.fit: empty vocabulary; index off.")
            self._vectorizer = None
            return

        # TruncatedSVD needs n_components < n_features and <= n_docs - 1.
        n_components = min(self._svd_dim, n_features - 1, n_docs - 1)
        if self._model_type == "tfidf_svd" and n_components >= 2:
            try:
                from sklearn.decomposition import TruncatedSVD

                self._svd = TruncatedSVD(n_components=int(n_components), random_state=42)
                doc_vecs = self._svd.fit_transform(tfidf)
            except Exception as exc:  # noqa: BLE001
                log.warning("Embedder.fit: SVD failed (%s); falling back to raw tf-idf.", exc)
                self._svd = None
                doc_vecs = tfidf.toarray()
        else:
            # Too small for a meaningful projection -- use raw tf-idf cosine.
            self._svd = None
            doc_vecs = tfidf.toarray()

        self._doc_matrix = _l2_normalize(np.asarray(doc_vecs, dtype=np.float64))
        self._ready = True
        log.info(
            "Embedder.fit: indexed %d songs (dim=%d, svd=%s).",
            len(self._ids), self._doc_matrix.shape[1], self._svd is not None,
        )

    # ------------------------------------------------------------------
    # Encode / search
    # ------------------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts`` into the (L2-normalised) embedding space."""
        clean = [str(t or "") for t in texts]
        if not self._ready or self._vectorizer is None or not clean:
            dim = self._doc_matrix.shape[1] if self._doc_matrix is not None else 0
            return np.zeros((len(clean), dim), dtype=np.float64)
        try:
            tfidf = self._vectorizer.transform(clean)
            vecs = self._svd.transform(tfidf) if self._svd is not None else tfidf.toarray()
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedder.encode failed: %s", exc)
            dim = self._doc_matrix.shape[1] if self._doc_matrix is not None else 0
            return np.zeros((len(clean), dim), dtype=np.float64)
        return _l2_normalize(np.asarray(vecs, dtype=np.float64))

    def search(self, query_text: str, top_k: int) -> list[EmbeddingMatch]:
        """Return up to ``top_k`` nearest songs to ``query_text``."""
        if not self._ready or self._doc_matrix is None:
            return []
        q = str(query_text or "").strip()
        if not q or int(top_k) <= 0:
            return []

        qvec = self.encode([q])[0]
        if not np.any(qvec):
            return []

        # Both sides are L2-normalised, so the dot product is the cosine.
        sims = self._doc_matrix @ qvec
        k = min(int(top_k), sims.shape[0])
        if k <= 0:
            return []

        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[EmbeddingMatch] = []
        for rank, idx in enumerate(top_idx):
            score = float(sims[idx])
            if score <= 0.0:
                continue
            out.append(EmbeddingMatch(
                song_id=int(self._ids[idx]),
                score=max(0.0, min(1.0, score)),
                rank=rank,
            ))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        norm = np.linalg.norm(mat)
        return mat / norm if norm > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms
