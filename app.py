"""
Local-only Flask demo server for the real-song recommendation product.

Two-layer architecture
----------------------
This Flask app exposes a *product layer* on top of a NetEase-backed
recommendation pipeline. The KGRec ALS / content / popularity *research
layer* stays entirely separate and is only reachable from one
deliberately-named debug route. None of the research artefacts -- the
trained ALS state, the preprocessing parquets, the validation results,
the evaluation pipeline, or the saved KGRec metrics -- are mutated by
this server.

Routes
------
* ``GET  /``                       -- static frontend.
* ``GET  /api/health``             -- service status + which layers are up.
* ``GET  /api/song-search?q=...``  -- live NetEase real-song search.
* ``POST /api/recommend``          -- real-song recommendations (NetEase
                                       pipeline; **the main user-facing route**).
* ``POST /api/kgrec-recommend``    -- developer-only: drives the KGRec
                                       :class:`RecommendationService`
                                       directly with KGRec item IDs.
                                       Hidden behind the Advanced panel
                                       in the UI.

CLI flags
---------
* ``--host``               bind host (default 127.0.0.1, local-only).
* ``--port``               bind port (default 5173).
* ``--no-kgrec``           skip loading the heavy KGRec service. Faster
                            startup; ``/api/kgrec-recommend`` answers 503.
* ``--netease-base-url``   override NetEase API base URL.
* ``--debug``              Flask debug mode (no auto-reload).

The server binds to ``127.0.0.1`` by default (local-only on purpose).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when launched as `python SongRecDemo/app.py`
# or `python -m SongRecDemo.app` from the project root.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

import config  # noqa: E402
from src.personalization.netease_enrichment import (  # noqa: E402
    NeteaseAPIClient,
    NeteaseAPIError,
    NeteaseCache,
)

# Local product-layer modules (live alongside this file).
from SongRecDemo.netease_pipeline import (  # noqa: E402
    NeteaseRecommender,
    RealSongRequest,
    TrackRef,
)


log = logging.getLogger("songrec_demo")


# ---------------------------------------------------------------------------
# Service container
# ---------------------------------------------------------------------------

class _ServiceHolder:
    """Lazy singleton wiring up:

    1. NetEase API client + on-disk query cache (always tried).
    2. The product-layer :class:`NeteaseRecommender`.
    3. Optionally, the KGRec :class:`RecommendationService` for the
       developer debug route.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inited = False

        # Product layer.
        self._client: Optional[NeteaseAPIClient] = None
        self._cache: Optional[NeteaseCache] = None
        self._recommender: Optional[NeteaseRecommender] = None
        self._netease_alive: bool = False
        self._netease_base_url: str = ""

        # Research layer (optional).
        self._kgrec_service: Any = None              # RecommendationService | None
        self._kgrec_error: Optional[str] = None
        self._kgrec_enabled: bool = False
        self._kgrec_attempted: bool = False

        # Cached freshness for the live ping done by ``/api/health``.
        # We re-probe NetEase a few seconds after the previous probe
        # so the status indicator picks up "service came online" or
        # "service went down" without requiring the user to restart
        # the Python demo. The ping itself is one ``/search`` call
        # with limit=1, so it costs a fraction of a second when the
        # service is up and exits at the configured timeout when down.
        self._last_ping_at: float = 0.0
        self._ping_ttl_seconds: float = 5.0

        # Test injection: when set, ``init`` skips constructing a real
        # NetEase HTTP client and uses the injected one instead. This
        # lets the smoke test run fully offline.
        self._injected_client: Any = None
        self._injected_cache: Any = None
        # Test injection: an in-memory SongFeatureStore so the smoke test
        # never touches the on-disk catalogue.
        self._injected_feature_store: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def inject_for_tests(
        self,
        *,
        client: Any,
        cache: Any,
        netease_alive: bool = True,
        feature_store: Any = None,
    ) -> None:
        """Smoke-test hook -- swap in a fake NetEase client + cache.

        ``feature_store`` lets the hermetic smoke test inject an in-memory
        :class:`SongFeatureStore` so the embedding recall channel can be
        exercised without writing to the on-disk catalogue.
        """
        with self._lock:
            self._injected_client = client
            self._injected_cache = cache
            self._netease_alive = bool(netease_alive)
            self._injected_feature_store = feature_store

    def init(
        self,
        *,
        netease_base_url: Optional[str] = None,
        load_kgrec: bool = True,
    ) -> None:
        with self._lock:
            if self._inited:
                return
            self._netease_base_url = netease_base_url or config.NETEASE_API_BASE_URL
            self._kgrec_enabled = bool(load_kgrec)

            # ----- Product layer ------------------------------------
            if self._injected_client is not None:
                self._client = self._injected_client
                self._cache = self._injected_cache
            else:
                self._client = NeteaseAPIClient(
                    base_url=self._netease_base_url,
                    timeout=float(config.NETEASE_TIMEOUT_SECONDS),
                    max_retries=int(config.NETEASE_MAX_RETRIES),
                )
                self._cache = NeteaseCache(config.NETEASE_CACHE_PATH)

                log.info("Probing NetEase API at %s ...", self._netease_base_url)
                try:
                    self._netease_alive = bool(self._client.ping())
                except Exception as exc:                # noqa: BLE001
                    log.warning("NetEase ping crashed: %s", exc)
                    self._netease_alive = False
                if self._netease_alive:
                    log.info("NetEase API reachable.")
                else:
                    log.warning(
                        "NetEase API at %s did not answer the ping; "
                        "search and recommend will rely on whatever is "
                        "in the on-disk cache.",
                        self._netease_base_url,
                    )

            self._recommender = NeteaseRecommender(
                client=self._client,
                cache=self._cache,
                feature_store=self._injected_feature_store,
            )

            # ----- Research layer (optional) ------------------------
            if load_kgrec:
                threading.Thread(
                    target=self._init_kgrec_background,
                    name="kgrec-loader",
                    daemon=True,
                ).start()

            self._inited = True

    def _init_kgrec_background(self) -> None:
        """Load the heavy KGRec service off the request thread.

        We don't want the homepage to wait ten seconds for the ALS
        artefacts to deserialise; the debug route waits for this thread
        instead, and other routes never depend on it.
        """
        try:
            from src.data.artifacts import load_processed_artifacts
            from src.personalization import (
                InternalFeaturesEnricher,
                RecommendationService,
            )
            log.info("Loading KGRec processed artefacts (research layer) ...")
            arts = load_processed_artifacts()
            enricher = InternalFeaturesEnricher(arts.item_features)
            log.info("Building KGRec RecommendationService ...")
            svc = RecommendationService.from_artifacts(enricher=enricher)
            with self._lock:
                self._kgrec_service = svc
            log.info(
                "KGRec service ready (debug-only): %d items, vocab=%d",
                svc.catalogue_size(), svc.vocabulary_size(),
            )
        except Exception as exc:                       # noqa: BLE001
            log.warning("KGRec service failed to load: %s", exc)
            with self._lock:
                self._kgrec_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._kgrec_attempted = True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def recommender(self) -> NeteaseRecommender:
        if self._recommender is None:
            raise RuntimeError("Service not initialised; call init() first.")
        return self._recommender

    @property
    def client(self) -> NeteaseAPIClient:
        if self._client is None:
            raise RuntimeError("Service not initialised; call init() first.")
        return self._client

    @property
    def netease_alive(self) -> bool:
        return self._netease_alive

    def refresh_netease_alive(self) -> bool:
        """Re-probe the NetEase API if the previous result is stale.

        Used by ``/api/health`` so the status indicator reflects the
        current state -- if the user starts the NetEase service after
        the Flask app has already booted, the next health request
        flips the indicator to green without a Python restart.
        """
        if self._injected_client is not None:
            return self._netease_alive
        client = self._client
        if client is None:
            return False

        import time as _time
        now = _time.time()
        if now - self._last_ping_at < self._ping_ttl_seconds:
            return self._netease_alive
        self._last_ping_at = now
        try:
            self._netease_alive = bool(client.ping())
        except Exception:                                  # noqa: BLE001
            self._netease_alive = False
        return self._netease_alive

    @property
    def kgrec_enabled(self) -> bool:
        return self._kgrec_enabled

    @property
    def kgrec_service(self) -> Any:
        return self._kgrec_service

    @property
    def kgrec_error(self) -> Optional[str]:
        return self._kgrec_error

    @property
    def kgrec_attempted(self) -> bool:
        return self._kgrec_attempted

    @property
    def netease_base_url(self) -> str:
        return self._netease_base_url


HOLDER = _ServiceHolder()


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def _coerce_str_list(value: Any, field: str, max_len: int = 50) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"`{field}` must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if raw is None:
            continue
        s = str(raw).strip()
        if s and s.lower() not in seen:
            out.append(s)
            seen.add(s.lower())
        if len(out) >= max_len:
            break
    return out


def _coerce_int_list(value: Any, field: str, max_len: int = 50) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"`{field}` must be a list of ints")
    out: list[int] = []
    seen: set[int] = set()
    for raw in value:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        if n and n not in seen:
            out.append(n)
            seen.add(n)
        if len(out) >= max_len:
            break
    return out


def _coerce_float(value: Any, field: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"`{field}` must be a number")
    return float(max(lo, min(hi, f)))


def _coerce_int(value: Any, field: str, default: int, lo: int = 1, hi: int = 100) -> int:
    if value is None or value == "":
        return int(default)
    try:
        i = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"`{field}` must be an integer")
    return int(max(lo, min(hi, i)))


def _coerce_track_list(value: Any, field: str, max_len: int = 30) -> list[TrackRef]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"`{field}` must be a list of song objects")
    out: list[TrackRef] = []
    seen: set[int] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        try:
            sid = int(raw.get("netease_song_id"))
        except (TypeError, ValueError):
            continue
        if not sid or sid in seen:
            continue
        artists_raw = raw.get("artists") or []
        if not isinstance(artists_raw, list):
            artists_raw = []
        artists = [str(a).strip() for a in artists_raw if a]
        out.append(TrackRef(
            netease_song_id=sid,
            title=str(raw.get("title") or "").strip(),
            artist=str(raw.get("artist") or "").strip(),
            artists=artists,
            album=str(raw.get("album") or "").strip(),
            cover_url=str(raw.get("cover_url") or "").strip(),
        ))
        seen.add(sid)
        if len(out) >= max_len:
            break
    return out


def _build_real_song_request(payload: dict[str, Any]) -> RealSongRequest:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    return RealSongRequest(
        liked_songs=_coerce_track_list(payload.get("liked_songs"), "liked_songs"),
        liked_artists=_coerce_str_list(payload.get("liked_artists"), "liked_artists", max_len=15),
        genres=_coerce_str_list(payload.get("genres"), "genres", max_len=20),
        moods=_coerce_str_list(payload.get("moods"), "moods", max_len=20),
        tags=_coerce_str_list(payload.get("tags"), "tags", max_len=20),
        excluded_song_ids=_coerce_int_list(payload.get("excluded_song_ids"), "excluded_song_ids"),
        content_weight=_coerce_float(payload.get("content_weight"), "content_weight", 0.50),
        novelty=_coerce_float(payload.get("novelty"), "novelty", 0.30),
        diversity=_coerce_float(payload.get("diversity"), "diversity", 0.30),
        k=_coerce_int(payload.get("k", payload.get("n")), "k", 10, lo=1, hi=30),
        request_id=payload.get("request_id"),
    )


# ---------------------------------------------------------------------------
# /api/song-search payload shaping
# ---------------------------------------------------------------------------

def _shape_search_hit(row: dict[str, Any]) -> dict[str, Any]:
    """Map a NetEaseAPIClient.search_songs row to the JSON shape the
    frontend wants to render in a search-result card."""
    try:
        sid = int(row.get("netease_song_id"))
    except (TypeError, ValueError):
        sid = 0
    return {
        "netease_song_id": sid,
        "title":           row.get("title") or "",
        "artist":          row.get("artist") or "",
        "artists":         list(row.get("artists") or []),
        "album":           row.get("album") or "",
        "cover_url":       row.get("cover_url") or "",
        "netease_url":     f"https://music.163.com/#/song?id={sid}" if sid else "",
        "duration_ms":     row.get("duration_ms"),
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    *,
    netease_base_url: Optional[str] = None,
    load_kgrec: bool = True,
    eager: bool = True,
) -> Flask:
    """Construct the Flask app.

    Parameters
    ----------
    netease_base_url
        Override the ``config.NETEASE_API_BASE_URL`` for tests / non-default
        deployments.
    load_kgrec
        If False, skip loading the heavy KGRec :class:`RecommendationService`.
        ``/api/kgrec-recommend`` will answer 503. Useful for the smoke
        test where we don't want to depend on artefacts being trained.
    eager
        If True, initialise the service container before returning the
        app (default; production startup). If False, the holder
        initialises lazily on first request.
    """
    static_dir = _HERE / "static"
    app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="",
    )
    app.config["JSON_SORT_KEYS"] = False

    if eager:
        HOLDER.init(netease_base_url=netease_base_url, load_kgrec=load_kgrec)

    def _ensure() -> None:
        HOLDER.init(netease_base_url=netease_base_url, load_kgrec=load_kgrec)

    # ------------------------------------------------------------------
    # Static / index
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> Any:
        return send_from_directory(str(static_dir), "index.html")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> Any:
        _ensure()
        kgrec_status: dict[str, Any] = {
            "enabled":   HOLDER.kgrec_enabled,
            "ready":     HOLDER.kgrec_service is not None,
            "attempted": HOLDER.kgrec_attempted,
            "error":     HOLDER.kgrec_error,
        }
        if HOLDER.kgrec_service is not None:
            kgrec_status["model"] = HOLDER.kgrec_service.MODEL_NAME
            kgrec_status["catalogue_size"] = HOLDER.kgrec_service.catalogue_size()
            kgrec_status["vocabulary_size"] = HOLDER.kgrec_service.vocabulary_size()

        # Re-probe NetEase here so the status indicator flips when
        # the user starts/stops the upstream service mid-run.
        netease_alive = HOLDER.refresh_netease_alive()

        return jsonify({
            "ok":              True,
            "product_layer": {
                "name":           NeteaseRecommender.MODEL_NAME,
                "netease_alive":  netease_alive,
                "netease_base":   HOLDER.netease_base_url,
            },
            "research_layer": kgrec_status,
        })

    # ------------------------------------------------------------------
    # /api/song-search -- live real-song search via NetEase
    # ------------------------------------------------------------------

    @app.get("/api/song-search")
    def song_search() -> Any:
        _ensure()
        q = (request.args.get("q") or "").strip()
        try:
            limit = max(1, min(20, int(request.args.get("limit") or 10)))
        except (TypeError, ValueError):
            limit = 10

        if not q:
            return jsonify({"ok": True, "query": "", "items": []})

        # Cache lookup first (shared cache: any prior successful run
        # serves results even when NetEase is currently down).
        cache_key = f"songrec_demo:search-ui:{q}::{limit}"
        try:
            cached = HOLDER._cache.get_query(cache_key) if HOLDER._cache is not None else None  # noqa: SLF001
        except Exception:                                 # noqa: BLE001
            cached = None

        if cached is not None:
            return jsonify({
                "ok": True, "query": q, "cached": True,
                "items": [_shape_search_hit(r) for r in cached],
            })

        try:
            hits = HOLDER.client.search_songs(q, limit=limit)
        except NeteaseAPIError as exc:
            return jsonify({
                "ok":     False,
                "error":  f"NetEase API unavailable: {exc}",
                "items":  [],
            }), 503
        except Exception as exc:                          # noqa: BLE001
            log.exception("/api/song-search crashed")
            return jsonify({
                "ok":     False,
                "error":  f"Internal error: {exc}",
                "items":  [],
            }), 500

        try:
            if HOLDER._cache is not None:                  # noqa: SLF001
                HOLDER._cache.set_query(cache_key, hits)   # noqa: SLF001
        except Exception:                                  # noqa: BLE001
            pass

        return jsonify({
            "ok":    True,
            "query": q,
            "items": [_shape_search_hit(r) for r in hits],
        })

    # ------------------------------------------------------------------
    # /api/recommend -- main user-facing route (NetEase pipeline)
    # ------------------------------------------------------------------

    @app.post("/api/recommend")
    def recommend() -> Any:
        _ensure()

        if not request.is_json:
            return jsonify({
                "ok": False,
                "error": "Content-Type must be application/json",
            }), 415
        try:
            payload = request.get_json(silent=False) or {}
        except Exception as exc:                           # noqa: BLE001
            return jsonify({"ok": False, "error": f"Invalid JSON: {exc}"}), 400

        try:
            req = _build_real_song_request(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        try:
            resp = HOLDER.recommender.recommend(req)
        except Exception as exc:                           # noqa: BLE001
            log.exception("recommend() raised")
            return jsonify({"ok": False, "error": f"Internal error: {exc}"}), 500

        return jsonify({"ok": True, "data": resp.to_dict()})

    # ------------------------------------------------------------------
    # /api/kgrec-recommend -- developer-only debug route.
    #
    # Never advertised in the main UI flow. Hidden behind the Advanced
    # panel. Lets the developer poke the KGRec ALS service directly
    # with KGRec item IDs, bypassing the NetEase product layer.
    # ------------------------------------------------------------------

    @app.post("/api/kgrec-recommend")
    def kgrec_recommend() -> Any:
        _ensure()
        if not HOLDER.kgrec_enabled:
            return jsonify({
                "ok": False,
                "error": "KGRec debug route disabled (server started with --no-kgrec).",
            }), 503
        if HOLDER.kgrec_service is None:
            if not HOLDER.kgrec_attempted:
                return jsonify({
                    "ok": False,
                    "error": "KGRec service still loading; retry shortly.",
                }), 503
            return jsonify({
                "ok": False,
                "error": (
                    "KGRec service failed to load: "
                    f"{HOLDER.kgrec_error or 'unknown error'}. "
                    "Run preprocessing + ALS training first."
                ),
            }), 503
        if not request.is_json:
            return jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415
        try:
            payload = request.get_json(silent=False) or {}
        except Exception as exc:                           # noqa: BLE001
            return jsonify({"ok": False, "error": f"Invalid JSON: {exc}"}), 400

        from src.personalization import RecommendationRequest, SeedInput

        try:
            seed_ids = _coerce_str_list(payload.get("seed_ids"), "seed_ids", max_len=200)
            favorite_ids = _coerce_str_list(payload.get("favorite_ids"), "favorite_ids", max_len=200)
            tags = _coerce_str_list(payload.get("tags"), "tags", max_len=20)
            exclude_ids = _coerce_str_list(payload.get("exclude_ids"), "exclude_ids", max_len=200)
            content_weight = _coerce_float(payload.get("content_weight"), "content_weight", 0.25)
            novelty = _coerce_float(payload.get("novelty"), "novelty", 0.20)
            diversity = _coerce_float(payload.get("diversity"), "diversity", 0.0)
            k = _coerce_int(payload.get("k", payload.get("n")), "k", 10, lo=1, hi=50)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        try:
            kgrec_req = RecommendationRequest(
                seeds=SeedInput(
                    item_ids=seed_ids,
                    favorite_ids=favorite_ids,
                    tags=tags,
                    exclude_ids=exclude_ids,
                ),
                n=k,
                novelty=novelty,
                content_weight=content_weight,
                diversity=diversity,
                request_id=payload.get("request_id"),
            )
            resp = HOLDER.kgrec_service.recommend(kgrec_req)
        except Exception as exc:                           # noqa: BLE001
            log.exception("/api/kgrec-recommend crashed")
            return jsonify({"ok": False, "error": f"Internal error: {exc}"}), 500

        return jsonify({
            "ok": True,
            "warning": (
                "Developer debug route. KGRec item IDs are internal "
                "research-layer identifiers; the user-facing demo uses "
                "/api/recommend instead."
            ),
            "data": dataclasses.asdict(resp),
        })

    # ------------------------------------------------------------------
    # 404 fallback (SPA-ish)
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(_e: Any) -> Any:
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Unknown endpoint"}), 404
        return send_from_directory(str(static_dir), "index.html")

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind host (default: 127.0.0.1, local-only).")
    p.add_argument("--port", type=int, default=5173,
                   help="Bind port (default: 5173).")
    p.add_argument("--netease-base-url", default=None,
                   help=f"Override NETEASE_API_BASE_URL "
                        f"(default: {config.NETEASE_API_BASE_URL}).")
    p.add_argument("--no-kgrec", action="store_true",
                   help="Skip loading the KGRec research-layer service. "
                        "/api/kgrec-recommend will answer 503.")
    p.add_argument("--debug", action="store_true",
                   help="Enable Flask debug mode.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # IDE / piped runs often buffer stdout; unbuffered hints help when
    # startup looks "stuck" on the NetEase ping or heavy imports.
    print(
        "SongRecDemo: starting (imports + NetEase probe; "
        "this can take ~5–15s if http://localhost:3000 is down) …",
        file=sys.stderr,
        flush=True,
    )
    app = create_app(
        netease_base_url=args.netease_base_url,
        load_kgrec=(not args.no_kgrec),
        eager=True,
    )

    url = f"http://{args.host}:{args.port}"
    log.info("Starting demo server on %s", url)
    log.info("Open the URL in your browser. Press Ctrl+C to stop.")
    print(
        f"SongRecDemo: server ready — open {url} in a browser. "
        "This terminal stays busy until you press Ctrl+C.",
        file=sys.stderr,
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
