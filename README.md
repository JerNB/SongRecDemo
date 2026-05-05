# SongRecDemo &mdash; Real-Song Music Recommender

A local-only web demo that recommends **real songs** &mdash; with title,
artist, album, cover art, and a NetEase Cloud Music link &mdash; from
user-friendly inputs (songs you like, artists you like, genres / moods
/ tags). The demo is split into two clearly separate layers and one
of them deliberately stays untouched.

```
┌───────────────────────────────────────────────────────────────────┐
│                      RESEARCH LAYER (untouched)                    │
│                                                                    │
│  src/personalization/RecommendationService                         │
│    - ALS (implicit MF), content (tag TF-IDF), popularity           │
│    - Trained on KGRec-music interaction data                       │
│    - Evaluated against KGRec test splits                           │
│    - Reachable only from /api/kgrec-recommend (developer debug)    │
└───────────────────────────────────────────────────────────────────┘
                                ▲
                                │ (debug route only)
                                │
┌───────────────────────────────┴──────────────────────────────────┐
│                       PRODUCT LAYER (this demo)                   │
│                                                                   │
│  SongRecDemo/netease_pipeline.py                                  │
│    - Profile builder       (liked songs + artists + tags)         │
│    - Candidate retrieval   (artist / tag / title / discovery)     │
│    - Score + MMR rerank    (KGRec-inspired logic, NetEase data)   │
│    - Explanations          (safe / exploratory / diverse picks)   │
│                                                                   │
│  SongRecDemo/app.py                                               │
│    - GET  /api/song-search   (live NetEase real-song search)      │
│    - POST /api/recommend     (real-song recommendations)          │
│    - GET  /api/health                                             │
│    - POST /api/kgrec-recommend  (research-layer debug)            │
└───────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP /search
                                ▼
                  Local NetEase Cloud Music API
                  (https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced)
```

> **Important:** the product layer is **not** trained on NetEase user
> behaviour data. It does not learn from user listens, plays, likes,
> playlists, or any private NetEase signal. It uses NetEase only as a
> public real-song catalog (via the `/search` endpoint) and ranks
> candidates with logic *inspired by* what the KGRec research layer
> demonstrates &mdash; profile building, content blending, novelty,
> diversity, and explanations.

---

## 1. What this demo does (and doesn't) change

The following research-layer artefacts are completely untouched:

- `src/recommenders/` &mdash; popularity, content-based, ALS.
- `src/personalization/` &mdash; engine, profile builder, explainer,
  enrichment Protocol, NetEase enricher (used as a side-channel by the
  research-layer evaluator only).
- `src/data/` &mdash; preprocessing, splits, artefact loader.
- `src/evaluation/` and the `run_*_eval.py` scripts.
- `artifacts/splits/`, `artifacts/models/als_state.pkl`,
  `artifacts/results/`, `artifacts/sample_request.json`,
  `artifacts/sample_response.json`.
- `config.py` &mdash; pipeline + NetEase configuration.
- All saved KGRec validation / evaluation metrics.

The demo:

- **Adds** `SongRecDemo/netease_pipeline.py` &mdash; the new product-layer
  pipeline.
- **Rewrites** `SongRecDemo/app.py`, `SongRecDemo/static/index.html`, and
  `SongRecDemo/static/app.js` to expose the new pipeline as the
  primary user-facing flow, and exposes the KGRec model only behind
  an Advanced (developer) panel.
- **Reuses** the existing on-disk NetEase SQLite cache
  (`artifacts/netease_cache.sqlite`) so repeated calls are fast and
  offline-friendly.

---

## 2. Folder layout

```
SongRecDemo/
├─ app.py                   Flask backend (search / recommend / health / kgrec-debug)
├─ netease_pipeline.py      Product-layer pipeline + FakeNeteaseClient
├─ catalog.py               (legacy helper; unused by the new flow but kept on disk)
├─ smoke_test.py            Hermetic smoke test (no network)
├─ requirements.txt         Demo-only dependency: flask>=3.0
└─ static/
   ├─ index.html            Real-song search + picks + sliders + results
   ├─ styles.css            Calm dark UI
   └─ app.js                Real-song flow + KGRec debug button
```

The UI binds to **127.0.0.1 only** (local-only on purpose).

---

## 3. Prerequisites

You only need a **local NetEase Cloud Music API service** running for
the product layer to do anything useful. Nothing else is required for
the main user flow.

This repo ships a one-shot helper that clones the upstream service
into `vendor/api-enhanced/` (gitignored), installs its dependencies
through the npmmirror.com registry (much faster from China), and
starts the Node.js app on `http://localhost:3000`:

```powershell
scripts\start_netease_api.bat
```

Leave that terminal window open while you're using the demo. To stop
the NetEase service, close the window or hit `Ctrl+C` in it.

If you'd rather drive the upstream service by hand, the equivalent
commands are:

```powershell
# One-off:
git clone https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced vendor\api-enhanced
cd vendor\api-enhanced
npm install --omit=dev --ignore-scripts ^
    --registry=https://registry.npmmirror.com --no-audit --no-fund

# Every run:
node app.js                  # listens on localhost:3000 by default
```

Quick sanity check that the API is up:

```powershell
curl "http://localhost:3000/search?keywords=kanye&type=1&limit=1"
```

You should see a JSON blob with `result.songs[0].name`.

The Python demo speaks to the service via `config.NETEASE_API_BASE_URL`
(default `http://localhost:3000`). `/api/health` re-probes the URL on
every request, so you can start/stop the NetEase service without
restarting the Python demo &mdash; the green/amber dot updates in the
browser within seconds.

If you also want the **KGRec research-layer debug route** to work,
you need the trained ALS state on disk. Skip this section entirely if
you only care about the real-song flow.

```powershell
pip install -r requirements.txt
python run_preprocessing.py            # parquet splits + TF-IDF
python run_train_personalized.py       # writes artifacts/models/als_state.pkl
```

---

## 4. Install the demo dependency

```powershell
pip install -r SongRecDemo/requirements.txt
```

This adds **only** `flask>=3.0`.

---

## 5. Run the server

From the project root:

```powershell
python SongRecDemo/app.py
```

Default bind: `http://127.0.0.1:5173` (local-only). Open that URL in
your browser. The status pill in the top-right shows:

- **green** &mdash; product layer up, NetEase reachable.
- **amber** &mdash; product layer up, NetEase offline (search / recommend
  fall back to the on-disk cache; uncached queries return 503 until
  NetEase is reachable again).
- **red** &mdash; backend itself is unreachable.

### CLI flags

| Flag                        | Default                                   | Purpose                                                            |
| --------------------------- | ----------------------------------------- | ------------------------------------------------------------------ |
| `--host`                    | `127.0.0.1`                               | Bind host. Local-only by default.                                  |
| `--port`                    | `5173`                                    | Bind port.                                                         |
| `--netease-base-url URL`    | `http://localhost:3000`                   | Override the NetEase API URL.                                      |
| `--no-kgrec`                | off                                       | Skip the heavy research-layer load. `/api/kgrec-recommend` -> 503. |
| `--debug`                   | off                                       | Flask debug mode.                                                  |

`--no-kgrec` is convenient for fast iteration on the product layer
when you don't care about the debug route. The KGRec service loads
in a background thread on startup so the homepage is responsive even
when the artefacts are large.

---

## 6. Use the UI

1. **Search real songs** &mdash; type a song name, artist, or vibe in
   the search box. Results come straight from NetEase via
   `GET /api/song-search`.
2. **Add picks** &mdash; click `+ Add` to put a song into "Songs I
   like". Each pick is a real song with cover art + title + artist.
3. **Profile fields (optional)** &mdash; comma-separated
   - **Favourite artists**
   - **Genres** (e.g. `indie, folk`)
   - **Moods** (e.g. `mellow, energetic`)
   - **Other tags** (e.g. `summer, 80s`)
   The backend merges genres + moods + tags into one `tags` bag.
4. **Sliders** &mdash; content similarity, novelty, diversity, k.
5. **Get recommendations** &mdash; `POST /api/recommend`. Each card
   shows title, artist, album, cover art, NetEase link, an
   explanation, a list of reasons, the matched tags, and a pick-type
   badge.

Pick-type badges:

| Badge                                     | Meaning                                                             |
| ----------------------------------------- | ------------------------------------------------------------------- |
| <code>safe</code>                         | Same artist as one of your picks, or strong content overlap.        |
| <code>exploratory</code>                  | Long-tail pick &mdash; different artist, deeper search rank.        |
| <code>diverse</code>                      | MMR pulled this card up to broaden the result list.                 |

### Advanced (developer) panel

Hidden behind a `<details>` toggle in the form. Lets you call
`POST /api/kgrec-recommend` directly with raw KGRec item IDs. This
is the **research-layer** debug route &mdash; the KGRec model can only
recommend KGRec items, so the cards there carry KGRec IDs and a
"KGRec id ###" badge instead of pretending to be real songs.

This is a tool for inspecting the trained model, not for end users.
Running the server with `--no-kgrec` disables the route entirely.

---

## 7. JSON contracts

### `GET /api/song-search?q=...&limit=N`

Real-song search via NetEase, with on-disk cache hit-through.

```jsonc
{
  "ok": true,
  "query": "bon iver",
  "items": [
    {
      "netease_song_id": 12345,
      "title": "Holocene",
      "artist": "Bon Iver",
      "artists": ["Bon Iver"],
      "album": "Bon Iver, Bon Iver",
      "cover_url": "https://...",
      "netease_url": "https://music.163.com/#/song?id=12345",
      "duration_ms": 337000
    }
  ]
}
```

If NetEase is unreachable and the query isn't cached, returns 503
with a clear error message and an empty `items` list.

### `POST /api/recommend`

Real-song recommendations. The whole request is **NetEase-shaped**:

```jsonc
{
  "liked_songs": [
    {
      "netease_song_id": 12345,
      "title": "Holocene",
      "artist": "Bon Iver",
      "artists": ["Bon Iver"],
      "album": "Bon Iver, Bon Iver",
      "cover_url": "https://..."
    }
  ],
  "liked_artists": ["Phoebe Bridgers"],
  "genres":        ["indie folk"],
  "moods":         ["mellow"],
  "tags":          [],
  "content_weight": 0.50,
  "novelty":        0.30,
  "diversity":      0.30,
  "k":              10
}
```

Response:

```jsonc
{
  "ok": true,
  "data": {
    "request_id": "....",
    "items": [
      {
        "rank": 1,
        "netease_song_id": 22345,
        "title": "Motion Sickness",
        "artist": "Phoebe Bridgers",
        "artists": ["Phoebe Bridgers"],
        "album": "Stranger in the Alps",
        "cover_url": "https://...",
        "netease_url": "https://music.163.com/#/song?id=22345",
        "score": 0.812,
        "score_breakdown": {
          "final": 0.812,
          "content": 0.560,
          "artist_match": 1.000,
          "tag_match": 0.500,
          "title_match": 0.000,
          "retrieval": 1.000,
          "multi_source": 0.000,
          "novelty_term": 0.000
        },
        "explanation": "Safe pick from Phoebe Bridgers, who is in your liked artists.",
        "reasons": [
          "Same artist as someone you like: Phoebe Bridgers",
          "Matches your tags: indie folk, mellow",
          "Found via multiple signals: artist, tag"
        ],
        "matched_tags": ["indie folk", "mellow"],
        "sources": ["artist:Phoebe Bridgers", "tag:indie folk"],
        "pick_type": "safe"
      }
    ],
    "control":           { "content_weight": 0.5, "novelty": 0.3, "diversity": 0.3, "k": 10 },
    "candidate_summary": { "artist": 4, "tag": 3, "title": 4, "discovery": 4, "total_unique": 11 },
    "profile":           { "liked_song_ids": [12345], "liked_artists": ["Phoebe Bridgers"], "tags": ["indie folk", "mellow"], ... },
    "model_info": {
      "name": "NetEase-Pipeline-v1",
      "research_layer": "ALS-Personalized-v1 (KGRec, untouched)",
      "source": "NetEase /search"
    },
    "fallback_used": null
  }
}
```

The `data.items` payload **never** carries a KGRec `item_id`. The
smoke test asserts this.

### `GET /api/health`

```jsonc
{
  "ok": true,
  "product_layer":  {
    "name": "NetEase-Pipeline-v1",
    "netease_alive": true,
    "netease_base":  "http://localhost:3000"
  },
  "research_layer": {
    "enabled":   true,
    "ready":     true,
    "attempted": true,
    "error":     null,
    "model":            "ALS-Personalized-v1",
    "catalogue_size":   8640,
    "vocabulary_size":  8477
  }
}
```

### `POST /api/kgrec-recommend` (developer debug)

Same input shape as the old KGRec `/api/recommend` (`seed_ids`,
`favorite_ids`, `tags`, sliders, `k`), wraps `RecommendationService`
unchanged, returns the standard `RecommendationResponse` plus a
`warning` string. 503 when the research layer is disabled or still
loading.

---

## 8. The recommendation pipeline (logic notes)

The product layer is intentionally model-light: it relies on NetEase
search as a retrieval engine and on simple, well-justified content
features for ranking. The high-level loop is:

1. **Profile** &mdash; liked songs contribute their artists + title
   tokens; explicit liked artists are added; tags / genres / moods
   merge into one bag of tokens.
2. **Retrieve** &mdash; up to 4 search channels:
   - per-artist (top liked artists),
   - per-tag (top tag phrases),
   - per-title (top liked song titles),
   - discovery (one broad combo query).
   Each candidate remembers which channels surfaced it and its rank
   in each channel.
3. **Score** &mdash; per-candidate normalised sub-scores in [0, 1]:
   - `artist_match` = 1.0 on exact match, else token Jaccard.
   - `tag_match` = fraction of profile tag tokens in the candidate's
     metadata bag (title + artist + album).
   - `title_match` = liked-title token overlap, scaled by sqrt of
     vocabulary size.
   - `retrieval` = position-decayed average across channels, plus a
     multi-source bonus (capped).
   Final score blends these with the user's sliders:

   ```
   content   = 0.50 * artist_match + 0.30 * tag_match + 0.20 * title_match
   final     = (1 - content_weight) * retrieval
             + content_weight       * content
             + novelty              * (1 - artist_in_liked) * (1 - retrieval)
   ```
4. **MMR rerank** &mdash; greedy diversification with `lambda = diversity`.
   Similarity is `0.7 * same-artist + 0.3 * tag-Jaccard`. A
   per-artist cap (default 2) prevents the result list from being
   dominated by one artist's catalogue.
5. **Explain** &mdash; each card gets a one-line explanation, a list of
   reasons, the matched tags, and a `pick_type` of `safe`,
   `exploratory`, or `diverse`.

Caching: every NetEase `/search` call is cached on disk in
`artifacts/netease_cache.sqlite` (the same cache the research-layer
enricher uses; the keys are namespaced so the two callers never
collide). Repeated demo runs with the same artists / tags are
near-instant.

---

## 9. Smoke test

```powershell
python SongRecDemo/smoke_test.py
```

The smoke test is **fully hermetic**: it injects a
`FakeNeteaseClient` (in `netease_pipeline.py`) with canned responses
for the queries it exercises, and constructs the Flask app with
`load_kgrec=False` so it does not depend on trained artefacts being
present.

Checks:

1. `GET /api/health` &mdash; product layer up, NetEase reported alive.
2. `GET /api/song-search?q=...` &mdash; returns NetEase-shaped real-song
   hits with `netease_song_id`, `title`, `artist`, `album`,
   `cover_url`, `netease_url`. Empty query -> 200 + empty list.
3. `POST /api/recommend` (real-song flow) &mdash; selecting real songs
   and tags returns ranked cards.
4. Recommendation cards include title, artist, album, NetEase link,
   explanation, score breakdown, matched tags, and a `pick_type`
   label.
5. The main `/api/recommend` response **never** carries a KGRec
   `item_id` field (recursive walk asserts).
6. Empty input -> response with `fallback_used = "no_input"`,
   200 OK.
7. Bad body -> 400 / wrong content type -> 415.
8. `/api/kgrec-recommend` answers 503 when the research layer is
   disabled (`--no-kgrec`).

Exit code is `0` on full pass and `1` otherwise.

---

## 10. Why the KGRec model isn't the user-facing recommender

The KGRec ALS model has one row per KGRec item in its `item_factors`
matrix. It can only score those 8&nbsp;640 KGRec items, and KGRec
items don't all map to a real-song NetEase match (the catalog of
matched items is sparse and depends on what the offline NetEase
enrichment was able to find). So:

- If we made the KGRec model the user-facing recommender, the user
  would pick songs (or KGRec IDs) and then receive KGRec IDs back.
  Many of those would have no real-song metadata and would render as
  "internal-only" cards &mdash; a description blurb instead of a
  song.
- If we filtered KGRec output to only matched items, the candidate
  pool collapses and many recommendations become repetitive.

Instead, the demo uses the KGRec model where it is honest: as the
*research artifact* that demonstrates the personalization logic, and
exposes that model only via a developer-facing debug route. The
user-facing experience is built on top of NetEase search directly,
with the same shape of logic the research demonstrates &mdash;
profile, content blend, novelty, diversity, explanation &mdash; but
operating over real-song retrieval.
