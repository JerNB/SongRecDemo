# NetEase Cloud Music API — local setup for the recommender demo

The personalized recommender uses an **optional** NetEase Cloud Music
API service for *display-side* metadata enrichment only — song titles,
artist names, album names, cover-art URLs, and a NetEase song-page
link. None of this metadata is read by the KGRec training pipeline,
the ALS model, or the validation/test evaluator. Turning the service
off changes nothing about ranking quality; it only changes how
recommendations look in the UI.

The reference implementation we depend on is:

> https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced

Two equivalent ways to run it locally are documented below. Pick
whichever you find easier to keep alive in the background while the
recommender demo is running.

---

## Option A — Local Node.js (preferred for development)

Requirements:

- Node.js **18+**
- `pnpm` (`npm install -g pnpm` if you don't have it)

Steps:

```bash
git clone https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced
cd api-enhanced
pnpm install
node app.js
```

You should see a banner that ends with something like:

```
server running @ http://0.0.0.0:3000
```

Leave that terminal running. The recommender demo will talk to it at
`http://localhost:3000` by default.

Quick sanity check from another terminal (PowerShell or bash):

```bash
curl "http://localhost:3000/search?keywords=hello%20adele&type=1&limit=1"
```

You should get back a JSON blob whose `result.songs[0]` has a `name`
and an `artists` array.

---

## Option B — Docker (preferred when the local toolchain is annoying)

The upstream project ships an image that exposes the same API on port
3000:

```bash
docker run -d --name netease-api -p 3000:3000 \
    binaryify/netease_cloud_music_api
```

(Or build from source with the `Dockerfile` in the repo if you want
the latest commits.) Verify with the same `curl` call as above. To
stop / restart:

```bash
docker stop netease-api
docker start netease-api
```

---

## Wiring the recommender demo to the API

### Default port (`http://localhost:3000`)

```bash
python run_personalized_demo.py --interactive --netease
```

The `--netease` flag attaches `NeteaseMetadataEnricher` on top of the
existing `InternalFeaturesEnricher`. For each ranked item the enricher

1. builds a best-effort search query from the KGRec
   description-snippet + tags,
2. calls `GET /search?keywords=...&type=1`,
3. picks the highest-scoring candidate that exceeds
   `NETEASE_MIN_CONFIDENCE` (default 0.35),
4. and merges the NetEase fields into `ScoredItem.metadata`.

If any step fails — service down, timeout, low confidence — the
metadata silently falls back to the internal enricher, so the demo
keeps working.

### Custom URL (Docker on a different port, remote host, etc.)

```bash
python run_personalized_demo.py --interactive --netease \
    --netease-base-url http://192.168.1.42:3000
```

Or set the environment variable once and forget about it:

```bash
# PowerShell
$env:NETEASE_API_BASE_URL = "http://192.168.1.42:3000"
python run_personalized_demo.py --interactive --netease
```

### Tunable knobs (all read from `config.py` or env vars)

| Setting                   | Default                              | Purpose                                      |
| ------------------------- | ------------------------------------ | -------------------------------------------- |
| `NETEASE_API_BASE_URL`    | `http://localhost:3000`              | Where the API service listens.               |
| `NETEASE_TIMEOUT_SECONDS` | `5.0`                                | HTTP timeout per call.                       |
| `NETEASE_MAX_RETRIES`     | `1`                                  | Retries on 5xx / network errors.             |
| `NETEASE_SEARCH_LIMIT`    | `5`                                  | How many candidates to fetch per query.      |
| `NETEASE_MIN_CONFIDENCE`  | `0.35`                               | Below this we keep internal metadata only.   |
| `NETEASE_CACHE_PATH`      | `artifacts/netease_cache.sqlite`     | On-disk cache of API responses + decisions.  |

The cache makes the demo fast (no repeat HTTP hits) and offline
friendly (once an item is enriched, the result survives the API going
away). Delete the SQLite file to force a refresh.

---

## Smoke-test the integration

Before relying on `--netease` in a live demo, run:

```bash
python run_netease_enrichment_smoke.py
```

It exercises four paths:

1. **API available** — pings `/search` and validates the JSON shape.
   Reported as `[SKIP]` (not `[FAIL]`) when the service is not
   running, so the script doubles as a "is the NetEase server up?"
   probe.
2. **API unavailable → fallback** — the enricher is pointed at a
   dead URL; metadata must still come back from the internal enricher
   with no NetEase fields leaking in.
3. **Low-confidence → fallback** — a stub client returns a
   deliberately irrelevant candidate; the score has to fall below
   threshold and the enricher has to record `netease_attempted=True`
   without polluting the metadata.
4. **Successful enrichment** — picks two KGRec items whose tags hint
   at well-known artists (`best-coast`, `bon-iver`, ...) and verifies
   that at least one ends up with a NetEase song id and title.

A passing run looks like:

```
[ OK ]  1) API available + valid /search response
[ OK ]  2) API unavailable -> fallback to internal metadata
[ OK ]  3) Low-confidence match -> fallback to internal metadata
[ OK ]  4) Successful end-to-end NetEase enrichment
SUMMARY: PASS=4  FAIL=0  SKIP=0  TOTAL=4
```

When the API is not running, tests 1 and 4 will report `[SKIP]` while
tests 2 and 3 still pass — which is the correct, graceful behaviour.

---

## Optional: enrichment quality report

`scripts/netease_quality_report.py` runs a fan of recommendation
requests against the live API and prints a match-rate / confidence
breakdown plus example matches. Use it whenever you change the
scoring heuristics in `src/personalization/netease_enrichment.py` to
make sure the change actually helps.

```bash
python scripts/netease_quality_report.py
```

Typical output on the current heuristics (KGRec catalogue, 8 mixed
seed scenarios, ~85 items):

- ~80% NetEase match rate, ~20% fallback to internal metadata.
- Confidence median ≈ 0.50, max ≈ 0.85.
- Top-confidence matches resolve both the artist and the album to
  the right values (e.g. *Dropkick Murphys — Rose Tattoo (Signed and
  Sealed In Blood)*, *Sufjan Stevens — Age of Adz*).
- Lowest-confidence accepted matches (≈0.40) tend to get the artist
  right but pick a different track — acceptable for a demo as long
  as the frontend surfaces `metadata.match_confidence`.
- Items with no proper-noun phrase in the description fall back to
  internal-only metadata, which is the right conservative behaviour.

A per-scenario JSON dump is written to
`artifacts/netease_quality_report.json` for further inspection.

---

## What this integration explicitly does **not** do

These are deliberate constraints, not future work:

- **No login / cookies.** The first version is read-only and uses
  only the open endpoints.
- **No playable URLs.** Stream availability depends on copyright,
  region, and login status; we do not surface `/song/url` results.
- **No KGRec ID == NetEase ID assumption.** The NetEase song id is
  stored *alongside* the KGRec item id under
  `metadata.netease_song_id`. Frontend code that wants to deep-link
  must use that field, not `item_id`.
- **No model/eval impact.** The training pipeline, ALS factors,
  validation/test splits, and metric computations are independent of
  this module. Disabling `--netease` (or letting the API go down) is
  guaranteed not to change ranking quality or evaluation numbers.
