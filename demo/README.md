# Demo scenarios

Three fixed profiles for live demos. Each `*.json` is a ready-to-POST
`/api/recommend` body (the `_demo_id` / `_label` keys are ignored by the
server, they are just labels for humans).

The exact songs returned depend on the live NetEase catalogue and on what is
already in the local feature store, so these scenarios deliberately do **not**
assert specific songs. Instead, each one lists the **trace** fields and
**score_breakdown** fields you should look at to show that the corresponding
subsystem is working.

## How to run a scenario

Start the server (`python SongRecDemo/app.py`), then:

```powershell
# PowerShell
curl -Method POST http://127.0.0.1:5173/api/recommend `
  -ContentType "application/json" `
  -InFile SongRecDemo\demo\content_heavy.json
```

```bash
# bash / curl
curl -s -X POST http://127.0.0.1:5173/api/recommend \
  -H "Content-Type: application/json" \
  --data @SongRecDemo/demo/content_heavy.json | python -m json.tool
```

Every response carries `data.trace` and, per card, `data.items[i].score_breakdown`.

---

## 1. `content_heavy.json` — stay close to my taste

High `content_weight` (0.85), low `novelty` / `diversity`. The list should lean
hard on the user's stated artists / songs / tags rather than on retrieval
consensus or surprise.

What to look at:

- `items[*].score_breakdown.content_score` — should be **high** for the top
  cards, and clearly above `retrieval_score` for most of them.
- `items[*].score_breakdown.content_weight` — echoes `0.85`, confirming the
  blend leaned on content.
- `items[*].score_breakdown.artist_match` / `tag_match` — non-zero where the
  candidate matches a liked artist or selected tag.
- `items[*].score_breakdown.novelty_bonus` — should be **small** (novelty
  slider is 0.1), i.e. the list is not being pushed toward unfamiliar songs.
- `items[*].pick_type` — expect mostly `safe` (and `balanced`), few/no
  `exploratory`.
- `candidate_summary` — `final_candidate_count > 0`; filters trimmed the pool.

---

## 2. `discovery_high_novelty.json` — discovery / high novelty

Low `content_weight` (0.4), very high `novelty` (0.9), high `diversity` (0.6).
The list should surface less obvious, more varied songs while still being
gated by relevance (novelty never promotes pure noise).

What to look at:

- `items[*].score_breakdown.novelty_score` and `novelty_bonus` — should be
  meaningfully **larger** than in the content-heavy scenario.
- `items[*].score_breakdown.relevance_gate` — shows novelty is still gated:
  low-relevance candidates get a smaller effective novelty bonus.
- `items[*].pick_type` — expect more `exploratory` / `diverse` labels.
- `trace.diversity` — echoes `0.6`; the MMR reranker spread artists/albums.
- `coverage` (via `/SongRecDemo/evaluation`) or simply scan `items[*].artist`
  — fewer repeated artists than the content-heavy run.

---

## 3. `embedding_recall.json` — local embedding recall channel

A taste profile (indie folk / mellow / acoustic guitar) with **no** liked
songs or artists, so retrieval leans on genre/tag search **plus** the local
embedding channel.

> Warm the catalogue first. Embedding recall only fires once the local feature
> store holds at least `EMBEDDING_MIN_CORPUS_SIZE` (default 20) songs. The
> store grows on every recommendation, so run a few recommendations first (or
> reuse an existing `data/song_feature_store.sqlite`).

What to look at:

- `trace.embedding_recall_enabled` — `true`.
- `trace.num_feature_store_songs` — `>= 20` once the catalogue is warm.
- `trace.embedding_index_ready` — `true` when the channel actually ran.
- `trace.num_embedding_candidates` — `>= 1`: songs pulled from the local
  catalogue, not from live NetEase search.
- `items[*].source_types` — at least one card includes `"embedding"`.
- `items[*].score_breakdown.multi_source_agreement` — songs found by both the
  search channel and the embedding channel get a higher value (consensus).

---

## Shadow learned ranker (optional overlay on any scenario)

If a trained model exists at `data/learned_ranker.joblib` and
`LEARNED_RANKER_SHADOW_MODE` is on (default), every scenario above also gets:

- `items[*].score_breakdown.learned_score` — the model's engagement
  probability in `[0, 1]`.
- `items[*].learned_rank_position` — where the learned model would have placed
  the card (1 = best).
- `trace.learned_ranker_loaded` — `true`; `trace.num_learned_scored` —
  number of cards scored.

Crucially the **displayed order does not change**: `rank` / `final_score` /
`rank_score` are still decided by the P0 rule ranker. Compare `rank` with
`learned_rank_position` to see where the model agrees or disagrees with the
rules.
