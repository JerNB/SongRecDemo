/* ----------------------------------------------------------------------
 * Real-song music recommender -- demo frontend.
 *
 * Talks to the local Flask backend (`SongRecDemo/app.py`):
 *   GET  /api/health                     -> service status banner
 *   GET  /api/song-search?q=...&limit=N  -> live NetEase real-song search
 *   POST /api/recommend                  -> NetEase-pipeline real-song recs
 *   POST /api/kgrec-recommend            -> developer-only KGRec debug route
 *
 * UX shape:
 *   1. The user types a song / artist / mood in the search box.
 *   2. The frontend hits /api/song-search and renders real-song result
 *      cards with cover + title + artist + album.
 *   3. The user clicks "+ Add" to add songs to "Songs I like".
 *   4. Optional comma-separated fields collect favourite artists,
 *      genres, moods, and other tags.
 *   5. Sliders + (k) compose the rest of the request.
 *   6. /api/recommend returns ranked real-song cards with title,
 *      artist, album, cover, NetEase link, explanation, score
 *      breakdown, matched tags, and a pick-type badge.
 *
 * The KGRec ALS model is the *research* layer and is reachable only
 * from the Advanced / developer panel. Normal users never see KGRec
 * item IDs in the main flow.
 * ---------------------------------------------------------------------- */

(function () {
  "use strict";

  // ============================== state ==============================

  const state = {
    picks: new Map(),          // netease_song_id -> TrackRef-shaped object
    lastSearch: { q: "", items: [] },
  };

  // ============================== DOM handles =========================

  const $form        = document.getElementById("rec-form");
  const $submit      = document.getElementById("submit-btn");
  const $example     = document.getElementById("example-btn");
  const $error       = document.getElementById("error");
  const $cards       = document.getElementById("cards");
  const $kgrecCards  = document.getElementById("kgrec-cards");
  const $empty       = document.getElementById("empty-state");
  const $loading     = document.getElementById("loading");
  const $resultsMeta = document.getElementById("results-meta");
  const $profileSum  = document.getElementById("profile-summary");
  const $modelInfo   = document.getElementById("model-info");
  const $statusDot   = document.getElementById("status-dot");
  const $statusText  = document.getElementById("status-text");

  const $searchQ      = document.getElementById("search-q");
  const $searchClear  = document.getElementById("search-clear");
  const $searchResults = document.getElementById("search-results");
  const $searchStatus = document.getElementById("search-status");

  const $picksChips  = document.getElementById("picks-chips");
  const $picksCount  = document.getElementById("picks-count");
  const $picksEmpty  = document.getElementById("picks-empty");

  const $kgrecBtn   = document.getElementById("kgrec-btn");

  // Slider value mirrors.
  const SLIDERS = ["content_weight", "novelty", "diversity", "k"];
  for (const id of SLIDERS) {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "_v");
    if (!el || !out) continue;
    const fmt = (id === "k")
      ? (v) => String(parseInt(v, 10))
      : (v) => Number(v).toFixed(2);
    out.textContent = fmt(el.value);
    el.addEventListener("input", () => { out.textContent = fmt(el.value); });
  }

  // ============================== utilities ==========================

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function parseCsv(raw) {
    if (!raw) return [];
    return raw.split(",").map((s) => s.trim()).filter(Boolean);
  }

  function debounce(fn, ms) {
    let t = 0;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  function httpsify(url) {
    return url ? String(url).replace(/^http:\/\//, "https://") : "";
  }

  function coverHtml(coverUrl, klass = "cover") {
    if (coverUrl) {
      const u = httpsify(coverUrl);
      return `<img class="${klass}" src="${escapeHtml(u)}" alt="" loading="lazy"
                 onerror="this.outerHTML='<div class=&quot;${klass}-fallback&quot;>—</div>'" />`;
    }
    return `<div class="${klass}-fallback">—</div>`;
  }

  // ============================== picks (chips) ======================

  function renderPickChip(item) {
    const li = document.createElement("span");
    li.className = "chip";
    const title  = item.title || "(untitled)";
    const artist = item.artist || "—";
    const cover  = item.cover_url
      ? `<img class="chip-cover" src="${escapeHtml(httpsify(item.cover_url))}"
              alt="" onerror="this.outerHTML='<div class=&quot;chip-cover-fallback&quot;></div>'" />`
      : `<div class="chip-cover-fallback"></div>`;
    li.innerHTML = `
      ${cover}
      <span class="chip-text">
        <span class="chip-title">${escapeHtml(title)}</span>
        <span class="chip-sub">${escapeHtml(artist)}</span>
      </span>
      <button type="button" class="chip-remove" aria-label="Remove" title="Remove">×</button>
    `;
    li.querySelector(".chip-remove").addEventListener("click", () => {
      state.picks.delete(item.netease_song_id);
      renderPicks();
      syncSearchAddButtons();
    });
    return li;
  }

  function renderPicks() {
    $picksChips.innerHTML = "";
    for (const item of state.picks.values()) {
      $picksChips.appendChild(renderPickChip(item));
    }
    $picksCount.textContent = String(state.picks.size);
    $picksEmpty.hidden = state.picks.size > 0;
  }

  function addPick(item) {
    if (!item || !item.netease_song_id) return;
    state.picks.set(item.netease_song_id, item);
    renderPicks();
    syncSearchAddButtons();
  }

  function syncSearchAddButtons() {
    $searchResults.querySelectorAll("button.add").forEach((btn) => {
      const id = parseInt(btn.dataset.songId, 10);
      const inSet = state.picks.has(id);
      btn.disabled = inSet;
      btn.textContent = inSet ? "✓ added" : "+ Add";
    });
  }

  // ============================== search =============================

  const runSearch = debounce(async (q) => {
    if (!q) {
      $searchResults.hidden = true;
      $searchResults.innerHTML = "";
      $searchStatus.textContent = "";
      state.lastSearch = { q: "", items: [] };
      return;
    }
    $searchStatus.textContent = "searching NetEase…";
    try {
      const url = `/api/song-search?q=${encodeURIComponent(q)}&limit=10`;
      const res = await fetch(url);
      const body = await res.json().catch(() => ({}));
      if (!res.ok || body.ok === false) {
        $searchStatus.textContent =
          body.error || `Search failed (${res.status}).`;
        $searchResults.hidden = true;
        return;
      }
      state.lastSearch = { q, items: body.items || [] };
      renderSearchResults(body.items || []);
      $searchStatus.textContent = (body.items.length
        ? `${body.items.length} matches`
        : "no matches")
        + (body.cached ? " · cached" : "");
    } catch (err) {
      $searchStatus.textContent = `network error: ${err.message || err}`;
      $searchResults.hidden = true;
    }
  }, 240);

  function renderSearchResults(items) {
    $searchResults.innerHTML = "";
    if (!items.length) { $searchResults.hidden = true; return; }
    for (const it of items) {
      const sub = [it.artist, it.album].filter(Boolean).join(" · ") || "—";
      const cover = it.cover_url
        ? `<img class="cover-thumb" src="${escapeHtml(httpsify(it.cover_url))}" alt=""
                onerror="this.outerHTML='<div class=&quot;cover-thumb-fallback&quot;></div>'" />`
        : `<div class="cover-thumb-fallback"></div>`;
      const div = document.createElement("div");
      div.className = "search-result";
      div.innerHTML = `
        ${cover}
        <div class="meta-line">
          <span class="name" title="${escapeHtml(it.title)}">${escapeHtml(it.title || "(untitled)")}</span>
          <span class="sub">${escapeHtml(sub)}</span>
        </div>
        <div class="actions">
          <button type="button" class="add"
                  data-song-id="${escapeHtml(String(it.netease_song_id))}">+ Add</button>
        </div>
      `;
      $searchResults.appendChild(div);
    }
    $searchResults.hidden = false;

    const byId = new Map(state.lastSearch.items.map((x) => [String(x.netease_song_id), x]));
    $searchResults.querySelectorAll("button.add").forEach((btn) => {
      btn.addEventListener("click", () => {
        const it = byId.get(btn.dataset.songId);
        if (!it) return;
        addPick(it);
      });
    });
    syncSearchAddButtons();
  }

  $searchQ.addEventListener("input", (e) => runSearch(e.target.value.trim()));
  $searchClear.addEventListener("click", () => {
    $searchQ.value = ""; runSearch("");
    $searchQ.focus();
  });

  // ============================== health =============================

  fetch("/api/health")
    .then((r) => r.json())
    .then((data) => {
      if (!data || data.ok === false) throw new Error("health failed");
      const product = data.product_layer || {};
      const research = data.research_layer || {};
      $statusDot.className = product.netease_alive ? "dot dot-ok" : "dot dot-warn";
      const aliveText = product.netease_alive
        ? "NetEase ✓"
        : "NetEase offline (cache only)";
      const researchText = research.enabled
        ? (research.ready ? `KGRec debug ✓` : "KGRec loading…")
        : "KGRec disabled";
      $statusText.textContent = `${product.name || "product"} · ${aliveText} · ${researchText}`;
      $modelInfo.textContent =
        `product: ${product.name || "—"} · `
        + `research: ${research.model || (research.enabled ? "loading…" : "disabled")}`;
    })
    .catch(() => {
      $statusDot.className = "dot dot-bad";
      $statusText.textContent = "backend unreachable";
    });

  // ============================== build payload ======================

  function buildPayload() {
    return {
      liked_songs: Array.from(state.picks.values()).map((p) => ({
        netease_song_id: p.netease_song_id,
        title:           p.title,
        artist:          p.artist,
        artists:         p.artists || [],
        album:           p.album || "",
        cover_url:       p.cover_url || "",
      })),
      liked_artists:  parseCsv(document.getElementById("liked_artists").value),
      genres:         parseCsv(document.getElementById("genres").value),
      moods:          parseCsv(document.getElementById("moods").value),
      tags:           parseCsv(document.getElementById("tags").value),
      content_weight: parseFloat(document.getElementById("content_weight").value),
      novelty:        parseFloat(document.getElementById("novelty").value),
      diversity:      parseFloat(document.getElementById("diversity").value),
      k:              parseInt(document.getElementById("k").value, 10),
    };
  }

  function setLoading(on) {
    $submit.disabled = on;
    $submit.textContent = on ? "Searching NetEase…" : "Get recommendations";
    $loading.hidden = !on;
    if (on) { $empty.hidden = true; $error.hidden = true; }
  }

  function showError(msg) {
    $error.textContent = msg;
    $error.hidden = false;
    $empty.hidden = ($cards.children.length + $kgrecCards.children.length) > 0;
  }

  // ============================== rendering: profile summary =========

  function renderProfileSummary(data) {
    const profile = data.profile || {};
    const ctrl = data.control || {};
    const cs = data.candidate_summary || {};
    const fb = data.fallback_used;

    const parts = [];
    if (profile.liked_song_ids?.length) {
      const titles = Array.from(state.picks.values())
        .map((p) => p.title || `#${p.netease_song_id}`);
      parts.push(`<strong>liked songs</strong> <span class="pill">${escapeHtml(titles.join(", "))}</span>`);
    }
    if (profile.liked_artists?.length) {
      parts.push(`<strong>artists</strong> <span class="pill">${escapeHtml(profile.liked_artists.join(", "))}</span>`);
    }
    if (profile.tags?.length) {
      parts.push(`<strong>tags</strong> <span class="pill">${escapeHtml(profile.tags.join(", "))}</span>`);
    }
    parts.push(
      `<strong>candidates</strong> <span class="pill">`
      + `${cs.total_unique || 0} unique · `
      + `artist ${cs.artist || 0} · `
      + `tag ${cs.tag || 0} · `
      + `title ${cs.title || 0} · `
      + `discovery ${cs.discovery || 0}`
      + `</span>`
    );
    if (fb) {
      parts.push(`<span class="badge-fallback">fallback: ${escapeHtml(fb)}</span>`);
    }

    $profileSum.innerHTML = parts.join(" · ");
    $profileSum.hidden = false;

    $resultsMeta.innerHTML =
      `k=${ctrl.k} · content=${(ctrl.content_weight||0).toFixed(2)} · `
      + `novelty=${(ctrl.novelty||0).toFixed(2)} · diversity=${(ctrl.diversity||0).toFixed(2)}`;
    $resultsMeta.hidden = false;
  }

  // ============================== rendering: cards ===================

  function bar(label, value, klass) {
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    const pct = (v * 100).toFixed(0) + "%";
    return `
      <div class="bar-row">
        <span>${label}</span>
        <span class="bar ${klass}"><span style="width:${pct}"></span></span>
        <span class="num">${v.toFixed(2)}</span>
      </div>`;
  }

  function pickBadge(pickType) {
    const pt = pickType || "safe";
    return `<span class="pick pick-${escapeHtml(pt)}">${escapeHtml(pt)}</span>`;
  }

  function renderHead(item) {
    const title = item.title || "(untitled)";
    const artist = item.artist || "Unknown artist";
    const album = item.album || "";
    const linkHtml = item.netease_url
      ? `<a class="netease-link" href="${escapeHtml(httpsify(item.netease_url))}"
            target="_blank" rel="noopener noreferrer">open on NetEase →</a>`
      : "";
    return `
      <div class="card-head">
        <h3 class="title">${escapeHtml(title)}</h3>
        <span class="artist">${escapeHtml(artist)}</span>
        ${album ? `<span class="album">· ${escapeHtml(album)}</span>` : ""}
        ${pickBadge(item.pick_type)}
        ${linkHtml}
      </div>`;
  }

  function renderBodyExtras(item) {
    let html = "";
    if (item.explanation) {
      html += `<p class="explanation">${escapeHtml(item.explanation)}</p>`;
    }
    if (Array.isArray(item.reasons) && item.reasons.length) {
      html += `<ul class="reasons">${
        item.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")
      }</ul>`;
    }
    if (Array.isArray(item.matched_tags) && item.matched_tags.length) {
      html += `<div class="tags">${
        item.matched_tags.slice(0, 6).map((t) =>
          `<span class="tag tag-strong">${escapeHtml(t)}</span>`).join("")
      }</div>`;
    }
    return html;
  }

  function renderBreakdown(item) {
    const sb = item.score_breakdown || {};
    return `
      <div class="breakdown">
        <div class="score-final">${(item.score ?? 0).toFixed(3)}</div>
        ${bar("artist",   sb.artist_match, "bar-als")}
        ${bar("tag",      sb.tag_match,    "bar-content")}
        ${bar("title",    sb.title_match,  "bar-content")}
        ${bar("retrieval",sb.retrieval,    "bar-pop")}
        ${bar("novelty",  sb.novelty_term, "bar-nov")}
      </div>`;
  }

  function renderCard(item) {
    const li = document.createElement("li");
    li.className = `card mode-full pick-${(item.pick_type || "safe")}`;
    li.innerHTML = `
      <div class="rank">${item.rank}</div>
      ${coverHtml(item.cover_url, "cover")}
      <div class="card-body">
        ${renderHead(item)}
        ${renderBodyExtras(item)}
      </div>
      ${renderBreakdown(item)}
    `;
    return li;
  }

  function renderResponse(data) {
    $cards.innerHTML = "";
    $kgrecCards.hidden = true;
    $kgrecCards.innerHTML = "";
    $empty.hidden = true;
    renderProfileSummary(data);
    if (!Array.isArray(data.items) || data.items.length === 0) {
      $cards.innerHTML = `<li class="empty">
        No real-song candidates came back. Try adding a liked song,
        an artist, or a different tag.
      </li>`;
      return;
    }
    const frag = document.createDocumentFragment();
    for (const item of data.items) frag.appendChild(renderCard(item));
    $cards.appendChild(frag);
  }

  // ============================== submit =============================

  async function submitMain(payload) {
    setLoading(true);
    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok || body.ok === false) {
        showError(body.error || `Request failed (${res.status}).`);
        return;
      }
      renderResponse(body.data);
    } catch (err) {
      showError(`Network error: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  }

  $form.addEventListener("submit", (e) => {
    e.preventDefault();
    submitMain(buildPayload());
  });

  // ============================== KGRec debug =========================

  function renderKgrecCard(item) {
    const md = item.metadata || {};
    const sb = item.score_breakdown || {};
    const li = document.createElement("li");
    li.className = "card mode-internal";
    const title = md.title || `KGRec item #${item.item_id}`;
    const artist = md.artist || "(no NetEase artist match)";
    li.innerHTML = `
      <div class="rank">${item.rank}</div>
      ${coverHtml(md.cover_url, "cover")}
      <div class="card-body">
        <div class="card-head">
          <h3 class="title">${escapeHtml(title)}</h3>
          <span class="artist">${escapeHtml(artist)}</span>
          <span class="confidence confidence-internal">KGRec id ${escapeHtml(item.item_id)}</span>
        </div>
        <p class="explanation">${escapeHtml(item.explanation || "")}</p>
        ${Array.isArray(item.reasons) && item.reasons.length
          ? `<ul class="reasons">${item.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
          : ""}
      </div>
      <div class="breakdown">
        <div class="score-final">${(item.score ?? 0).toFixed(3)}</div>
        ${bar("ALS",     sb.als,     "bar-als")}
        ${bar("content", sb.content, "bar-content")}
        ${bar("pop",     sb.popularity, "bar-pop")}
        ${bar("nov pen", sb.novelty_penalty, "bar-nov")}
      </div>
    `;
    return li;
  }

  $kgrecBtn.addEventListener("click", async () => {
    const seed_ids   = parseCsv(document.getElementById("seed_ids_raw").value);
    const fav_ids    = parseCsv(document.getElementById("favorite_ids_raw").value);
    const kgrec_tags = parseCsv(document.getElementById("kgrec_tags").value);
    if (!seed_ids.length && !fav_ids.length && !kgrec_tags.length) {
      showError("Provide at least one KGRec seed/favourite/tag for the debug route.");
      return;
    }
    $error.hidden = true;
    $kgrecBtn.disabled = true;
    $kgrecBtn.textContent = "Running KGRec…";
    try {
      const res = await fetch("/api/kgrec-recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed_ids, favorite_ids: fav_ids, tags: kgrec_tags,
          k: parseInt(document.getElementById("k").value, 10),
          content_weight: parseFloat(document.getElementById("content_weight").value),
          novelty: parseFloat(document.getElementById("novelty").value),
          diversity: parseFloat(document.getElementById("diversity").value),
        }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok || body.ok === false) {
        showError(body.error || `KGRec debug failed (${res.status}).`);
        return;
      }
      $kgrecCards.innerHTML = "";
      $kgrecCards.hidden = false;
      const banner = document.createElement("li");
      banner.className = "kgrec-banner";
      banner.innerHTML = `
        <strong>KGRec research-layer output (debug)</strong>
        <p class="muted">
          ${escapeHtml(body.warning || "Developer-only route. KGRec item IDs are research identifiers.")}
        </p>`;
      $kgrecCards.appendChild(banner);
      const items = (body.data && body.data.items) || [];
      for (const it of items) $kgrecCards.appendChild(renderKgrecCard(it));
      $empty.hidden = true;
    } catch (err) {
      showError(`Network error: ${err.message || err}`);
    } finally {
      $kgrecBtn.disabled = false;
      $kgrecBtn.textContent = "Run KGRec model (debug)";
    }
  });

  // ============================== example button =====================

  $example.addEventListener("click", async () => {
    state.picks.clear();
    document.getElementById("liked_artists").value = "Bon Iver, Phoebe Bridgers";
    document.getElementById("genres").value = "indie folk";
    document.getElementById("moods").value = "mellow";
    document.getElementById("tags").value = "";
    document.getElementById("content_weight").value = "0.50";
    document.getElementById("novelty").value        = "0.30";
    document.getElementById("diversity").value      = "0.30";
    document.getElementById("k").value              = "10";
    for (const id of SLIDERS) {
      document.getElementById(id).dispatchEvent(new Event("input"));
    }

    // Pull a known artist as a starter pick so the recommend call has
    // a concrete real-song seed.
    try {
      const res = await fetch("/api/song-search?q=Bon%20Iver&limit=4");
      const body = await res.json();
      const items = body.items || [];
      if (items[0]) addPick(items[0]);
    } catch { /* ignore: profile fields alone are enough */ }

    renderPicks();
    submitMain(buildPayload());
  });

  // Initial render.
  renderPicks();
})();
