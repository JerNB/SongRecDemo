/* Real-song music recommender frontend. Main flow hides KGRec item IDs. */
(function () {
  "use strict";

  const state = {
    selectedSongs: new Map(),
    lastSearchItems: [],
    isSearching: false,
    loadingTimer: 0,
    loadingStageIndex: 0,
  };
  const $ = (id) => document.getElementById(id);
  const els = {
    form: $("rec-form"), introSubmit: $("intro-submit"), submit: $("submit-btn"),
    error: $("error"), statusDot: $("status-dot"), statusText: $("status-text"),
    modelInfo: $("model-info"), searchQ: $("search-q"), searchBtn: $("search-btn"),
    searchStatus: $("search-status"), searchResults: $("search-results"),
    picksList: $("picks-list"), picksCount: $("picks-count"), picksEmpty: $("picks-empty"),
    tasteChips: $("taste-chips"), loading: $("loading"), loadingStage: $("loading-stage"), empty: $("empty-state"),
    cards: $("cards"), resultsMeta: $("results-meta"), resultsSubtitle: $("results-subtitle"),
    technicalDetails: $("technical-details"), technicalBody: $("technical-body"),
  };
  const sliders = { content_weight: $("content_weight"), novelty: $("novelty"), diversity: $("diversity") };
  const presets = {
    safe: { content_weight: 0.85, novelty: 0.15, diversity: 0.20 },
    balanced: { content_weight: 0.60, novelty: 0.40, diversity: 0.50 },
    discovery: { content_weight: 0.50, novelty: 0.80, diversity: 0.75 },
  };
  const examples = {
    "alt-rock": { query: "Radiohead Karma Police", artists: "Radiohead, Coldplay", genres: "alternative rock, indie rock", moods: "sad, mellow", tags: "piano, alternative", preset: "safe" },
    mandopop: { query: "Jay Chou", artists: "Jay Chou, Eason Chan", genres: "Mandopop", moods: "nostalgic, romantic", tags: "", preset: "balanced" },
    broad: { query: "electronic indie pop", artists: "", genres: "electronic, indie pop", moods: "energetic, dreamy", tags: "synth, dance", preset: "discovery" },
  };
  const loadingStages = [
    "Reading taste anchors...",
    "Finding close matches...",
    "Adding discovery picks...",
    "Ranking playlist...",
  ];
  const scoreLabels = {
    final: "Overall fit", final_score: "Overall fit", content: "Taste match", artist_match: "Artist affinity",
    tag_match: "Genre and mood match", title_match: "Liked-song similarity",
    retrieval: "Retrieval confidence", multi_source: "Multi-source agreement",
    novelty_term: "Novelty", diversity_boost: "Diversity promotion",
    metadata_quality_score: "Metadata quality",
    // collaborative_proxy_score is a legacy alias of multi_source_agreement;
    // it is a retrieval-consensus signal, not collaborative filtering.
    collaborative_proxy_score: "Multi-source agreement", multi_source_agreement: "Multi-source agreement",
    content_score: "Taste match (content)", retrieval_score: "Retrieval score", quality_score: "Quality score",
    personalized_relevance: "Personalized relevance", base_relevance: "Base relevance", rank_score: "List rank score",
    retrieval_confidence_score: "Retrieval confidence", artist_affinity_score: "Artist affinity",
    novelty_score: "Discovery value",
  };
  const NETEASE_OFFLINE_HINT = "NetEase service is offline. Start the local NetEase API on port 3000, then try again.";
  const BACKEND_HINT = "Backend is unreachable. Start the Flask demo server, then try again.";

  async function fetchJson(url, options = {}, timeoutMs = 8000) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      let body;
      try {
        body = await response.json();
      } catch (error) {
        const parseError = new Error("The backend returned invalid JSON. Restart the Flask demo server, then try again.");
        parseError.code = "invalid_json";
        parseError.response = response;
        parseError.cause = error;
        throw parseError;
      }
      return { response, body };
    } catch (error) {
      if (error && error.name === "AbortError") {
        const timeoutError = new Error(`Request timed out after ${Math.round(timeoutMs / 1000)} seconds.`);
        timeoutError.code = "timeout";
        throw timeoutError;
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function parseCsv(raw) { return String(raw || "").split(",").map((s) => s.trim()).filter(Boolean); }
  function uniq(values) {
    const out = [], seen = new Set();
    for (const value of values) {
      const s = String(value).trim(), key = s.toLowerCase();
      if (s && !seen.has(key)) { seen.add(key); out.push(s); }
    }
    return out;
  }
  function httpsify(url) { return url ? String(url).replace(/^http:\/\//i, "https://") : ""; }
  function formatDuration(ms) {
    const n = Number(ms);
    if (!Number.isFinite(n) || n <= 0) return "";
    const total = Math.round(n / 1000);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  }
  function debounce(fn, delay) {
    let timer = 0;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
  }
  function coverMarkup(url, className) {
    const safeUrl = httpsify(url);
    if (!safeUrl) return `<div class="${className} cover-fallback" aria-hidden="true"></div>`;
    return `<img class="${className}" src="${escapeHtml(safeUrl)}" alt="" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'), {className: '${className} cover-fallback'}))" />`;
  }
  function songId(song) { return String(song?.netease_song_id || ""); }
  function normalizeSong(song) {
    return {
      netease_song_id: song.netease_song_id, title: song.title || "",
      artist: song.artist || "", artists: Array.isArray(song.artists) ? song.artists : [],
      album: song.album || "", cover_url: song.cover_url || "",
      netease_url: song.netease_url || (song.netease_song_id ? `https://music.163.com/#/song?id=${song.netease_song_id}` : ""),
      duration_ms: song.duration_ms || null,
    };
  }
  function showError(message) { els.error.textContent = message; els.error.hidden = false; }
  function clearError() { els.error.textContent = ""; els.error.hidden = true; }

  function updateSliderOutputs() {
    for (const id of Object.keys(sliders)) $(`${id}_v`).textContent = Number(sliders[id].value).toFixed(2);
    const k = Math.max(1, Math.min(30, Number($("k").value) || 10));
    $("k").value = String(k);
    $("k_v").textContent = `${k} song${k === 1 ? "" : "s"}`;
  }
  function setPreset(name) {
    const preset = presets[name] || presets.balanced;
    for (const [key, value] of Object.entries(preset)) sliders[key].value = String(value);
    document.querySelectorAll(".preset").forEach((btn) => btn.classList.toggle("active", btn.dataset.preset === name));
    updateSliderOutputs();
  }
  function renderTasteChips() {
    const chips = [
      ...parseCsv($("liked_artists").value).map((v) => ["Artist", v]),
      ...parseCsv($("genres").value).map((v) => ["Genre", v]),
      ...parseCsv($("moods").value).map((v) => ["Mood", v]),
      ...parseCsv($("tags").value).map((v) => ["Tag", v]),
    ];
    els.tasteChips.innerHTML = chips.map(([kind, value], index) =>
      `<span class="taste-chip" style="animation-delay:${Math.min(index * 24, 160)}ms"><b>${escapeHtml(kind)}</b>${escapeHtml(value)}</span>`).join("");
  }

  function renderSelectedSongs() {
    const songs = Array.from(state.selectedSongs.values());
    els.picksCount.textContent = `${songs.length} selected`;
    els.picksCount.classList.remove("bump");
    void els.picksCount.offsetWidth;
    els.picksCount.classList.add("bump");
    els.picksEmpty.hidden = songs.length > 0;
    els.picksList.innerHTML = "";
    for (const [index, song] of songs.entries()) {
      const node = document.createElement("div");
      node.className = "selected-song";
      node.style.animationDelay = `${Math.min(index * 35, 180)}ms`;
      node.innerHTML = `
        ${coverMarkup(song.cover_url, "mini-cover")}
        <div class="song-copy">
          <strong>${escapeHtml(song.title || "Untitled song")}</strong>
          <span>${escapeHtml([song.artist, song.album].filter(Boolean).join(" - ") || "Unknown artist")}</span>
        </div>
        <button type="button" class="icon-btn" title="Remove selected song" aria-label="Remove ${escapeHtml(song.title || "song")}">Remove</button>`;
      node.querySelector("button").addEventListener("click", () => {
        state.selectedSongs.delete(songId(song)); renderSelectedSongs(); syncSearchButtons();
      });
      els.picksList.appendChild(node);
    }
  }
  function addSong(song) {
    const normalized = normalizeSong(song);
    if (!songId(normalized)) return;
    state.selectedSongs.set(songId(normalized), normalized);
    renderSelectedSongs(); syncSearchButtons();
  }
  function syncSearchButtons() {
    els.searchResults.querySelectorAll("[data-add-song]").forEach((btn) => {
      const added = state.selectedSongs.has(btn.dataset.addSong);
      btn.disabled = added || state.isSearching;
      btn.textContent = added ? "Added" : "Add as liked song";
    });
  }
  function renderSearchResults(items) {
    state.lastSearchItems = items.map(normalizeSong);
    els.searchResults.innerHTML = "";
    if (!items.length) {
      els.searchResults.hidden = true;
      els.searchStatus.textContent = "No songs found. Try artist + song title, such as 'Adele Hello'.";
      return;
    }
    const lookup = new Map(state.lastSearchItems.map((item) => [songId(item), item]));
    for (const [index, item] of state.lastSearchItems.entries()) {
      const duration = formatDuration(item.duration_ms);
      const card = document.createElement("article");
      card.className = "search-card";
      card.style.animationDelay = `${Math.min(index * 28, 180)}ms`;
      card.innerHTML = `
        ${coverMarkup(item.cover_url, "search-cover")}
        <div class="song-copy">
          <strong>${escapeHtml(item.title || "Untitled song")}</strong>
          <span>${escapeHtml(item.artist || "Unknown artist")}</span>
          <em>${escapeHtml([item.album, duration].filter(Boolean).join(" - "))}</em>
        </div>
        <button type="button" class="secondary compact-btn" data-add-song="${escapeHtml(songId(item))}">Add as liked song</button>`;
      card.querySelector("button").addEventListener("click", (event) => addSong(lookup.get(event.currentTarget.dataset.addSong)));
      els.searchResults.appendChild(card);
    }
    els.searchResults.hidden = false;
    els.searchStatus.textContent = `${items.length} song${items.length === 1 ? "" : "s"} found`;
    syncSearchButtons();
  }
  async function searchSongs(query) {
    const q = String(query || "").trim();
    if (!q) {
      els.searchResults.hidden = true; els.searchResults.innerHTML = "";
      els.searchStatus.textContent = ""; state.lastSearchItems = []; return;
    }
    state.isSearching = true; els.searchBtn.disabled = true;
    els.searchStatus.textContent = "Searching songs..."; syncSearchButtons();
    try {
      const { response, body } = await fetchJson(`/api/song-search?q=${encodeURIComponent(q)}&limit=10`, {}, 7000);
      if (!response.ok || body.ok === false) {
        const offline = response.status === 503 || /netease/i.test(body.error || "");
        if (!state.lastSearchItems.length) els.searchResults.hidden = true;
        els.searchStatus.textContent = offline
          ? NETEASE_OFFLINE_HINT
          : (body.error || `Search failed (${response.status}).`);
        return;
      }
      renderSearchResults(body.items || []);
    } catch (error) {
      if (!state.lastSearchItems.length) els.searchResults.hidden = true;
      if (error.code === "timeout") {
        els.searchStatus.textContent = `Song search timed out. ${NETEASE_OFFLINE_HINT}`;
      } else if (error.code === "invalid_json") {
        els.searchStatus.textContent = error.message;
      } else {
        els.searchStatus.textContent = `Could not search songs. ${NETEASE_OFFLINE_HINT}`;
      }
    } finally {
      state.isSearching = false; els.searchBtn.disabled = false; syncSearchButtons();
    }
  }
  const debouncedSearch = debounce(() => searchSongs(els.searchQ.value), 320);

  function buildPayload() {
    return {
      liked_songs: Array.from(state.selectedSongs.values()).map((song) => ({
        netease_song_id: song.netease_song_id, title: song.title, artist: song.artist,
        artists: song.artists, album: song.album, cover_url: song.cover_url,
        netease_url: song.netease_url,
      })),
      liked_artists: uniq(parseCsv($("liked_artists").value)),
      genres: uniq(parseCsv($("genres").value)),
      moods: uniq(parseCsv($("moods").value)),
      tags: uniq(parseCsv($("tags").value)),
      content_weight: Number(sliders.content_weight.value),
      novelty: Number(sliders.novelty.value),
      diversity: Number(sliders.diversity.value),
      k: Number($("k").value) || 10,
    };
  }
  function hasInput(payload) {
    return Boolean(payload.liked_songs.length || payload.liked_artists.length || payload.genres.length || payload.moods.length || payload.tags.length);
  }
  function reasonChips(item) {
    const chips = [], reasons = (item.reasons || []).join(" ").toLowerCase();
    const sources = item.sources || [], breakdown = item.score_breakdown || {};
    if (reasons.includes("artist") || breakdown.artist_match > 0.5) chips.push("same artist");
    if ((item.matched_tags || []).length || breakdown.tag_match > 0) chips.push("matched genre");
    if (reasons.includes("mood")) chips.push("matched mood");
    if (sources.length > 1 || breakdown.multi_source > 0) chips.push("multi-source match");
    if (item.pick_type === "exploratory" || breakdown.novelty_term > 0.25) chips.push("discovery pick");
    if (item.pick_type === "diverse") chips.push("diversity pick");
    if (!chips.length && item.pick_type) chips.push(`${item.pick_type} pick`);
    return uniq(chips).slice(0, 6);
  }
  function readableScoreRows(scoreBreakdown) {
    const sb = scoreBreakdown || {};
    const ordered = [
      ["content_score", "Taste match (content)"], ["retrieval_score", "Retrieval score"],
      ["multi_source_agreement", "Multi-source agreement"], ["quality_score", "Quality score"],
      ["artist_match", "Artist affinity"], ["tag_match", "Genre and mood match"],
      ["title_match", "Liked-song similarity"], ["novelty_score", "Discovery value"],
      ["metadata_quality_score", "Metadata quality"],
    ];
    // Legacy aliases / control echoes are kept in the API payload for
    // backward compatibility but hidden here so the panel shows each
    // signal once under its clearer name.
    const skip = new Set([
      "final", "final_score", "content", "retrieval", "multi_source",
      "collaborative_proxy_score", "novelty_term", "content_weight", "mmr_objective",
    ]);
    const used = new Set(), rows = [];
    for (const [key, label] of ordered) {
      if (typeof sb[key] !== "number") continue;
      used.add(key); rows.push([label, sb[key]]);
    }
    for (const [key, value] of Object.entries(sb)) {
      if (!used.has(key) && !skip.has(key) && typeof value === "number") rows.push([scoreLabels[key] || key.replaceAll("_", " "), value]);
    }
    return rows;
  }
  function scoreRow(label, value) {
    const numeric = Number(value) || 0;
    return `<div class="score-row"><span>${escapeHtml(label)}</span><div class="score-bar"><span style="width: ${Math.max(0, Math.min(1, numeric)) * 100}%"></span></div><b>${numeric.toFixed(2)}</b></div>`;
  }
  function renderRecommendationCard(item, index) {
    const card = document.createElement("li");
    const featured = Number(item.rank || index + 1) === 1;
    card.className = `rec-card pick-${escapeHtml(item.pick_type || "balanced")}${featured ? " featured" : ""}`;
    card.style.animationDelay = `${Math.min(index * 55, 420)}ms`;
    const chips = reasonChips(item), neteaseUrl = httpsify(item.netease_url);
    const rows = readableScoreRows(item.score_breakdown);
    card.innerHTML = `
      <div class="rank">${escapeHtml(item.rank || "")}</div>
      ${coverMarkup(item.cover_url, "rec-cover")}
      <div class="rec-main">
        <div class="rec-title-row">
          <div><h3>${escapeHtml(item.title || "Untitled song")}</h3><p>${escapeHtml(item.artist || "Unknown artist")}${item.album ? ` - ${escapeHtml(item.album)}` : ""}</p></div>
          <span class="pick-badge ${escapeHtml(item.pick_type || "balanced")}">${escapeHtml(item.pick_type || "balanced")}</span>
        </div>
        ${neteaseUrl ? `<a class="netease-link" href="${escapeHtml(neteaseUrl)}" target="_blank" rel="noopener noreferrer">Open on NetEase</a>` : ""}
        <p class="explanation">${escapeHtml(item.explanation || "Recommended because it matches several parts of your taste profile.")}</p>
        <div class="reason-chips">${chips.map((chip, chipIndex) => `<span style="animation-delay:${Math.min(120 + chipIndex * 35, 320)}ms">${escapeHtml(chip)}</span>`).join("")}</div>
        <details class="why"><summary>Why this song?</summary><div class="score-list">${rows.map(([label, value]) => scoreRow(label, value)).join("")}</div></details>
      </div>`;
    return card;
  }
  function renderTechnicalDetails(data) {
    const details = {
      candidate_summary: data.candidate_summary || {}, model_info: data.model_info || {},
      request_id: data.request_id || "", control: data.control || {},
      items: (data.items || []).map((item) => ({
        rank: item.rank, title: item.title, artist: item.artist, pick_type: item.pick_type,
        netease_song_id: item.netease_song_id, sources: item.sources || [],
        score_breakdown: item.score_breakdown || {},
      })),
      kgrec_research_mode: "/api/kgrec-recommend",
    };
    els.technicalBody.innerHTML = `<p>Technical fields are hidden from the main experience and shown here for inspection.</p><pre>${escapeHtml(JSON.stringify(details, null, 2))}</pre>`;
    els.technicalDetails.hidden = false;
  }
  function renderResponse(data) {
    els.cards.innerHTML = ""; els.empty.hidden = true;
    els.resultsSubtitle.textContent = "Ranked songs based on your profile.";
    const items = data.items || [];
    if (!items.length) {
      els.empty.hidden = false;
      els.empty.innerHTML = "<strong>No recommendations yet.</strong><p>Try adding at least one song, artist, genre, mood, or tag.</p>";
      els.technicalDetails.hidden = true; return;
    }
    const summary = data.candidate_summary || {};
    els.resultsMeta.hidden = false;
    els.resultsMeta.textContent = `${items.length} songs - ${summary.total_unique || 0} candidates`;
    const fragment = document.createDocumentFragment();
    items.forEach((item, index) => fragment.appendChild(renderRecommendationCard(item, index)));
    els.cards.appendChild(fragment);
    renderTechnicalDetails(data);
    if (data.fallback_used === "no_input") showError("Could not generate recommendations. Try adding at least one song, artist, genre, mood, or tag.");
  }
  function setRecommendLoading(on) {
    els.submit.disabled = on; els.introSubmit.disabled = on;
    els.submit.textContent = on ? "Building your recommendation list..." : "Build recommendation list";
    els.introSubmit.textContent = on ? "Building playlist..." : "Start with a song";
    els.loading.hidden = !on;
    if (on) {
      state.loadingStageIndex = 0;
      els.loadingStage.textContent = loadingStages[0];
      window.clearInterval(state.loadingTimer);
      state.loadingTimer = window.setInterval(() => {
        state.loadingStageIndex = (state.loadingStageIndex + 1) % loadingStages.length;
        els.loadingStage.textContent = loadingStages[state.loadingStageIndex];
      }, 850);
    } else {
      window.clearInterval(state.loadingTimer);
      state.loadingTimer = 0;
    }
    if (on) { clearError(); els.empty.hidden = true; }
  }
  async function recommend() {
    const payload = buildPayload();
    if (!hasInput(payload)) {
      showError("Add at least one song, artist, genre, mood, or tag before building the playlist.");
      els.searchQ.scrollIntoView({ behavior: "smooth", block: "center" });
      els.searchQ.focus();
      return;
    }
    setRecommendLoading(true);
    try {
      const { response, body } = await fetchJson("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }, 12000);
      if (!response.ok || body.ok === false) {
        const offline = response.status === 503 || /netease/i.test(body.error || "");
        showError(offline
          ? `Could not generate recommendations. ${NETEASE_OFFLINE_HINT}`
          : "Could not generate recommendations. Try adding at least one song, artist, genre, mood, or tag.");
        if (!offline && body.error) els.error.textContent += ` (${body.error})`;
        return;
      }
      renderResponse(body.data || {});
    } catch (error) {
      if (error.code === "timeout") {
        showError(`Recommendation timed out. NetEase may be offline or busy. ${NETEASE_OFFLINE_HINT}`);
      } else if (error.code === "invalid_json") {
        showError(error.message);
      } else {
        showError(`Could not generate recommendations. ${BACKEND_HINT}`);
      }
    } finally {
      setRecommendLoading(false);
    }
  }
  async function loadHealth() {
    try {
      const { response, body: data } = await fetchJson("/api/health", {}, 1800);
      if (!response.ok || data.ok === false) throw new Error(data.error || `Health failed (${response.status}).`);
      const product = data.product_layer || {}, research = data.research_layer || {};
      const alive = Boolean(product.netease_alive);
      els.statusDot.className = alive ? "dot dot-ok" : "dot dot-warn";
      els.statusText.textContent = alive ? "NetEase connected" : NETEASE_OFFLINE_HINT;
      els.modelInfo.textContent = `${product.name || "Real Song Mode"} - KGRec ${research.enabled ? (research.ready ? "ready" : "loading") : "debug off"}`;
    } catch (error) {
      els.statusDot.className = "dot dot-bad";
      els.statusText.textContent = error.code === "timeout"
        ? "Backend health check timed out. Start the Flask demo server, then try again."
        : BACKEND_HINT;
      els.modelInfo.textContent = "Health check unavailable";
    }
  }
  function applyExample(name) {
    const example = examples[name];
    if (!example) return;
    $("liked_artists").value = example.artists; $("genres").value = example.genres;
    $("moods").value = example.moods; $("tags").value = example.tags;
    setPreset(example.preset); renderTasteChips();
    els.searchQ.value = example.query; searchSongs(example.query); els.searchQ.focus();
  }

  function startFromIntro() {
    if (hasInput(buildPayload())) {
      recommend();
      return;
    }
    clearError();
    els.searchQ.scrollIntoView({ behavior: "smooth", block: "center" });
    els.searchQ.focus();
  }

  els.form.addEventListener("submit", (event) => { event.preventDefault(); recommend(); });
  els.introSubmit.addEventListener("click", startFromIntro);
  els.searchBtn.addEventListener("click", () => searchSongs(els.searchQ.value));
  els.searchQ.addEventListener("input", debouncedSearch);
  els.searchQ.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); searchSongs(els.searchQ.value); } });
  for (const id of ["liked_artists", "genres", "moods", "tags"]) $(id).addEventListener("input", renderTasteChips);
  for (const slider of Object.values(sliders)) slider.addEventListener("input", updateSliderOutputs);
  $("k").addEventListener("input", updateSliderOutputs);
  document.querySelectorAll(".preset").forEach((button) => button.addEventListener("click", () => setPreset(button.dataset.preset)));
  document.querySelectorAll(".example-card").forEach((button) => button.addEventListener("click", () => applyExample(button.dataset.example)));
  setPreset("balanced"); updateSliderOutputs(); renderTasteChips(); renderSelectedSongs(); loadHealth();
  setInterval(loadHealth, 10000);
})();
