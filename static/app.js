/* =========================================================================
   Local Trip Suggester - dashboard frontend.

   Stack:
     - Vanilla JS (no framework, no build step). Easy to read in an interview
       and easy to deploy: every API call is a one-line fetch().
     - Leaflet for the map (free + OSM tiles).

   Five concerns, organised top-to-bottom in this file:
     1. tiny logger + DOM helpers
     2. map setup (initMap, renderMap, distance-line styling)
     3. submit -> POST /suggest-trip -> render + scroll + highlight
     4. /health/detailed pill + grid (auto-refresh every 30s)
     5. /history list AND /logs live feed (auto-refresh every 2.5s)
   ========================================================================= */

(function () {
  "use strict";

  // ---------------------------------------------------------------------
  // 1. Tiny logger + DOM helpers
  // ---------------------------------------------------------------------
  const log = (level, msg, extra) => {
    const line = `[ui ${level}] ${msg}`;
    if (level === "error") console.error(line, extra ?? "");
    else if (level === "warn") console.warn(line, extra ?? "");
    else console.log(line, extra ?? "");
  };
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------------------------------------------------------------------
  // 2. Map
  // ---------------------------------------------------------------------
  let map;
  let mapLayers = []; // markers + polylines we wipe between requests

  function initMap() {
    map = L.map("map", {
      zoomControl: true,
      attributionControl: true,
      // Snap-fit when fitBounds is used, with a tighter min so distance
      // lines are visible without zooming all the way in.
      worldCopyJump: false,
    }).setView([20.5937, 78.9629], 5); // India centroid as default

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    // CRITICAL: Leaflet sometimes initialises before the parent flexbox
    // has finished computing its size, which leaves grey bands inside
    // the card. invalidateSize() forces a re-measure once the layout
    // has settled. We also re-measure on window resize.
    requestAnimationFrame(() => map.invalidateSize());
    setTimeout(() => map.invalidateSize(), 250);
    window.addEventListener("resize", () => map.invalidateSize());

    log("info", "map initialised");
  }

  function clearMap() {
    mapLayers.forEach((l) => map.removeLayer(l));
    mapLayers = [];
  }

  /** Build a small inline-SVG marker so the user pin is visually distinct
   *  from the place pins without bringing in extra image assets. */
  function makeIcon(color, label) {
    const html = `
      <div style="
        background:${color};
        width:30px;height:30px;border-radius:50% 50% 50% 0;
        transform:rotate(-45deg);
        box-shadow:0 2px 8px rgba(0,0,0,0.5);
        display:grid;place-items:center;
        border:2px solid #fff;">
        <span style="
          transform:rotate(45deg);
          color:#fff;font-weight:700;font-size:12px;
          font-family:ui-monospace,monospace;">
          ${label}
        </span>
      </div>`;
    return L.divIcon({
      html,
      className: "",
      iconSize: [30, 30],
      iconAnchor: [15, 30],
      popupAnchor: [0, -30],
    });
  }

  /**
   * Render the full result on the map: user pin + place pins + a line
   * from the user to every place, with the haversine distance shown
   * as a permanent label on every line. Centres + zooms to fit the
   * entire set with comfortable padding.
   */
  function renderMap(userPoint, suggestions) {
    clearMap();

    if (!suggestions || suggestions.length === 0) {
      map.setView([20.5937, 78.9629], 5);
      map.invalidateSize();
      return;
    }

    const bounds = [];

    // 1. User marker (only if locality was provided).
    if (userPoint && userPoint.lat != null && userPoint.lng != null) {
      const userMarker = L.marker([userPoint.lat, userPoint.lng], {
        icon: makeIcon("#22c55e", "U"),
        zIndexOffset: 1000,
      })
        .bindPopup(
          `<strong>You are here</strong><br/>(${userPoint.lat.toFixed(4)}, ${userPoint.lng.toFixed(4)})`
        )
        .addTo(map);
      mapLayers.push(userMarker);
      bounds.push([userPoint.lat, userPoint.lng]);
    }

    // 2. Place markers + polylines from user -> place.
    suggestions.forEach((s, idx) => {
      const lat = s.coords?.lat;
      const lng = s.coords?.lng;
      if (lat == null || lng == null) return;

      const marker = L.marker([lat, lng], {
        icon: makeIcon("#3b82f6", String(idx + 1)),
      })
        .bindPopup(buildPopupHTML(s, idx + 1))
        .addTo(map);
      marker.on("click", () => highlightSuggestion(idx));
      mapLayers.push(marker);
      bounds.push([lat, lng]);

      // Distance line: SOLID, BOLD, AMBER. We deliberately don't dash
      // the line and we add a permanent yellow pill at its midpoint so
      // distances are readable at any zoom without hovering.
      if (userPoint && userPoint.lat != null) {
        const line = L.polyline(
          [
            [userPoint.lat, userPoint.lng],
            [lat, lng],
          ],
          {
            color: "#fbbf24",
            weight: 4,
            opacity: 0.9,
            lineCap: "round",
          }
        ).addTo(map);
        mapLayers.push(line);

        // Permanent label at the midpoint with the distance in km.
        if (s.distance_km != null) {
          const midLat = (userPoint.lat + lat) / 2;
          const midLng = (userPoint.lng + lng) / 2;
          const label = L.tooltip({
            permanent: true,
            direction: "center",
            className: "distance-label",
          })
            .setLatLng([midLat, midLng])
            .setContent(`${s.distance_km} km`);
          map.addLayer(label);
          mapLayers.push(label);
        }
      }
    });

    // 3. Fit the map to whatever pins we ended up with - generous
    //    padding so the markers + labels aren't cropped.
    if (bounds.length === 1) {
      map.setView(bounds[0], 13);
    } else if (bounds.length > 1) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
    }

    // One more invalidate after the data lands - guarantees no grey area
    // even if the user resized the window before submitting.
    setTimeout(() => map.invalidateSize(), 50);
  }

  function buildPopupHTML(s, idx) {
    const dist =
      s.distance_km != null ? `${s.distance_km} km from you` : "";
    const total = s.estimated_budget?.total;
    const budget = total != null ? `~&#8377;${total} total` : "";
    return `
      <div style="min-width:200px;">
        <strong>#${idx} ${escapeHTML(s.name)}</strong><br/>
        <small>${(s.categories || []).slice(0, 3).map(escapeHTML).join(", ") || ""}</small>
        <p style="margin:6px 0; font-size:12px;">${escapeHTML(s.reasoning || "")}</p>
        <small style="color:#8a93ad;">${dist}${dist && budget ? " &middot; " : ""}${budget}</small>
      </div>`;
  }

  // ---------------------------------------------------------------------
  // 3. Form -> /suggest-trip
  // ---------------------------------------------------------------------
  const form = $("#trip-form");
  const submitBtn = $("#submit-btn");
  const errBox = $("#form-error");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errBox.hidden = true;
    setSubmitting(true);

    const formData = new FormData(form);
    const payload = {
      city: (formData.get("city") || "").toString().trim(),
      preference: (formData.get("preference") || "").toString().trim() || null,
      locality: (formData.get("locality") || "").toString().trim() || null,
      max_results: Number(formData.get("max_results")) || 5,
    };
    log("info", "submitting /suggest-trip", payload);

    const t0 = performance.now();
    try {
      const res = await fetch("/suggest-trip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText.slice(0, 300)}`);
      }
      const body = await res.json();
      const t1 = performance.now();
      log(
        "info",
        `received ${body.suggestions?.length ?? 0} suggestions in ${Math.round(t1 - t0)} ms (server: ${body.meta?.elapsed_ms} ms)`
      );
      renderResponse(body);
      revealResults(); // ScrollTo + brief highlight pulse
      loadHistory();
      loadLogs(); // pull in the brand-new server-side log lines
    } catch (err) {
      log("error", "suggest-trip failed", err);
      errBox.textContent = err.message || String(err);
      errBox.hidden = false;
    } finally {
      setSubmitting(false);
    }
  });

  function setSubmitting(on) {
    submitBtn.disabled = on;
    submitBtn.querySelector(".btn-spinner").hidden = !on;
    submitBtn.querySelector(".btn-label").textContent = on
      ? "Working..."
      : "Suggest places";
  }

  /**
   * Smooth-scroll to the results card and pulse it briefly so the user
   * cannot miss that new results landed - especially helpful on small
   * screens where suggestions otherwise show up below the fold.
   */
  function revealResults() {
    const target = $("#weather-card");
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    const card = $("#suggestions-card");
    card.classList.remove("is-pulse");
    void card.offsetWidth; // force reflow so the animation re-triggers
    card.classList.add("is-pulse");
  }

  // ---------------------------------------------------------------------
  // Render the /suggest-trip response
  // ---------------------------------------------------------------------
  function renderResponse(body) {
    // Weather card
    const w = body.weather || {};
    $("#weather-card").hidden = false;
    $("#weather-city").textContent = body.city ? `- ${body.city}` : "";
    $("#w-temp").textContent = w.temp_c != null ? w.temp_c.toFixed(1) : "--";
    $("#w-feels").textContent =
      w.feels_like_c != null ? `feels like ${w.feels_like_c.toFixed(1)}°C` : "";
    $("#w-cond").textContent = w.condition || "--";
    $("#w-desc").textContent = w.description || "";
    $("#w-hum").textContent = w.humidity != null ? `${w.humidity}%` : "--";
    $("#w-wind").textContent = w.wind_kph != null ? `${w.wind_kph} kph` : "--";

    // Meta card
    const meta = body.meta || {};
    $("#meta-card").hidden = false;
    $("#m-elapsed").textContent = meta.elapsed_ms ?? "--";
    $("#m-llm").textContent = meta.llm_provider || "--";

    // "Understood as" - show how the prompt was parsed (category, mood,
    // and which keywords we sent to the places API). Empty/default is
    // shown as a low-key chip so the user knows the parser ran.
    const intent = meta.intent || {};
    const intentChips = [];
    if (intent.category) intentChips.push(intent.category);
    if (intent.mood) intentChips.push(`mood:${intent.mood}`);
    if (intent.source) intentChips.push(`via ${intent.source}`);
    if (Array.isArray(intent.search_keywords) && intent.search_keywords.length) {
      intentChips.push(`-> ${intent.search_keywords.slice(0, 4).join(", ")}`);
    }
    renderTags($("#m-intent"), intentChips, false);

    renderTags(
      $("#m-curate"),
      [meta.llm_curate_used ? "used" : "rule-based only"],
      !meta.llm_curate_used
    );
    renderTags($("#m-cache"), meta.cache_hits || [], false);
    renderTags($("#m-degraded"), meta.degraded || [], true);

    // Suggestions list
    const list = $("#suggestions-list");
    const empty = $("#suggestions-empty");
    list.innerHTML = "";
    const suggestions = body.suggestions || [];
    $("#sugg-count").textContent = suggestions.length
      ? `${suggestions.length} place${suggestions.length === 1 ? "" : "s"}`
      : "";
    if (!suggestions.length) {
      empty.hidden = false;
      empty.textContent =
        "No places returned for this query. Check the System health panel - both providers may be down.";
    } else {
      empty.hidden = true;
      suggestions.forEach((s, idx) => list.appendChild(buildSuggestionCard(s, idx)));
    }

    // Map
    const hint = $("#map-hint");
    if (suggestions.length) {
      hint.textContent = body.user_location
        ? `${body.city}: ${suggestions.length} pins, lines from your area show distance.`
        : `${body.city}: ${suggestions.length} pins (no locality set, so no distance lines).`;
    } else {
      hint.textContent = "No places to plot for this query.";
    }
    renderMap(body.user_location, suggestions);
  }

  function buildSuggestionCard(s, idx) {
    const li = document.createElement("li");
    li.className = "suggestion";
    li.dataset.idx = String(idx);
    li.tabIndex = 0;

    const cats = (s.categories || []).slice(0, 3).join(" - ") || "";
    const dist = s.distance_km != null ? `${s.distance_km} km` : "n/a";
    const total = s.estimated_budget?.total;
    const budget = total != null ? `&#8377;${total}` : "n/a";
    const score = (s.score ?? 0).toFixed(2);
    const website = s.website
      ? `<a href="${escapeAttr(s.website)}" target="_blank" rel="noreferrer">website</a>`
      : "";

    li.innerHTML = `
      <div class="name">
        <span>#${idx + 1} ${escapeHTML(s.name)}</span>
        <span class="badge">score ${score}</span>
      </div>
      <small class="muted">${escapeHTML(cats)}</small>
      <p class="reason">${escapeHTML(s.reasoning || "")}</p>
      <div class="meta">
        <span>Distance:<strong>${dist}</strong></span>
        <span>Est. budget:<strong>${budget}</strong></span>
        ${website ? `<span>${website}</span>` : ""}
      </div>`;

    li.addEventListener("click", () => focusSuggestion(idx, s));
    li.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        focusSuggestion(idx, s);
      }
    });
    return li;
  }

  function highlightSuggestion(idx) {
    $$(".suggestion").forEach((el) =>
      el.classList.toggle("is-active", Number(el.dataset.idx) === idx)
    );
  }

  function focusSuggestion(idx, s) {
    highlightSuggestion(idx);
    const lat = s.coords?.lat;
    const lng = s.coords?.lng;
    if (lat != null && lng != null) {
      map.setView([lat, lng], 14, { animate: true });
      const marker = mapLayers.find(
        (l) =>
          l instanceof L.Marker &&
          l.getLatLng().lat === lat &&
          l.getLatLng().lng === lng
      );
      if (marker) marker.openPopup();
    }
  }

  function renderTags(container, items, asWarn) {
    container.innerHTML = "";
    if (!items.length) {
      container.innerHTML = '<small class="muted">none</small>';
      return;
    }
    items.forEach((t) => {
      const span = document.createElement("span");
      span.className = `tag ${asWarn ? "tag--warn" : ""}`;
      span.textContent = t;
      container.appendChild(span);
    });
  }

  // ---------------------------------------------------------------------
  // 4. Health pill + grid
  // ---------------------------------------------------------------------
  const pill = $("#health-toggle");
  const grid = $("#health-grid");

  pill.addEventListener("click", () =>
    document
      .querySelector(".health-card")
      ?.scrollIntoView({ behavior: "smooth", block: "center" })
  );
  $("#refresh-health").addEventListener("click", loadHealth);

  async function loadHealth() {
    log("info", "GET /health/detailed");
    grid.innerHTML = '<div class="muted">Refreshing...</div>';
    try {
      const res = await fetch("/health/detailed");
      const body = await res.json();
      renderHealth(body);
    } catch (err) {
      log("error", "health probe failed", err);
      setPill("down", "unreachable");
      grid.innerHTML = `<div class="muted">Health endpoint unreachable.</div>`;
    }
  }

  function renderHealth(body) {
    setPill(body.status, body.status);

    grid.innerHTML = "";
    const checks = body.checks || {};
    Object.values(checks).forEach((c) => {
      // Three-state LED: ok (green), disabled (amber, intentionally
      // off), down (red, configured but failing). Anything else stays
      // grey so the user can tell it's just "not configured" vs broken.
      let ledClass;
      if (c.status === "ok") ledClass = "led--ok";
      else if (c.status === "disabled") ledClass = "led--noconfig";
      else if (c.configured) ledClass = "led--down";
      else ledClass = "led--noconfig";
      const cell = document.createElement("div");
      cell.className = "health-cell";
      cell.innerHTML = `
        <span class="led ${ledClass}"></span>
        <div class="body">
          <span class="name">${escapeHTML(c.name)}</span>
          <span class="note">${escapeHTML(c.note || "")} - ${c.elapsed_ms} ms</span>
        </div>`;
      grid.appendChild(cell);
    });

    const stats = body.stats || {};
    $("#hs-db").textContent = body.config?.db_kind || "?";
    $("#hs-history").textContent = stats.history_rows ?? "-";
    $("#hs-places").textContent = stats.place_cache_rows ?? "-";
  }

  function setPill(state, label) {
    pill.classList.remove(
      "health-pill--ok",
      "health-pill--degraded",
      "health-pill--down",
      "health-pill--unknown"
    );
    if (state === "ok") pill.classList.add("health-pill--ok");
    else if (state === "degraded") pill.classList.add("health-pill--degraded");
    else if (state === "down") pill.classList.add("health-pill--down");
    else pill.classList.add("health-pill--unknown");
    pill.querySelector(".label").textContent = label;
  }

  // ---------------------------------------------------------------------
  // 5a. Recent queries
  // ---------------------------------------------------------------------
  $("#refresh-history").addEventListener("click", loadHistory);

  async function loadHistory() {
    log("info", "GET /history");
    try {
      const res = await fetch("/history?limit=10");
      const body = await res.json();
      const list = $("#history-list");
      list.innerHTML = "";
      if (!body.items?.length) {
        list.innerHTML = '<li class="muted">No queries yet.</li>';
        return;
      }
      body.items.forEach((item) => {
        const li = document.createElement("li");
        const top = item.top_suggestion ? ` -> ${item.top_suggestion}` : "";
        li.innerHTML = `
          <span>
            <span class="h-city">${escapeHTML(item.city || "?")}</span>
            <span class="h-meta">${escapeHTML(item.preference || "no pref")}${escapeHTML(top)}</span>
          </span>
          <span class="h-meta">
            ${item.suggestion_count} place(s) - ${item.latency_ms} ms
          </span>`;
        list.appendChild(li);
      });
    } catch (err) {
      log("warn", "history fetch failed", err);
    }
  }

  // ---------------------------------------------------------------------
  // 5b. Live logs panel
  //
  // We poll /logs every 2.5s (default). The user can:
  //   - toggle "errors only" to filter to WARNING+
  //   - pause polling so the stream stops scrolling while they read
  //   - toggle auto-scroll so they can scroll back without being snapped
  // ---------------------------------------------------------------------
  const logsStream = $("#logs-stream");
  const errorsOnly = $("#logs-errors-only");
  const autoScroll = $("#logs-autoscroll");
  const pauseBtn = $("#logs-pause");
  let logsPaused = false;

  pauseBtn.addEventListener("click", () => {
    logsPaused = !logsPaused;
    pauseBtn.textContent = logsPaused ? "Resume" : "Pause";
  });

  async function loadLogs() {
    if (logsPaused) return;
    const level = errorsOnly.checked ? "WARNING" : "";
    try {
      const url = `/logs?limit=200${level ? `&level=${level}` : ""}`;
      const res = await fetch(url);
      const body = await res.json();
      renderLogs(body.items || []);
    } catch (err) {
      // We deliberately swallow errors here - the panel must never
      // throw a wall of red just because one poll missed.
      log("warn", "logs fetch failed", err);
    }
  }

  function renderLogs(items) {
    if (!items.length) {
      logsStream.textContent = "No log entries yet.";
      return;
    }
    // We rebuild the inner HTML each tick. 200 lines is small enough
    // that it's cheaper than diffing. We escape every dynamic field.
    const html = items
      .map((it) => {
        const lvl = (it.level || "INFO").toUpperCase();
        return (
          `<span class="log-line lvl-${escapeAttr(lvl)}">` +
          `<span class="lvl">${escapeHTML(lvl)}</span> ` +
          `<span class="lname">${escapeHTML(it.logger || "")}</span> ` +
          `<span class="rid">[${escapeHTML(it.rid || "-")}]</span> ` +
          `<span class="msg">${escapeHTML(it.msg || "")}</span>` +
          `</span>`
        );
      })
      .join("\n");
    logsStream.innerHTML = html;
    if (autoScroll.checked) {
      logsStream.scrollTop = logsStream.scrollHeight;
    }
  }

  // ---------------------------------------------------------------------
  // Tiny escape helpers (we render server JSON into innerHTML in places).
  // ---------------------------------------------------------------------
  function escapeHTML(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }
  function escapeAttr(s) {
    return escapeHTML(s).replaceAll("'", "&#39;");
  }

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initMap();
    loadHealth();
    loadHistory();
    loadLogs();
    // Cheap probes - keep the dashboard live without hammering the API.
    setInterval(loadHealth, 30_000);
    setInterval(loadLogs, 2_500);
  });
})();
