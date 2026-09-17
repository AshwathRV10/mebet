/* mebet frontend.
 *
 * No build step and no framework: the page is served straight from the local
 * backend, which keeps the application genuinely local-first and removes a
 * toolchain the user would otherwise have to install and maintain.
 *
 * Presentation rule followed throughout: lead with the headline numbers, then
 * let the user open the detail. Withheld markets are shown as withheld, with
 * the reason, rather than hidden - the absence of a prediction is information.
 */

const $ = (sel) => document.querySelector(sel);
const fmtPct = (v) => (v === null || v === undefined ? "-" : (v * 100).toFixed(1) + "%");
const fmtNum = (v, d = 2) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let currentMatchId = null;

/* ------------------------------------------------------------------ setup */
async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* keep status */ }
    throw new Error(detail);
  }
  return response.json();
}

async function loadSources() {
  try {
    const { sources } = await api("/api/sources");
    $("#source-status").innerHTML = sources
      .map((s) => `<span class="pill ${s.available ? "up" : "down"}" title="${esc(s.detail)}">
        ${esc(s.key)}</span>`)
      .join("");
  } catch (e) {
    $("#source-status").innerHTML = `<span class="pill down">sources unavailable</span>`;
  }
}

async function loadCompetitions() {
  const { stored, available } = await api("/api/competitions");
  const select = $("#competition");
  const storedKeys = new Set(stored.map((c) => c.key));
  const options = stored.map(
    (c) => `<option value="${esc(c.key)}">${esc(c.name)} (${esc(c.key)}) — ${c.matches} matches</option>`
  );
  available
    .filter((k) => !storedKeys.has(k))
    .forEach((k) => options.push(`<option value="${esc(k)}">${esc(k)} — no data loaded yet</option>`));
  select.innerHTML = options.join("");
  if (stored.length) await loadTeams(select.value);
}

async function loadTeams(competition) {
  try {
    const { teams } = await api(`/api/teams?competition=${encodeURIComponent(competition)}`);
    $("#team-list").innerHTML = teams.map((t) => `<option value="${esc(t.name)}">`).join("");
  } catch (e) { /* datalist is a convenience; typing still works */ }
}

/* --------------------------------------------------------------- progress */
function setProgress(step, note) {
  const order = ["collect", "validate", "analyse", "models", "predict"];
  const index = order.indexOf(step);
  document.querySelectorAll("#pipeline-steps li").forEach((li, i) => {
    li.classList.toggle("active", i === index);
    li.classList.toggle("done", i < index);
  });
  $("#progress-note").textContent = note || "";
}

/* ---------------------------------------------------------------- render */
function renderOutcome(prediction) {
  const { outcome, home_team, away_team } = prediction;
  if (!outcome || !Object.keys(outcome).length) {
    $("#outcome").innerHTML =
      `<div class="note bad">No match-result prediction could be produced from the
       available data.</div>`;
    return;
  }
  const rows = [
    { key: "home", name: home_team },
    { key: "draw", name: "Draw" },
    { key: "away", name: away_team },
  ];
  $("#outcome").innerHTML = rows
    .map(
      (r) => `<div class="outcome-row">
        <span class="outcome-name" title="${esc(r.name)}">${esc(r.name)}</span>
        <span class="bar-track"><span class="bar-fill ${r.key}"
              style="width:${(outcome[r.key] * 100).toFixed(1)}%"></span></span>
        <span class="outcome-pct">${fmtPct(outcome[r.key])}</span>
      </div>`
    )
    .join("");
}

/* Raw market keys ("over_under_2.5") are precise but unfriendly; these are the
   labels shown to the reader. Anything unlisted falls back to a tidied key. */
const MARKET_LABELS = {
  "1x2": "Match result",
  double_chance: "Double chance",
  correct_score: "Correct score",
  most_likely_score_given: "Likeliest score given outcome",
  expected_goals: "Expected goals",
  btts: "Both teams to score",
  clean_sheet: "Clean sheet",
  winning_margin: "Winning margin",
  expected_first_half_goals: "Expected first-half goals",
  expected_corners: "Expected corners",
  expected_cards: "Expected cards (red = 2)",
  player_goals: "Player goals",
  player_assists: "Player assists",
  player_involvement: "Goal or assist",
  player_minutes: "Expected minutes",
  player_shots: "Player shots",
  player_key_passes: "Player key passes",
};

function marketLabel(key) {
  if (MARKET_LABELS[key]) return MARKET_LABELS[key];
  let m = key.match(/^over_under_([\d.]+)$/);
  if (m) return `Over/under ${m[1]} goals`;
  m = key.match(/^(corners|cards)_over_under_([\d.]+)$/);
  if (m) return `Over/under ${m[2]} ${m[1]}`;
  m = key.match(/^(home|away)_goals_over_under_([\d.]+)$/);
  if (m) return `${m[1] === "home" ? "Home" : "Away"} goals over/under ${m[2]}`;
  m = key.match(/^first_half_over_under_([\d.]+)$/);
  if (m) return `First half over/under ${m[1]}`;
  return key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

function section(title, sub, bodyHtml, open = false) {
  return `<details class="section" ${open ? "open" : ""}>
    <summary><span>${esc(title)}</span><span class="sub">${esc(sub || "")}</span></summary>
    <div class="section-body">${bodyHtml}</div>
  </details>`;
}

function marketTable(markets, keys, label) {
  const rows = [];
  keys.forEach((key) => {
    (markets[key] || []).forEach((t) => {
      if (!t.sufficient_data) {
        rows.push(`<tr><td>${esc(marketLabel(key))}</td>
          <td colspan="2" class="unavailable">${esc(t.note)}</td></tr>`);
        return;
      }
      const value = t.probability !== null ? fmtPct(t.probability)
        : t.expected_value !== null ? fmtNum(t.expected_value)
        : "-";
      const interval = t.interval
        ? ` <span class="muted">(80% range ${t.interval[0]}–${t.interval[1]})</span>` : "";
      // The subject (a team or player name) qualifies the selection rather
      // than replacing the market, so a row always says what it is about.
      const selection = t.subject_name
        ? `${esc(t.subject_name)} — ${esc(t.selection)}`
        : esc(t.selection);
      const note = t.note && t.market === "most_likely_score_given"
        ? ` <span class="muted">(${esc(t.note)})</span>` : "";
      rows.push(`<tr>
        <td>${esc(marketLabel(key))}</td>
        <td>${selection}${note}</td>
        <td class="num">${value}${interval}</td>
      </tr>`);
    });
  });
  if (!rows.length) return `<p class="unavailable">Nothing to report for ${esc(label)}.</p>`;
  return `<table><thead><tr><th>Market</th><th>Selection</th>
    <th class="num">Value</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function renderSections(prediction) {
  const m = prediction.markets || {};
  const out = [];

  /* --- scorelines and goals ------------------------------------------- */
  const scores = (m.correct_score || [])
    .map((t) => `<tr><td>${esc(t.selection)}</td><td class="num">${fmtPct(t.probability)}</td></tr>`)
    .join("");
  out.push(section("Score & goals", "scorelines, over/under, both teams to score",
    `<table><thead><tr><th>Scoreline</th><th class="num">Probability</th></tr></thead>
       <tbody>${scores || '<tr><td colspan="2" class="unavailable">unavailable</td></tr>'}</tbody></table>
     <div style="height:14px"></div>` +
    marketTable(m, ["expected_goals", "most_likely_score_given",
      "over_under_0.5", "over_under_1.5", "over_under_2.5",
      "over_under_3.5", "over_under_4.5", "btts", "clean_sheet",
      "home_goals_over_under_1.5", "away_goals_over_under_1.5",
      "expected_first_half_goals", "first_half_over_under_0.5", "first_half_over_under_1.5",
      "winning_margin", "double_chance"], "Market"), true));

  /* --- corners --------------------------------------------------------- */
  out.push(section("Corners", "expected totals and lines",
    marketTable(m, ["expected_corners", "corners_over_under_7.5", "corners_over_under_8.5",
      "corners_over_under_9.5", "corners_over_under_10.5", "corners_over_under_11.5",
      "corners_over_under_12.5", "corners"], "Corners")));

  /* --- cards ----------------------------------------------------------- */
  out.push(section("Cards", "card points, where the data supports it",
    marketTable(m, ["expected_cards", "cards_over_under_2.5", "cards_over_under_3.5",
      "cards_over_under_4.5", "cards_over_under_5.5", "cards_over_under_6.5", "cards"], "Cards")));

  /* --- players --------------------------------------------------------- */
  const players = prediction.players || [];
  const playerRows = players.map((p) => {
    if (!p.sufficient_data) {
      return `<tr><td>${esc(p.player)}</td><td colspan="5" class="unavailable">${esc(p.note)}</td></tr>`;
    }
    return `<tr>
      <td>${esc(p.player)}</td>
      <td>${esc(p.position || "")}</td>
      <td class="num">${fmtNum(p.expected_minutes, 0)}</td>
      <td class="num">${fmtPct(p.prob_scores)}</td>
      <td class="num">${fmtPct(p.prob_assists)}</td>
      <td class="num">${fmtPct(p.prob_goal_or_assist)}</td>
    </tr>`;
  }).join("");
  const withheld = (m.player_shots || []).concat(m.player_key_passes || [])
    .map((t) => `<div class="note info">${esc(t.note)}</div>`).join("");
  out.push(section("Players", `${players.filter((p) => p.sufficient_data).length} projected`,
    (playerRows
      ? `<table><thead><tr><th>Player</th><th>Pos</th><th class="num">Mins</th>
         <th class="num">Scores</th><th class="num">Assists</th><th class="num">G or A</th>
         </tr></thead><tbody>${playerRows}</tbody></table>`
      : `<p class="unavailable">No player-level data available for these teams.</p>`) + withheld));

  /* --- form and team statistics ---------------------------------------- */
  const f = prediction.features || {};
  const formTable = (side) => {
    const t = f[side];
    if (!t) return "";
    const w = t.windows || {};
    const row = (label, stat) => `<tr><td>${esc(label)}</td>
      <td class="num">${fmtNum(w.last5?.[stat]?.value)}</td>
      <td class="num">${fmtNum(w.last10?.[stat]?.value)}</td>
      <td class="num">${fmtNum(w.last20?.[stat]?.value)}</td>
      <td class="num">${fmtNum(w.all?.[stat]?.value)}</td></tr>`;
    return `<h4>${esc(t.team_name)} <span class="muted small">(${t.matches_available} matches on record,
      league position ${t.league_position ?? "n/a"})</span></h4>
      <table><thead><tr><th>Per match</th><th class="num">L5</th><th class="num">L10</th>
      <th class="num">L20</th><th class="num">All</th></tr></thead><tbody>
      ${row("Goals scored", "goals_for")}
      ${row("Goals conceded", "goals_against")}
      ${row("Shots", "shots")}
      ${row("Shots on target", "shots_on_target")}
      ${row("Corners", "corners")}
      ${row("Fouls", "fouls")}
      ${row("Yellow cards", "yellows")}
      </tbody></table>
      <dl class="kv" style="margin-top:10px">
        <dt>Points per match (last 5 / all)</dt><dd>${fmtNum(t.form_points?.last5)} / ${fmtNum(t.form_points?.all)}</dd>
        <dt>Clean sheet rate</dt><dd>${fmtPct(t.clean_sheet_rate?.value)}</dd>
        <dt>Both teams scored</dt><dd>${fmtPct(t.btts_rate?.value)}</dd>
        <dt>Opponent-adjusted attack index</dt><dd>${fmtNum(t.opponent_adjusted?.attack_index)}</dd>
        <dt>Opponent-adjusted defence index</dt><dd>${fmtNum(t.opponent_adjusted?.defence_index)}</dd>
        <dt>Rest days</dt><dd>${t.rest_days ?? "-"}</dd>
        <dt>Matches in last 14 days</dt><dd>${t.matches_last_14_days ?? "-"}</dd>
      </dl>`;
  };
  out.push(section("Team statistics & form", "multiple windows, not just the last five",
    formTable("home") + `<div style="height:18px"></div>` + formTable("away")));

  /* --- head to head ----------------------------------------------------- */
  const h2h = f.h2h || {};
  out.push(section("Head-to-head", h2h.matches ? `${h2h.matches} meetings` : "no meetings on record",
    h2h.matches
      ? `<dl class="kv">
          <dt>Meetings</dt><dd>${h2h.matches}</dd>
          <dt>Home-team wins</dt><dd>${h2h.home_team_wins}</dd>
          <dt>Draws</dt><dd>${h2h.draws}</dd>
          <dt>Away-team wins</dt><dd>${h2h.away_team_wins}</dd>
          <dt>Average total goals</dt><dd>${fmtNum(h2h.avg_total_goals)}</dd>
          <dt>Most recent</dt><dd>${esc(h2h.most_recent)}</dd>
        </dl>
        <table style="margin-top:10px"><thead><tr><th>Date</th><th>Venue</th><th>Score</th></tr></thead>
        <tbody>${(h2h.results || []).map((r) =>
          `<tr><td>${esc(r.date)}</td><td>${r.home ? "home" : "away"}</td>
           <td class="num">${esc(r.score)}</td></tr>`).join("")}</tbody></table>`
      : `<p class="unavailable">${esc(h2h.note || "No previous meetings on record.")}</p>`));

  /* --- lineups and availability ----------------------------------------- */
  const q = prediction.data_quality || {};
  out.push(section("Lineups & availability", `lineups: ${q.lineups || "unavailable"}`,
    `<dl class="kv">
      <dt>Lineup status</dt><dd>${esc(q.lineups || "unavailable")}</dd>
      <dt>Team news retrieved</dt><dd>${q.availability_data ? "yes" : "no"}</dd>
      <dt>Player statistics</dt><dd>${q.player_data ? "yes" : "no"}</dd>
      <dt>Weather</dt><dd>${q.weather ? "retrieved" : "not retrieved"}</dd>
    </dl>
    ${q.lineups !== "confirmed"
      ? `<div class="note warn">Lineups are not confirmed. When official lineups become
         available, use Refresh Data &amp; Recalculate to regenerate the prediction.</div>` : ""}`));

  /* --- model reasoning --------------------------------------------------- */
  const factors = (prediction.explanation || []).map((fct) =>
    `<div class="factor">
      <span class="factor-tag ${esc(fct.favours)}">${esc(fct.favours)}</span>
      <div><div class="factor-text">${esc(fct.statement)}</div>
      <div class="evidence">${esc(JSON.stringify(fct.evidence).slice(0, 220))}</div></div>
    </div>`).join("");
  const diag = prediction.model_diagnostics || {};
  const weights = diag.ensemble?.weights || {};
  const components = diag.ensemble?.components || {};
  out.push(section("Model reasoning", `${(prediction.explanation || []).length} factors`,
    (factors || `<p class="unavailable">No factors could be traced to the data.</p>`) +
    `<h4 style="margin-top:18px">Model agreement</h4>
     <table><thead><tr><th>Model</th><th class="num">Weight</th><th class="num">Home</th>
     <th class="num">Draw</th><th class="num">Away</th></tr></thead><tbody>
     ${Object.keys(components).map((k) => `<tr><td>${esc(k)}</td>
       <td class="num">${fmtNum(weights[k], 3)}</td>
       <td class="num">${fmtPct(components[k].home)}</td>
       <td class="num">${fmtPct(components[k].draw)}</td>
       <td class="num">${fmtPct(components[k].away)}</td></tr>`).join("")}
     </tbody></table>
     <p class="muted small">Weights source: ${esc(diag.ensemble?.weight_source || "equal")}.
     Measured weights come from the most recent backtest of this competition; without one,
     the models are weighted equally.</p>`));

  /* --- data quality ------------------------------------------------------- */
  const issues = (q.issues || []).map((i) =>
    `<div class="note ${i.severity === "serious" ? "bad" : i.severity === "warning" ? "warn" : "info"}">
      ${esc(i.message)}</div>`).join("");
  out.push(section("Data quality", `${q.tier || "?"} — ${q.historical_matches || 0} matches analysed`,
    `<dl class="kv">
      <dt>Tier</dt><dd class="tier-${esc(q.tier)}">${esc(q.tier)}</dd>
      <dt>Quality score</dt><dd>${fmtNum(q.score, 3)}</dd>
      <dt>Confidence multiplier</dt><dd>${fmtNum(q.confidence_multiplier, 3)}</dd>
      <dt>Historical matches</dt><dd>${q.historical_matches}</dd>
      <dt>Home / away samples</dt><dd>${q.home_matches} / ${q.away_matches}</dd>
      <dt>Head-to-head meetings</dt><dd>${q.h2h_matches}</dd>
      <dt>Most recent data</dt><dd>${q.freshness_days ?? "-"} days old</dd>
    </dl><div style="height:12px"></div>${issues}`));

  /* --- sources ------------------------------------------------------------ */
  const acq = prediction.__acquisition || {};
  const used = (q.sources_used || []).map((s) => `<li>${esc(s)}</li>`).join("");
  const failed = (acq.failed || []).map((f2) =>
    `<li>${esc(f2.source)} — ${esc(f2.error)}</li>`).join("");
  out.push(section("Data sources", `${(q.sources_used || []).length} used`,
    `<h4>Used</h4><ul>${used || "<li class='unavailable'>none recorded</li>"}</ul>
     ${failed ? `<h4>Unavailable at analysis time</h4><ul>${failed}</ul>` : ""}
     <p class="muted small">Betting odds are excluded from every model by design; odds
     columns are stripped at the parsing boundary and never reach the feature layer.</p>`));

  $("#sections").innerHTML = out.join("");
}

function renderResult(payload) {
  const p = payload.prediction;
  if (!p) throw new Error(payload.message || "no prediction returned");
  p.__acquisition = payload.acquisition || {};
  currentMatchId = payload.match_id;

  $("#match-teams").textContent = `${p.home_team} vs ${p.away_team}`;
  $("#match-meta").textContent =
    `${p.competition}${p.kickoff ? " · " + new Date(p.kickoff).toLocaleString() : ""}`;
  $("#generated-at").textContent =
    `generated ${new Date(p.generated_at + "Z").toLocaleString()}`;
  $("#version").textContent = "v" + p.version;
  $("#change-summary").textContent = p.change_summary || "";

  const warnings = (payload.warnings || []).concat(p.warnings || []);
  $("#warnings").innerHTML = [...new Set(warnings)]
    .map((w) => `<div class="note warn">${esc(w)}</div>`).join("");

  renderOutcome(p);
  $("#likely-score").textContent = p.most_likely_score || "-";

  const expected = (p.markets?.expected_goals || []).find((t) => t.selection === "total");
  const home = (p.markets?.expected_goals || []).find((t) => t.selection === "home");
  const away = (p.markets?.expected_goals || []).find((t) => t.selection === "away");
  $("#expected-goals").textContent = expected
    ? `${fmtNum(expected.expected_value)} (${fmtNum(home?.expected_value)}–${fmtNum(away?.expected_value)})`
    : "-";

  const tier = p.data_quality?.tier || "?";
  const tierEl = $("#quality-tier");
  tierEl.textContent = tier;
  tierEl.className = "value tier-" + tier;

  renderSections(p);
  $("#results").classList.remove("hidden");
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------------------ flows */
async function runAnalysis(body) {
  $("#progress-card").classList.remove("hidden");
  $("#results").classList.add("hidden");
  $("#analyze-btn").disabled = true;
  setProgress("collect", "Contacting data sources. First run for a competition downloads "
    + "several seasons of history and can take a minute.");
  try {
    // The backend runs the whole sequence in one request; the steps below
    // reflect that sequence rather than polling a progress endpoint.
    const pending = api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
    setTimeout(() => setProgress("validate", "Validating what came back."), 900);
    setTimeout(() => setProgress("analyse", "Building features from match history."), 2200);
    setTimeout(() => setProgress("models", "Fitting models at the prediction cutoff."), 4200);
    const payload = await pending;
    setProgress("predict", "Done.");
    renderResult(payload);
  } catch (err) {
    $("#progress-note").innerHTML = `<span class="note bad">${esc(err.message)}</span>`;
  } finally {
    $("#analyze-btn").disabled = false;
    setTimeout(() => $("#progress-card").classList.add("hidden"), 700);
  }
}

$("#analyze-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const date = $("#date").value;
  const time = $("#time").value;
  runAnalysis({
    sport: $("#sport").value,
    competition: $("#competition").value,
    home_team: $("#home_team").value.trim(),
    away_team: $("#away_team").value.trim(),
    date: date,
    kickoff: time ? `${date}T${time}:00` : null,
    refresh: $("#refresh").checked,
    include_players: $("#players").checked,
    history_seasons: Number($("#seasons").value) || 8,
  });
});

$("#refresh-btn").addEventListener("click", async () => {
  if (!currentMatchId) return;
  $("#refresh-btn").disabled = true;
  $("#progress-card").classList.remove("hidden");
  setProgress("collect", "Fetching the newest available data for this fixture.");
  try {
    const payload = await api(`/api/matches/${currentMatchId}/refresh`, { method: "POST" });
    setProgress("predict", "Recalculated.");
    renderResult(payload);
  } catch (err) {
    $("#progress-note").innerHTML = `<span class="note bad">${esc(err.message)}</span>`;
  } finally {
    $("#refresh-btn").disabled = false;
    setTimeout(() => $("#progress-card").classList.add("hidden"), 700);
  }
});

$("#competition").addEventListener("change", (e) => loadTeams(e.target.value));

$("#fixtures-btn").addEventListener("click", async () => {
  const panel = $("#fixtures-panel");
  panel.classList.toggle("hidden");
  if (panel.classList.contains("hidden")) return;
  panel.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const { fixtures } = await api(
      `/api/fixtures?competition=${encodeURIComponent($("#competition").value)}`);
    panel.innerHTML = fixtures.length
      ? fixtures.map((f) => `<div class="fixture-row" data-home="${esc(f.home)}"
          data-away="${esc(f.away)}" data-date="${esc(f.date)}">
          <span>${esc(f.home)} vs ${esc(f.away)}</span><span class="muted">${esc(f.date)}</span>
        </div>`).join("")
      : `<p class="unavailable">No upcoming fixtures stored for this competition.
         Analyse a match to fetch them, or enter the fixture manually.</p>`;
    panel.querySelectorAll(".fixture-row").forEach((row) => {
      row.addEventListener("click", () => {
        $("#home_team").value = row.dataset.home;
        $("#away_team").value = row.dataset.away;
        $("#date").value = row.dataset.date;
        panel.classList.add("hidden");
      });
    });
  } catch (err) {
    panel.innerHTML = `<div class="note bad">${esc(err.message)}</div>`;
  }
});

/* ------------------------------------------------------------------- init */
$("#date").value = new Date(Date.now() + 86400000).toISOString().slice(0, 10);
loadSources();
loadCompetitions().catch((e) => {
  $("#competition").innerHTML = `<option>could not load competitions</option>`;
});
