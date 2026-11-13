// Blindspot story: 8 chapters, claim on the left, one visual on the right. Every number is read from results/ui
// (via /api/data); nothing numeric is hardcoded here. Keys: Up/Down chapters, Left/Right switch crowd or example, R restart.
(async () => {
  const { load, pct, esc, reduced, countTo, markSample } = window.BS;
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";   // the story owns scrolling (snap + hash)
  const $ = (id) => document.getElementById(id);
  const get = async (name) => { try { const r = await load(name); markSample(r.mock); return r; } catch (_) { return null; } };

  const CHAPTERS = [...document.querySelectorAll(".chapter")];
  const [sumR, repR, boardR, fairR, exR, recR] = await Promise.all(
    ["summary", "repairs_gss", "scoreboard_gss", "fairness_gss", "examples_gss", "realrecruit_gss"].map(get));
  const summary = sumR?.data;
  const gssInfo = summary?.datasets?.gss, twinInfo = summary?.datasets?.twin;
  const mapCache = {};
  const map = async (name) => (mapCache[name] ??= (await load(name).then((r) => { markSample(r.mock); return r.data; })));
  const src = (ds, c) => {
    const d = summary.datasets[ds], e = d.crowds[c] || {};
    const ci = e.coverage_ci ? `; 95% interval for coverage ${pct(e.coverage_ci[0])}–${pct(e.coverage_ci[1])}` : "";
    return `${d.label}, ${d.split} split, n = ${e.n ?? d.n} real people${e.items || d.items ? `, ${e.items ?? d.items} questions` : ""}${ci}.`;
  };
  const numbers = {};                       // chapter id -> function that runs when the chapter is shown
  const set = (el, text) => { if (el) el.textContent = text; };
  const title = (ids, text) => ids.forEach((id) => $(id) && ($(id).title = text));
  const where = () => `${gssInfo?.label ?? "GSS 2024"}, ${gssInfo?.split ?? "test"} split, n = ${gssInfo?.n ?? "–"}`;

  // ---------- 1 · the gap ----------
  (() => {
    const tw = twinInfo?.crowds?.gpt41mini_twin;
    if (!tw) { set($("gapSource"), "Twin results are not exported yet."); return; }
    const held = tw.holdout_coverage != null, cov = held ? tw.holdout_coverage : tw.coverage;
    const missed = 1 - cov, nMiss = Math.round(missed * 100);
    const order = [...Array(100).keys()].sort((a, b) => ((a * 37) % 101) - ((b * 37) % 101));   // spread the missed dots around
    const flag = new Set(order.slice(0, nMiss));
    $("people").innerHTML = [...Array(100).keys()].map((i) => `<i class="${flag.has(i) ? "miss" : ""}" style="--k:${i}"></i>`).join("");
    set($("gapLegendMiss"), nMiss); set($("gapLegendOk"), 100 - nMiss);
    const how = held ? "the one-time held-out split of respondents no method ever saw" : `the ${twinInfo.split} split`;
    title(["gapMiss"], `Share of real people with nobody in the AI crowd like them, measured on ${how}. ${twinInfo.label}; crowd = ${tw.label}.`);
    title(["gapReal"], `Realism = share of AI twins that fall inside some real person's neighbourhood (precision), ${twinInfo.split} split, n = ${twinInfo.n}.`);
    set($("gapSource"), `${twinInfo.label}. “Missed” = 1 − coverage on ${how}. Realism: ${twinInfo.split} split, n = ${twinInfo.n}. The twins are GPT-4.1-mini, built from each person’s own survey answers. Covered = someone in the crowd is as close as the person’s 5th most similar real person.`);
    numbers.ch1 = () => { $("gapMiss").dataset.v = 0; $("gapReal").dataset.v = 0; countTo($("gapMiss"), missed, pct, 1300); countTo($("gapReal"), tw.precision, pct, 1300); };
  })();

  // ---------- 2 · the map ----------
  const AI_KEYS = ["claude_mode", "claude_sampled", "gpt41mini_twin", "flashpoint_panel"];
  const REAL_KEYS = ["gaussian_copula", "independent_marginals", "other_real_people", "retest"];
  const DESC = {
    claude_mode: "Claude answers as the typical person with each profile would.",
    claude_sampled: "Claude draws every answer at random from its own probabilities.",
    gpt41mini_twin: "GPT-4.1-mini digital twins, built from each person’s own survey answers.",
    flashpoint_panel: "A synthetic panel from Flashpoint (uncalibrated).",
    gaussian_copula: "A statistical model fitted on real people’s answers.",
    independent_marginals: "Real answers shuffled between people: right shares, no structure.",
    other_real_people: "A different set of real people: the best any crowd can do.",
    retest: "The same people answering the survey a second time."};
  let ds = "gss", crowd = "claude_mode", revealed = false, mapPlayed = false, main;
  const crowdsOf = (d) => AI_KEYS.concat(REAL_KEYS).filter((k) => summary.datasets[d].crowds[k]);
  function applyMapNumbers() {
    const e = summary.datasets[ds].crowds[crowd];
    if (!revealed) { set($("mapMiss"), "–"); set($("mapPrec"), "–"); $("mapMiss").dataset.v = 0; $("mapPrec").dataset.v = 0; return; }
    countTo($("mapMiss"), 1 - e.coverage, pct, 600); countTo($("mapPrec"), e.precision, pct, 600);
    title(["mapMiss", "mapPrec"], src(ds, crowd));
  }
  async function setCrowd(k) {
    crowd = k;
    const d = summary.datasets[ds];
    const chip = (c, ai) => `<button class="chip ${ai ? "ai" : ""} ${c === k ? "on" : ""}" data-c="${c}" aria-pressed="${c === k}">${esc(d.crowds[c].label)}</button>`;
    $("aiChips").innerHTML = AI_KEYS.filter((c) => d.crowds[c]).map((c) => chip(c, true)).join("");
    $("realChips").innerHTML = REAL_KEYS.filter((c) => d.crowds[c]).map((c) => chip(c, false)).join("");
    document.querySelectorAll("#ch2 .chip").forEach((b) => (b.onclick = () => setCrowd(b.dataset.c)));
    set($("crowdDesc"), DESC[k] || "");
    set($("mapSource"), `${src(ds, k)} Covered = someone in the crowd is as close as the person’s 5th most similar real person. Believable = the crowd person falls inside some real person’s neighbourhood.`);
    applyMapNumbers();
    try { main.set(await map(`map_${ds}_${k}`)); } catch (_) { /* keep previous map */ }
  }
  async function setDataset(d) {
    ds = d;
    $("dsSeg").innerHTML = Object.keys(summary.datasets).map((k) => `<button class="${k === ds ? "on" : ""}" data-d="${k}">${esc(summary.datasets[k].label)}</button>`).join("");
    document.querySelectorAll("#dsSeg button").forEach((b) => (b.onclick = () => setDataset(b.dataset.d)));
    await setCrowd(crowdsOf(ds).includes(crowd) ? crowd : crowdsOf(ds)[0]);
  }
  if (summary) {
    main = new Constellation($("mapMain"), { pad: 26, padT: 56, padB: 64, onStage: (s) => {
      document.querySelectorAll("#steps span").forEach((sp) => sp.classList.toggle("on", +sp.dataset.s <= s));
      if (s === 3 && !revealed) { revealed = true; applyMapNumbers(); }
    } });
    await setDataset("gss");
    numbers.ch2 = () => { if (!mapPlayed) { mapPlayed = true; revealed = false; applyMapNumbers(); main.play(); } };
  }

  // ---------- 3 · one person ----------
  let exIdx = 0;
  function renderExample() {
    const list = exR?.data?.examples;
    if (!list) { $("compare").innerHTML = `<div class="empty">Examples are not exported yet.</div>`; return; }
    const ex = list[exIdx];
    $("exTabs").innerHTML = list.map((e, i) => `<button class="chip ${i === exIdx ? "on" : ""}" data-i="${i}" aria-pressed="${i === exIdx}">${esc(e.label)}</button>`).join("")
      + `<span class="hintline">← → switch person</span>`;
    document.querySelectorAll("#exTabs .chip").forEach((b) => (b.onclick = () => { exIdx = +b.dataset.i; renderExample(); }));
    const diff = ex.rows.filter((r) => r.real !== r.ai), same = ex.rows.filter((r) => r.real === r.ai);
    const row = (r, cls) => `<div class="rowx ${cls}"><div class="q">${esc(r.q)}</div><div class="a real">${esc(r.real)}</div><div class="a ai">${esc(r.ai)}</div></div>`;
    $("compare").innerHTML = `<div class="compare"><div class="head"><div>Question</div><div class="h-real"><i class="sw real"></i>${esc(ex.label)} (real)</div><div class="h-ai"><i class="sw syn"></i>Closest AI person</div></div>`
      + (diff.length ? `<div class="rowx sep"><div>Where they differ</div><div></div><div></div></div>${diff.map((r) => row(r, "diff")).join("")}` : "")
      + (same.length ? `<div class="rowx sep"><div>Where they agree</div><div></div><div></div></div>${same.map((r) => row(r, "same")).join("")}` : "") + `</div>`;
    $("exDiff").dataset.v = 0;
    countTo($("exDiff"), ex.n_diff, (v) => Math.round(v), 700);
    set($("exDiffL"), `of ${ex.n_items} answers differ from the closest AI person`);
    title(["exDiff"], `${ex.label}: ${where()}. Closest AI person = nearest person in the Claude “typical answers” crowd by the same family-weighted distance used for coverage. Nobody in the crowd is as close as this person’s 5th most similar real person, which is why they count as missed.`);
    set($("exSource"), `${where()}; ${ex.n_items} questions. Anonymous respondent, no identifying details. Only the largest differences and two agreements are shown. Nobody in the crowd is as close as this person’s 5th most similar real person, so they count as missed.`);
  }
  numbers.ch3 = renderExample;

  // ---------- 4 · three failures ----------
  const FAIL = [
    ["claude_mode", "Too alike", "Every person looks real, but they all look alike, so many real people have no match."],
    ["claude_sampled", "Incoherent", "The right spread of answers, in combinations nobody actually gives."],
    ["flashpoint_panel", "Wrongly combined", "Spread like real people, but the answers do not combine the way real people’s do."]];
  let failIdx = 0;
  const failCards = [];
  function setFail(i) {
    failIdx = i;
    failCards.forEach((c, j) => { c.el.classList.toggle("on", j === i); c.el.setAttribute("aria-pressed", j === i); });
    const e = gssInfo.crowds[failCards[i].key];
    set($("failMissL"), `real people missed by “${e.label}”`);
    title(["failMiss", "failPrec"], src("gss", failCards[i].key));
    $("failMiss").dataset.v = 0; $("failPrec").dataset.v = 0;
    countTo($("failMiss"), 1 - e.coverage, pct, 600); countTo($("failPrec"), e.precision, pct, 600);
  }
  if (gssInfo) {
    const present = FAIL.filter(([k]) => gssInfo.crowds[k]);
    $("trio").innerHTML = present.map(([k, name, why]) => {
      const e = gssInfo.crowds[k];
      return `<button class="fcard" data-k="${k}" aria-pressed="false"><div class="mapbox"><canvas aria-label="Map for ${esc(e.label)}"></canvas></div>
        <div class="tagline">${esc(name)}</div><div class="who">${esc(e.label)}${e.n ? ` · n = ${e.n}` : ""}</div>
        <div class="nums"><span><b class="num">${pct(1 - e.coverage)}</b> missed</span><span><b class="num">${pct(e.precision)}</b> believable</span></div>
        <p class="why">${esc(why)}</p></button>`;
    }).join("");
    document.querySelectorAll("#trio .fcard").forEach((el, i) => {
      const key = present[i][0], m = new Constellation(el.querySelector("canvas"), { pad: 10 });
      map(`map_gss_${key}`).then((d) => m.set(d)).catch(() => {});
      el.onclick = () => setFail(i);
      failCards.push({ el, key });
    });
    set($("failSource"), `${where()}; ${gssInfo.items ?? ""} questions. Flashpoint panel: n = ${gssInfo.crowds.flashpoint_panel?.n ?? "–"}. Hover a number for its source.`);
    numbers.ch4 = () => setFail(failIdx);
  }

  // ---------- 5 · self-repair ----------
  let repairView = 0;
  const VIEWS = ["What the agent tried", "How many real people"];
  function chartRepairs(d) {
    const ai = d.items.filter((i) => i.kind !== "real"), real = d.items.filter((i) => i.kind === "real");
    const best = ai.reduce((a, b) => (b.coverage > a.coverage ? b : a), ai[0]);
    const W = 990, x0 = 330, x1 = 770, rowH = 46, head = 40, x = (v) => x0 + v * (x1 - x0);
    const H = head * 2 + (ai.length + real.length) * rowH + 56;
    let y = 8, s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Coverage of each repair on people the agent never saw">`;
    for (let v = 0; v <= 1.0001; v += 0.25) s += `<line x1="${x(v)}" x2="${x(v)}" y1="0" y2="${H - 30}" stroke="rgba(244,240,232,.08)"/><text x="${x(v)}" y="${H - 8}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    const group = (name, dot, rows) => {
      s += `<circle cx="14" cy="${y + 14}" r="6" fill="${dot}"/><text class="gt" x="30" y="${y + 19}">${name}</text>`;
      y += head - 6;
      for (const it of rows) {
        const isBest = it === best, isStar = it === real[0];
        const col = it.kind === "real" ? "#f4f0e8" : it.kind === "ai_filtered" ? "#7ff0dc" : "#3ee0c6";
        s += `<text class="lbl" x="${x0 - 14}" y="${y + 18}" text-anchor="end">${esc(it.label)}</text>`;
        if (it.note) s += `<text x="${x0 - 14}" y="${y + 35}" text-anchor="end" style="font-size:14px;fill:#8089a3">${esc(it.note)}</text>`;
        s += `<rect x="${x(0)}" y="${y + 4}" width="${Math.max(2, x(it.coverage) - x(0))}" height="26" rx="5" fill="${col}" opacity="${it.kind === "real" ? 0.95 : 0.78}"${isBest || isStar ? ` stroke="#ffc24b" stroke-width="3"` : ""}/>`;
        if (it.lo != null && it.hi != null) s += `<line x1="${x(it.lo)}" x2="${x(it.hi)}" y1="${y + 17}" y2="${y + 17}" stroke="#060914" stroke-width="3"/><line x1="${x(it.lo)}" x2="${x(it.lo)}" y1="${y + 11}" y2="${y + 23}" stroke="#060914" stroke-width="3"/><line x1="${x(it.hi)}" x2="${x(it.hi)}" y1="${y + 11}" y2="${y + 23}" stroke="#060914" stroke-width="3"/>`;
        s += `<text class="val" x="${x(it.coverage) + 10}" y="${y + 24}">${pct(it.coverage)}</text>`;
        if (isBest) s += `<text class="tag" x="${x(it.coverage) + 62}" y="${y + 23}">◄ best AI repair</text>`;
        if (isStar) s += `<text class="tag" x="${x(it.coverage) + 62}" y="${y + 23}">◄ ${/\d+/.exec(it.label)?.[0] ?? ""} real people</text>`;
        y += rowH;
      }
      y += 12;
    };
    group("The agent tries to fix itself", "#3ee0c6", ai);
    group("The agent asks real people", "#f4f0e8", real);
    if (d.baseline != null) s += `<line x1="${x(d.baseline)}" x2="${x(d.baseline)}" y1="0" y2="${H - 30}" stroke="#ff6b4a" stroke-width="2" stroke-dasharray="6 6"/><text x="${x(d.baseline) + 6}" y="${H - 34}" style="fill:#ff6b4a;font-weight:600">AI alone ${pct(d.baseline)}</text>`;
    return { svg: s + "</svg>", best, real };
  }
  function chartBoard(sb) {
    const W = 900, H = 470, L = 56, Rm = 270, T = 18, B = 52, xs = sb.budgets;
    const x = (b) => L + ((b - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - L - Rm);
    const vals = []; Object.values(sb.methods).forEach((m) => vals.push(...m.lo, ...m.hi)); Object.values(sb.references || {}).forEach((r) => vals.push(r.value));
    const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.04) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.02) * 20) / 20);
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const colors = { copula_on_asked: "#f4f0e8", claude_plus_asked: "#ffc24b", blindspot_select_fp: "#3ee0c6", blindspot_select: "#1f9f8d" };
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Coverage by number of real people">`;
    for (let v = lo; v <= hi + 1e-9; v += 0.1) s += `<line x1="${L}" x2="${W - Rm}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(244,240,232,.08)"/><text x="${L - 10}" y="${y(v) + 5}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    xs.forEach((b) => (s += `<text x="${x(b)}" y="${H - B + 24}" text-anchor="middle">${b}</text>`));
    s += `<text x="${(L + W - Rm) / 2}" y="${H - 8}" text-anchor="middle">real people asked</text>`;
    const labels = [];
    for (const r of Object.values(sb.references || {})) { s += `<line x1="${L}" x2="${W - Rm}" y1="${y(r.value)}" y2="${y(r.value)}" stroke="#8089a3" stroke-dasharray="5 6"/>`; labels.push({ y: y(r.value), t: `${r.label} ${pct(r.value)}`, c: "#b0b7ca" }); }
    for (const [k, m] of Object.entries(sb.methods)) {
      const c = colors[k] || "#b0b7ca", band = xs.map((b, i) => `${x(b)},${y(m.hi[i])}`).join(" ") + " " + xs.map((b, i) => `${x(b)},${y(m.lo[i])}`).reverse().join(" ");
      s += `<polygon points="${band}" fill="${c}" opacity=".13"/><polyline points="${xs.map((b, i) => `${x(b)},${y(m.mean[i])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="${k === "copula_on_asked" ? 4 : 2.5}"/>`;
      xs.forEach((b, i) => (s += `<circle cx="${x(b)}" cy="${y(m.mean[i])}" r="4.5" fill="${c}"/>`));
      labels.push({ y: y(m.mean[m.mean.length - 1]), t: `${m.label} ${pct(m.mean[m.mean.length - 1])}`, c });
    }
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 19) labels[i].y = labels[i - 1].y + 19;
    labels.forEach((l) => (s += `<text x="${W - Rm + 12}" y="${l.y + 5}" style="fill:${l.c};font-size:14px">${esc(l.t)}</text>`));
    return s + "</svg>";
  }
  function renderRepair() {
    $("repairSeg").innerHTML = VIEWS.map((v, i) => `<button class="${i === repairView ? "on" : ""}" data-i="${i}">${v}</button>`).join("");
    document.querySelectorAll("#repairSeg button").forEach((b) => (b.onclick = () => { repairView = +b.dataset.i; renderRepair(); }));
    const rep = repR?.data, board = boardR?.data;
    if (repairView === 0) {
      if (!rep) { $("repairChart").innerHTML = `<div class="empty">Repair results not exported yet.</div>`; return; }
      const { svg, best, real } = chartRepairs(rep), last = real[real.length - 1];
      $("repairChart").innerHTML = svg; set($("repairCaption"), rep.caption || "");
      $("repAi").dataset.v = 0; $("repReal").dataset.v = 0;
      countTo($("repAi"), best.coverage, pct, 700); countTo($("repReal"), last.coverage, pct, 700);
      set($("repAiL"), `best AI repair: ${best.label.toLowerCase()}${best.note ? ` (${best.note})` : ""}`);
      set($("repRealL"), `statistics on the same real people: ${last.label.toLowerCase()}`);
      set($("repairExplain"), `The best AI repair still needed real people as a filter${best.note ? ` (${best.note})` : ""}. Statistics fitted on the same number of real people do better (${pct(last.coverage)}). Even ${/\d+/.exec(real[0].label)?.[0] ?? "a few"} real people reach ${pct(real[0].coverage)}. Every repair was scored on real people it had never seen.`);
      title(["repAi", "repReal"], `${where()}. Bars show the seed range, or a 95% interval for single runs.`);
      set($("repairSource"), `${where()}. Coverage as in chapter 2. Rows with a note use a different item set or number of real people. Ranges: min–max over seeds; single runs show a 95% interval.`);
    } else {
      if (!board) { $("repairChart").innerHTML = `<div class="empty">Scoreboard not exported yet.</div>`; return; }
      $("repairChart").innerHTML = chartBoard(board); set($("repairCaption"), board.caption || "");
      set($("repairSource"), `${where()}. Same real people used by every method; lines = mean, band = seed range (10 seeds).`);
    }
  }
  numbers.ch5 = renderRepair;

  // ---------- 6 · who is missed ----------
  (() => {
    const d = fairR?.data;
    if (!d) { $("fairChart").innerHTML = `<div class="empty">Group results not exported yet.</div>`; return; }
    const g = d.groups, lowG = g.reduce((a, b) => (b.coverage < a.coverage ? b : a)), highG = g.reduce((a, b) => (b.coverage > a.coverage ? b : a));
    const dims = [...new Set(g.map((q) => q.dimension))];
    const W = 900, L = 250, R = 150, rowH = 27, dimH = 36, T = 12, H = T + g.length * rowH + dims.length * dimH + 40;
    const lo = 0.3, x = (v) => L + ((v - lo) / (1 - lo)) * (W - L - R);
    let y = T, s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of each group the AI crowd covers">`;
    for (let v = 0.3; v <= 1.0001; v += 0.1) s += `<line x1="${x(v)}" x2="${x(v)}" y1="0" y2="${H - 30}" stroke="rgba(244,240,232,.08)"/><text x="${x(v)}" y="${H - 8}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    s += `<line x1="${x(d.overall)}" x2="${x(d.overall)}" y1="0" y2="${H - 30}" stroke="#b0b7ca" stroke-dasharray="5 6"/><text x="${x(d.overall) + 6}" y="${T + 6}" style="fill:#b0b7ca">overall ${pct(d.overall)}</text>`;
    for (const dim of dims) {
      const inc0 = /income/i.test(dim), n0 = g.filter((q) => q.dimension === dim).length;
      if (inc0) s += `<rect x="6" y="${y - 4}" width="${W - 12}" height="${dimH + n0 * rowH - 2}" rx="12" fill="rgba(255,107,74,.07)" stroke="rgba(255,107,74,.55)" stroke-width="1.5"/>`;
      s += `<text class="gt" x="${L - 16}" y="${y + 18}" text-anchor="end" style="font-size:14px${inc0 ? ";fill:#ffb9a6" : ""}">${esc(dim)}</text>`; y += dimH;
      for (const it of g.filter((q) => q.dimension === dim)) {
        const low = it.coverage < d.overall - 0.05, c = low ? "#ff6b4a" : "#f4f0e8", inc = /income/i.test(dim);
        s += `<text class="lbl" x="${L - 16}" y="${y + 5}" text-anchor="end" style="font-size:${inc ? 16 : 14}px;${inc ? "font-weight:700" : "opacity:.85"}">${esc(it.label)}</text>`;
        if (it.lo != null) s += `<line x1="${x(it.lo)}" x2="${x(it.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-opacity=".55" stroke-width="3" stroke-linecap="round"/>`;
        s += `<circle cx="${x(it.coverage)}" cy="${y}" r="${inc ? 9 : 7}" fill="${c}"/>${low ? `<circle cx="${x(it.coverage)}" cy="${y}" r="12" fill="none" stroke="#ff6b4a" stroke-width="1.8"/>` : ""}`;
        s += `<text class="val" x="${W - R + 14}" y="${y + 6}" style="font-size:16px">${pct(it.coverage)}</text><text x="${W - R + 62}" y="${y + 5}" style="font-size:14px;fill:#8089a3">n = ${it.n}</text>`;
        y += rowH;
      }
    }
    $("fairChart").innerHTML = s + "</svg>"; set($("fairCaption"), d.caption || "");
    set($("fairLowL"), `${lowG.dimension.toLowerCase()}: ${lowG.label.toLowerCase()}`); set($("fairHighL"), `${highG.dimension.toLowerCase()}: ${highG.label.toLowerCase()}`);
    const t = (q) => `${q.dimension}: ${q.label}, n = ${q.n}, 95% interval ${pct(q.lo)}–${pct(q.hi)}. Overall ${pct(d.overall)}. ${where()}.`;
    title(["fairLow"], t(lowG)); title(["fairHigh"], t(highG));
    set($("fairSource"), `${where()}, Claude “typical answers” crowd, groups of 30+ people, bootstrap 95% intervals. Ringed coral = more than 5 points below overall; white = about the same or above. Household income is highlighted because it shows the widest gap. Descriptive, not causal.`);
    numbers.ch6 = () => { $("fairLow").dataset.v = 0; $("fairHigh").dataset.v = 0; countTo($("fairLow"), lowG.coverage, pct, 700); countTo($("fairHigh"), highG.coverage, pct, 700); };
  })();

  // ---------- 7 · the product ----------
  (() => {
    const cop = boardR?.data?.methods?.copula_on_asked, b = boardR?.data?.budgets;
    if (cop && b) {
      set($("prodLowL"), `covered by statistics fitted on ${b[0]} real people`); set($("prodHighL"), `with ${b[b.length - 1]} real people`);
      title(["prodLow", "prodHigh"], `${where()}; mean over 10 seeds.`);
      set($("prodSource"), `${where()}; mean over 10 seeds of random real samples.`);
      numbers.ch7 = () => { $("prodLow").dataset.v = 0; $("prodHigh").dataset.v = 0; countTo($("prodLow"), cop.mean[0], pct, 800); countTo($("prodHigh"), cop.mean[cop.mean.length - 1], pct, 800); };
    }
    const d = recR && !recR.mock ? recR.data : null;
    $("recruit").className = "recruit rv" + (d ? " done" : "");
    $("recruit").innerHTML = d
      ? `<span class="pill">Real study</span><div><div class="rn num">${d.n}</div><div class="fine">real people, ${esc(d.provider)}${d.cost_usd ? `, $${d.cost_usd}` : ""}</div></div>
         <div><div class="rn num" style="color:var(--miss)">${pct(d.coverage_before)}</div><div class="fine">covered before</div></div>
         <div><div class="rn num" style="color:var(--syn)">${pct(d.coverage_after)}</div><div class="fine">after ${esc(d.method || "the fix")}</div></div>
         <p>${esc(d.caption || "")}</p>`
      : `<span class="pill">Next</span><p>Recruiting real respondents through Flashpoint, then fitting the repair on them and re-scoring the AI crowd against the national survey.</p>`;
  })();

  // ---------- 8 · your turn ----------
  const params = new URLSearchParams(location.search);
  const joinUrl = params.get("join") || `${location.origin}/join`;
  set($("joinUrl"), joinUrl);
  try { const q = qrcode(0, "M"); q.addData(joinUrl); q.make(); $("qr").innerHTML = q.createSvgTag({ cellSize: 8, margin: 0, scalable: true }); }
  catch (_) { $("qr").textContent = joinUrl; }
  set($("liveSource"), "Live count since this page opened. Answers are scored in memory and never stored.");
  let after = 0, nJoin = 0, nMiss = 0;
  async function poll() {
    try {
      const d = await (await fetch(`/api/live/recent?after=${after}`, { cache: "no-store" })).json();
      for (const p of d.points) { after = Math.max(after, p.seq); nJoin++; if (!p.covered) nMiss++; }
      countTo($("liveN"), nJoin, (v) => Math.round(v), 500); countTo($("liveMiss"), nMiss, (v) => Math.round(v), 500);
    } catch (_) { /* server busy: try again */ }
    setTimeout(poll, 2000);
  }
  poll();

  // ---------- navigation ----------
  const story = $("story"), bar = $("chapters"), line = $("progressLine");
  bar.innerHTML = CHAPTERS.map((c, i) => `<button data-i="${i}" title="${esc(c.dataset.short)}"><span class="n">${i + 1}</span><span class="t">${esc(c.dataset.short)}</span></button>`).join("");
  const btns = [...bar.children];
  let cur = 0, shown = -1;
  const go = (i) => { cur = Math.max(0, Math.min(CHAPTERS.length - 1, i)); CHAPTERS[cur].scrollIntoView({ behavior: reduced() ? "auto" : "smooth" }); };
  btns.forEach((b, i) => (b.onclick = () => go(i)));
  function activate(i) {
    shown = i; cur = i;
    const f = numbers[CHAPTERS[i].id]; if (f) f();
    btns.forEach((b, j) => { b.classList.toggle("on", j === i); if (j <= i) b.classList.add("seen"); });
    line.style.width = `${((i + 1) / CHAPTERS.length) * 100}%`;
    bar.scrollTo({ left: btns[i].offsetLeft - bar.clientWidth / 2 + btns[i].clientWidth / 2, behavior: reduced() ? "auto" : "smooth" });   // keep the current chapter visible on narrow screens
    history.replaceState(null, "", `#${CHAPTERS[i].id}`);
  }
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    const i = CHAPTERS.indexOf(e.target);
    if (e.isIntersecting && e.intersectionRatio >= 0.25) e.target.classList.add("in");
    if (e.isIntersecting && e.intersectionRatio >= 0.55 && shown !== i) activate(i);
  }), { root: story, threshold: [0.25, 0.55, 0.8] });
  CHAPTERS.forEach((c) => io.observe(c));

  function restart() {
    CHAPTERS.forEach((c) => c.classList.remove("in"));
    mapPlayed = false; revealed = false; failIdx = 0; exIdx = 0; repairView = 0; shown = -1;
    if (summary) setDataset("gss").then(() => setCrowd("claude_mode"));
    story.scrollTo({ top: 0, behavior: "auto" });
    setTimeout(() => { CHAPTERS[0].classList.add("in"); activate(0); }, 90);
  }
  const cycle = (list, now, d) => list[(list.indexOf(now) + d + list.length) % list.length];
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const id = CHAPTERS[cur].id, d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (["ArrowDown", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(cur + 1); }
    else if (["ArrowUp", "PageUp"].includes(e.key)) { e.preventDefault(); go(cur - 1); }
    else if (d) {
      e.preventDefault();
      if (id === "ch2" && summary) setCrowd(cycle(crowdsOf(ds), crowd, d));
      else if (id === "ch3" && exR?.data?.examples) { exIdx = (exIdx + d + exR.data.examples.length) % exR.data.examples.length; renderExample(); }
      else if (id === "ch4" && failCards.length) setFail((failIdx + d + failCards.length) % failCards.length);
      else if (id === "ch5") { repairView = (repairView + d + VIEWS.length) % VIEWS.length; renderRepair(); }
    } else if (e.key === "r" || e.key === "R") restart();
    else if (/^[1-8]$/.test(e.key)) go(+e.key - 1);
  });
  setTimeout(() => { CHAPTERS[0].classList.add("in"); if (shown < 0) activate(0); }, 60);
  if (location.hash) { const i = CHAPTERS.findIndex((c) => `#${c.id}` === location.hash); if (i > 0) setTimeout(() => go(i), 200); }
})();
