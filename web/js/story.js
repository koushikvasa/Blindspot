(async () => {
  const { load, pct, countTo, markSample } = window.BS;
  const $ = (id) => document.getElementById(id);
  const CROWD_ORDER = ["retest", "other_real_people", "gaussian_copula", "independent_marginals", "flashpoint_panel", "claude_mode", "claude_sampled", "gpt41mini_twin"];
  const mapCache = {};

  async function map(name) {
    if (!mapCache[name]) {
      const r = await load(name);
      markSample(r.mock);
      mapCache[name] = r.data;
    }
    return mapCache[name];
  }

  // ---------- summary + hook ----------
  const sumRes = await load("summary");
  markSample(sumRes.mock);
  const summary = sumRes.data;
  const twin = summary.datasets.twin?.crowds?.gpt41mini_twin;
  if (twin) {
    countTo($("hookBig"), 1 - twin.coverage, (v) => pct(v), 1200);
    $("hookLine").textContent = "of real people have no one like them in a crowd of AI twins.";
    $("hookSub").innerHTML = `Yet <b>${pct(twin.precision)}</b> of those twins look like perfectly realistic people.`;
    const ds = summary.datasets.twin;
    $("hookSource").textContent = `${ds.label}, ${ds.split} split, n = ${ds.n}. Twins: GPT-4.1-mini, built by the dataset's authors from each person's own survey answers. Covered = someone in the crowd is as close as the person's 5th most similar real person.`;
  }

  // ---------- scene 2: constellation ----------
  const main = new Constellation($("mapMain"));
  const datasets = Object.keys(summary.datasets);
  let ds = datasets.includes("twin") ? "twin" : datasets[0];
  let crowd = null;

  function crowdsOf(d) {
    const c = summary.datasets[d].crowds;
    return CROWD_ORDER.filter((k) => c[k]).concat(Object.keys(c).filter((k) => !CROWD_ORDER.includes(k)));
  }

  function renderDs() {
    $("dsSeg").innerHTML = "";
    for (const d of datasets) {
      const b = document.createElement("button");
      b.textContent = summary.datasets[d].label;
      b.className = d === ds ? "on" : "";
      b.onclick = () => { ds = d; renderDs(); setCrowd(defaultCrowd()); };
      $("dsSeg").appendChild(b);
    }
  }
  const defaultCrowd = () => (crowdsOf(ds).includes("gpt41mini_twin") ? "gpt41mini_twin" : "claude_mode");

  async function setCrowd(k) {
    crowd = k;
    const info = summary.datasets[ds].crowds[k];
    $("crowdChips").innerHTML = "";
    for (const c of crowdsOf(ds)) {
      const b = document.createElement("button");
      b.className = "chip" + (c === k ? " on" : "");
      b.textContent = summary.datasets[ds].crowds[c].label || c;
      b.onclick = () => setCrowd(c);
      $("crowdChips").appendChild(b);
    }
    countTo($("statMiss"), 1 - info.coverage);
    countTo($("statPrec"), info.precision);
    const ci = info.coverage_ci ? ` (95% CI ${pct(1 - info.coverage_ci[1])}–${pct(1 - info.coverage_ci[0])} missed)` : "";
    $("mainSource").textContent = `${summary.datasets[ds].label}, ${summary.datasets[ds].split} split, n = ${summary.datasets[ds].n}${ci}.`;
    try { main.set(await map(`map_${ds}_${k}`)); } catch (_) { /* map missing: keep previous */ }
  }
  renderDs();
  await setCrowd(defaultCrowd());

  // ---------- scene 3: three failure modes ----------
  const dsDuo = summary.datasets.gss?.crowds?.claude_sampled ? "gss" : "twin";
  const crowdsDuo = summary.datasets[dsDuo].crowds;
  const cap = (el, c) => { if (c) $(el).innerHTML = `missed <b>${pct(1 - c.coverage)}</b> · realistic <b>${pct(c.precision)}</b>`; };
  const trioMaps = [["mapModeC", "claude_mode", "capMode"], ["mapSampC", "claude_sampled", "capSamp"], ["mapFpC", "flashpoint_panel", "capFp"]];
  for (const [canvas, key, capId] of trioMaps) {
    const c = crowdsDuo[key];
    if (!c) { if (key === "flashpoint_panel") $("fpCol").style.display = "none"; continue; }
    cap(capId, c);
    const m = new Constellation($(canvas), { pad: 16 });
    map(`map_${dsDuo}_${key}`).then((d) => m.set(d)).catch(() => {});
  }

  // ---------- scene 4: repair attempts ----------
  try {
    const r = await load("repairs_gss"); markSample(r.mock);
    drawRepairs(r.data);
  } catch (_) { $("repairs").innerHTML = '<div class="empty">Repair results not exported yet.</div>'; }

  function drawRepairs(d) {
    const rows = d.items, W = 1100, rowH = 40, L = 360, R = 90, T = 30, H = T + rows.length * rowH + 30;
    const x = (v) => L + v * (W - L - R);
    const col = { ai: "#2DD4BF", ai_filtered: "#7FE3D6", hybrid: "#C9B6FF", real: "#E8ECF5" };
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Coverage of each repair attempt">`;
    for (let v = 0; v <= 1.0001; v += 0.2) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${T - 10}" y2="${H - 24}" stroke="rgba(232,236,245,.07)"/>`;
      s += `<text x="${x(v)}" y="${H - 6}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    }
    if (d.baseline != null) {
      s += `<line x1="${x(d.baseline)}" x2="${x(d.baseline)}" y1="${T - 14}" y2="${H - 24}" stroke="#FF7A59" stroke-dasharray="4 5"/>`;
      s += `<text x="${x(d.baseline) + 6}" y="${T - 16}" style="fill:#FF7A59">AI alone ${pct(d.baseline)}</text>`;
    }
    rows.forEach((it, i) => {
      const y = T + i * rowH + 8, c = col[it.kind] || "#9AA3B8";
      s += `<text x="${L - 14}" y="${y + 14}" text-anchor="end" style="fill:${it.kind === "real" ? "#E8ECF5" : "#C7CDDB"};font-size:14px">${esc(it.label)}</text>`;
      s += `<rect x="${x(0)}" y="${y}" width="${Math.max(2, x(it.coverage) - x(0))}" height="22" rx="4" fill="${c}" opacity="${it.kind === "real" ? 0.95 : 0.8}"/>`;
      if (it.lo != null && it.hi != null) s += `<line x1="${x(it.lo)}" x2="${x(it.hi)}" y1="${y + 11}" y2="${y + 11}" stroke="#070B16" stroke-width="2"/>`;
      s += `<text x="${x(it.coverage) + 8}" y="${y + 15}" style="fill:#E8ECF5;font-size:13px">${pct(it.coverage)}${it.note ? `  ·  ${esc(it.note)}` : ""}</text>`;
    });
    s += "</svg>";
    $("repairs").innerHTML = s;
    $("repairsCaption").textContent = d.caption || "";
  }

  function esc(t) { return String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch])); }

  // ---------- scene 5: scoreboard ----------
  try {
    const r = await load("scoreboard_gss"); markSample(r.mock);
    drawChart(r.data);
  } catch (_) { $("chart").innerHTML = '<div class="empty">Scoreboard not exported yet.</div>'; }

  function drawChart(sb) {
    const W = 1100, H = 440, L = 70, Rm = 230, T = 20, B = 50;
    const xs = sb.budgets, x = (b) => L + ((b - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - L - Rm);
    const vals = [];
    Object.values(sb.methods).forEach((m) => vals.push(...m.lo, ...m.hi));
    Object.values(sb.references || {}).forEach((r) => vals.push(r.value));
    const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.05) * 10) / 10), hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.02) * 10) / 10);
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const palette = { blindspot: "#2DD4BF", claude_plus_asked: "#E8ECF5", copula_on_asked: "#A78BFA", other: "#9AA3B8" };
    const color = (k) => (k.startsWith("blindspot") ? (k.includes("random") ? "#1F8F80" : "#2DD4BF") : palette[k] || palette.other);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Coverage by number of real people asked">`;
    for (let v = lo; v <= hi + 1e-9; v += 0.05) {
      s += `<line x1="${L}" x2="${W - Rm}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(232,236,245,.07)"/>`;
      s += `<text x="${L - 10}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    }
    xs.forEach((b) => (s += `<text x="${x(b)}" y="${H - B + 22}" text-anchor="middle">${b}</text>`));
    s += `<text x="${(L + W - Rm) / 2}" y="${H - 6}" text-anchor="middle">real people asked</text>`;
    const labels = [];
    for (const [k, r] of Object.entries(sb.references || {})) {
      s += `<line x1="${L}" x2="${W - Rm}" y1="${y(r.value)}" y2="${y(r.value)}" stroke="#6B748A" stroke-dasharray="5 6"/>`;
      labels.push({ y: y(r.value), text: `${r.label} ${pct(r.value)}`, c: "#9AA3B8" });
    }
    for (const [k, m] of Object.entries(sb.methods)) {
      const c = color(k);
      const band = xs.map((b, i) => `${x(b)},${y(m.hi[i])}`).join(" ") + " " + xs.map((b, i) => `${x(b)},${y(m.lo[i])}`).reverse().join(" ");
      s += `<polygon points="${band}" fill="${c}" opacity=".12"/>`;
      s += `<polyline points="${xs.map((b, i) => `${x(b)},${y(m.mean[i])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="${k.startsWith("blindspot") && !k.includes("random") ? 3 : 2}"/>`;
      xs.forEach((b, i) => (s += `<circle cx="${x(b)}" cy="${y(m.mean[i])}" r="3.5" fill="${c}"/>`));
      labels.push({ y: y(m.mean[m.mean.length - 1]), text: `${m.label} ${pct(m.mean[m.mean.length - 1])}`, c });
    }
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 17) labels[i].y = labels[i - 1].y + 17;
    labels.forEach((l) => (s += `<text x="${W - Rm + 10}" y="${l.y + 4}" style="fill:${l.c}">${l.text}</text>`));
    s += "</svg>";
    $("chart").innerHTML = s;
    $("chartCaption").textContent = sb.caption || "";
  }

  // ---------- scene 6: who is missed ----------
  try {
    const r = await load("fairness_gss"); markSample(r.mock);
    drawFairness(r.data);
  } catch (_) { $("fairness").innerHTML = '<div class="empty">Group results not exported yet.</div>'; }

  function drawFairness(d) {
    const groups = d.groups, W = 1100, rowH = 30, L = 300, R = 120, T = 20;
    const dims = [...new Set(groups.map((g) => g.dimension))];
    const H = T + groups.length * rowH + dims.length * 18 + 34;
    const x = (v) => L + ((v - 0.3) / 0.7) * (W - L - R);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="AI crowd coverage by group">`;
    for (let v = 0.3; v <= 1.0001; v += 0.1) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H - 26}" stroke="rgba(232,236,245,.07)"/>`;
      s += `<text x="${x(v)}" y="${H - 8}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    }
    if (d.overall != null) {
      s += `<line x1="${x(d.overall)}" x2="${x(d.overall)}" y1="${T}" y2="${H - 26}" stroke="#9AA3B8" stroke-dasharray="4 5"/>`;
      s += `<text x="${x(d.overall) + 6}" y="${T + 10}" style="fill:#9AA3B8">overall ${pct(d.overall)}</text>`;
    }
    let y = T + 8;
    for (const dim of dims) {
      s += `<text x="${L - 14}" y="${y + 6}" text-anchor="end" style="fill:#6B748A;font-size:12px">${esc(dim)}</text>`;
      y += 18;
      for (const g of groups.filter((g) => g.dimension === dim)) {
        const low = d.overall != null && g.coverage < d.overall - 0.05;
        const c = low ? "#FF7A59" : "#E8ECF5";
        s += `<text x="${L - 14}" y="${y + 5}" text-anchor="end" style="fill:#C7CDDB;font-size:14px">${esc(g.label)}</text>`;
        if (g.lo != null && g.hi != null) s += `<line x1="${x(g.lo)}" x2="${x(g.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-opacity=".5" stroke-width="2"/>`;
        s += `<circle cx="${x(g.coverage)}" cy="${y}" r="6" fill="${c}"/>`;
        s += `<text x="${x(1) + 14}" y="${y + 5}" style="font-size:13px">${pct(g.coverage)}${g.n ? `  n=${g.n}` : ""}</text>`;
        y += rowH;
      }
    }
    s += "</svg>";
    $("fairness").innerHTML = s;
    $("fairnessCaption").textContent = d.caption || "";
  }

  // ---------- scene 7: real people through Flashpoint ----------
  try {
    const r = await load("realrecruit_gss");
    if (r.mock) throw new Error("no sample data for real recruitment");
    const d = r.data;
    $("realBody").innerHTML = `
      <div class="card"><div class="stat miss num">${pct(d.coverage_before)}</div><div class="stat-label">covered by the AI crowd alone</div></div>
      <div class="card"><div class="stat num">${d.n}</div><div class="stat-label">real people recruited via Flashpoint (${esc(d.provider)})${d.cost_usd ? `, $${d.cost_usd}` : ""}</div></div>
      <div class="card"><div class="stat syn num">${pct(d.coverage_after)}</div><div class="stat-label">covered after Blindspot's fix${d.method ? ` (${esc(d.method)})` : ""}</div></div>`;
    if (d.caption) $("realBody").insertAdjacentHTML("afterend", `<p class="lede">${esc(d.caption)}</p>`);
  } catch (_) {
    document.querySelector("#realScene h2").textContent = "Next: real people, recruited through Flashpoint.";
    $("realBody").outerHTML = `<p class="pending">Real respondents are being recruited through Flashpoint today. When their answers arrive, Blindspot fits its repair on them and re-scores the crowd against the national survey.</p>`;
  }

  // ---------- scene 6: QR ----------
  const params = new URLSearchParams(location.search);
  const joinUrl = params.get("join") || `${location.origin}/join`;
  $("joinUrl").textContent = joinUrl;
  try {
    const q = qrcode(0, "M"); q.addData(joinUrl); q.make();
    $("qr").innerHTML = q.createSvgTag({ cellSize: 8, margin: 0, scalable: true });
  } catch (_) { $("qr").textContent = joinUrl; }

  // ---------- navigation ----------
  const scenes = [...document.querySelectorAll(".scene")];
  const pager = $("pager");
  scenes.forEach((sc, i) => {
    const b = document.createElement("button");
    b.title = sc.dataset.name; b.setAttribute("aria-label", sc.dataset.name);
    b.onclick = () => go(i);
    pager.appendChild(b);
  });
  let cur = 0;
  const go = (i) => { cur = Math.max(0, Math.min(scenes.length - 1, i)); scenes[cur].scrollIntoView({ behavior: "smooth" }); };
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { cur = scenes.indexOf(e.target); [...pager.children].forEach((b, j) => b.classList.toggle("on", j === cur)); }
  }), { threshold: 0.6 });
  scenes.forEach((s) => io.observe(s));

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.key.startsWith("Arrow") && (e.key === "ArrowLeft" || e.key === "ArrowRight")) e.preventDefault();
    const name = scenes[cur].dataset.name;
    if (["ArrowDown", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(cur + 1); }
    else if (["ArrowUp", "PageUp"].includes(e.key)) { e.preventDefault(); go(cur - 1); }
    else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const d = e.key === "ArrowRight" ? 1 : -1;
      if (name === "Constellation") { const list = crowdsOf(ds); setCrowd(list[(list.indexOf(crowd) + d + list.length) % list.length]); }
    } else if (e.key === "r" || e.key === "R") { go(0); }
  });
})();
