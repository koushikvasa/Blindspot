(async () => {
  const { load, pct, countTo, markSample } = window.BS;
  const $ = (id) => document.getElementById(id);
  const CROWD_ORDER = ["retest", "other_real_people", "gaussian_copula", "independent_marginals", "claude_mode", "claude_sampled", "gpt41mini_twin"];
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

  // ---------- scene 3: typical vs random ----------
  const dsDuo = summary.datasets.gss?.crowds?.claude_sampled ? "gss" : "twin";
  const mMode = new Constellation($("mapModeC"), { pad: 18 });
  const mSamp = new Constellation($("mapSampC"), { pad: 18 });
  const cm = summary.datasets[dsDuo].crowds.claude_mode, cs = summary.datasets[dsDuo].crowds.claude_sampled;
  if (cm && cs) {
    $("capMode").innerHTML = `missed <b>${pct(1 - cm.coverage)}</b> · realistic <b>${pct(cm.precision)}</b>`;
    $("capSamp").innerHTML = `missed <b>${pct(1 - cs.coverage)}</b> · realistic <b>${pct(cs.precision)}</b>`;
    map(`map_${dsDuo}_claude_mode`).then((d) => mMode.set(d)).catch(() => {});
    map(`map_${dsDuo}_claude_sampled`).then((d) => mSamp.set(d)).catch(() => {});
  }

  // ---------- scene 4: agent rounds ----------
  const agentMap = new Constellation($("mapAgent"));
  let agent = null, round = 0;
  try { const r = await load("agent_gss"); markSample(r.mock); agent = r.data; } catch (_) {}
  const STATUS = { learned: "learned", rejected: "not yet", cannot_imagine: "can't imagine: asking people" };

  async function setRound(r) {
    if (!agent) return;
    round = Math.max(0, Math.min(agent.rounds.length - 1, r));
    const R = agent.rounds[round];
    $("roundSlider").value = round; $("roundNum").textContent = round;
    $("roundLabel").textContent = round === 0 ? "Round 0: AI crowd only" : `Round ${round}`;
    countTo($("agentCov"), R.coverage);
    countTo($("agentAsked"), R.asked, (v) => Math.round(v).toString());
    const box = $("gaps");
    box.innerHTML = "";
    if (!R.gaps.length) box.innerHTML = '<div class="empty">Round 0: the plain AI crowd, before asking anyone.</div>';
    for (const g of R.gaps) {
      const el = document.createElement("div");
      el.className = "gap";
      const meta = g.status === "cannot_imagine"
        ? `${g.recipes_tried} recipes tried, none worked · recruited ${g.recruited} real people` + (g.observed_rate != null ? ` · ${pct(g.observed_rate)} missed in that group` : "")
        : `${g.size} people · ${g.recipes_accepted}/${g.recipes_tried} recipes passed verification`;
      el.innerHTML = `<div class="top"><span class="name"></span><span class="status ${g.status}">${STATUS[g.status] || g.status}</span></div>
        <div class="summary"></div><div class="meta">${meta}</div>`;
      el.querySelector(".name").textContent = g.name;
      el.querySelector(".summary").textContent = g.summary || "";
      box.appendChild(el);
    }
    try { agentMap.set(await map(R.map.replace(/\.json$/, ""))); } catch (_) {}
  }
  if (agent) {
    $("roundSlider").max = agent.rounds.length - 1;
    $("roundSlider").oninput = (e) => setRound(+e.target.value);
    setRound(0);
  }

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
      if (name === "Agent") setRound(round + d);
      if (name === "Constellation") { const list = crowdsOf(ds); setCrowd(list[(list.indexOf(crowd) + d + list.length) % list.length]); }
    } else if (e.key === "r" || e.key === "R") { setRound(0); go(0); }
  });
})();
