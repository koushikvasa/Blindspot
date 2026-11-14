/* Room version 2 of "can the AI imagine you?".
   Changes from v1 come from 8 simulated testers (busy judge, Gen Z student, privacy engineer, retiree, gamer,
   tired attendee, non-native speaker, survey methodologist): they read ~20-30% of the words, satisficed on 0-10 rows,
   and skipped the fine print. So: 7 questions, 3-6 words each, 5 big labelled buttons, a one-line verdict. */
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pct = (v) => `${Math.round(v * 100)}%`;
  let items = [], answers = {}, result = null, busy = false, meta = null;

  function setProgress(k) {
    $("bar").style.width = `${(k / items.length) * 100}%`;
    $("count").textContent = k < items.length ? `${k + 1} of ${items.length}` : "";
  }
  function questionHtml(k, inInvite) {
    const it = items[k], left = items.length - k;
    const opts = it.choices.map((c) => `<button class="opt${answers[it.code] === c.value ? " on" : ""}" data-v="${c.value}">${esc(c.label)}</button>`).join("");
    return `
      <div class="${inInvite ? "" : "slide"}">
        ${inInvite ? "" : `<p class="tease num">${left === 1 ? "Last one" : `${left} to go`} → your verdict</p>`}
        <h2 class="qtext">${esc(it.question)}</h2>
        <div class="opts n${it.choices.length}" role="group" aria-label="Answers">${opts}</div>
        <div class="qfoot">${k > 0 ? `<button class="back" data-back>← Back</button>` : "<span></span>"}<button class="skip" data-skip>Skip</button></div>
      </div>`;
  }
  function bind(root, k) {
    root.querySelectorAll(".opt").forEach((b) => (b.onclick = () => pick(k, Number(b.dataset.v), b)));
    root.querySelector("[data-skip]").onclick = () => pick(k, null);
    const back = root.querySelector("[data-back]");
    if (back) back.onclick = () => show(k - 1);
  }
  function openInvite() {
    setProgress(0);
    $("invite-q").innerHTML = questionHtml(0, true);
    bind($("invite-q"), 0);
    $("invite").hidden = false;
  }
  function show(k) {
    setProgress(k);
    $("invite").hidden = true; $("host").hidden = true; $("game").hidden = false;
    $("game").innerHTML = questionHtml(k, false);
    bind($("game"), k);
    window.scrollTo({ top: 0 });
  }
  function pick(k, v, el) {
    if (busy) return;
    answers[items[k].code] = v;
    if (el) { el.parentElement.querySelectorAll(".opt").forEach((b) => b.classList.remove("on")); el.classList.add("on"); }
    busy = true;
    setTimeout(() => { busy = false; k + 1 < items.length ? show(k + 1) : submit(); }, reduced ? 0 : 200);
  }
  document.addEventListener("keydown", (e) => {          // 1-5 pick answers on a laptop
    if (e.metaKey || e.ctrlKey || e.altKey || !/^[1-9]$/.test(e.key)) return;
    const root = !$("invite").hidden ? $("invite-q") : !$("game").hidden ? $("game") : null;
    const opt = root && root.querySelectorAll(".opt")[Number(e.key) - 1];
    if (opt) opt.click();
  });

  async function submit() {
    $("bar").style.width = "100%"; $("count").textContent = "";
    $("game").innerHTML = `<div class="decode"><div class="scanner"></div><p>Looking for you in two AI crowds…</p></div>`;
    try {
      const req = fetch("/api/room/answer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ answers }) })
        .then(async (r) => { const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Something went wrong"); return d; });
      const [d] = await Promise.all([req, new Promise((r) => setTimeout(r, reduced ? 0 : 1600))]);
      answers = {}; result = d; verdict(d);
    } catch (e) {
      $("game").innerHTML = `<div class="decode"><h2 class="qtext">That didn't go through</h2><p id="err"></p>
        <div class="actions"><button class="btn line" id="b">Back</button><button class="btn line" id="t">Try again</button></div></div>`;
      $("err").textContent = e.message; $("b").onclick = () => show(items.length - 1); $("t").onclick = submit;
    }
  }

  function outcome(d) {
    const c = d.verdicts.claude.covered, f = d.verdicts.flashpoint.covered;
    if (!c && !f) return { cls: "unreachable", word: "Unreachable.", sub: "Neither AI could imagine you." };
    if (!c || !f) return { cls: "half", word: "Half‑reachable.", sub: `${f ? "Flashpoint" : "Claude"}'s AI found you. ${f ? "Claude" : "Flashpoint"}'s didn't.` };
    return { cls: "imaginable", word: "Imaginable.", sub: "Both AIs found someone like you. You're the person AI already gets." };
  }
  function verdict(d) {
    const o = outcome(d), w = d.why, v = d.verdicts;
    const why = w.differs ? `You said <b>“${esc(w.yours)}”</b> to ${esc(w.topic)}. ${w.real_share < 0.15 ? "Only " : ""}<span class="pct">${pct(w.real_share)}</span> of ${d.real_n} real survey-takers did${w.real_share < 0.15 ? "" : " too"}.` : "";
    const pill = (name, r) => `<span class="chip ${r.covered ? "cov" : "miss"}">${r.covered ? "✓" : "✕"} ${name}'s AI ${r.covered ? "found you" : "didn't find you"}</span>`;
    $("count").textContent = "Verdict";
    $("game").innerHTML = `
      <div class="verdict">
        <h1 class="v-word ${o.cls}">${o.word}</h1>
        <p class="v-sub">${esc(o.sub)}</p>
        <div class="chips">
          ${pill("Claude", v.claude)}
          ${pill("Flashpoint", v.flashpoint)}
        </div>
        ${why ? `<p class="why">${why}</p>` : ""}
        <p class="stat">On these ${items.length} questions, Claude's AI misses <b>${pct(v.claude.miss_rate)}</b> of real people. Flashpoint's misses <b>${pct(v.flashpoint.miss_rate)}</b>.</p>
        <div class="actions">
          <button class="btn primary" id="share">Send to a friend</button>
          <button class="btn line" id="save">Save image</button>
        </div>
        <details class="how"><summary>How it works</summary>
          <p>Two AIs each made a crowd of pretend Americans: Claude made ${d.crowd_n.claude}, Flashpoint made ${d.crowd_n.flashpoint}. You count as "found" if someone in a crowd answers as much like you as real people like you do (your 5th closest of ${d.real_n} real people from the 2024 General Social Survey). On these ${items.length} questions Claude's crowd misses ${pct(v.claude.miss_rate)} of real people and Flashpoint's misses ${pct(v.flashpoint.miss_rate)}. Questions are shortened from the survey, so these numbers are separate from Blindspot's full 20-question results. Nothing you answered was saved.</p>
        </details>
      </div>`;
    $("share").onclick = share; $("save").onclick = saveImage;
    if (o.cls !== "imaginable") confetti(o.cls === "unreachable" ? ["#ff6b4a", "#ffc24b", "#f4f0e8"] : ["#ffc24b", "#3ee0c6", "#f4f0e8"]);
  }

  function shareText() {
    const o = outcome(result);
    return `${o.word.replace("‑", "-")} ${o.sub} Can AI imagine you?`;
  }
  function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200); }
  async function share() {
    const text = shareText(), url = `${location.origin}/you`;
    try {
      if (navigator.share) { await navigator.share({ title: "Can AI imagine you?", text, url }); return; }
      await navigator.clipboard.writeText(`${text} ${url}`); toast("Copied. Paste it anywhere.");
    } catch (_) { /* cancelled */ }
  }
  async function saveImage() {
    await document.fonts.ready;
    const W = 1080, H = 1350, P = 90, c = document.createElement("canvas"); c.width = W; c.height = H;
    const x = c.getContext("2d"), o = outcome(result), v = result.verdicts;
    const disp = "Fraunces, Georgia, serif", sans = "'Instrument Sans', system-ui, sans-serif";
    const color = { unreachable: "#ff6b4a", half: "#ffc24b", imaginable: "#3ee0c6" }[o.cls];
    let g = x.createLinearGradient(0, 0, 0, H); g.addColorStop(0, "#121a36"); g.addColorStop(1, "#060914"); x.fillStyle = g; x.fillRect(0, 0, W, H);
    g = x.createRadialGradient(W, 0, 0, W, 0, 900); g.addColorStop(0, color + "55"); g.addColorStop(1, color + "00"); x.fillStyle = g; x.fillRect(0, 0, W, H);
    x.fillStyle = "#ffc24b"; x.font = `700 32px ${sans}`; x.fillText("CAN AI IMAGINE YOU?", P, 200);
    x.fillStyle = color; x.font = `600 ${o.cls === "half" ? 118 : 150}px ${disp}`; x.fillText(o.word, P, 420);
    x.fillStyle = "#f4f0e8"; x.font = `400 48px ${sans}`; x.fillText(o.sub, P, 520);
    let cx = P;
    for (const [name, r] of [["Claude", v.claude], ["Flashpoint", v.flashpoint]]) {
      const col = r.covered ? "#3ee0c6" : "#ff6b4a", label = `${r.covered ? "✓" : "✕"} ${name}`;
      x.font = `700 44px ${sans}`; const w = x.measureText(label).width + 72;
      x.fillStyle = col + "22"; x.strokeStyle = col; x.lineWidth = 3; x.beginPath(); x.roundRect(cx, 600, w, 96, 48); x.fill(); x.stroke();
      x.fillStyle = col; x.fillText(label, cx + 36, 664); cx += w + 24;
    }
    x.fillStyle = "#8a92aa"; x.font = `400 28px ${sans}`;
    x.fillText("Blindspot · Hack the Unreachable", P, H - 110); x.fillText(`${location.host}/you`, P, H - 70);
    const a = document.createElement("a"); a.download = "can-ai-imagine-me.png"; a.href = c.toDataURL("image/png"); a.click();
    toast("Image saved.");
  }
  function confetti(colors) {
    if (reduced) return;
    const cv = $("confetti"), ctx = cv.getContext("2d"), dpr = devicePixelRatio || 1;
    cv.width = innerWidth * dpr; cv.height = innerHeight * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bits = Array.from({ length: 160 }, () => ({ x: innerWidth * (.3 + Math.random() * .4), y: innerHeight * .3, vx: (Math.random() - .5) * 16,
      vy: -Math.random() * 14 - 4, r: Math.random() * 3, vr: (Math.random() - .5) * .3, w: 7 + Math.random() * 7, h: 9 + Math.random() * 9,
      c: colors[Math.floor(Math.random() * colors.length)] }));
    let t = 0;
    (function frame() {
      ctx.clearRect(0, 0, innerWidth, innerHeight);
      for (const b of bits) { b.vy += .35; b.x += b.vx; b.y += b.vy; b.r += b.vr; ctx.save(); ctx.translate(b.x, b.y); ctx.rotate(b.r);
        ctx.globalAlpha = Math.max(0, 1 - t / 170); ctx.fillStyle = b.c; ctx.fillRect(-b.w / 2, -b.h / 2, b.w, b.h); ctx.restore(); }
      if (++t < 180) requestAnimationFrame(frame); else ctx.clearRect(0, 0, innerWidth, innerHeight);
    })();
  }

  fetch("/api/room/questions", { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject()))
    .then((d) => { meta = d; items = d.items; $("n-q").textContent = items.length; setTimeout(openInvite, reduced ? 0 : 300); })
    .catch(() => { document.querySelector(".host-title").textContent = "Live mode is offline. Reload in a moment."; });
})();
