(async () => {
  const $ = (id) => document.getElementById(id);
  const screen = $("screen");
  let items = [], answers = {}, i = -1, crowdLabel = "the AI crowd";

  try {
    const r = await fetch("/api/live/questions", { cache: "no-store" });
    if (!r.ok) throw new Error();
    const d = await r.json();
    items = d.items; crowdLabel = d.crowd_label || crowdLabel;
  } catch (_) {
    screen.innerHTML = `<div class="result"><h1>Live mode is offline</h1><p>Ask the presenter to check the connection, then reload this page.</p></div>`;
    return;
  }

  const pct = (v) => `${Math.round(v * 100)}%`;
  const setBar = () => {
    $("bar").style.width = `${Math.max(0, i) / items.length * 100}%`;
    $("count").textContent = i >= 0 && i < items.length ? `${i + 1} / ${items.length}` : "";
  };

  function intro() {
    i = -1; setBar();
    screen.innerHTML = `
      <div class="q">Can the AI imagine someone like you?</div>
      <p class="consent" style="margin-top:14px">${items.length} quick questions from a national survey, about a minute. Your answers are compared with an AI-generated crowd and then discarded. Anonymous. Nothing is stored.</p>
      <div class="navrow"><span></span><button class="btn primary" id="start">Start</button></div>`;
    $("start").onclick = () => show(0);
  }

  function show(k) {
    i = k; setBar();
    const it = items[i];
    screen.innerHTML = `<div class="q"></div><div class="opts" id="opts"></div>
      <div class="navrow"><button class="btn ghost" id="back">Back</button><button class="btn ghost" id="skip">Skip</button></div>`;
    screen.querySelector(".q").textContent = it.wording;
    it.options.forEach((o, j) => {
      const b = document.createElement("button");
      b.className = "opt" + (answers[it.code] === j ? " on" : "");
      b.textContent = o;
      b.onclick = () => { answers[it.code] = j; next(); };
      $("opts").appendChild(b);
    });
    $("back").onclick = () => (i === 0 ? intro() : show(i - 1));
    $("skip").onclick = () => { answers[it.code] = null; next(); };
  }

  const next = () => (i + 1 < items.length ? show(i + 1) : submit());

  async function submit() {
    i = items.length; setBar();
    screen.innerHTML = `<div class="result"><p>Looking for someone like you…</p></div>`;
    try {
      const r = await fetch("/api/live/answer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Something went wrong");
      answers = {};   // forget locally too
      const base = d.base_rate_uncovered != null
        ? `For context: ${pct(d.base_rate_uncovered)} of real people in the survey are also missed by this crowd.` : "";
      const stat = d.base_rate_uncovered != null
        ? `<div class="stat3 num">${pct(d.base_rate_uncovered)}</div><p>of real people in the survey are missed by this AI crowd too</p>` : "";
      screen.innerHTML = d.covered
        ? `<div class="result"><div class="icon cov" aria-hidden="true">✓</div><h1>Someone in the crowd is like you</h1><p>The AI crowd can imagine a person who answers the way you do.</p>${stat}</div>`
        : `<div class="result"><div class="icon miss" aria-hidden="true">!</div><h1>The AI couldn’t imagine you.</h1><p>No one in the AI crowd answers the way you do. You are exactly the kind of person Blindspot goes looking for.</p>${stat}</div>`;
      screen.innerHTML += `<div class="navrow"><span></span><button class="btn ghost" id="again">Done</button></div>`;
      $("again").onclick = intro;
    } catch (e) {
      screen.innerHTML = `<div class="result"><h1>That didn't go through</h1><p class="err"></p><div class="navrow"><button class="btn ghost" id="b">Back</button><button class="btn primary" id="t">Try again</button></div></div>`;
      screen.querySelector(".err").textContent = e.message;
      $("b").onclick = () => show(items.length - 1);
      $("t").onclick = submit;
    }
  }

  intro();
})();
