// /audit page: upload two CSVs, check the column matching, read the report. Files are read in the browser and posted as
// text; the server keeps nothing. All user-supplied text goes through esc() before it touches innerHTML.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = BS.esc;
  const pct = (v) => `${Math.round(v * 100)}%`;
  const MAX_BYTES = 5 * 1024 * 1024;
  const state = { panel: null, real: null, preview: null, body: null, map: null };

  function showError(msg) {
    const box = $("error");
    box.textContent = msg || "";
    box.hidden = !msg;
    if (msg) box.scrollIntoView({ block: "nearest" });
  }

  function showStep(n) {
    for (const id of ["s1", "s2", "busy", "s3"]) $(id).hidden = true;
    $(n === 1 ? "s1" : n === 2 ? "s2" : n === 3 ? "s3" : "busy").hidden = false;
    document.querySelectorAll("#steps li").forEach((li) => {
      const s = Number(li.dataset.s), cur = n === "busy" ? 2 : n;
      li.classList.toggle("on", s === cur);
      li.classList.toggle("done", s < cur);
    });
    window.scrollTo({ top: 0 });
  }

  async function api(path, body) {
    let r;
    try {
      r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (_) {
      throw new Error("Could not reach the server. Check your connection and try again.");
    }
    if (!r.ok) {
      let detail = null;
      try { detail = (await r.json()).detail; } catch (_) { /* not JSON */ }
      throw new Error(typeof detail === "string" ? detail : "The files could not be read (5 MB per file, 5,000 rows, 100 columns at most).");
    }
    return r;
  }

  // ---- step 1: files
  function setFile(slot, name, text) {
    state[slot] = { name, text };
    const drop = $(slot === "panel" ? "dropPanel" : "dropReal");
    drop.classList.add("ok");
    $(slot === "panel" ? "namePanel" : "nameReal").textContent = `${name} (${text.split("\n").length - 1} rows)`;
    $("toMatch").disabled = !(state.panel && state.real);
  }

  async function readFile(file, slot) {
    showError("");
    if (!file) return;
    if (!/\.(csv|tsv|txt)$/i.test(file.name) && !/csv|text/.test(file.type)) {
      return showError(`“${file.name}” is not a CSV file. Export your data as CSV and try again.`);
    }
    if (file.size > MAX_BYTES) return showError(`“${file.name}” is larger than 5 MB. Please upload at most 5,000 rows.`);
    try {
      setFile(slot, file.name, await file.text());
    } catch (_) {
      showError(`Could not read “${file.name}”.`);
    }
  }

  function wireDrop(slot, dropId, inputId) {
    const drop = $(dropId), input = $(inputId);
    input.addEventListener("change", () => { readFile(input.files[0], slot); });
    for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
    for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); readFile(e.dataTransfer.files[0], slot); });
  }

  // ---- step 2: matching
  async function goMatch() {
    showError("");
    state.body = { panel: state.panel.text, real: state.real.text };
    showStep("busy");
    $("busyText").textContent = "Reading your files…";
    try {
      state.preview = await (await api("/api/audit/preview", state.body)).json();
    } catch (err) {
      showStep(1);
      return showError(err.message);
    }
    renderMatch(state.preview);
    showStep(2);
  }

  function renderMatch(p) {
    const on = p.columns.filter((c) => c.on).length;
    $("matchLede").textContent = `${p.columns.length} columns match between the two files; ${on} will be compared. Untick any question you want to leave out.`;
    const notes = [`<p>AI panel: ${p.n_panel.toLocaleString()} respondents. Real sample: ${p.n_real.toLocaleString()} people.</p>`];
    const auto = p.columns.filter((c) => !c.on);
    if (auto.length) {
      notes.push(`<p>Ignored automatically: ${auto.map((c) => `<b>${esc(c.label)}</b> (${esc(c.reason || "not comparable")})`).join(", ")}.</p>`);
    }
    if (p.unmatched_real.length) notes.push(`<p>Only in your real sample, not compared: ${p.unmatched_real.map(esc).join(", ")}.</p>`);
    if (p.unmatched_panel.length) notes.push(`<p>Only in your AI panel, not compared: ${p.unmatched_panel.map(esc).join(", ")}.</p>`);
    const e = p.expected_error;
    if (!e.reliable) {
      notes.push(`<p><b>Small real sample.</b> ${p.n_real} people is below the reliable minimum of ${e.min_reliable_n}. The report still works, but its numbers are a rough guide (typically within ±${Math.round(100 * e.rmse)} points).</p>`);
    }
    $("matchNotes").innerHTML = notes.join("");
    const rows = p.columns.map((c, i) => {
      const opts = c.options.length ? `${c.options.map(esc).join(" · ")}${c.more_options ? ` · +${c.more_options} more` : ""}` : "";
      const how = c.method === "exact" ? "" : ` <span class="tag">${esc(c.method)} match</span>`;
      return `<tr class="${c.on ? "" : "off"}"><td><input type="checkbox" data-i="${i}" aria-label="Use ${esc(c.label)}" ${c.on ? "checked" : ""} ${c.usable ? "" : "disabled"}></td>` +
        `<td><b class="qlabel">${esc(c.label)}</b>${how}${c.label !== c.name ? `<div class="opts full">${esc(c.name)}</div>` : ""}${c.panel_header !== c.name ? `<div class="opts">AI file: ${esc(c.panel_header)}</div>` : ""}</td>` +
        `<td>${c.kind ? `${esc(c.kind)}, ${c.n_options} options` : "—"}</td><td class="opts">${opts || esc(c.reason || "")}</td></tr>`;
    });
    $("matchTable").innerHTML = `<tr><th></th><th>Question</th><th>Type</th><th>Answer options found</th></tr>${rows.join("")}`;
  }

  // ---- step 3: report
  async function runAudit() {
    showError("");
    const use = [...document.querySelectorAll("#matchTable input:checked")].map((b) => state.preview.columns[Number(b.dataset.i)].name);
    if (!use.length) return showError("Tick at least one question to compare.");
    state.body = { panel: state.panel.text, real: state.real.text, use };
    showStep("busy");
    $("busyText").textContent = "Comparing your panel with real people…";
    let report;
    try {
      report = await (await api("/api/audit/run", state.body)).json();
    } catch (err) {
      showStep(2);
      return showError(err.message);
    }
    renderReport(report);
    showStep(3);
    drawMap(report.map);
  }

  function drawMap(m) {
    if (!state.map) state.map = new Constellation($("repMap"), { pad: 18 });
    state.map.resize();
    state.map.set(m);
    state.map.play();
  }

  function renderReport(r) {
    const s = r.summary, e = s.expected_error;
    $("repHead").innerHTML = esc(r.recommendation.headline).replace(/misses about (\d+%)/, "misses about <em>$1</em>");
    BS.countTo($("repMissed"), s.missed, (v) => pct(v));
    BS.countTo($("repReal"), s.realistic, (v) => pct(v));
    $("repMissedL").textContent = e.rmse != null ? `of real people missed, typically within ±${Math.round(100 * e.rmse)} points` : "of real people missed";
    $("repRealL").textContent = `of AI respondents look like a real person${s.realistic_error != null ? ` (±${Math.round(100 * s.realistic_error)} points; reads low with small samples)` : ""}`;
    const warn = $("repWarn");
    warn.hidden = e.reliable;
    warn.textContent = e.reliable ? "" : `Small sample: your real sample has ${e.n} people. ${e.min_reliable_n} or more give a reliable number, so treat this report as a rough guide.`;
    $("repRec").innerHTML = `<b>What to do</b>${r.recommendation.paragraphs.map((p) => `<p>${esc(p)}</p>`).join("")}${r.notes.map((n) => `<p class="fine">${esc(n)}</p>`).join("")}`;

    const b = r.bunching;
    $("repBunch").innerHTML = (b.questions.length
      ? b.questions.map((q) => `<div class="gap"><div>${esc(q.question)}<small>${pct(q.ai_top_share)} of the AI panel answer “${esc(q.ai_top_answer)}”</small></div><div class="gl">real sample: ${pct(q.real_top_share)}</div></div>`).join("")
      : `<p class="fine">No question has more than ${Math.round(100 * b.threshold)}% of the AI panel giving the same answer.</p>`) +
      `<p class="fine">${b.n_ai_bunched} of ${b.n_questions} questions bunch in the AI panel, ${b.n_real_bunched} in your real sample.</p>`;

    const track = (v, colour) => `<div class="track"><i style="width:${Math.round(100 * v)}%;background:${colour}"></i></div>`;
    $("repGaps").innerHTML = r.gaps.map((g) => `<div class="gap"><div>${esc(g.question)}<small>biggest difference: “${esc(g.answer)}”</small></div>` +
      `<div>${track(g.ai_share, "var(--syn)")}<span class="gl">AI ${pct(g.ai_share)}</span>${track(g.real_share, "#f4f0e8")}<span class="gl">real ${pct(g.real_share)}</span></div></div>`).join("");

    $("repEx").innerHTML = r.examples.length ? r.examples.map((x) => `<div class="person"><b>${esc(x.label)}</b><div class="fine">differs from the closest AI person on ${x.n_diff} of ${x.n_questions} questions</div>` +
      `<table><tr><th>Question</th><th>Real</th><th>Closest AI</th></tr>${x.rows.map((row) => `<tr><td>${esc(row.q)}</td><td class="a">${esc(row.real || "(blank)")}</td><td class="b">${esc(row.ai || "(blank)")}</td></tr>`).join("")}</table></div>`).join("")
      : `<p class="fine">No missed real people to show.</p>`;

    $("repairedNote").textContent = r.repaired_note;
    $("repHow").textContent = r.how_to_read;
    const c = r.columns;
    $("repCols").textContent = `Compared ${s.n_real} real people with ${s.n_ai.toLocaleString()} AI respondents on ${s.n_questions} questions.` +
      (c.ignored.length ? ` Ignored: ${c.ignored.map((x) => `${x.label} (${x.reason})`).join(", ")}.` : "") +
      (c.fuzzy.length ? ` Matched despite different spelling: ${c.fuzzy.map((x) => `${x.panel} = ${x.real}`).join(", ")}.` : "");
  }

  async function download(path, filename, button) {
    showError("");
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Preparing…";
    try {
      const blob = await (await api(path, state.body)).blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    } catch (err) {
      showError(err.message);
    } finally {
      button.disabled = false;
      button.textContent = label;
    }
  }

  async function trySample() {
    showError("");
    try {
      const r = await fetch("/api/audit/sample");
      if (!r.ok) throw new Error();
      const s = await r.json();
      setFile("panel", s.panel_name, s.panel);
      setFile("real", s.real_name, s.real);
    } catch (_) {
      return showError("Could not load the sample data.");
    }
    goMatch();
  }

  function reset() {
    state.panel = state.real = state.preview = state.body = null;
    for (const id of ["filePanel", "fileReal"]) $(id).value = "";
    for (const [d, n] of [["dropPanel", "namePanel"], ["dropReal", "nameReal"]]) {
      $(d).classList.remove("ok");
      $(n).textContent = "Drop a file or click to choose";
    }
    $("toMatch").disabled = true;
    showError("");
    showStep(1);
  }

  wireDrop("panel", "dropPanel", "filePanel");
  wireDrop("real", "dropReal", "fileReal");
  $("toMatch").addEventListener("click", goMatch);
  $("trySample").addEventListener("click", trySample);
  $("backTo1").addEventListener("click", () => { showError(""); showStep(1); });
  $("runAudit").addEventListener("click", runAudit);
  $("dlReport").addEventListener("click", (ev) => download("/api/audit/report.html", "blindspot-audit-report.html", ev.currentTarget));
  $("dlRepaired").addEventListener("click", (ev) => download("/api/audit/repaired.csv", "blindspot-repaired-crowd.csv", ev.currentTarget));
  $("again").addEventListener("click", reset);
  showStep(1);
})();
