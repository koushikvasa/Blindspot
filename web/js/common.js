// Shared helpers: data loading (API first, static sample files as fallback), number formatting, tiny DOM helpers.
window.BS = (() => {
  async function load(name) {
    try {
      const r = await fetch(`/api/data/${name}`, { cache: "no-store" });
      if (r.ok) return await r.json();
    } catch (_) { /* fall through */ }
    const r = await fetch(`/static/mock/${name}.json`);
    if (!r.ok) throw new Error(`missing ${name}`);
    return { mock: true, data: await r.json() };
  }

  const pct = (v, d) => {
    if (v == null) return "–";
    if (d == null) d = v > 0.99 && v < 1 ? 1 : 0;   // 0.997 shows as 99.7%, not 100%
    return `${(v * 100).toFixed(d)}%`;
  };
  const esc = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  function countTo(el, target, fmt = pct, ms = 700) {
    if (!el) return;
    const start = parseFloat(el.dataset.v ?? "0") || 0;
    el.dataset.v = target;
    if (reduced() || start === target) { el.textContent = fmt(target); return; }
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(start + (target - start) * e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // Any panel that fell back to sample data raises one visible badge (never shown as real data).
  function markSample(isMock) {
    if (!isMock) return;
    const slot = document.getElementById("sampleBadge");
    if (slot) { slot.textContent = "Sample data in use"; return; }
    if (document.querySelector(".sample-flag")) return;
    const b = document.createElement("span");
    b.className = "badge sample sample-flag";
    b.title = "Some panels are using sample data until real results are exported to results/ui/";
    b.textContent = "Sample data in use";
    document.body.appendChild(b);
  }

  return { load, pct, esc, reduced, countTo, markSample };
})();
