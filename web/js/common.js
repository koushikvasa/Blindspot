// Shared helpers: data loading (API first, static sample files as fallback) and number formatting.
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

  function countTo(el, target, fmt = pct, ms = 600) {
    const start = parseFloat(el.dataset.v || target);
    el.dataset.v = target;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || start === target) { el.textContent = fmt(target); return; }
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(start + (target - start) * e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  let sampleShown = false;
  function markSample(isMock) {
    if (!isMock || sampleShown) return;
    sampleShown = true;
    const b = document.createElement("span");
    b.className = "badge sample sample-flag";
    b.title = "Some panels are using sample data until real results are exported to results/ui/";
    b.textContent = "Sample data in use";
    document.body.appendChild(b);
  }

  return { load, pct, countTo, markSample };
})();
