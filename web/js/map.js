// Constellation renderer: real people (white), synthetic crowd (teal glow), uncovered real people (coral + ring),
// live visitors (gold pulse). Real positions are fixed per dataset; colour changes animate between crowds.
class Constellation {
  constructor(canvas, opts = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.pad = opts.pad ?? 28;
    this.real = [];        // [{x,y,cov,covFrom,covTo}]
    this.syn = [];         // [{x,y}]
    this.synPrev = [];
    this.visitors = [];    // [{x,y,covered,t}]
    this.mix = 1;          // 0..1 transition progress
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  resize() {
    const r = this.c.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = r.width; this.h = r.height;
    this.c.width = Math.max(1, r.width * dpr); this.c.height = Math.max(1, r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // data: {real: [[x,y,cov]], synthetic: [[x,y]]}
  set(data) {
    const prevCov = this.real.map((p) => p.cov);
    const sameLayout = this.real.length === data.real.length;
    this.real = data.real.map(([x, y, cov], i) => ({
      x, y, covTo: cov, covFrom: sameLayout ? prevCov[i] : cov, cov,
    }));
    this.synPrev = this.syn;
    this.syn = (data.synthetic || []).map(([x, y]) => ({ x, y }));
    this.mix = this.reduced ? 1 : 0;
    this.t0 = performance.now();
  }

  addVisitor(v) { this.visitors.push({ ...v, t: performance.now() }); }

  px(x) { return this.pad + x * (this.w - 2 * this.pad); }
  py(y) { return this.pad + (1 - y) * (this.h - 2 * this.pad); }

  _loop(t) {
    if (this.mix < 1) this.mix = Math.min(1, (t - this.t0) / 650);
    this.draw(t);
    requestAnimationFrame(this._loop);
  }

  draw(t) {
    const ctx = this.ctx, e = 1 - Math.pow(1 - this.mix, 3);
    ctx.clearRect(0, 0, this.w, this.h);

    // synthetic glow: previous crowd fades out, new crowd fades in
    const glow = (pts, a) => {
      if (a <= 0) return;
      ctx.fillStyle = `rgba(45,212,191,${0.10 * a})`;
      for (const p of pts) { ctx.beginPath(); ctx.arc(this.px(p.x), this.py(p.y), 7, 0, 6.283); ctx.fill(); }
      ctx.fillStyle = `rgba(45,212,191,${0.75 * a})`;
      for (const p of pts) { ctx.beginPath(); ctx.arc(this.px(p.x), this.py(p.y), 1.6, 0, 6.283); ctx.fill(); }
    };
    glow(this.synPrev, 1 - e);
    glow(this.syn, e);

    // real people
    for (const p of this.real) {
      const cov = p.covFrom + (p.covTo - p.covFrom) * e;   // 1 = covered, 0 = missed
      const x = this.px(p.x), y = this.py(p.y);
      const miss = 1 - cov;
      if (miss > 0.01) {
        ctx.strokeStyle = `rgba(255,122,89,${0.55 * miss})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(x, y, 5.5, 0, 6.283); ctx.stroke();
      }
      const r = Math.round(232 + (255 - 232) * miss), g = Math.round(236 + (122 - 236) * miss), b = Math.round(245 + (89 - 245) * miss);
      ctx.fillStyle = `rgba(${r},${g},${b},${0.55 + 0.4 * miss})`;
      ctx.beginPath(); ctx.arc(x, y, 2.3, 0, 6.283); ctx.fill();
    }

    // live visitors
    const now = performance.now();
    for (const v of this.visitors) {
      const age = (now - v.t) / 1000, x = this.px(v.x), y = this.py(v.y);
      if (age < 6 && !this.reduced) {
        const k = (age % 1.2) / 1.2;
        ctx.strokeStyle = `rgba(251,191,36,${0.8 * (1 - k)})`;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 6 + 26 * k, 0, 6.283); ctx.stroke();
      }
      ctx.fillStyle = "#FBBF24";
      ctx.beginPath(); ctx.arc(x, y, age < 6 ? 6 : 3.5, 0, 6.283); ctx.fill();
    }
  }

  coverage() {
    if (!this.real.length) return null;
    return this.real.reduce((s, p) => s + p.covTo, 0) / this.real.length;
  }
}
window.Constellation = Constellation;
