// Constellation renderer: real people (paper white), AI crowd (teal glow), real people nobody in the crowd answers like
// (coral dot + ring, so colour is never the only cue), live visitors (gold pulse).
// Real positions are fixed per dataset; switching crowds only recolours and moves the glow.
// play() runs the staged reveal: 1 real people appear, 2 the AI crowd fades in, 3 the missed people turn coral.
class Constellation {
  constructor(canvas, opts = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.pad = opts.pad ?? 28;
    this.padT = opts.padT ?? this.pad;      // extra room for overlay chips inside the map card
    this.padB = opts.padB ?? this.pad;
    this.real = [];        // [{x,y,covFrom,covTo,d}]
    this.syn = [];         // [{x,y}]
    this.synPrev = [];
    this.visitors = [];    // [{x,y,covered,t}]
    this.mix = 1;          // 0..1 crowd-switch progress
    this.rev = { real: 1, syn: 1, miss: 1 };   // staged reveal progress (all 1 = fully shown)
    this.onStage = opts.onStage || null;
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
    const prevCov = this.real.map((p) => p.covTo);
    const sameLayout = this.real.length === data.real.length;
    this.real = data.real.map(([x, y, cov], i) => ({
      x, y, covTo: cov, covFrom: sameLayout ? prevCov[i] : cov, d: ((i * 2654435761) % 1000) / 1000,
    }));
    this.synPrev = this.syn;
    this.syn = (data.synthetic || []).map(([x, y]) => ({ x, y }));
    this.mix = this.reduced ? 1 : 0;
    this.t0 = performance.now();
  }

  // Staged reveal (only the first time the map is seen, and again on restart).
  play() {
    this.stageT = null;
    if (this.reduced) { this.rev = { real: 1, syn: 1, miss: 1 }; this._stage(3); return; }
    this.rev = { real: 0, syn: 0, miss: 0 };
    this.stageT = performance.now();
    this._stageSent = 0;
    this._stage(1);
  }
  _stage(s) { if (this._stageSent !== s) { this._stageSent = s; if (this.onStage) this.onStage(s); } }

  addVisitor(v) { this.visitors.push({ ...v, t: performance.now() }); }
  px(x) { return this.pad + x * (this.w - 2 * this.pad); }
  py(y) { return this.padT + (1 - y) * (this.h - this.padT - this.padB); }

  _loop(t) {
    if (this.mix < 1) this.mix = Math.min(1, (t - this.t0) / 650);
    if (this.stageT != null) {
      const s = (t - this.stageT) / 1000;                       // seconds since the reveal began
      this.rev.real = Math.min(1, s / 1.4);
      this.rev.syn = Math.max(0, Math.min(1, (s - 1.7) / 1.0));
      this.rev.miss = Math.max(0, Math.min(1, (s - 3.3) / 0.9));
      if (s > 1.7) this._stage(2);
      if (s > 3.3) this._stage(3);
      if (s > 4.4) { this.rev = { real: 1, syn: 1, miss: 1 }; this.stageT = null; }
    }
    this.draw(t);
    requestAnimationFrame(this._loop);
  }

  draw(t) {
    const ctx = this.ctx, e = 1 - Math.pow(1 - this.mix, 3);
    ctx.clearRect(0, 0, this.w, this.h);
    const u = Math.max(1.7, Math.min(3.4, Math.min(this.w, this.h) / 190));   // base dot radius adapts to the canvas

    // AI crowd glow: previous crowd fades out, new crowd fades in
    const glow = (pts, a) => {
      a *= this.rev.syn;
      if (a <= 0.01) return;
      ctx.fillStyle = `rgba(62,224,198,${0.12 * a})`;
      for (const p of pts) { ctx.beginPath(); ctx.arc(this.px(p.x), this.py(p.y), u * 3.1, 0, 6.283); ctx.fill(); }
      ctx.fillStyle = `rgba(62,224,198,${0.85 * a})`;
      for (const p of pts) { ctx.beginPath(); ctx.arc(this.px(p.x), this.py(p.y), u * 0.78, 0, 6.283); ctx.fill(); }
    };
    glow(this.synPrev, 1 - e);
    glow(this.syn, e);

    // real people
    for (const p of this.real) {
      const alpha = Math.max(0, Math.min(1, (this.rev.real - p.d * 0.7) / 0.3));
      if (alpha <= 0) continue;
      const cov = p.covFrom + (p.covTo - p.covFrom) * e;          // 1 = covered, 0 = missed
      const miss = (1 - cov) * this.rev.miss;
      const x = this.px(p.x), y = this.py(p.y);
      if (miss > 0.01) {
        ctx.strokeStyle = `rgba(255,107,74,${0.85 * miss * alpha})`;
        ctx.lineWidth = Math.max(1.1, u * 0.5);
        ctx.beginPath(); ctx.arc(x, y, u * 2.5, 0, 6.283); ctx.stroke();
      }
      const r = Math.round(244 + (255 - 244) * miss), g = Math.round(240 + (107 - 240) * miss), b = Math.round(232 + (74 - 232) * miss);
      ctx.fillStyle = `rgba(${r},${g},${b},${(0.6 + 0.38 * miss) * alpha})`;
      ctx.beginPath(); ctx.arc(x, y, u * (1 + 0.25 * miss), 0, 6.283); ctx.fill();
    }

    // live visitors
    const now = performance.now();
    for (const v of this.visitors) {
      const age = (now - v.t) / 1000, x = this.px(v.x), y = this.py(v.y);
      if (age < 6 && !this.reduced) {
        const k = (age % 1.2) / 1.2;
        ctx.strokeStyle = `rgba(255,194,75,${0.8 * (1 - k)})`;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 6 + 26 * k, 0, 6.283); ctx.stroke();
      }
      ctx.fillStyle = "#FFC24B";
      ctx.beginPath(); ctx.arc(x, y, age < 6 ? 6 : 3.5, 0, 6.283); ctx.fill();
    }
  }

  coverage() {
    if (!this.real.length) return null;
    return this.real.reduce((s, p) => s + p.covTo, 0) / this.real.length;
  }
}
window.Constellation = Constellation;
