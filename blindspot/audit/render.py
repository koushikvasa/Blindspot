"""Self-contained HTML report (one file: inline CSS and SVG, no scripts, no network) built from `build_report` output."""
from html import escape as e

STYLE = """
:root{--bg:#060914;--panel:#0c1328;--line:rgba(244,240,232,.14);--text:#f4f0e8;--muted:#b0b7ca;--syn:#3ee0c6;--miss:#ff6b4a;--gold:#ffc24b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 "Instrument Sans","Segoe UI",system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:40px 20px 80px}h1,h2{font-family:Georgia,"Iowan Old Style",serif;font-weight:500;line-height:1.15}
h1{font-size:2.3rem;margin:.2em 0}h2{font-size:1.4rem;margin:2em 0 .5em}.kicker{color:var(--gold);letter-spacing:.14em;text-transform:uppercase;font-size:.8rem}
.bigs{display:flex;gap:32px;flex-wrap:wrap;margin:18px 0}.big .n{font-family:Georgia,serif;font-size:3.6rem;line-height:1}.big .l{color:var(--muted)}
.miss{color:var(--miss)}.syn{color:var(--syn)}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:14px 0}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);font-size:.95rem}th{color:var(--muted);font-weight:500}
.bar{height:10px;border-radius:5px;background:rgba(255,255,255,.08);position:relative;min-width:120px}.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px}
.fine{color:var(--muted);font-size:.88rem}.warn{border-color:rgba(255,194,75,.6)}.rec p{margin:.5em 0}svg{max-width:100%;height:auto}
.ex{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.ex td.a{color:var(--miss)}.ex td.b{color:var(--syn)}
"""


def pct(v: float) -> str:
    return f"{round(100 * v)}%"


def map_svg(m: dict, size: int = 460) -> str:
    pad = 14

    def f(v: float) -> float:
        return pad + v * (size - 2 * pad)

    parts = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Map of real people (white), AI panel (teal) and missed real people (coral rings)">']
    for x, y in m["synthetic"]:
        parts.append(f'<circle cx="{f(x):.1f}" cy="{f(1 - y):.1f}" r="2.4" fill="#3ee0c6" opacity=".75"/>')
    for x, y, c in m["real"]:
        cx, cy = f(x), f(1 - y)
        if not c:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="none" stroke="#ff6b4a" stroke-width="1.6"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{"#f4f0e8" if c else "#ff6b4a"}"/>')
    return "".join(parts) + "</svg>"


def bunching_table(b: dict) -> str:
    if not b["questions"]:
        return "<p class='fine'>No question has more than 80% of the AI panel giving the same answer.</p>"
    rows = "".join(f"<tr><td>{e(r['question'])}</td><td>{e(r['ai_top_answer'])}</td><td>{pct(r['ai_top_share'])}</td>"
                   f"<td>{pct(r['real_top_share'])}</td></tr>" for r in b["questions"])
    return ("<table><tr><th>Question</th><th>Answer almost everyone in the AI panel gives</th><th>AI panel</th>"
            f"<th>Real sample</th></tr>{rows}</table>")


def gaps_table(gaps: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{e(g['question'])}</td><td>{e(g['answer'])}</td>"
        f"<td><div class='bar'><i style='width:{100 * g['ai_share']:.0f}%;background:var(--syn)'></i></div>{pct(g['ai_share'])} AI</td>"
        f"<td><div class='bar'><i style='width:{100 * g['real_share']:.0f}%;background:#f4f0e8'></i></div>{pct(g['real_share'])} real</td></tr>"
        for g in gaps)
    return f"<table><tr><th>Question</th><th>Answer with the biggest gap</th><th></th><th></th></tr>{rows}</table>"


def examples_html(exs: list[dict]) -> str:
    if not exs:
        return "<p class='fine'>No missed real people to show.</p>"
    cards = []
    for x in exs:
        rows = "".join(f"<tr><td>{e(r['q'])}</td><td class='a'>{e(r['real'] or '(blank)')}</td><td class='b'>{e(r['ai'] or '(blank)')}</td></tr>"
                       for r in x["rows"])
        cards.append(f"<div class='card'><b>{e(x['label'])}</b> <span class='fine'>differs from the closest AI person on "
                     f"{x['n_diff']} of {x['n_questions']} questions</span><table><tr><th>Question</th><th>Real person</th>"
                     f"<th>Closest AI person</th></tr>{rows}</table></div>")
    return f"<div class='ex'>{''.join(cards)}</div>"


def render_html(report: dict) -> str:
    s, err, rec = report["summary"], report["summary"]["expected_error"], report["recommendation"]
    pm = f"typically within ±{round(100 * err['rmse'])} points" if err.get("rmse") is not None else ""
    pr = s.get("realistic_error")
    prm = f"±{round(100 * pr)} points" if pr is not None else "uncertain"
    warn = "" if err.get("reliable") else (
        f"<div class='card warn'><b>Small sample.</b> Your real sample has {err['n']} people; "
        f"{err['min_reliable_n']} or more give a reliable number. Treat this as a rough guide.</div>")
    notes = "".join(f"<p class='fine'>{e(n)}</p>" for n in report["notes"])
    paras = "".join(f"<p>{e(p)}</p>" for p in rec["paragraphs"])
    ignored = ", ".join(f"{e(c['label'])} ({e(c['reason'])})" for c in report["columns"]["ignored"]) or "none"
    b = report["bunching"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blindspot audit report</title><style>{STYLE}</style></head><body><main>
<p class="kicker">Blindspot audit</p><h1>{e(rec['headline'])}</h1>
<div class="bigs"><div class="big"><div class="n miss">{pct(s['missed'])}</div><div class="l">of real people missed ({pm})</div></div>
<div class="big"><div class="n syn">{pct(s['realistic'])}</div><div class="l">of AI respondents look realistic ({prm}; tends to read low with small samples)</div></div></div>
{warn}{notes}
<div class="card rec"><b>What to do</b>{paras}</div>
<h2>The map</h2><div class="card">{map_svg(report['map'])}<p class="fine">White dots are real people; teal dots are AI respondents; coral rings are real people nobody in the AI panel resembles. Closer means more alike in their answers.</p></div>
<h2>Questions where the AI panel bunches</h2>{bunching_table(b)}
<p class="fine">{b['n_ai_bunched']} of {b['n_questions']} questions in the AI panel vs {b['n_real_bunched']} in the real sample.</p>
<h2>Biggest differences in answers</h2>{gaps_table(report['gaps'])}
<h2>Three real people the AI panel missed</h2>{examples_html(report['examples'])}
<h2>How to read this</h2><p class="fine">{e(report['how_to_read'])}</p>
<p class="fine">Compared {s['n_real']} real people with {s['n_ai']} AI respondents on {s['n_questions']} questions. Ignored columns: {ignored}.</p>
<p class="fine">{e(report['repaired_note'])}</p>
<p class="fine">Made with Blindspot. Your files were processed in memory and not stored.</p>
</main></body></html>"""
