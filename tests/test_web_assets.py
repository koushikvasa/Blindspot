"""The demo must work with the network off: fonts and scripts are self-hosted, nothing points at a CDN."""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
PAGES = [WEB / "index.html", WEB / "join.html", WEB / "stage.html"]
EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.I)


def test_pages_load_nothing_from_the_network():
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        assert not EXTERNAL.search(text), f"{page.name} references an external URL"
        assert "googleapis" not in text and "gstatic" not in text, page.name


def test_stylesheets_have_no_external_imports_or_urls():
    for css in [WEB / "css" / "app.css", WEB / "vendor" / "fonts" / "fonts.css"]:
        text = css.read_text(encoding="utf-8")
        assert not re.search(r"url\(\s*[\"']?https?://", text), css.name
        assert "@import" not in text, css.name


def test_every_font_face_points_at_a_file_that_exists():
    css = (WEB / "vendor" / "fonts" / "fonts.css").read_text(encoding="utf-8")
    files = re.findall(r'url\("/static/vendor/fonts/([^"]+)"\)', css)
    assert len(files) >= 3
    for name in files:
        path = WEB / "vendor" / "fonts" / name
        assert path.exists() and path.stat().st_size > 10_000, name


def test_body_text_is_never_below_13px_when_the_scale_is_one():
    css = (WEB / "css" / "app.css").read_text(encoding="utf-8")
    sizes = [float(m) for m in re.findall(r"font-size:\s*calc\(([\d.]+) \* var\(--u\)\)", css)]
    assert sizes and min(sizes) >= 13, min(sizes)
