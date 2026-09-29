"""Post-render: add og:url, og:type and a canonical link to every page.

Quarto 1.10 writes og:title/og:image but not og:url or og:type, which Apple's
LinkPresentation (iMessage) and Facebook use to identify the page. The block is
inserted right after og:title so it sits with the rest of the Open Graph tags.
"""
import html
import os
import pathlib
import re

SITE_URL = "https://javorraca.github.io/duckboost/"

out_dir = pathlib.Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
og_title = re.compile(r'^<meta property="og:title"[^>]*>\n', re.M)

for page in out_dir.rglob("*.html"):
    text = page.read_text(encoding="utf-8")
    if 'property="og:url"' in text:
        continue
    match = og_title.search(text)
    if not match:
        continue
    rel = page.relative_to(out_dir).as_posix()
    if rel == "index.html" or rel.endswith("/index.html"):
        rel = rel[: -len("index.html")]
    url = html.escape(SITE_URL + rel, quote=True)
    block = (
        f'<meta property="og:url" content="{url}">\n'
        '<meta property="og:type" content="website">\n'
        f'<link rel="canonical" href="{url}">\n'
    )
    page.write_text(text[: match.end()] + block + text[match.end() :], encoding="utf-8")
