"""Build the duckboost logo assets in site/images/.

Writes the hex sticker, the favicon, the apple-touch-icon and the 2400x1260
link-preview card as SVG, then
rasterizes them to PNG with headless Chrome. Text is converted to outlines so
the SVGs render identically without the fonts installed.

    pip install fonttools uharfbuzz
    python3 site/_helpers/make_logo.py [--chrome /usr/bin/google-chrome-stable]
"""

import argparse
import math
import pathlib
import subprocess
import tempfile

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = pathlib.Path(__file__).resolve().parent.parent / "images"

BLACK = "#0d0d0d"
WHITE = "#f2f2f2"
GREY = "#b2b2b2"
YELLOW = "#fff100"

# An original rubber-duck silhouette facing right, drawn in a 100x100 box.
DUCK = f"""
<symbol id="duck" viewBox="0 0 100 100">
  <path fill="{YELLOW}" d="M6 46 C14 52 22 56 34 56 L52 56
    C50 50 47 44 47 36 C47 20 58 10 71 10 C84 10 93 20 93 32
    C93 40 90 45 85 49 C94 55 97 64 95 74 C92 88 78 94 58 94
    L40 94 C20 94 8 84 6 68 C5 60 5 52 6 46 Z"/>
  <path fill="{YELLOW}" d="M88 26 C95 23 100 24 100 28 C100 33 95 37 88 38 Z"/>
  <circle cx="75" cy="27" r="4.2" fill="{BLACK}"/>
</symbol>
"""


def font_file(pattern):
    return subprocess.run(
        ["fc-match", "-f", "%{file}", pattern], check=True, capture_output=True, text=True
    ).stdout


class TextOutliner:
    def __init__(self, path):
        self.hb_font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
        tt = TTFont(path)
        self.glyphs = tt.getGlyphSet()
        self.order = tt.getGlyphOrder()
        self.upem = tt["head"].unitsPerEm

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {})
        return buf.glyph_infos, buf.glyph_positions

    def width(self, text, size):
        _, positions = self._shape(text)
        return sum(p.x_advance for p in positions) * size / self.upem

    def path(self, text, size, x, y, anchor="start", fill=WHITE):
        infos, positions = self._shape(text)
        scale = size / self.upem
        if anchor == "middle":
            x -= self.width(text, size) / 2
        pen = SVGPathPen(self.glyphs)
        cursor = 0
        for info, pos in zip(infos, positions):
            gx = x + (cursor + pos.x_offset) * scale
            gy = y - pos.y_offset * scale
            self.glyphs[self.order[info.codepoint]].draw(
                TransformPen(pen, (scale, 0, 0, -scale, gx, gy))
            )
            cursor += pos.x_advance
        return f'<path fill="{fill}" d="{pen.getCommands()}"/>'


def hex_points(cx, cy, r):
    pts = []
    for k in range(6):
        a = math.radians(90 + 60 * k)
        pts.append(f"{cx + r * math.cos(a):.2f},{cy - r * math.sin(a):.2f}")
    return " ".join(pts)


def hex_frame(cx, cy, r, border):
    inner = r - border / 2 / math.cos(math.radians(30))
    return (
        f'<polygon points="{hex_points(cx, cy, inner)}" fill="{BLACK}" '
        f'stroke="{YELLOW}" stroke-width="{border}" stroke-linejoin="round"/>'
    )


def duck(cx, top, size):
    return f'<use href="#duck" x="{cx - size / 2:.2f}" y="{top:.2f}" width="{size:.2f}" height="{size:.2f}"/>'


def family_tree(cx, top, scale=1.0):
    """Root duck, two ducks, four ducklings, joined by elbow connectors."""
    s = scale
    levels = [
        (top, 40 * s, [0]),
        (top + 52 * s, 30 * s, [-32 * s, 32 * s]),
        (top + 96 * s, 22 * s, [-50 * s, -14 * s, 14 * s, 50 * s]),
    ]
    parts = []
    stroke = 2.4 * s
    for (y0, size0, xs0), (y1, _, xs1) in zip(levels, levels[1:]):
        for i, px in enumerate(xs0):
            stem_top = y0 + size0 * 0.93
            fork = (stem_top + y1) / 2 - 1 * s
            children = xs1[2 * i : 2 * i + 2]
            d = f"M{cx + px:.2f} {stem_top:.2f} V{fork:.2f}"
            for kx in children:
                d += f" M{cx + px:.2f} {fork:.2f} L{cx + kx:.2f} {y1 + 1.5 * s:.2f}"
            parts.append(
                f'<path d="{d}" fill="none" stroke="{WHITE}" stroke-width="{stroke:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
    for y, size, xs in levels:
        parts.extend(duck(cx + x, y, size) for x in xs)
    return "\n".join(parts)


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n<defs>{DUCK}</defs>\n{body}\n</svg>\n'
    )


def hex_sticker(sans):
    w, h = 2 * 100 * math.cos(math.radians(30)), 200
    cx = w / 2
    body = "\n".join(
        [
            hex_frame(cx, 100, 100, 7),
            family_tree(cx, 27, 0.88),
            sans.path("duckboost", 19, cx, 164, anchor="middle"),
        ]
    )
    return svg(round(w, 2), h, body)


def social_card(sans, sans_regular, mono):
    w, h = 1200, 630
    hex_r = 215
    hex_w = 2 * hex_r * math.cos(math.radians(30))
    hx, hy = 80 + hex_w / 2, h / 2
    k = hex_r / 100
    sticker = (
        f'<g transform="translate({hx - 86.6 * k:.2f} {hy - 100 * k:.2f}) scale({k:.4f})">'
        + hex_frame(86.6, 100, 100, 7)
        + family_tree(86.6, 27, 0.88)
        + sans.path("duckboost", 19, 86.6, 164, anchor="middle")
        + "</g>"
    )
    tx = 530
    code = "SELECT duckboost_predict(model, ...)"
    code_size = 24
    code_w = mono.width(code, code_size)
    body = "\n".join(
        [
            f'<rect width="{w}" height="{h}" fill="{BLACK}"/>',
            sticker,
            sans.path("duckboost", 96, tx - 5, 280),
            sans_regular.path("Gradient-boosted trees,", 34, tx, 346, fill=GREY),
            sans_regular.path("trained and scored in DuckDB SQL", 34, tx, 390, fill=GREY),
            f'<rect x="{tx - 10}" y="428" width="{code_w + 20:.1f}" height="42" rx="5" fill="{YELLOW}"/>',
            mono.path(code, code_size, tx, 457, fill=BLACK),
        ]
    )
    return svg(w, h, body)


def touch_icon():
    # Full-bleed square; iOS rounds the corners itself and would show
    # transparency as black anyway.
    size = 180
    k = 0.8
    hw, hh = 2 * 100 * math.cos(math.radians(30)), 200
    scale = size * k / hh
    body = (
        f'<rect width="{size}" height="{size}" fill="{BLACK}"/>'
        f'<g transform="translate({(size - hw * scale) / 2:.2f} {(size - hh * scale) / 2:.2f}) scale({scale:.4f})">'
        + hex_frame(hw / 2, 100, 100, 14)
        + duck(hw / 2, 45, 110)
        + "</g>"
    )
    return svg(size, size, body)


def icon():
    size = 200
    hw = 2 * 100 * math.cos(math.radians(30))
    body = (
        f'<g transform="translate({(size - hw) / 2:.2f} 0)">'
        + hex_frame(hw / 2, 100, 100, 14)
        + duck(hw / 2, 45, 110)
        + "</g>"
    )
    return svg(size, size, body)


def rasterize(chrome, svg_path, png_path, width, height, transparent):
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "page.html"
        page.write_text(
            "<!doctype html><style>html,body{margin:0;background:transparent}"
            f"img{{display:block;width:{width}px;height:{height}px}}</style>"
            f'<img src="{svg_path.as_uri()}">'
        )
        cmd = [
            chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
            f"--user-data-dir={tmp}/profile", f"--window-size={width},{height}",
            f"--screenshot={png_path}", "--virtual-time-budget=2000",
        ]
        if transparent:
            cmd.append("--default-background-color=00000000")
        subprocess.run(cmd + [page.as_uri()], check=True, capture_output=True, timeout=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", default="google-chrome-stable")
    args = parser.parse_args()

    sans = TextOutliner(font_file("Inter:style=SemiBold"))
    sans_regular = TextOutliner(font_file("Inter:style=Regular"))
    mono = TextOutliner(font_file("JetBrains Mono:style=Medium"))

    OUT.mkdir(exist_ok=True)
    assets = {
        "duckboost-hex.svg": hex_sticker(sans),
        "duckboost-icon.svg": icon(),
        "duckboost-touch-icon.svg": touch_icon(),
        "duckboost-banner.svg": social_card(sans, sans_regular, mono),
    }
    for name, text in assets.items():
        (OUT / name).write_text(text)

    rasterize(args.chrome, OUT / "duckboost-hex.svg", OUT / "duckboost-hex.png", 520, 600, True)
    rasterize(args.chrome, OUT / "duckboost-icon.svg", OUT / "favicon.png", 64, 64, True)
    rasterize(args.chrome, OUT / "duckboost-touch-icon.svg", OUT / "apple-touch-icon.png", 180, 180, False)
    # 2400 wide: iOS only draws the full-width preview for images at least 2400x1256
    rasterize(args.chrome, OUT / "duckboost-banner.svg", OUT / "duckboost-banner.png", 2400, 1260, False)


if __name__ == "__main__":
    main()
