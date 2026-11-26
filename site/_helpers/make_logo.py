"""Render the Field Guide hex sticker from SVG to the PNG used in the README.

    python3 site/_helpers/make_logo.py [--chrome /usr/bin/google-chrome-stable]
"""

import argparse
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default="google-chrome-stable")
    args = parser.parse_args()

    images = pathlib.Path(__file__).resolve().parent.parent / "images"
    sticker = images / "duckboost-hex.svg"
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "page.html"
        page.write_text(
            "<!doctype html><style>html,body{margin:0;background:transparent}"
            "img{display:block;width:520px;height:600px}</style>"
            f'<img src="{sticker.as_uri()}">'
        )
        subprocess.run(
            [
                args.chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--user-data-dir={tmp}/profile",
                "--window-size=520,600",
                f"--screenshot={sticker.with_suffix('.png')}",
                "--virtual-time-budget=2000",
                "--default-background-color=00000000",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )


if __name__ == "__main__":
    main()
