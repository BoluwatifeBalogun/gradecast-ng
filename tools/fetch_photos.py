"""Download the default landing-page photographs into webapp/static/img/
so the site shows them without an internet connection.

    python tools/fetch_photos.py

Run it once on a computer that is online. Slots that already have a local
file are left alone.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webapp.photos import EXTENSIONS, IMG_DIR, SLOTS  # noqa: E402


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for slot, meta in SLOTS.items():
        if any((IMG_DIR / f"{slot}{ext}").exists() for ext in EXTENSIONS):
            print(f"{slot}: already present, skipped")
            continue
        if not meta["remote"]:
            print(f"{slot}: no default picture. Add your own as "
                  f"webapp/static/img/{slot}.jpg")
            continue
        target = IMG_DIR / f"{slot}.jpg"
        request = urllib.request.Request(
            meta["remote"], headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if len(data) < 10_000:
                raise ValueError("the download was too small to be a photo")
            target.write_bytes(data)
            print(f"{slot}: saved {target.name} ({len(data) // 1024} KB), "
                  f"credit {meta['credit']}")
        except Exception as exc:
            print(f"{slot}: could not download ({exc})")


if __name__ == "__main__":
    main()
