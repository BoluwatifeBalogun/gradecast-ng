"""Photographs on the public pages.

Each slot looks for a file you have placed in webapp/static/img/ first
(hero.jpg, campus-1.jpg, campus-2.jpg, campus-3.jpg; .jpeg, .png and .webp
also work). If there is no local file, the slot falls back to its `remote`
address when it has one, and is left out of the page when it has none.

To use your own pictures, drop them in webapp/static/img/ with those names
and edit the alt text, caption and credit below. Landscape photos about
1600 pixels wide work best. Only use pictures you have permission to use.
"""

from pathlib import Path

from flask import url_for

IMG_DIR = Path(__file__).resolve().parent / "static" / "img"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

SLOTS = {
    "hero": {
        # Free to use under the Unsplash License.
        # https://unsplash.com/photos/sgQM3gsGD9s
        "remote": ("https://images.unsplash.com/photo-1769905226600-"
                   "1d447fe7d020?auto=format&fit=crop&w=1600&q=70"),
        "alt": "Graduands in caps and gowns seated in an auditorium in Lagos",
        "caption": "",
        "credit": "Blessfield John on Unsplash",
        "credit_url": "https://unsplash.com/@blessfield",
    },
    "campus-1": {
        "remote": None,
        "alt": "Students in a lecture",
        "caption": "",
        "credit": "", "credit_url": "",
    },
    "campus-2": {
        "remote": None,
        "alt": "Students studying together",
        "caption": "",
        "credit": "", "credit_url": "",
    },
    "campus-3": {
        "remote": None,
        "alt": "Students on campus",
        "caption": "",
        "credit": "", "credit_url": "",
    },
}


def _local(slot):
    for ext in EXTENSIONS:
        if (IMG_DIR / f"{slot}{ext}").exists():
            return f"img/{slot}{ext}"
    return None


def photo(slot):
    """Details for one slot, or None when there is nothing to show."""
    meta = SLOTS.get(slot)
    if meta is None:
        return None
    local = _local(slot)
    if local:
        return {**meta, "src": url_for("static", filename=local),
                "local": True}
    if meta["remote"]:
        return {**meta, "src": meta["remote"], "local": False}
    return None


def gallery():
    """Every slot that has a picture, hero first."""
    return [p for p in (photo(s) for s in SLOTS) if p]
