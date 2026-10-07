"""GradeCast NG web application (Flask)."""

import os
import re
import secrets
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, g, render_template, request, session
from markupsafe import Markup

from ml import schema, service

from . import db, photos

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = Path(__file__).resolve().parent / "static" / "icons"


def _secret_key(instance: Path) -> str:
    """Use SECRET_KEY from the environment, or keep one on disk."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    key_file = instance / "secret.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    return key_file.read_text().strip()


@lru_cache(maxsize=None)
def _icon_body(name: str) -> str:
    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        return ""
    text = path.read_text()
    return "".join(re.findall(r"<(?:path|circle|rect|line|polyline|polygon|ellipse)[^>]*/>", text))


def icon(name, size=18, cls=""):
    """Inline Lucide icon. Decorative by default (hidden from readers)."""
    return Markup(
        f'<svg class="icon {cls}" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true" focusable="false">{_icon_body(name)}</svg>')


def create_app(test_config=None):
    app = Flask(__name__, instance_path=str(ROOT / "instance"))
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    app.config.update(
        SECRET_KEY=_secret_key(Path(app.instance_path)),
        DATABASE=os.environ.get(
            "DATABASE_PATH", str(Path(app.instance_path) / "gradecast.db")),
        DATABASE_URL=os.environ.get("DATABASE_URL", ""),
        UPLOAD_DIR=str(ROOT / "data" / "uploads"),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SEED_SAMPLE=os.environ.get("SEED_SAMPLE", "1") == "1",
        SHOW_DEMO_ACCOUNTS=os.environ.get("SHOW_DEMO_ACCOUNTS", "1") == "1",
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    db.configure(app)
    fresh = db.init_db(app)
    app.teardown_appcontext(db.close_db)
    service.load()
    if fresh and app.config["SEED_SAMPLE"]:
        db.add_sample_cohort(app)

    # ---- request hooks ---------------------------------------------------
    @app.before_request
    def load_user():
        g.user = None
        uid = session.get("uid")
        if uid:
            g.user = db.get_db().execute(
                "SELECT * FROM users WHERE id=?", (uid,)).fetchone()
            if g.user is None:
                session.clear()

    @app.before_request
    def csrf_protect():
        if request.method != "POST" or request.endpoint == "main.api_preview":
            return
        sent = (request.form.get("csrf_token")
                or request.headers.get("X-CSRF-Token"))
        if not sent or not secrets.compare_digest(
                sent, session.get("csrf", "")):
            abort(400, "This form has expired. Go back, reload the page and "
                       "try again.")

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    # ---- template helpers ------------------------------------------------
    def fmt_dt(value, style="long"):
        if not value:
            return ""
        try:
            moment = datetime.fromisoformat(value) + timedelta(hours=1)  # WAT
        except ValueError:
            return value
        # Day without a leading zero, built by hand so it also works on
        # Windows (which has no %-d).
        if style == "date":
            return f"{moment.day} {moment:%b %Y}"
        if style == "short":
            return f"{moment.day} {moment:%b, %H:%M}"
        return f"{moment.day} {moment:%b %Y, %H:%M}"

    app.jinja_env.filters["dt"] = fmt_dt
    app.jinja_env.filters["pct"] = lambda v, d=0: f"{100 * float(v):.{d}f}%"
    app.jinja_env.filters["slug"] = lambda v: str(v).lower().replace(" ", "-")
    app.jinja_env.globals.update(
        icon=icon, csrf_token=csrf_token, photo=photos.photo,
        gallery=photos.gallery, CLASSES=schema.CLASSES,
        model_ready=service.ready, app_name="GradeCast NG")

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def handle_error(err):
        code = getattr(err, "code", 500)
        messages = {
            400: ("That request could not be processed",
                  getattr(err, "description", "")),
            403: ("You do not have access to this page",
                  "Sign in with an account that has the right role."),
            404: ("This page does not exist",
                  "The link may be old, or the record was deleted."),
            413: ("That file is too large",
                  "Upload a CSV smaller than 10 MB."),
            500: ("Something went wrong on our side",
                  "Try again. If it keeps happening, check the server log."),
        }
        title, detail = messages.get(code, messages[500])
        return render_template("error.html", code=code, title=title,
                               detail=detail), code

    from . import admin, auth, views
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(admin.bp)
    return app
