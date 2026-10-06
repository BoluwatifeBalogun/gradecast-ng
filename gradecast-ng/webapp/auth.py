"""Sign in, registration and role checks."""

import re
from functools import wraps

from flask import (Blueprint, abort, flash, g, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from . import db

bp = Blueprint("auth", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("auth.login", next=request.path))
        if g.user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def _safe_next(target):
    return target if target and target.startswith("/") \
        and not target.startswith("//") else None


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("main.dashboard"))
    errors, email = {}, ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db.get_db().execute(
            "SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user is None or not check_password_hash(user["password_hash"],
                                                   password):
            errors["form"] = ("That email and password do not match an "
                              "account. Check both and try again.")
        else:
            csrf = session.get("csrf")
            session.clear()
            session["uid"] = user["id"]
            session["csrf"] = csrf
            session.permanent = True
            return redirect(_safe_next(request.args.get("next"))
                            or url_for("main.dashboard"))
    return render_template("auth/login.html", errors=errors, email=email)


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("main.dashboard"))
    errors, form = {}, {"role": "student"}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip()
                for k in ("name", "email", "institution", "role")}
        form["email"] = form["email"].lower()
        password = request.form.get("password", "")
        if len(form["name"]) < 2:
            errors["name"] = "Enter your full name."
        if not EMAIL_RE.match(form["email"]):
            errors["email"] = "Enter a valid email address."
        if form["role"] not in ("student", "officer"):
            errors["role"] = "Choose student or academic officer."
        if len(password) < 8:
            errors["password"] = "Use at least 8 characters."
        conn = db.get_db()
        if "email" not in errors and conn.execute(
                "SELECT 1 FROM users WHERE email=?",
                (form["email"],)).fetchone():
            errors["email"] = ("An account with this email already exists. "
                               "Sign in instead.")
        if not errors:
            cur = conn.execute(
                "INSERT INTO users (name,email,password_hash,role,institution,"
                "created_at) VALUES (?,?,?,?,?,?)",
                (form["name"], form["email"],
                 generate_password_hash(password), form["role"],
                 form["institution"] or None, db.now()))
            conn.commit()
            csrf = session.get("csrf")
            session.clear()
            session["uid"] = cur.lastrowid
            session["csrf"] = csrf
            session.permanent = True
            flash("Account created. You can run your first prediction now.",
                  "success")
            return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html", errors=errors, form=form)


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("main.landing"))
