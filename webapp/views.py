"""Public pages and the prediction workflow."""

import csv
import io
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

from flask import (Blueprint, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, url_for)

from ml import advice, schema, service

from . import db, nigeria
from .auth import login_required

bp = Blueprint("main", __name__)
PAGE_SIZE = 12


def _scope():
    """Admins see every record, everyone else sees their own."""
    if g.user["role"] == "admin":
        return "1=1", []
    return "p.user_id = ?", [g.user["id"]]


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
@bp.route("/")
def landing():
    return render_template("landing.html", metrics=service.metrics(),
                           features=schema.FEATURES,
                           defaults=schema.defaults())


@bp.route("/api/preview", methods=["POST"])
def api_preview():
    """Live demo on the landing page: three attributes vary, the other
    sixteen stay at the values of a typical student."""
    if not service.ready():
        return jsonify(error="The model is not trained yet."), 503
    payload = request.get_json(silent=True) or {}
    profile = schema.defaults()
    for key in ("Hours_Studied", "Attendance", "Previous_Scores"):
        if key in payload:
            profile[key] = payload[key]
    clean, errors = schema.validate(profile)
    if errors:
        return jsonify(error=next(iter(errors.values()))), 400
    result = service.predict(clean)
    return jsonify(label=result["label"], index=result["index"],
                   confidence=result["confidence"], proba=result["proba"],
                   agree=result["agree"],
                   votes={k: v["label"] for k, v in result["votes"].items()})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@bp.route("/dashboard")
@login_required
def dashboard():
    conn = db.get_db()
    where, params = _scope()
    rows = conn.execute(
        f"SELECT label, label_index, risk, zone, institution_type, "
        f"created_at, is_sample FROM predictions p WHERE {where}",
        params).fetchall()

    total = len(rows)
    by_class = Counter(r["label"] for r in rows)
    by_risk = Counter(r["risk"] for r in rows)

    zones = {}
    for r in rows:
        if r["zone"]:
            z = zones.setdefault(r["zone"], {"n": 0, "risk": 0})
            z["n"] += 1
            z["risk"] += r["risk"] == "At risk"
    zone_rows = sorted(
        ({"zone": k, "n": v["n"], "risk": v["risk"],
          "share": v["risk"] / v["n"]} for k, v in zones.items()),
        key=lambda z: z["share"], reverse=True)

    today = (datetime.now(timezone.utc) + timedelta(hours=1)).date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    per_day = Counter()
    for r in rows:
        try:
            d = (datetime.fromisoformat(r["created_at"])
                 + timedelta(hours=1)).date()
            per_day[d] += 1
        except ValueError:
            pass

    recent = conn.execute(
        f"SELECT p.*, u.name AS owner FROM predictions p "
        f"JOIN users u ON u.id = p.user_id WHERE {where} "
        f"ORDER BY p.created_at DESC, p.id DESC LIMIT 6", params).fetchall()

    return render_template(
        "app/dashboard.html", total=total,
        by_class=[by_class.get(c, 0) for c in schema.CLASSES],
        by_risk={k: by_risk.get(k, 0)
                 for k in ("At risk", "Watch", "On track")},
        zone_rows=zone_rows,
        timeline={"labels": [f"{d.day} {d:%b}" for d in days],
                  "values": [per_day.get(d, 0) for d in days]},
        recent=recent, has_sample=any(r["is_sample"] for r in rows),
        metrics=service.metrics())


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def _record_from_form(form):
    record = {
        "student_name": form.get("student_name", "").strip(),
        "student_ref": form.get("student_ref", "").strip() or None,
        "institution_type": form.get("institution_type", "").strip() or None,
        "state": form.get("state", "").strip() or None,
    }
    errors = {}
    if len(record["student_name"]) < 2:
        errors["student_name"] = "Enter the student's name."
    if record["state"] and record["state"] not in nigeria.STATE_ZONE:
        errors["state"] = "Choose a state from the list."
    if (record["institution_type"]
            and record["institution_type"] not in nigeria.INSTITUTION_TYPES):
        errors["institution_type"] = "Choose an institution type from the list."
    record["zone"] = nigeria.STATE_ZONE.get(record["state"])
    return record, errors


@bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    values = schema.defaults()
    record = {"student_name": g.user["name"]
              if g.user["role"] == "student" else "",
              "student_ref": "", "institution_type": "", "state": ""}
    errors = {}

    if request.method == "POST":
        if not service.ready():
            abort(400, "The model has not been trained yet. Ask the "
                       "administrator to train it.")
        record, errors = _record_from_form(request.form)
        clean, field_errors = schema.validate(request.form)
        errors.update(field_errors)
        values.update({k: request.form.get(k, v) for k, v in values.items()})
        if not errors:
            result = service.predict(clean)
            conn = db.get_db()
            pid = db.save_prediction(conn, g.user["id"], record, clean,
                                     result)
            conn.commit()
            return redirect(url_for("main.prediction", pid=pid))

    step_of = {"student_name": 0, "student_ref": 0, "institution_type": 0,
               "state": 0}
    for i, (key, _, _) in enumerate(schema.GROUPS, start=1):
        for f in schema.FEATURES:
            if f["group"] == key:
                step_of[f["name"]] = i
    first_error_step = min((step_of.get(k, 0) for k in errors), default=0)

    return render_template(
        "app/predict.html", groups=schema.GROUPS, features=schema.FEATURES,
        values=values, record=record, errors=errors,
        states=nigeria.STATES, institution_types=nigeria.INSTITUTION_TYPES,
        first_error_step=first_error_step,
        ranges=service.training_ranges()), (400 if errors else 200)


def _get_prediction(pid):
    where, params = _scope()
    row = db.get_db().execute(
        f"SELECT p.*, u.name AS owner FROM predictions p "
        f"JOIN users u ON u.id = p.user_id WHERE p.id = ? AND {where}",
        [pid] + params).fetchone()
    if row is None:
        abort(404)
    return row


@bp.route("/predictions/<int:pid>")
@login_required
def prediction(pid):
    row = _get_prediction(pid)
    inputs = json.loads(row["inputs_json"])
    votes = json.loads(row["votes_json"])
    explanation = None
    if service.ready():
        try:
            explanation = service.explain(inputs)
        except Exception:  # an old record that no longer fits the schema
            explanation = None
    return render_template(
        "app/result.html", p=row, inputs=inputs,
        proba=json.loads(row["proba_json"]), votes=votes["votes"],
        agree=votes["agree"], explanation=explanation,
        features=schema.FEATURES, groups=schema.GROUPS,
        support_note=advice.SUPPORT_NOTE
        if inputs.get("Learning_Disabilities") == "Yes" else None,
        stale=service.version() != row["model_version"],
        out_of_range=service.out_of_range(inputs))


@bp.route("/predictions/<int:pid>/delete", methods=["POST"])
@login_required
def prediction_delete(pid):
    row = _get_prediction(pid)
    conn = db.get_db()
    conn.execute("DELETE FROM predictions WHERE id=?", (row["id"],))
    conn.commit()
    flash(f"Deleted the prediction for {row['student_name']}.", "success")
    return redirect(url_for("main.history"))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def _history_query():
    where, params = _scope()
    q = request.args.get("q", "").strip()
    label = request.args.get("label", "")
    risk = request.args.get("risk", "")
    if q:
        where += (" AND (LOWER(p.student_name) LIKE ?"
                  " OR LOWER(p.student_ref) LIKE ?)")
        params += [f"%{q.lower()}%", f"%{q.lower()}%"]
    if label in schema.CLASSES:
        where += " AND p.label = ?"
        params.append(label)
    if risk in ("At risk", "Watch", "On track"):
        where += " AND p.risk = ?"
        params.append(risk)
    return where, params, {"q": q, "label": label, "risk": risk}


@bp.route("/history")
@login_required
def history():
    where, params, filters = _history_query()
    conn = db.get_db()
    total = conn.execute(
        f"SELECT COUNT(*) FROM predictions p WHERE {where}",
        params).fetchone()[0]
    pages = max(1, -(-total // PAGE_SIZE))
    page = min(max(request.args.get("page", 1, type=int), 1), pages)
    rows = conn.execute(
        f"SELECT p.*, u.name AS owner FROM predictions p "
        f"JOIN users u ON u.id = p.user_id WHERE {where} "
        f"ORDER BY p.created_at DESC, p.id DESC LIMIT ? OFFSET ?",
        params + [PAGE_SIZE, (page - 1) * PAGE_SIZE]).fetchall()
    return render_template("app/history.html", rows=rows, total=total,
                           page=page, pages=pages, filters=filters,
                           filtered=any(filters.values()))


@bp.route("/history/export.csv")
@login_required
def history_export():
    where, params, _ = _history_query()
    rows = db.get_db().execute(
        f"SELECT p.* FROM predictions p WHERE {where} "
        f"ORDER BY p.created_at DESC", params).fetchall()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Date (UTC)", "Student", "Reference", "Institution type",
                     "State", "Zone", "Predicted class", "Confidence",
                     "Risk level"] + schema.FEATURE_NAMES)
    for r in rows:
        inputs = json.loads(r["inputs_json"])
        writer.writerow(
            [r["created_at"], r["student_name"], r["student_ref"] or "",
             r["institution_type"] or "", r["state"] or "", r["zone"] or "",
             r["label"], f"{r['confidence']:.3f}", r["risk"]]
            + [inputs.get(n, "") for n in schema.FEATURE_NAMES])
    return Response(
        out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=gradecast-predictions.csv"})


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
@bp.route("/insights")
@login_required
def insights():
    return render_template("app/insights.html", m=service.metrics(),
                           load_error=service.load_error())
