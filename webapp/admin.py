"""System Administrator area: datasets, training, users (Figure 3.3)."""

import threading
import uuid
from pathlib import Path

from flask import (Blueprint, abort, current_app, flash, g, jsonify, redirect,
                   render_template, request, url_for)

from ml import pipeline, service

from . import db
from .auth import admin_required

bp = Blueprint("admin", __name__, url_prefix="/admin")

# One training job at a time, tracked in memory. Run the app with a single
# worker process (threads are fine) so every request sees the same state.
_job = {"running": False, "percent": 0, "message": "", "error": None,
        "run_id": None}
_job_lock = threading.Lock()


def _dataset_path(row) -> Path:
    if row["is_builtin"]:
        return pipeline.DEFAULT_DATASET
    return Path(current_app.config["UPLOAD_DIR"]) / row["stored_name"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@bp.route("/datasets")
@admin_required
def datasets():
    conn = db.get_db()
    rows = conn.execute(
        "SELECT d.*, u.name AS uploader FROM datasets d "
        "LEFT JOIN users u ON u.id = d.uploaded_by "
        "ORDER BY d.is_active DESC, d.created_at DESC").fetchall()
    active = next((r for r in rows if r["is_active"]), None)
    preview = None
    if active is not None:
        try:
            df = pipeline.load_dataset(_dataset_path(active))
            preview = {"columns": list(df.columns),
                       "rows": df.head(6).fillna("").astype(str)
                       .values.tolist()}
        except Exception as exc:
            flash(f"The active dataset could not be opened: {exc}", "error")
    sample_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE is_sample=1").fetchone()[0]
    return render_template("admin/datasets.html", rows=rows, active=active,
                           preview=preview, sample_count=sample_count,
                           required=pipeline.schema.REQUIRED_COLUMNS)


@bp.route("/datasets/upload", methods=["POST"])
@admin_required
def dataset_upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        flash("Choose a CSV file to upload.", "error")
        return redirect(url_for("admin.datasets"))
    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files can be uploaded.", "error")
        return redirect(url_for("admin.datasets"))
    stored = f"{uuid.uuid4().hex}.csv"
    path = Path(current_app.config["UPLOAD_DIR"]) / stored
    file.save(path)
    try:
        df = pipeline.load_dataset(path)
        _, report, _ = pipeline.clean(df)
        pipeline.choose_cutoffs(
            pipeline.clean(df)[0][pipeline.schema.TARGET])
    except pipeline.DatasetError as exc:
        path.unlink(missing_ok=True)
        flash(f"Upload rejected. {exc}", "error")
        return redirect(url_for("admin.datasets"))
    conn = db.get_db()
    conn.execute(
        "INSERT INTO datasets (original_name, stored_name, rows,"
        " missing_cells, uploaded_by, created_at) VALUES (?,?,?,?,?,?)",
        (Path(file.filename).name[:120], stored, report["rows_raw"],
         report["missing_cells"], g.user["id"], db.now()))
    conn.commit()
    flash(f"Uploaded {Path(file.filename).name} with "
          f"{report['rows_raw']:,} rows. Set it as active to train on it.",
          "success")
    return redirect(url_for("admin.datasets"))


@bp.route("/datasets/<int:did>/activate", methods=["POST"])
@admin_required
def dataset_activate(did):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    if row is None:
        abort(404)
    conn.execute("UPDATE datasets SET is_active = "
                 "CASE WHEN id = ? THEN 1 ELSE 0 END", (did,))
    conn.commit()
    flash(f"{row['original_name']} is now the active dataset. Train the "
          f"model to use it.", "success")
    return redirect(url_for("admin.datasets"))


@bp.route("/datasets/<int:did>/delete", methods=["POST"])
@admin_required
def dataset_delete(did):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    if row is None:
        abort(404)
    if row["is_builtin"] or row["is_active"]:
        flash("The built-in dataset and the active dataset cannot be "
              "deleted.", "error")
    else:
        _dataset_path(row).unlink(missing_ok=True)
        conn.execute("DELETE FROM datasets WHERE id=?", (did,))
        conn.commit()
        flash(f"Deleted {row['original_name']}.", "success")
    return redirect(url_for("admin.datasets"))


@bp.route("/sample/add", methods=["POST"])
@admin_required
def sample_add():
    added = db.add_sample_cohort(current_app, seed=uuid.uuid4().int % 10_000)
    if added:
        flash(f"Added {added} sample records.", "success")
    else:
        flash("Sample records need a trained model and an academic officer "
              "account.", "error")
    return redirect(url_for("admin.datasets"))


@bp.route("/sample/clear", methods=["POST"])
@admin_required
def sample_clear():
    conn = db.get_db()
    n = conn.execute("DELETE FROM predictions WHERE is_sample=1").rowcount
    conn.commit()
    flash(f"Removed {n} sample records.", "success")
    return redirect(url_for("admin.datasets"))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_training(db_path, run_id, path, name, cv_folds):
    def progress(stage, percent, message):
        with _job_lock:
            if stage == "cv":
                _job["message"] = message
            else:
                _job.update(percent=percent, message=message)

    conn = db.connect(db_path)
    try:
        _, metrics = pipeline.train(path, progress=progress,
                                    cv_folds=cv_folds, dataset_name=name)
        service.load(force=True)
        res = metrics["results"]["ensemble"]
        conn.execute(
            "UPDATE training_runs SET finished_at=?, status='Completed',"
            " accuracy=?, f1_macro=?, rows_balanced=?, model_version=?,"
            " message=? WHERE id=?",
            (db.now(), res["accuracy"], res["f1_macro"],
             metrics["dataset"]["rows_train_balanced"], metrics["version"],
             f"Trained in {metrics['duration_seconds']} seconds.", run_id))
        with _job_lock:
            _job.update(running=False, percent=100,
                        message="Training complete", error=None)
    except Exception as exc:
        conn.execute(
            "UPDATE training_runs SET finished_at=?, status='Failed',"
            " message=? WHERE id=?", (db.now(), str(exc)[:400], run_id))
        with _job_lock:
            _job.update(running=False, error=str(exc))
    finally:
        conn.commit()
        conn.close()


@bp.route("/training")
@admin_required
def training():
    conn = db.get_db()
    # A run left "Running" by a restart can never finish; mark it.
    if not _job["running"]:
        conn.execute(
            "UPDATE training_runs SET status='Interrupted', finished_at=?,"
            " message='The server restarted before this run finished.'"
            " WHERE status='Running'", (db.now(),))
        conn.commit()
    runs = conn.execute(
        "SELECT r.*, u.name AS starter FROM training_runs r "
        "LEFT JOIN users u ON u.id = r.started_by "
        "ORDER BY r.started_at DESC LIMIT 12").fetchall()
    active = conn.execute(
        "SELECT * FROM datasets WHERE is_active=1").fetchone()
    return render_template("admin/training.html", runs=runs, active=active,
                           m=service.metrics(), job=dict(_job),
                           cfg=pipeline.CONFIG,
                           load_error=service.load_error())


@bp.route("/training/start", methods=["POST"])
@admin_required
def training_start():
    conn = db.get_db()
    active = conn.execute(
        "SELECT * FROM datasets WHERE is_active=1").fetchone()
    if active is None:
        return jsonify(error="Set an active dataset first."), 400
    with _job_lock:
        if _job["running"]:
            return jsonify(error="Training is already running."), 409
        _job.update(running=True, percent=1, message="Starting", error=None)
    cur = conn.execute(
        "INSERT INTO training_runs (dataset_id, dataset_name, started_by,"
        " started_at, status) VALUES (?,?,?,?, 'Running')",
        (active["id"], active["original_name"], g.user["id"], db.now()))
    conn.commit()
    _job["run_id"] = cur.lastrowid
    cv_folds = 5 if request.form.get("cross_validate") == "1" else 0
    threading.Thread(
        target=_run_training, daemon=True,
        args=(current_app.config["DATABASE"], cur.lastrowid,
              _dataset_path(active), active["original_name"],
              cv_folds)).start()
    return jsonify(started=True)


@bp.route("/training/status")
@admin_required
def training_status():
    with _job_lock:
        return jsonify(dict(_job))


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
@bp.route("/users")
@admin_required
def users():
    rows = db.get_db().execute(
        "SELECT u.*, (SELECT COUNT(*) FROM predictions p"
        " WHERE p.user_id = u.id) AS n FROM users u "
        "ORDER BY u.created_at").fetchall()
    return render_template("admin/users.html", rows=rows)


@bp.route("/users/<int:uid>/role", methods=["POST"])
@admin_required
def user_role(uid):
    role = request.form.get("role")
    if role not in ("student", "officer", "admin"):
        abort(400, "Unknown role.")
    if uid == g.user["id"]:
        flash("You cannot change your own role.", "error")
        return redirect(url_for("admin.users"))
    conn = db.get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if row is None:
        abort(404)
    conn.execute("UPDATE users SET role=? WHERE id=?", (role, uid))
    conn.commit()
    labels = {"student": "Student", "officer": "Academic officer",
              "admin": "Administrator"}
    flash(f"{row['name']} is now: {labels[role]}.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:uid>/delete", methods=["POST"])
@admin_required
def user_delete(uid):
    if uid == g.user["id"]:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin.users"))
    conn = db.get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if row is None:
        abort(404)
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    flash(f"Deleted {row['name']} and their predictions.", "success")
    return redirect(url_for("admin.users"))
