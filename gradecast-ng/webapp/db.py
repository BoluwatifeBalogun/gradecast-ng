"""SQLite storage (the relational database named in Section 1.5)."""

import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from flask import current_app, g
from werkzeug.security import generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('student','officer','admin')),
    institution   TEXT,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    student_name     TEXT NOT NULL,
    student_ref      TEXT,
    institution_type TEXT,
    state            TEXT,
    zone             TEXT,
    inputs_json      TEXT NOT NULL,
    label            TEXT NOT NULL,
    label_index      INTEGER NOT NULL,
    confidence       REAL NOT NULL,
    proba_json       TEXT NOT NULL,
    votes_json       TEXT NOT NULL,
    risk             TEXT NOT NULL,
    model_version    TEXT,
    is_sample        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT NOT NULL,
    stored_name   TEXT NOT NULL,
    rows          INTEGER NOT NULL,
    missing_cells INTEGER NOT NULL DEFAULT 0,
    is_builtin    INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 0,
    uploaded_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
    dataset_name  TEXT,
    started_by    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,
    accuracy      REAL,
    f1_macro      REAL,
    rows_balanced INTEGER,
    model_version TEXT,
    message       TEXT
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db():
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def save_prediction(conn, user_id, record, features, result, is_sample=0,
                    created_at=None):
    cur = conn.execute(
        """INSERT INTO predictions (user_id, student_name, student_ref,
               institution_type, state, zone, inputs_json, label, label_index,
               confidence, proba_json, votes_json, risk, model_version,
               is_sample, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, record["student_name"], record.get("student_ref"),
         record.get("institution_type"), record.get("state"),
         record.get("zone"), json.dumps(features), result["label"],
         result["index"], result["confidence"], json.dumps(result["proba"]),
         json.dumps({"votes": result["votes"], "agree": result["agree"]}),
         result["risk"], result["model_version"], is_sample,
         created_at or now()))
    return cur.lastrowid


def init_db(app):
    """Create tables and first-run records."""
    from ml import pipeline
    conn = connect(app.config["DATABASE"])
    conn.executescript(SCHEMA)

    if not conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        for name, email, password, role, inst in [
            ("System Administrator", "admin@gradecast.ng", "Admin@2026",
             "admin", "GradeCast NG"),
            ("Demo Academic Officer", "officer@gradecast.ng", "Officer@2026",
             "officer", "Yaba College of Technology"),
            ("Demo Student", "student@gradecast.ng", "Student@2026",
             "student", "Yaba College of Technology"),
        ]:
            conn.execute(
                "INSERT INTO users (name,email,password_hash,role,institution,"
                "created_at) VALUES (?,?,?,?,?,?)",
                (name, email, generate_password_hash(password), role, inst,
                 now()))

    if not conn.execute("SELECT 1 FROM datasets LIMIT 1").fetchone():
        try:
            df = pipeline.load_dataset(pipeline.DEFAULT_DATASET)
            _, report, _ = pipeline.clean(df)
            conn.execute(
                "INSERT INTO datasets (original_name, stored_name, rows,"
                " missing_cells, is_builtin, is_active, created_at)"
                " VALUES (?,?,?,?,1,1,?)",
                (pipeline.DEFAULT_DATASET.name, pipeline.DEFAULT_DATASET.name,
                 report["rows_raw"], report["missing_cells"], now()))
        except Exception as exc:  # the app still runs; admin can upload
            app.logger.warning("Built-in dataset not registered: %s", exc)
    conn.commit()
    conn.close()


def add_sample_cohort(app, count=48, seed=7):
    """Score real rows from the dataset under invented Nigerian names so the
    dashboard has something to show at a demonstration. Clearly flagged as
    sample records and removable from the admin area."""
    import pandas as pd
    from ml import pipeline, schema, service
    from . import nigeria

    if not service.ready():
        return 0
    conn = connect(app.config["DATABASE"])
    officer = conn.execute(
        "SELECT id FROM users WHERE role='officer' ORDER BY id LIMIT 1"
    ).fetchone()
    if officer is None:
        conn.close()
        return 0
    rng = random.Random(seed)
    df, _, _ = pipeline.clean(pipeline.load_dataset(pipeline.DEFAULT_DATASET))
    rows = df.sample(n=count, random_state=seed)
    start = datetime.now(timezone.utc) - timedelta(days=27)
    added = 0
    for i, (_, row) in enumerate(rows.iterrows()):
        features = {}
        for f in schema.FEATURES:
            v = row[f["name"]]
            features[f["name"]] = int(v) if f["kind"] == "num" else str(v)
        result = service.predict(features)
        state = rng.choice(nigeria.STATES)
        record = {
            "student_name": f"{rng.choice(nigeria.SAMPLE_FIRST)} "
                            f"{rng.choice(nigeria.SAMPLE_LAST)}",
            "student_ref": f"SAMPLE/{2026}/{1000 + i}",
            "institution_type": rng.choice(nigeria.INSTITUTION_TYPES),
            "state": state, "zone": nigeria.STATE_ZONE[state],
        }
        when = start + timedelta(days=i * 27 / count,
                                 hours=rng.randint(8, 16),
                                 minutes=rng.randint(0, 59))
        save_prediction(conn, officer["id"], record, features, result,
                        is_sample=1,
                        created_at=when.isoformat(timespec="seconds"))
        added += 1
    conn.commit()
    conn.close()
    return added
