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


# ---------------------------------------------------------------------------
# Two back ends, one interface.
#
# With no DATABASE_URL the app uses a SQLite file (laptop, defence demo).
# With DATABASE_URL set it uses PostgreSQL, so accounts and predictions
# survive on hosts that wipe local files, such as Render's free plan.
# The rest of the code writes SQLite-style SQL ("?" placeholders) and the
# small wrapper below translates it for PostgreSQL.
# ---------------------------------------------------------------------------
_pg = {"url": None, "pool": None}


def configure(app):
    """Choose the back end. Call once, before init_db."""
    url = app.config.get("DATABASE_URL") or None
    if _pg["pool"] is not None and _pg["url"] != url:
        _pg["pool"].close()
        _pg["pool"] = None
    _pg["url"] = url
    if url and _pg["pool"] is None:
        from psycopg_pool import ConnectionPool
        _pg["pool"] = ConnectionPool(
            url, min_size=1, max_size=4, timeout=20, open=True,
            check=ConnectionPool.check_connection)


def using_postgres():
    return _pg["url"] is not None


class _Row:
    """A result row readable by column name or position, like sqlite3.Row."""
    __slots__ = ("_keys", "_values")

    def __init__(self, keys, values):
        self._keys, self._values = keys, values

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        try:
            return self._values[self._keys.index(key)]
        except ValueError:
            raise KeyError(key)

    def keys(self):
        return list(self._keys)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _PgCursor:
    def __init__(self, cur, lastrowid=None):
        self._cur, self.lastrowid = cur, lastrowid
        self.rowcount = cur.rowcount
        self._keys = ([c.name for c in cur.description]
                      if cur.description else [])

    def fetchone(self):
        row = self._cur.fetchone()
        return None if row is None else _Row(self._keys, row)

    def fetchall(self):
        return [_Row(self._keys, r) for r in self._cur.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class _PgConnection:
    def __init__(self, raw, pool=None):
        self._raw, self._pool = raw, pool

    def execute(self, sql, params=()):
        text = sql.replace("%", "%%").replace("?", "%s")
        is_insert = text.lstrip().upper().startswith("INSERT")
        if is_insert and "RETURNING" not in text.upper():
            text += " RETURNING id"
        cur = self._raw.cursor()
        cur.execute(text, tuple(params))
        if is_insert:
            count = cur.rowcount
            new_id = cur.fetchone()[0]
            out = _PgCursor(cur, lastrowid=new_id)
            out.rowcount = count
            return out
        return _PgCursor(cur)

    def executescript(self, script):
        script = (script
                  .replace("INTEGER PRIMARY KEY AUTOINCREMENT",
                           "SERIAL PRIMARY KEY")
                  .replace(" REAL", " DOUBLE PRECISION"))
        cur = self._raw.cursor()
        for statement in script.split(";"):
            if statement.strip():
                cur.execute(statement)
        self._raw.commit()

    def commit(self):
        self._raw.commit()

    def close(self):
        if self._pool is not None:
            self._raw.rollback()        # drop anything left uncommitted
            self._pool.putconn(self._raw)
        else:
            self._raw.close()


def connect(path=None):
    """A stand-alone connection (start-up work and the training thread)."""
    if using_postgres():
        import psycopg
        return _PgConnection(psycopg.connect(_pg["url"]))
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db():
    if "db" not in g:
        if using_postgres():
            pool = _pg["pool"]
            g.db = _PgConnection(pool.getconn(), pool)
        else:
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
    """Create tables and first-run records. Returns True on a brand-new
    database (no users yet), which is when sample records are added."""
    from ml import pipeline
    conn = connect(app.config["DATABASE"])
    conn.executescript(SCHEMA)

    fresh = not conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
    if fresh:
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
    return fresh


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
