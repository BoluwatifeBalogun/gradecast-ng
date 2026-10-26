"""Recommendations and drivers for a saved prediction.

Working them out takes a fraction of a second per student, too slow to
repeat for every row of a bulk export, so the result is saved with the
prediction and reused by the result page, the PDF report and the CSV
export. It is worked out again only when the model has been retrained
since it was saved.
"""

import json

from ml import service

# Raise this when the way recommendations are worked out changes, so saved
# ones are refreshed.
REVISION = "2"


def _key():
    return f"{service.version()}/r{REVISION}"


def save(conn, pid, explanation):
    conn.execute(
        "UPDATE predictions SET explain_json = ?, explain_version = ?"
        " WHERE id = ?", (json.dumps(explanation), _key(), pid))


def for_row(conn, row, commit=True):
    """Explanation for a prediction row, or None if no model is loaded."""
    keys = row.keys()
    stored = row["explain_json"] if "explain_json" in keys else None
    if stored and row["explain_version"] == _key():
        return json.loads(stored)
    if not service.ready():
        return json.loads(stored) if stored else None
    try:
        explanation = service.explain(json.loads(row["inputs_json"]))
    except Exception:  # an old record that no longer fits the schema
        return json.loads(stored) if stored else None
    save(conn, row["id"], explanation)
    if commit:
        conn.commit()
    return explanation


# ---------------------------------------------------------------------------
# Plain-text forms, shared by the CSV export and the PDF report.
# ---------------------------------------------------------------------------
def fmt_value(name, value):
    from ml import schema
    f = schema.BY_NAME.get(name, {})
    unit = f.get("unit", "") if f.get("kind") == "num" else ""
    if unit == "hrs":
        return f"{value} hrs"
    return f"{value}{unit}"


def as_text(explanation, scale_key):
    """'1. Raise class attendance (80% to 95%); 2. ...' for one cell."""
    from . import grading
    if not explanation:
        return ""
    levers = explanation.get("levers") or []
    if not levers:
        return "No change needed: the habits the student can change are " \
               "already strong."
    parts = [f"{i}. {l['advice']['title']} "
             f"({fmt_value(l['name'], l['current'])} to "
             f"{fmt_value(l['name'], l['value'])})"
             for i, l in enumerate(levers, start=1)]
    combined = explanation.get("combined")
    text = "; ".join(parts)
    if combined:
        text += (". If all are followed, the model expects: "
                 f"{grading.label(combined['index'], scale_key=scale_key)}.")
    return text
