"""Prediction service used by the web application.

Only the lightweight path runs at prediction time: validate, encode, scale,
predict. The dataset is not reloaded and SMOTE is not involved; both belong
to training (see pipeline.py).
"""

from __future__ import annotations

import json
import threading

import joblib
import numpy as np
import pandas as pd

from . import advice, pipeline, schema

_lock = threading.Lock()
_bundle = None
_metrics = None
_load_error = None


def load(force=False):
    """Load the saved model once. Returns True when a model is ready."""
    global _bundle, _metrics, _load_error
    with _lock:
        if _bundle is not None and not force:
            return True
        try:
            bundle = joblib.load(pipeline.MODEL_PATH)
            if bundle.get("features") != schema.FEATURE_NAMES:
                raise ValueError("saved model does not match the schema")
            _bundle = bundle
            _metrics = json.loads(pipeline.METRICS_PATH.read_text())
            _load_error = None
            return True
        except FileNotFoundError:
            _bundle, _metrics = None, None
            _load_error = "No trained model was found."
        except Exception as exc:  # version mismatch, corrupt file ...
            _bundle, _metrics = None, None
            _load_error = f"The saved model could not be loaded ({exc})."
        return False


def ready():
    return _bundle is not None or load()


def load_error():
    return _load_error


def metrics():
    ready()
    return _metrics


def version():
    return _bundle["version"] if ready() else None


def _matrix(rows):
    df = pd.DataFrame(rows, columns=schema.FEATURE_NAMES)
    return pipeline.scale(pipeline.encode(df), _bundle["scaler"])


def _expected(proba):
    """Position on the class ladder, 0 (Fail) to 4 (First Class)."""
    return proba @ np.arange(proba.shape[-1])


def predict(features: dict) -> dict:
    """Predict one validated student profile."""
    if not ready():
        raise RuntimeError(_load_error or "Model not available.")
    model = _bundle["model"]
    X = _matrix([features])
    proba = model.predict_proba(X)[0]
    rf = model.named_estimators_["rf"].predict_proba(X)[0]
    gb = model.named_estimators_["gb"].predict_proba(X)[0]
    idx = int(np.argmax(proba))
    classes = _bundle["classes"]
    return {
        "label": classes[idx],
        "index": idx,
        "confidence": float(proba[idx]),
        "proba": [float(p) for p in proba],
        "ladder": float(_expected(proba)),
        "votes": {
            "random_forest": {"label": classes[int(np.argmax(rf))],
                              "index": int(np.argmax(rf)),
                              "proba": [float(p) for p in rf]},
            "gradient_boosting": {"label": classes[int(np.argmax(gb))],
                                  "index": int(np.argmax(gb)),
                                  "proba": [float(p) for p in gb]},
        },
        "agree": bool(np.argmax(rf) == np.argmax(gb)),
        "risk": risk_level(idx),
        "model_version": _bundle["version"],
    }


def out_of_range(features: dict) -> list:
    """Numeric inputs outside what the training data covers.

    Tree models cannot extrapolate: attendance of 40% is treated exactly
    like the lowest attendance in the data. The result page says so.
    """
    if not ready():
        return []
    notes = []
    for f in schema.FEATURES:
        ref = _bundle["reference"].get(f["name"], {})
        if f["kind"] != "num" or "min" not in ref:
            continue
        value = features.get(f["name"])
        if value is None:
            continue
        if value < ref["min"] or value > ref["max"]:
            edge = ref["min"] if value < ref["min"] else ref["max"]
            notes.append({"label": f["label"], "value": value,
                          "edge": int(edge), "unit": f.get("unit", ""),
                          "low": int(ref["min"]), "high": int(ref["max"])})
    return notes


def training_ranges() -> dict:
    if not ready():
        return {}
    return {name: (int(r["min"]), int(r["max"]))
            for name, r in _bundle["reference"].items() if "min" in r}


def risk_level(index: int) -> str:
    if index <= 1:
        return "At risk"
    if index == 2:
        return "Watch"
    return "On track"


# A suggestion is one realistic step, not a jump to the best possible value:
# a student at 61% attendance is shown what 76% would do, not 96%.
STEP = {"Hours_Studied": 6, "Attendance": 15, "Tutoring_Sessions": 2,
        "Physical_Activity": 2}


def _better_value(f, current, ref):
    """The value an attribute would take after one realistic improvement."""
    if f["kind"] == "num":
        if f["better"] == "high":
            ceiling = min(f["max"], round(ref["strong"]))
            if current >= ceiling:
                return None
            floor = max(current, ref.get("min", current))  # see out_of_range
            return int(min(ceiling, floor + STEP.get(f["name"], 1)))
        if f["better"] == "mid":
            target = round(ref["typical"])
            return target if abs(current - target) >= 1 else None
        return None
    options = f["options"]
    i = options.index(current)
    if f["better"] == "high":
        return options[i + 1] if i + 1 < len(options) else None
    if f["better"] == "low":
        return options[i - 1] if i > 0 else None
    return None


def explain(features: dict) -> dict:
    """What-if analysis around one student.

    Every attribute is swapped, one at a time, for (a) the value of a typical
    student and (b) an improved value, and the ensemble is asked again. The
    movement on the class ladder tells us what is driving the prediction and
    which changes would help most (chosen one after another, best first). This is a sensitivity analysis of the
    trained model, not a causal claim about the student.
    """
    if not ready():
        raise RuntimeError(_load_error or "Model not available.")
    model, ref = _bundle["model"], _bundle["reference"]
    classes = _bundle["classes"]

    rows, plan = [dict(features)], []
    for f in schema.FEATURES:
        name, current = f["name"], features[f["name"]]
        typical = ref[name]["typical"]
        if f["kind"] == "num":
            typical = round(typical)
        if typical != current:
            rows.append({**features, name: typical})
            plan.append(("typical", f, typical))
        if f["actionable"]:
            target = _better_value(f, current, ref[name])
            if target is not None:
                rows.append({**features, name: target})
                plan.append(("better", f, target))

    proba = model.predict_proba(_matrix(rows))
    ladder = _expected(proba)
    base = float(ladder[0])

    drivers, levers = [], []
    for i, (kind, f, value) in enumerate(plan, start=1):
        delta = float(ladder[i]) - base
        item = {"name": f["name"], "label": f["label"],
                "current": features[f["name"]], "value": value,
                "unit": f.get("unit", "")}
        if kind == "typical":
            # Positive effect: the student's own value beats the typical one.
            item["effect"] = -delta
            drivers.append(item)
        else:
            item["gain"] = delta
            item["new_label"] = classes[int(np.argmax(proba[i]))]
            item["advice"] = advice.for_feature(f["name"])
            levers.append(item)

    drivers.sort(key=lambda d: abs(d["effect"]), reverse=True)
    standalone = {l["name"]: l for l in levers}

    # Greedy sequence. When a student sits at the bottom of the ladder one
    # change on its own often cannot move the prediction (a floor effect), so
    # after the single best change is applied the remaining ones are tested
    # again on top of it, and so on. Each gain is the extra movement that
    # step adds.
    chosen, current, level = [], dict(features), base
    remaining = dict(standalone)
    while remaining and len(chosen) < 5:
        names = list(remaining)
        trial = [{**current, n: remaining[n]["value"]} for n in names]
        gains = _expected(model.predict_proba(_matrix(trial))) - level
        best = int(np.argmax(gains))
        if gains[best] <= 0.02:
            break
        step = remaining.pop(names[best])
        step["gain"] = float(gains[best])
        step["alone_label"] = step.pop("new_label")
        current[step["name"]] = step["value"]
        level += float(gains[best])
        chosen.append(step)
    levers = chosen

    combined = None
    if levers:
        p = model.predict_proba(_matrix([current]))[0]
        combined = {"label": classes[int(np.argmax(p))],
                    "index": int(np.argmax(p)),
                    "confidence": float(p.max()),
                    "changes": [l["label"] for l in levers]}

    return {"drivers": [d for d in drivers if abs(d["effect"]) >= 0.02][:8],
            "levers": levers, "combined": combined}
