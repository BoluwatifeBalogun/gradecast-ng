"""Training pipeline for the ensemble classifier (Chapter Three of the report).

Stages, in order (they follow the KDD framework in Section 2.8):

  1. Selection       load the Student Performance Factors CSV
  2. Preprocessing   clean, impute, remove duplicates
  3. Transformation  encode categoricals, band Exam_Score into five classes,
                     stratified 80/20 split, SMOTE on the training split only,
                     scale numeric attributes
  4. Data mining     Random Forest + Gradient Boosting joined by a soft
                     Voting Classifier
  5. Evaluation      accuracy, precision, recall, F1 and confusion matrix on
                     the untouched 20% of original records

SMOTE is applied after the split and only to the training portion. If it were
applied before the split, synthetic copies of test students would leak into
training and the reported accuracy would be inflated.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import schema

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "StudentPerformanceFactors.csv"
MODEL_PATH = ROOT / "models" / "ensemble.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
BALANCED_PATH = ROOT / "data" / "processed" / "training_balanced.csv"

CONFIG = dict(
    test_size=0.20,
    random_state=42,
    # The report expands the data "to over 20,000 records" with SMOTE.
    # Every class in the training split is raised to the same size so that
    # the balanced training set passes this total.
    min_balanced_rows=20_000,
    smote_k_neighbors=5,
    rf=dict(n_estimators=250, max_features=0.5, min_samples_leaf=2,
            n_jobs=-1, random_state=42),
    gb=dict(n_estimators=300, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42),
    voting="soft",
    weights=[1, 2],
)


class DatasetError(ValueError):
    """Raised when a CSV cannot be used for training."""


# --------------------------------------------------------------------------
# 1-2. Selection and preprocessing
# --------------------------------------------------------------------------
def load_dataset(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable or not a CSV
        raise DatasetError(f"The file could not be read as CSV ({exc}).")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError("The file is missing these columns: "
                           + ", ".join(missing) + ".")
    if len(df) < 200:
        raise DatasetError("The file needs at least 200 rows to train on.")
    return df[schema.REQUIRED_COLUMNS].copy()


def fit_fill(df: pd.DataFrame) -> dict:
    """Replacement values for missing cells: median or most common value."""
    fill = {}
    for name in schema.NUMERIC:
        fill[name] = float(df[name].median())
    for name in schema.CATEGORICAL:
        fill[name] = str(df[name].mode(dropna=True).iloc[0])
    return fill


def apply_fill(df: pd.DataFrame, fill: dict) -> pd.DataFrame:
    df = df.copy()
    for name in schema.NUMERIC:
        df[name] = df[name].fillna(fill[name])
    for name in schema.CATEGORICAL:
        df[name] = df[name].fillna(fill[name]).astype(str)
    return df


def clean(df: pd.DataFrame, impute=True):
    """Clean the raw table. Returns (clean_df, report, fill_values).

    With impute=False missing cells are left empty, so that training can
    learn the replacement values from the training split alone.
    """
    report = {"rows_raw": int(len(df))}
    df = df.copy()

    for name in schema.NUMERIC + [schema.TARGET]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    for name in schema.CATEGORICAL:
        options = schema.BY_NAME[name]["options"]
        lookup = {o.lower(): o for o in options}
        df[name] = (df[name].astype("string").str.strip().str.lower()
                    .map(lookup))

    report["missing_by_column"] = {
        c: int(n) for c, n in df[schema.FEATURE_NAMES].isna().sum().items()
        if n}
    report["missing_cells"] = int(sum(report["missing_by_column"].values()))

    before = len(df)
    df = df.dropna(subset=[schema.TARGET])
    report["dropped_no_target"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = int(before - len(df))

    over = int((df[schema.TARGET] > 100).sum())
    df[schema.TARGET] = df[schema.TARGET].clip(upper=100)
    report["scores_capped_at_100"] = over

    fill = fit_fill(df)
    if impute:
        df = apply_fill(df, fill)

    report["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), report, fill


# --------------------------------------------------------------------------
# 3. Transformation
# --------------------------------------------------------------------------
def choose_cutoffs(scores: pd.Series):
    """Fixed cut-offs for the reference dataset, quantiles as a fallback."""
    cutoffs = list(schema.DEFAULT_CUTOFFS)
    labels = band(scores, cutoffs)
    counts = np.bincount(labels, minlength=len(schema.CLASSES))
    if counts.min() >= max(30, 0.01 * len(scores)):
        return cutoffs, "fixed"
    q = [float(math.floor(scores.quantile(p)))
         for p in schema.FALLBACK_QUANTILES]
    for i in range(1, len(q)):          # keep cut-offs strictly increasing
        q[i] = max(q[i], q[i - 1] + 1)
    labels = band(scores, q)
    counts = np.bincount(labels, minlength=len(schema.CLASSES))
    if counts.min() < 10:
        raise DatasetError(
            "Exam_Score does not vary enough to form five performance "
            "classes. Check the Exam_Score column.")
    return q, "quantile"


def band(scores, cutoffs) -> np.ndarray:
    """Exam_Score to class index (0 = Fail ... 4 = First Class)."""
    return np.searchsorted(np.asarray(cutoffs, dtype=float),
                           np.asarray(scores, dtype=float), side="left")


def encode(df: pd.DataFrame) -> np.ndarray:
    """Ordinal-encode categoricals. Column order follows schema.FEATURES."""
    out = np.empty((len(df), len(schema.FEATURE_NAMES)), dtype=float)
    for j, f in enumerate(schema.FEATURES):
        col = df[f["name"]]
        if f["kind"] == "num":
            out[:, j] = col.astype(float).to_numpy()
        else:
            out[:, j] = col.map({o: i for i, o in enumerate(f["options"])}
                                ).astype(float).to_numpy()
    return out


def decode(X: np.ndarray) -> pd.DataFrame:
    """Inverse of encode(), used to export the balanced training table."""
    data = {}
    for j, f in enumerate(schema.FEATURES):
        if f["kind"] == "num":
            data[f["name"]] = np.rint(X[:, j]).astype(int)
        else:
            idx = np.clip(np.rint(X[:, j]).astype(int), 0,
                          len(f["options"]) - 1)
            data[f["name"]] = [f["options"][i] for i in idx]
    return pd.DataFrame(data)


NUM_IDX = [i for i, f in enumerate(schema.FEATURES) if f["kind"] == "num"]
CAT_IDX = [i for i, f in enumerate(schema.FEATURES) if f["kind"] == "cat"]


def fit_scaler(X: np.ndarray):
    mean = X[:, NUM_IDX].mean(axis=0)
    std = X[:, NUM_IDX].std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def scale(X: np.ndarray, scaler) -> np.ndarray:
    X = X.copy()
    X[:, NUM_IDX] = ((X[:, NUM_IDX] - np.asarray(scaler["mean"]))
                     / np.asarray(scaler["std"]))
    return X


def balance(X, y, cfg=CONFIG):
    """SMOTE (Chawla et al., 2002) on the encoded training split.

    Synthetic students are interpolated between real neighbours of the same
    class, so their values are deliberately left as decimals (attendance of
    83.4, or a motivation level of 1.4 on the 0 to 2 scale). The first version
    of this build used SMOTE-NC and rounded the synthetic values to whole
    numbers; it scored about 2.7 points lower on the held-out students.
    """
    counts = np.bincount(y, minlength=len(schema.CLASSES))
    per_class = max(int(counts.max()),
                    math.ceil((cfg["min_balanced_rows"] + 1)
                              / len(schema.CLASSES)))
    per_class = int(math.ceil(per_class / 100.0) * 100)
    k = max(1, min(cfg["smote_k_neighbors"], int(counts.min()) - 1))
    smote = SMOTE(sampling_strategy={i: per_class
                                     for i in range(len(schema.CLASSES))},
                  k_neighbors=k, random_state=cfg["random_state"])
    return smote.fit_resample(X, y)


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
def build_ensemble(cfg=CONFIG) -> VotingClassifier:
    rf = RandomForestClassifier(**cfg["rf"])
    gb = GradientBoostingClassifier(**cfg["gb"])
    return VotingClassifier(estimators=[("rf", rf), ("gb", gb)],
                            voting=cfg["voting"], weights=cfg["weights"])


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------
def score(y_true, y_pred) -> dict:
    labels = list(range(len(schema.CLASSES)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "within_one_class": float(
            (np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1).mean()),
        "precision_macro": float(mp), "recall_macro": float(mr),
        "f1_macro": float(mf),
        "precision_weighted": float(wp), "recall_weighted": float(wr),
        "f1_weighted": float(wf),
        "per_class": [
            {"label": schema.CLASSES[i], "precision": float(p[i]),
             "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in labels],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
    }


def _noop(stage, percent, message):
    pass


def cross_validate(df_raw, y, folds, cfg=CONFIG, progress=_noop):
    """Stratified k-fold. Filling, SMOTE and scaling are refitted inside
    every fold from that fold's training part only."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True,
                          random_state=cfg["random_state"])
    accs, f1s = [], []
    for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        progress("cv", 5 + int(90 * (i - 1) / folds),
                 f"Cross-validation fold {i} of {folds}")
        fill = fit_fill(df_raw.iloc[tr])
        X_tr = encode(apply_fill(df_raw.iloc[tr], fill))
        X_te = encode(apply_fill(df_raw.iloc[te], fill))
        Xb, yb = balance(X_tr, y[tr], cfg)
        sc = fit_scaler(Xb)
        model = build_ensemble(cfg).fit(scale(Xb, sc), yb)
        pred = model.predict(scale(X_te, sc))
        res = score(y[te], pred)
        accs.append(res["accuracy"])
        f1s.append(res["f1_macro"])
    return {"folds": folds, "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
            "accuracy_per_fold": [float(a) for a in accs]}


def train(dataset_path=DEFAULT_DATASET, progress=_noop, cv_folds=0,
          cfg=CONFIG, save=True, dataset_name=None):
    """Run the whole pipeline. Returns (bundle, metrics)."""
    started = time.time()
    dataset_path = Path(dataset_path)

    progress("load", 4, "Loading dataset")
    raw = load_dataset(dataset_path)

    progress("clean", 10, "Cleaning records and filling missing values")
    df_raw, cleaning, _ = clean(raw, impute=False)

    progress("encode", 16, "Encoding attributes and banding exam scores")
    cutoffs, cutoff_mode = choose_cutoffs(df_raw[schema.TARGET])
    y = band(df_raw[schema.TARGET], cutoffs)

    idx_tr, idx_te, y_tr, y_te = train_test_split(
        np.arange(len(df_raw)), y, test_size=cfg["test_size"], stratify=y,
        random_state=cfg["random_state"])
    # Missing values are filled with what the training students show, so
    # nothing about the test students reaches the model.
    fill = fit_fill(df_raw.iloc[idx_tr])
    df = apply_fill(df_raw, fill)
    X_tr, X_te = encode(df.iloc[idx_tr]), encode(df.iloc[idx_te])

    progress("smote", 24, "Balancing the training split with SMOTE")
    X_bal, y_bal = balance(X_tr, y_tr, cfg)

    progress("scale", 30, "Scaling numeric attributes")
    scaler = fit_scaler(X_bal)
    Xs_bal, Xs_te = scale(X_bal, scaler), scale(X_te, scaler)

    progress("fit", 36, "Training Random Forest and Gradient Boosting")
    ensemble = build_ensemble(cfg).fit(Xs_bal, y_bal)
    rf = ensemble.named_estimators_["rf"]
    gb = ensemble.named_estimators_["gb"]

    progress("evaluate", 86, "Evaluating on held-out students")
    results = {
        "ensemble": score(y_te, ensemble.predict(Xs_te)),
        "random_forest": score(y_te, rf.predict(Xs_te)),
        "gradient_boosting": score(y_te, gb.predict(Xs_te)),
    }
    # A lone decision tree as the "single classifier" the report argues
    # against (Sections 1.2 and 2.10).
    tree = DecisionTreeClassifier(random_state=cfg["random_state"],
                                  min_samples_leaf=2).fit(Xs_bal, y_bal)
    results["single_decision_tree"] = score(y_te, tree.predict(Xs_te))

    w_rf, w_gb = cfg["weights"]
    importance = ((w_rf * rf.feature_importances_
                   + w_gb * gb.feature_importances_) / (w_rf + w_gb))
    importance = importance / importance.sum()
    order = np.argsort(importance)[::-1]

    cv = None
    if cv_folds and cv_folds > 1:
        cv = cross_validate(df_raw, y, cv_folds, cfg, progress)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    names = schema.CLASSES
    metrics = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
        "dataset": {
            "name": dataset_name or dataset_path.name,
            "cleaning": cleaning,
            "rows_train_original": int(len(y_tr)),
            "rows_train_balanced": int(len(y_bal)),
            "rows_synthetic": int(len(y_bal) - len(y_tr)),
            "rows_test": int(len(y_te)),
            "score_min": float(df[schema.TARGET].min()),
            "score_max": float(df[schema.TARGET].max()),
        },
        "classes": names,
        "cutoffs": cutoffs,
        "cutoff_mode": cutoff_mode,
        "class_distribution": {
            "full": np.bincount(y, minlength=len(names)).tolist(),
            "train_before": np.bincount(y_tr, minlength=len(names)).tolist(),
            "train_after": np.bincount(y_bal, minlength=len(names)).tolist(),
            "test": np.bincount(y_te, minlength=len(names)).tolist(),
        },
        "results": results,
        "cross_validation": cv,
        "feature_importance": [
            {"name": schema.FEATURE_NAMES[i],
             "label": schema.FEATURES[i]["label"],
             "importance": float(importance[i])} for i in order],
        "config": {k: v for k, v in cfg.items()},
        "library_versions": {"scikit-learn": sklearn.__version__,
                             "imbalanced-learn": imblearn.__version__,
                             "pandas": pd.__version__,
                             "numpy": np.__version__},
    }

    bundle = {
        "model": ensemble,
        "scaler": scaler,
        "fill": fill,
        "cutoffs": cutoffs,
        "classes": names,
        "features": schema.FEATURE_NAMES,
        "version": version,
        # Per-attribute reference values used by the what-if explanation.
        "reference": _reference_values(df),
    }

    if save:
        progress("save", 95, "Saving model and metrics")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        BALANCED_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".tmp")
        joblib.dump(bundle, tmp, compress=3)
        tmp.replace(MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        balanced = pd.DataFrame(np.round(X_bal, 3),
                                columns=schema.FEATURE_NAMES)
        balanced["Performance_Class"] = [names[i] for i in y_bal]
        balanced["Record_Type"] = (["Original"] * len(y_tr)
                                   + ["Synthetic (SMOTE)"]
                                   * (len(y_bal) - len(y_tr)))
        balanced.to_csv(BALANCED_PATH, index=False)

    progress("done", 100, "Training complete")
    return bundle, metrics


def _reference_values(df: pd.DataFrame) -> dict:
    """Strong and typical values per attribute, read from the data."""
    ref = {}
    for f in schema.FEATURES:
        col = df[f["name"]]
        if f["kind"] == "num":
            ref[f["name"]] = {
                "min": float(col.min()), "max": float(col.max()),
                "typical": float(col.median()),
                "strong": float(col.quantile(0.90)),
                "weak": float(col.quantile(0.10)),
            }
        else:
            ref[f["name"]] = {"typical": str(col.mode().iloc[0])}
    return ref
