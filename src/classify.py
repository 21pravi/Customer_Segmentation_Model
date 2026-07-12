"""
classify.py
===========
Notebook 4 territory. Where accuracy / precision / recall actually mean something.

TWO SUPERVISED MODELS, TWO DIFFERENT JOBS:

1. CHURN CLASSIFIER  (binary, real ground truth = the `churn` column)
   Predicts whether a subscriber leaves. Uses the 4 clustering features + the
   engineered ratios + the GMM segment as a categorical input. This is the model
   whose precision/recall you take to the retention team.

2. SEGMENT SURROGATE (multiclass, ground truth = GMM's own assignment)
   Predicts which segment a *new* customer belongs to, without re-fitting the
   GMM. Its "accuracy" measures how faithfully a fast tree reproduces the GMM's
   decision boundary - it is a fidelity metric, not a truth metric. Useful, but
   do not present it to the business as "our segmentation is 97% accurate".
   Nothing is 97% accurate about an unsupervised partition.

On threshold choice: the default 0.5 cut is almost never the right operating
point for churn. Retention budget is finite; you want the threshold that
maximises expected saved revenue, or at minimum the F1-optimal or
recall-constrained point. We report the sweep and pick F1-optimal.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


# ---------------------------------------------------------------- churn
def build_churn_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Feature matrix for churn. The GMM segment enters as one-hot: it is a
    behavioural summary the raw columns do not fully express, and trees pick it
    up as a cheap interaction term.
    """
    feats = C.CLUSTER_FEATURES + C.CHURN_EXTRA_FEATURES
    X = df[feats].copy()
    seg = pd.get_dummies(df["segment"], prefix="seg", dtype="int8")
    X = pd.concat([X, seg], axis=1)
    y = df["churn"].astype(int)
    return X, y, list(X.columns)


def train_churn_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit three models and let the ROC-AUC decide. No mystery, no hand-waving."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, random_state=C.RANDOM_STATE, class_weight="balanced"
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=13,
            min_samples_leaf=25,
            n_jobs=-1,
            random_state=C.RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
        # Histogram-based boosting, not the classic GradientBoostingClassifier.
        # Same algorithm family; it bins features once instead of sorting them at
        # every split, which is ~15x faster here and scales to the 57M-row case
        # where the exact implementation simply does not finish.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.08,
            random_state=C.RANDOM_STATE,
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "model": model,
            "auc": float(roc_auc_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, pred)),
            "f1": float(f1_score(yte, pred)),
            "proba": proba,
        }

    best = max(results, key=lambda k: results[k]["auc"])
    return {
        "results": results,
        "best_name": best,
        "best_model": results[best]["model"],
        "splits": (Xtr, Xte, ytr, yte),
        "best_proba": results[best]["proba"],
    }


def optimal_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds; return the F1-maximising cut and the full curve."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more prec/rec than thr
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    best_i = int(np.argmax(f1))
    curve = pd.DataFrame({"threshold": thr, "precision": prec, "recall": rec, "f1": f1})
    return float(thr[best_i]), curve


def full_metrics(y_true, proba, threshold: float) -> dict:
    """Every score worth reporting for a binary classifier, at one threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"negatives": int(tn + fp), "positives": int(fn + tp)},
    }


def cross_validate_churn(model, X, y) -> dict:
    """
    Stratified k-fold on a subsample. The point of CV here is a *variance*
    estimate - how much does AUC move between folds - and that estimate is
    already tight at 60k rows. Refitting a forest five times over the full base
    would buy a third decimal place at ten times the CPU.
    """
    if len(X) > C.CV_SUBSAMPLE:
        idx = (
            pd.Series(range(len(X)))
            .sample(C.CV_SUBSAMPLE, random_state=C.RANDOM_STATE)
            .to_numpy()
        )
        X, y = X.iloc[idx], y.iloc[idx]

    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return {
        "folds": C.CV_FOLDS,
        "cv_rows": int(len(X)),
        "roc_auc_mean": float(auc.mean()),
        "roc_auc_std": float(auc.std()),
        "roc_auc_folds": [float(v) for v in auc],
        "f1_mean": float(f1.mean()),
        "f1_std": float(f1.std()),
    }


# ---------------------------------------------------------------- segment surrogate
def train_segment_surrogate(df: pd.DataFrame) -> dict:
    """
    Multiclass model that reproduces the GMM's partition from raw features.
    Ground truth = GMM label. High accuracy here means "the tree learned the
    GMM's boundary", which is exactly what we want for cheap real-time scoring.
    """
    X = df[C.CLUSTER_FEATURES]
    y = df["segment"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=10, n_jobs=-1,
        random_state=C.RANDOM_STATE,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    labels = sorted(y.unique())
    return {
        "model": clf,
        "labels": labels,
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_precision": float(precision_score(yte, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yte, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yte, pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yte, pred)),
        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
        "per_class": classification_report(
            yte, pred, labels=labels, output_dict=True, zero_division=0
        ),
        "report_text": classification_report(yte, pred, labels=labels, zero_division=0),
    }
