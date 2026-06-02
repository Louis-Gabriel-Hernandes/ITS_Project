from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from .utils import ensure_dir, write_json


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def compute_metrics(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict[str, object]:
    proba = predict_proba(model, X)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, zero_division=0, output_dict=True),
    }


def evaluate_models(models: Dict[str, object], X_train, y_train, X_val, y_val) -> Dict[str, Dict]:
    results = {}
    for name, model in models.items():
        if name == "loss_history":
            continue
        results[name] = {
            "train": compute_metrics(model, X_train, y_train),
            "validation": compute_metrics(model, X_val, y_val),
        }
    return results


def save_loss_curve(loss_history: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history["epoch"], loss_history["train_loss"], label="Train loss")
    plt.plot(loss_history["epoch"], loss_history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_auc_curve(model, X, y, outdir: str | Path) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "auc_curve.png"
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_precision_recall_curve(model, X, y, outdir: str | Path) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "precision_recall_curve.png"
    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_estimator(model, X, y)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_confusion_matrix(model, X, y, outdir: str | Path, threshold: float = 0.5) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "confusion_matrix.png"
    proba = predict_proba(model, X)
    pred = (proba >= threshold).astype(int)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Struggle", "Success"])
    disp.ax_.set_title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_calibration_curve(model, X, y, outdir: str | Path) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "calibration_curve.png"
    proba = predict_proba(model, X)
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_feature_importance(model, X, y, feature_names: list[str], outdir: str | Path) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "feature_importance.png"

    importances = None
    fitted = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    if hasattr(fitted, "feature_importances_"):
        importances = fitted.feature_importances_
    elif hasattr(fitted, "coef_"):
        importances = np.abs(fitted.coef_).ravel()

    if importances is None:
        result = permutation_importance(model, X, y, n_repeats=8, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    order = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(9, 6))
    plt.barh(np.array(feature_names)[order][::-1], np.array(importances)[order][::-1])
    plt.xlabel("Importance")
    plt.title("Feature Importance / XAI")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_probability_histogram(model, X, y, outdir: str | Path) -> Path:
    outdir = ensure_dir(outdir)
    path = outdir / "predicted_probability_histogram.png"
    proba = predict_proba(model, X)
    plt.figure(figsize=(8, 5))
    plt.hist(proba[y.to_numpy() == 0], bins=20, alpha=0.7, label="Actual struggle")
    plt.hist(proba[y.to_numpy() == 1], bins=20, alpha=0.7, label="Actual success")
    plt.xlabel("Predicted probability of success")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_all_figures(model, X_val, y_val, loss_history, feature_names, outdir: str | Path) -> Dict[str, str]:
    outdir = ensure_dir(outdir)
    paths = {
        "loss_curve": save_loss_curve(loss_history, outdir),
        "auc_curve": save_auc_curve(model, X_val, y_val, outdir),
        "confusion_matrix": save_confusion_matrix(model, X_val, y_val, outdir),
        "precision_recall_curve": save_precision_recall_curve(model, X_val, y_val, outdir),
        "calibration_curve": save_calibration_curve(model, X_val, y_val, outdir),
        "feature_importance": save_feature_importance(model, X_val, y_val, feature_names, outdir),
        "predicted_probability_histogram": save_probability_histogram(model, X_val, y_val, outdir),
    }
    return {k: str(v) for k, v in paths.items()}


def leakage_diagnostic_text(safe_auc: float, leakage_auc: Optional[float]) -> str:
    if leakage_auc is None:
        return "Leakage audit was not run."
    gap = leakage_auc - safe_auc
    if gap > 0.08:
        return (
            "Leakage audit warning: adding current-row Attempts/CorrectEventually increases AUC substantially. "
            "Those fields should not be used in the deployed predictive tutor because they summarize the current outcome."
        )
    return "Leakage audit did not show a large AUC jump, but current-row outcome variables remain excluded from the deployed model."
