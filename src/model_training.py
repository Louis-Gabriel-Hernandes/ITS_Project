from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from .utils import ensure_dir, set_global_seed


def subject_wise_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
        groups.iloc[train_idx].copy(),
        groups.iloc[test_idx].copy(),
    )


def build_baseline_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_final_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=4,
                    max_depth=12,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_sgd_loss_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    epochs: int = 35,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Train an SGD logistic model incrementally to obtain train/validation loss curves."""
    set_global_seed(random_state)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xva = scaler.transform(imputer.transform(X_val))
    ytr = y_train.to_numpy(dtype=int)
    yva = y_val.to_numpy(dtype=int)

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=0.0005,
        l1_ratio=0.05,
        class_weight=weight_dict,
        learning_rate="optimal",
        random_state=random_state,
    )
    rows = []
    for epoch in range(1, epochs + 1):
        X_epoch, y_epoch = shuffle(Xtr, ytr, random_state=random_state + epoch)
        clf.partial_fit(X_epoch, y_epoch, classes=classes)
        train_proba = np.clip(clf.predict_proba(Xtr)[:, 1], 1e-6, 1 - 1e-6)
        val_proba = np.clip(clf.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": log_loss(ytr, train_proba),
                "val_loss": log_loss(yva, val_proba),
                "train_auc": roc_auc_score(ytr, train_proba),
                "val_auc": roc_auc_score(yva, val_proba),
            }
        )
    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
            ("model", clf),
        ]
    )
    return pipeline, pd.DataFrame(rows)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> Dict[str, object]:
    baseline = build_baseline_model(random_state=random_state)
    final = build_final_model(random_state=random_state)
    baseline.fit(X_train, y_train)
    final.fit(X_train, y_train)
    loss_model, loss_history = train_sgd_loss_curves(
        X_train, y_train, X_val, y_val, random_state=random_state
    )
    return {
        "baseline_model": baseline,
        "final_model": final,
        "loss_model": loss_model,
        "loss_history": loss_history,
    }


def save_model_package(
    model: Pipeline,
    features: list[str],
    metrics: Dict,
    output_path: str | Path,
    threshold: float = 0.5,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    package = {
        "model": model,
        "feature_columns": features,
        "threshold": threshold,
        "metrics": metrics,
        "model_note": "Subject-wise split. Safe features exclude current-row Attempts and CorrectEventually.",
    }
    joblib.dump(package, output_path)
    return output_path
