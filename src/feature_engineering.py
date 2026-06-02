from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

SAFE_FEATURE_COLUMNS = [
    "phase_code",
    "assignment_order_norm",
    "problem_order_norm",
    "estimated_difficulty",
    "student_prior_items",
    "student_prior_success_rate",
    "student_prior_mean_attempts",
    "student_prior_incorrect_rate",
    "student_recent_success_rate",
    "student_recent_mean_attempts",
    "student_current_streak",
    "student_prior_compile_success_rate",
]

LEAKAGE_AUDIT_COLUMNS = SAFE_FEATURE_COLUMNS + [
    "Attempts",
    "CorrectEventually",
]


@dataclass
class FeatureBundle:
    frame: pd.DataFrame
    features: List[str]
    target: str
    groups: pd.Series


def _streak_before(values: pd.Series) -> pd.Series:
    streaks = []
    streak = 0
    for v in values.astype(int).tolist():
        streaks.append(streak)
        if v == 1:
            streak = max(streak, 0) + 1
        else:
            streak = min(streak, 0) - 1
    return pd.Series(streaks, index=values.index)


def _rolling_shifted_mean(values: pd.Series, window: int, default: float) -> pd.Series:
    return (
        values.shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
        .fillna(default)
    )


def add_safe_chronological_features(labels: pd.DataFrame) -> pd.DataFrame:
    """Create features using only information available before each row.

    The row's own Attempts and CorrectEventually are intentionally not used in
    SAFE_FEATURE_COLUMNS because they are direct outcome/attempt signals for the
    current problem and can leak the target definition.
    """
    df = labels.copy()
    df["Label"] = df["Label"].astype(int)
    df["CorrectEventually"] = df["CorrectEventually"].astype(int)
    df["phase_code"] = df["Phase"].map({"early": 0, "late": 1}).fillna(0).astype(int)

    assignment_rank = df["AssignmentID"].rank(method="dense").astype(int)
    problem_rank = df["ProblemID"].rank(method="dense").astype(int)
    df["assignment_order_norm"] = (assignment_rank - assignment_rank.min()) / max(
        1, assignment_rank.max() - assignment_rank.min()
    )
    df["problem_order_norm"] = (problem_rank - problem_rank.min()) / max(
        1, problem_rank.max() - problem_rank.min()
    )

    # A deterministic, non-target difficulty proxy based on curricular order.
    # This is not derived from the current row's outcome.
    df["estimated_difficulty"] = np.clip(
        np.ceil(1 + 4 * (0.55 * df["assignment_order_norm"] + 0.45 * df["problem_order_norm"])),
        1,
        5,
    ).astype(int)

    df = df.sort_values(
        ["SubjectID", "phase_code", "AssignmentID", "ProblemID"],
        kind="mergesort",
    ).reset_index(drop=True)

    grp = df.groupby("SubjectID", sort=False)
    df["student_prior_items"] = grp.cumcount()
    df["student_prior_correct"] = grp["Label"].cumsum() - df["Label"]
    df["student_prior_attempt_sum"] = grp["Attempts"].cumsum() - df["Attempts"]

    df["student_prior_success_rate"] = (
        df["student_prior_correct"] / df["student_prior_items"].replace(0, np.nan)
    ).fillna(0.5)
    df["student_prior_mean_attempts"] = (
        df["student_prior_attempt_sum"] / df["student_prior_items"].replace(0, np.nan)
    ).fillna(df["Attempts"].median())
    df["student_prior_incorrect_rate"] = 1.0 - df["student_prior_success_rate"]
    df["student_recent_success_rate"] = grp["Label"].transform(
        lambda s: _rolling_shifted_mean(s, window=5, default=0.5)
    )
    df["student_recent_mean_attempts"] = grp["Attempts"].transform(
        lambda s: _rolling_shifted_mean(s, window=5, default=float(df["Attempts"].median()))
    )
    df["student_current_streak"] = grp["Label"].transform(_streak_before)

    # Placeholder for possible prior compile feature. It is kept model-compatible
    # with the app, but defaults to neutral when no safe prior compile history is used.
    df["student_prior_compile_success_rate"] = 0.5

    return df


def make_feature_bundle(labels: pd.DataFrame, leakage_audit: bool = False) -> FeatureBundle:
    frame = add_safe_chronological_features(labels)
    features = LEAKAGE_AUDIT_COLUMNS if leakage_audit else SAFE_FEATURE_COLUMNS
    return FeatureBundle(
        frame=frame,
        features=features,
        target="Label",
        groups=frame["SubjectID"],
    )


def split_xy(bundle: FeatureBundle) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = bundle.frame[bundle.features].copy()
    y = bundle.frame[bundle.target].astype(int).copy()
    groups = bundle.groups.copy()
    return X, y, groups


def app_feature_row(
    *,
    difficulty: int,
    prior_items: int,
    prior_success_rate: float,
    prior_mean_attempts: float,
    recent_success_rate: float,
    current_streak: int,
    phase_code: int = 1,
    assignment_order_norm: float = 0.5,
    problem_order_norm: float = 0.5,
    prior_compile_success_rate: float = 0.5,
) -> pd.DataFrame:
    row = {
        "phase_code": phase_code,
        "assignment_order_norm": assignment_order_norm,
        "problem_order_norm": problem_order_norm,
        "estimated_difficulty": int(difficulty),
        "student_prior_items": int(prior_items),
        "student_prior_success_rate": float(prior_success_rate),
        "student_prior_mean_attempts": float(prior_mean_attempts),
        "student_prior_incorrect_rate": float(1.0 - prior_success_rate),
        "student_recent_success_rate": float(recent_success_rate),
        "student_recent_mean_attempts": float(prior_mean_attempts),
        "student_current_streak": int(current_streak),
        "student_prior_compile_success_rate": float(prior_compile_success_rate),
    }
    return pd.DataFrame([row], columns=SAFE_FEATURE_COLUMNS)
