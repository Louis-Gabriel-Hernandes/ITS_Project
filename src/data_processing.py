from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .utils import ensure_dir, project_root

RAW_FILES = [
    "DatasetMetadata.csv",
    "MainTable.csv",
    "early.csv",
    "late.csv",
    "Subject.csv",
    "CodeStates.csv",
]


def raw_data_dir(root: Optional[str | Path] = None) -> Path:
    root = Path(root) if root is not None else project_root()
    return root / "data" / "raw"


def load_csv(
    filename: str,
    root: Optional[str | Path] = None,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    path = raw_data_dir(root) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, nrows=nrows, usecols=usecols, **kwargs)


def validate_raw_files(root: Optional[str | Path] = None) -> Dict[str, bool]:
    base = raw_data_dir(root)
    return {name: (base / name).exists() for name in RAW_FILES}


def inspect_raw_files(root: Optional[str | Path] = None, nrows: int = 5) -> Dict[str, Dict]:
    """Return lightweight schema and sample metadata for each CSV."""
    results: Dict[str, Dict] = {}
    base = raw_data_dir(root)
    for name in RAW_FILES:
        path = base / name
        if not path.exists():
            results[name] = {"exists": False}
            continue
        sample = pd.read_csv(path, nrows=nrows)
        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception:
            row_count = None
        results[name] = {
            "exists": True,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "rows_estimate": row_count,
            "columns": list(sample.columns),
            "dtypes": {c: str(t) for c, t in sample.dtypes.items()},
        }
    return results


def load_label_frames(root: Optional[str | Path] = None) -> pd.DataFrame:
    early = load_csv("early.csv", root=root)
    late = load_csv("late.csv", root=root)
    early["Phase"] = "early"
    late["Phase"] = "late"
    labels = pd.concat([early, late], ignore_index=True)
    labels["Label"] = labels["Label"].astype(bool).astype(int)
    labels["CorrectEventually"] = labels["CorrectEventually"].astype(bool).astype(int)
    return labels


def load_subjects(root: Optional[str | Path] = None) -> pd.DataFrame:
    subjects = load_csv("Subject.csv", root=root)
    return subjects


def build_event_aggregates(
    root: Optional[str | Path] = None,
    cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Aggregate MainTable event logs at the Subject/Assignment/Problem level.

    These aggregates are useful for exploratory analysis and leakage audits. For
    the thesis-safe predictive model, current-problem aggregates should not be
    used as features because they summarize behavior that happens during or after
    the target attempt.
    """
    root = Path(root) if root is not None else project_root()
    cache_path = root / "data" / "generated" / "main_event_aggregates.csv"
    if cache and cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    usecols = [
        "SubjectID",
        "AssignmentID",
        "ProblemID",
        "EventType",
        "Score",
        "Compile.Result",
        "Order",
    ]
    main = load_csv("MainTable.csv", root=root, usecols=usecols)
    main["is_run"] = (main["EventType"] == "Run.Program").astype(int)
    main["is_compile"] = (main["EventType"] == "Compile").astype(int)
    main["is_compile_error_event"] = (main["EventType"] == "Compile.Error").astype(int)
    main["is_compile_success"] = (main["Compile.Result"].astype(str) == "Success").astype(int)
    main["is_compile_error_result"] = (main["Compile.Result"].astype(str) == "Error").astype(int)
    main["has_score"] = main["Score"].notna().astype(int)
    main["score_filled"] = main["Score"].fillna(0)

    keys = ["SubjectID", "AssignmentID", "ProblemID"]
    agg = (
        main.groupby(keys, observed=True)
        .agg(
            event_count=("EventType", "size"),
            run_count=("is_run", "sum"),
            compile_count=("is_compile", "sum"),
            compile_error_event_count=("is_compile_error_event", "sum"),
            compile_error_result_count=("is_compile_error_result", "sum"),
            compile_success_count=("is_compile_success", "sum"),
            score_event_count=("has_score", "sum"),
            max_score=("Score", "max"),
            mean_score=("Score", "mean"),
            first_order=("Order", "min"),
            last_order=("Order", "max"),
        )
        .reset_index()
    )
    agg["compile_error_count"] = (
        agg["compile_error_event_count"] + agg["compile_error_result_count"]
    )
    agg["mean_score"] = agg["mean_score"].fillna(0)
    agg["max_score"] = agg["max_score"].fillna(0)

    if cache:
        ensure_dir(cache_path.parent)
        agg.to_csv(cache_path, index=False)
    return agg


def summarize_labels(labels: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(labels)),
        "students": int(labels["SubjectID"].nunique()),
        "assignments": int(labels["AssignmentID"].nunique()),
        "problems": int(labels["ProblemID"].nunique()),
        "label_distribution": labels["Label"].value_counts(dropna=False).to_dict(),
        "phase_distribution": labels["Phase"].value_counts(dropna=False).to_dict(),
        "attempts_summary": labels["Attempts"].describe().to_dict(),
    }


def code_state_sample(root: Optional[str | Path] = None, nrows: int = 1000) -> pd.DataFrame:
    """Load a small sample of CodeStates for qualitative/code-pattern analysis."""
    return load_csv("CodeStates.csv", root=root, nrows=nrows)
