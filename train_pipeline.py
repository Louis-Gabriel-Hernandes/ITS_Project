from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.content_generation import generate_all_content
from src.data_processing import (
    build_event_aggregates,
    inspect_raw_files,
    load_label_frames,
    summarize_labels,
)
from src.evaluation import (
    compute_metrics,
    leakage_diagnostic_text,
    save_all_figures,
)
from src.feature_engineering import make_feature_bundle, split_xy
from src.model_training import save_model_package, subject_wise_split, train_models
import joblib
from src.utils import ensure_dir, project_root, write_json


def main() -> None:
    root = project_root()
    ensure_dir(root / "models")
    ensure_dir(root / "reports" / "figures")
    ensure_dir(root / "data" / "generated")

    print("Generating question bank, misconception map, and pedagogical policy...")
    generate_all_content(root / "data" / "generated", min_questions=240)

    print("Inspecting raw files...")
    raw_summary = inspect_raw_files(root)
    write_json(raw_summary, root / "reports" / "raw_file_summary.json")

    print("Loading labels...")
    labels = load_label_frames(root)
    write_json(summarize_labels(labels), root / "reports" / "label_summary.json")

    print("Building event aggregates from MainTable for EDA/leakage audit documentation...")
    event_agg = build_event_aggregates(root, cache=True)
    write_json(
        {
            "rows": int(len(event_agg)),
            "students": int(event_agg["SubjectID"].nunique()),
            "assignments": int(event_agg["AssignmentID"].nunique()),
            "problems": int(event_agg["ProblemID"].nunique()),
            "note": "These current-problem aggregates are saved for EDA and should not be used in the safe deployed model.",
        },
        root / "reports" / "event_aggregate_summary.json",
    )

    print("Creating thesis-safe chronological feature set...")
    bundle = make_feature_bundle(labels, leakage_audit=False)
    X, y, groups = split_xy(bundle)
    X_train, X_val, y_train, y_val, g_train, g_val = subject_wise_split(X, y, groups)

    print("Training baseline, final model, and loss-curve model...")
    trained = train_models(X_train, y_train, X_val, y_val)
    baseline_model = trained["baseline_model"]
    final_model = trained["final_model"]
    loss_history = trained["loss_history"]

    print("Evaluating safe models...")
    safe_metrics = {
        "baseline_model": {
            "train": compute_metrics(baseline_model, X_train, y_train),
            "validation": compute_metrics(baseline_model, X_val, y_val),
        },
        "final_model": {
            "train": compute_metrics(final_model, X_train, y_train),
            "validation": compute_metrics(final_model, X_val, y_val),
        },
    }

    print("Running leakage audit model with current-row Attempts/CorrectEventually...")
    leakage_bundle = make_feature_bundle(labels, leakage_audit=True)
    X_leak, y_leak, groups_leak = split_xy(leakage_bundle)
    Xl_train, Xl_val, yl_train, yl_val, _, _ = subject_wise_split(X_leak, y_leak, groups_leak)
    leakage_trained = train_models(Xl_train, yl_train, Xl_val, yl_val)
    leak_model = leakage_trained["baseline_model"]
    leakage_metrics = compute_metrics(leak_model, Xl_val, yl_val)

    safe_auc = safe_metrics["final_model"]["validation"]["auc"]
    leak_auc = leakage_metrics["auc"]
    metrics = {
        "safe_modeling_note": "Final deployed model excludes current-row Attempts and CorrectEventually to reduce target leakage.",
        "split_strategy": "GroupShuffleSplit by SubjectID; no student appears in both train and validation sets.",
        "feature_columns": bundle.features,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "train_subjects": int(g_train.nunique()),
        "validation_subjects": int(g_val.nunique()),
        "models": safe_metrics,
        "leakage_audit_model_validation": leakage_metrics,
        "leakage_diagnostic": leakage_diagnostic_text(safe_auc, leak_auc),
    }
    write_json(metrics, root / "reports" / "metrics.json")
    loss_history.to_csv(root / "reports" / "loss_history.csv", index=False)

    print("Saving evaluation figures...")
    fig_paths = save_all_figures(
        final_model,
        X_val,
        y_val,
        loss_history,
        bundle.features,
        root / "reports" / "figures",
    )
    write_json(fig_paths, root / "reports" / "figure_paths.json")

    print("Saving model package...")
    save_model_package(
        final_model,
        features=bundle.features,
        metrics=safe_metrics["final_model"]["validation"],
        output_path=root / "models" / "its_model.pkl",
        threshold=0.5,
    )
    # Save lightweight preprocessing metadata. The full executable preprocessing
    # steps are also inside models/its_model.pkl as an sklearn Pipeline.
    joblib.dump(
        {
            "feature_columns": bundle.features,
            "safe_modeling_note": "Current-row Attempts and CorrectEventually are excluded from deployed features.",
            "preprocessing": "Median imputation inside the saved sklearn Pipeline.",
        },
        root / "models" / "preprocessor.pkl",
    )

    print("Done. Artifacts saved under models/, reports/, and data/generated/.")


if __name__ == "__main__":
    main()
