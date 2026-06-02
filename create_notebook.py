from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "train_evaluate_its_model.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

cells = []

cells.append(md("""# Adaptive Java ITS: Training, Evaluation, and Explainability

This notebook trains and evaluates the machine-learning component of the Java Intelligent Tutoring System using the attached CSEDM2021-style dataset.

The deployed app treats every launch as a new student session. The historical dataset is used to learn general patterns of student success/struggle, while generated Java questions, misconception maps, and pedagogical rules drive the interactive tutoring experience.
"""))

cells.append(md("""## 1. Setup

The notebook uses relative paths so it can be run from the project root or from the `notebooks/` folder. It saves models, metrics, and figures to the project directories.
"""))

cells.append(code("""from pathlib import Path
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.content_generation import generate_all_content
from src.data_processing import inspect_raw_files, load_label_frames, load_subjects, build_event_aggregates, summarize_labels, code_state_sample
from src.feature_engineering import make_feature_bundle, split_xy
from src.model_training import subject_wise_split, train_models, save_model_package
from src.evaluation import compute_metrics, save_all_figures, leakage_diagnostic_text
from src.utils import ensure_dir, write_json

pd.set_option('display.max_columns', 80)
ensure_dir(ROOT / 'reports' / 'figures')
ensure_dir(ROOT / 'models')
ensure_dir(ROOT / 'data' / 'generated')
print('Project root:', ROOT)
"""))

cells.append(md("""## 2. Generate tutoring content

The Java question bank, misconception map, and pedagogical policy are generated educational content. They are not treated as historical CSEDM data.
"""))

cells.append(code("""content_paths = generate_all_content(ROOT / 'data' / 'generated', min_questions=240)
content_paths
"""))

cells.append(md("""## 3. Raw dataset inspection

This section verifies the attached files, schemas, estimated row counts, and basic structure before modeling.
"""))

cells.append(code("""raw_summary = inspect_raw_files(ROOT)
write_json(raw_summary, ROOT / 'reports' / 'raw_file_summary.json')
pd.DataFrame(raw_summary).T
"""))

cells.append(code("""labels = load_label_frames(ROOT)
subjects = load_subjects(ROOT)
print(labels.shape)
display(labels.head())
display(pd.DataFrame([summarize_labels(labels)]).T)
"""))

cells.append(code("""print('Label distribution:')
display(labels['Label'].value_counts(normalize=True).rename('proportion'))
print('Phase distribution:')
display(labels['Phase'].value_counts())
print('Attempts summary by phase:')
display(labels.groupby('Phase')['Attempts'].describe())
"""))

cells.append(md("""## 4. Event-log aggregation from `MainTable.csv`

`MainTable.csv` contains event-level logs such as compile, run, score, and compiler error events. The current-problem aggregates are useful for exploratory analysis and leakage audits. They are **not** used as deployed model features because they summarize behavior that happens during the current problem.
"""))

cells.append(code("""event_agg = build_event_aggregates(ROOT, cache=True)
print(event_agg.shape)
display(event_agg.head())
write_json({
    'rows': int(len(event_agg)),
    'students': int(event_agg['SubjectID'].nunique()),
    'assignments': int(event_agg['AssignmentID'].nunique()),
    'problems': int(event_agg['ProblemID'].nunique()),
    'note': 'Current-problem event aggregates are used for EDA/leakage documentation, not safe deployed prediction.'
}, ROOT / 'reports' / 'event_aggregate_summary.json')
"""))

cells.append(code("""event_summary = event_agg[['event_count','run_count','compile_count','compile_error_count','compile_success_count','max_score']].describe()
display(event_summary)
"""))

cells.append(md("""## 5. CodeStates sample

`CodeStates.csv` can be large. The notebook samples it for qualitative inspection rather than loading every code state into the predictive model.
"""))

cells.append(code("""sample_code = code_state_sample(ROOT, nrows=5)
display(sample_code)
"""))

cells.append(md("""## 6. Thesis-safe feature engineering

The model uses chronological features that are available before each labeled row:

- prior number of items attempted by the same student
- prior success rate
- prior mean attempts
- recent success rate
- current streak before the item
- curricular order and estimated difficulty proxy

The deployed feature set deliberately excludes the row's own `Attempts` and `CorrectEventually`, because those are outcome-proximal and can cause leakage.
"""))

cells.append(code("""bundle = make_feature_bundle(labels, leakage_audit=False)
features_df = bundle.frame
X, y, groups = split_xy(bundle)
print('Feature matrix:', X.shape)
print('Target mean:', y.mean())
display(features_df[['SubjectID','Phase','AssignmentID','ProblemID','Label'] + bundle.features].head(10))
"""))

cells.append(md("""## 7. Subject-wise split

A subject-wise split is used so the same student does not appear in both training and validation sets. This better simulates predicting for unseen students.
"""))

cells.append(code("""X_train, X_val, y_train, y_val, g_train, g_val = subject_wise_split(X, y, groups, test_size=0.2, random_state=42)
print('Train rows:', len(X_train), 'Validation rows:', len(X_val))
print('Train students:', g_train.nunique(), 'Validation students:', g_val.nunique())
print('Student overlap:', len(set(g_train).intersection(set(g_val))))
"""))

cells.append(md("""## 8. Model training

The notebook trains:

1. a baseline logistic regression model,
2. a stronger random forest model used by the app,
3. an incremental SGD logistic model used to produce train/validation loss curves.
"""))

cells.append(code("""trained = train_models(X_train, y_train, X_val, y_val, random_state=42)
baseline_model = trained['baseline_model']
final_model = trained['final_model']
loss_model = trained['loss_model']
loss_history = trained['loss_history']
display(loss_history.head())
display(loss_history.tail())
loss_history.to_csv(ROOT / 'reports' / 'loss_history.csv', index=False)
"""))

cells.append(md("""## 9. Evaluation metrics

The evaluation includes accuracy, AUC, F1, precision, recall, Brier score, confusion matrix, and classification report.
"""))

cells.append(code("""metrics = {
    'baseline_model': {
        'train': compute_metrics(baseline_model, X_train, y_train),
        'validation': compute_metrics(baseline_model, X_val, y_val),
    },
    'final_model': {
        'train': compute_metrics(final_model, X_train, y_train),
        'validation': compute_metrics(final_model, X_val, y_val),
    },
}
summary_rows = []
for model_name, split_data in metrics.items():
    for split, vals in split_data.items():
        summary_rows.append({
            'model': model_name,
            'split': split,
            'accuracy': vals['accuracy'],
            'auc': vals['auc'],
            'f1': vals['f1'],
            'precision': vals['precision'],
            'recall': vals['recall'],
            'brier_score': vals['brier_score'],
        })
display(pd.DataFrame(summary_rows))
"""))

cells.append(md("""## 10. Curves, confusion matrix, calibration, and XAI

The following cell saves all thesis figures into `reports/figures/`.
"""))

cells.append(code("""fig_paths = save_all_figures(
    final_model, X_val, y_val, loss_history, bundle.features, ROOT / 'reports' / 'figures'
)
fig_paths
"""))

cells.append(code("""from IPython.display import Image, display
for name, path in fig_paths.items():
    print(name, path)
    display(Image(filename=path))
"""))

cells.append(md("""## 11. Leakage audit

This audit trains a comparison model with current-row `Attempts` and `CorrectEventually`. If the AUC jumps substantially, that is evidence that these fields leak the target and should not be used by the deployed tutor.
"""))

cells.append(code("""leakage_bundle = make_feature_bundle(labels, leakage_audit=True)
X_leak, y_leak, groups_leak = split_xy(leakage_bundle)
Xl_train, Xl_val, yl_train, yl_val, _, _ = subject_wise_split(X_leak, y_leak, groups_leak, test_size=0.2, random_state=42)
leakage_trained = train_models(Xl_train, yl_train, Xl_val, yl_val, random_state=42)
leak_model = leakage_trained['baseline_model']
leakage_metrics = compute_metrics(leak_model, Xl_val, yl_val)
print('Safe final validation AUC:', metrics['final_model']['validation']['auc'])
print('Leakage audit validation AUC:', leakage_metrics['auc'])
print(leakage_diagnostic_text(metrics['final_model']['validation']['auc'], leakage_metrics['auc']))
"""))

cells.append(md("""## 12. Save final artifacts

The app loads `models/its_model.pkl`. The package contains the trained model, feature columns, validation metrics, and a note about the subject-wise split.
"""))

cells.append(code("""all_metrics = {
    'safe_modeling_note': 'Final deployed model excludes current-row Attempts and CorrectEventually to reduce target leakage.',
    'split_strategy': 'GroupShuffleSplit by SubjectID; no student appears in both train and validation sets.',
    'feature_columns': bundle.features,
    'train_rows': int(len(X_train)),
    'validation_rows': int(len(X_val)),
    'train_subjects': int(g_train.nunique()),
    'validation_subjects': int(g_val.nunique()),
    'models': metrics,
    'leakage_audit_model_validation': leakage_metrics,
    'leakage_diagnostic': leakage_diagnostic_text(metrics['final_model']['validation']['auc'], leakage_metrics['auc']),
}
write_json(all_metrics, ROOT / 'reports' / 'metrics.json')
save_model_package(final_model, bundle.features, metrics['final_model']['validation'], ROOT / 'models' / 'its_model.pkl')
import joblib\njoblib.dump({'feature_columns': bundle.features, 'preprocessing': 'Median imputation inside models/its_model.pkl Pipeline.'}, ROOT / 'models' / 'preprocessor.pkl')
print('Saved model and metrics.')
"""))

cells.append(md("""## 13. Thesis discussion notes

- The subject-wise split is more realistic than a random row split because it evaluates generalization to unseen students.
- The deployed model intentionally excludes current-row attempts and final correctness fields to reduce leakage.
- The Streamlit app combines historical ML prediction with an online student model because generated Java tutoring questions are not identical to the original CSEDM problem IDs.
- The tutor uses cold-start diagnostic questions, progressive hints, mastery updates, misconception tracking, and adaptive item selection.
"""))

nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(NOTEBOOK_PATH)
