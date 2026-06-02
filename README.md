# Adaptive Java Intelligent Tutoring System

This project is a thesis-oriented Intelligent Tutoring System (ITS) for Java programming using the attached CSEDM2021-style dataset files.

The system contains two connected parts:

1. **Machine-learning notebook and training pipeline** using real student interaction/performance data.
2. **Streamlit ITS application** that starts every launch as a new student, asks diagnostic questions, then adapts topic, difficulty, hints, and remediation.

## Dataset files

Place these files in `data/raw/`:

- `DatasetMetadata.csv`
- `MainTable.csv`
- `early.csv`
- `late.csv`
- `Subject.csv`
- `CodeStates.csv`

The historical student interaction data is not generated. Only the Java question bank, misconception map, hints, and pedagogical policy are generated educational content.

## Project structure

```text
java_its_project/
├── app.py
├── train_pipeline.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── train_evaluate_its_model.ipynb
├── src/
│   ├── adaptive_engine.py
│   ├── content_generation.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── pedagogical_policy.py
│   ├── student_model.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── generated/
├── models/
└── reports/
    └── figures/
```

## Installation

```bash
pip install -r requirements.txt
```

## Train and evaluate the model

Either run the notebook:

```bash
jupyter notebook notebooks/train_evaluate_its_model.ipynb
```

Or run the full training script:

```bash
python train_pipeline.py
```

The pipeline creates:

- `models/its_model.pkl`
- `models/preprocessor.pkl`
- `reports/metrics.json`
- `reports/loss_history.csv`
- `reports/figures/loss_curve.png`
- `reports/figures/auc_curve.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/precision_recall_curve.png`
- `reports/figures/calibration_curve.png`
- `reports/figures/feature_importance.png`
- `data/generated/question_bank.json`
- `data/generated/misconception_hint_map.json`
- `data/generated/pedagogical_policy.json`

## Launch the ITS app

```bash
streamlit run app.py
```

The app starts each launch as a new student session. It first gives a diagnostic quiz, then switches into adaptive tutoring.

## Personalization behavior

The ITS maintains a session-based student model in `st.session_state`:

- topic mastery
- attempts
- correctness
- hint usage
- misconception counts
- recent performance
- current streak
- recommended next topic

After every answer, the app updates the student model. Correct answers without hints increase mastery more than correct answers after hints. Incorrect answers decrease mastery, activate misconception tracking, and make remediation more likely.

## Adaptive difficulty

The adaptive engine chooses questions using:

- weakest topics
- current topic mastery
- question difficulty
- recent success rate
- hint usage
- predicted success probability

When the trained model exists, the app blends the CSEDM-trained success prediction with live mastery estimates. If the model is missing, the app uses a transparent rule-based fallback so the tutor still works.

## Hint policy

Hints are progressive:

1. conceptual hint
2. strategic hint
3. worked-example hint
4. final explanation after attempts are exhausted

The system avoids revealing the answer immediately.

## ML methodology

The notebook uses `early.csv` and `late.csv` as real student-level outcome labels. It engineers chronological features from previous student performance only. The deployed model excludes current-row `Attempts` and `CorrectEventually` because those can leak the outcome.

Evaluation uses a subject-wise split, so the same student does not appear in both train and validation sets.

Reported metrics include:

- accuracy
- AUC
- F1
- precision
- recall
- confusion matrix
- ROC curve
- precision-recall curve
- calibration curve
- training and validation loss curves
- feature importance / XAI
- leakage audit

## Limitations

- The generated Java question bank is educational content, not original CSEDM problem text.
- The CSEDM-trained model is connected to generated app questions through general student-performance features, not identical historical problem IDs.
- The app uses session memory only; it resets on a new launch unless persistence is later added.
- More advanced knowledge tracing models such as DKT/SAKT could be added as future work.

## Future improvements

- Add persistent student accounts.
- Add real Java code execution/sandboxing.
- Add LLM-based feedback for free-form code answers.
- Use a sequence model such as Deep Knowledge Tracing.
- Add teacher dashboard and classroom analytics.
