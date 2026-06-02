from __future__ import annotations

from pathlib import Path
from typing import Dict

from .utils import read_json


def default_policy() -> Dict:
    return {
        "cold_start": {
            "diagnostic_question_count": 8,
            "diagnostic_strategy": "sample across core Java topics before personalization",
        },
        "target_success_probability": [0.65, 0.80],
        "difficulty_rules": {
            "correct_no_hint": "increase mastery and consider harder question",
            "correct_with_hint": "small mastery increase and keep difficulty stable",
            "incorrect": "decrease mastery and select remediation or easier question",
            "repeated_failure": "enter remediation mode for the weakest topic",
        },
        "hint_policy": [
            "conceptual hint",
            "strategic hint",
            "worked example",
            "final explanation after attempts are exhausted",
        ],
        "personalization": [
            "topic selection",
            "difficulty selection",
            "hint depth",
            "explanation style",
            "review recommendation",
        ],
    }


def load_policy(path: str | Path | None = None) -> Dict:
    if path is None:
        return default_policy()
    return read_json(path, default=default_policy())
