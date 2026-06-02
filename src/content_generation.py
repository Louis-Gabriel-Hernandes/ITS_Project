from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .beginner_question_bank import (
    BEGINNER_TOPICS as TOPIC_SPECS,
    QUESTION_TYPES,
    generate_all_beginner_content,
    generate_beginner_hint_map,
    generate_beginner_policy,
    generate_beginner_question_bank,
)


def generate_question_bank(min_questions: int = 1100) -> List[Dict[str, Any]]:
    return generate_beginner_question_bank(min_questions=min_questions)


def generate_misconception_hint_map() -> Dict[str, Any]:
    return generate_beginner_hint_map()


def generate_pedagogical_policy() -> Dict[str, Any]:
    return generate_beginner_policy()


def generate_all_content(output_dir: str | Path, min_questions: int = 1100) -> Dict[str, Path]:
    return generate_all_beginner_content(output_dir, min_questions=min_questions)
