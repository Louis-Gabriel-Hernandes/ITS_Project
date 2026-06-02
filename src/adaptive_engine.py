from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import app_feature_row
from .student_model import mastery_to_difficulty, prior_mean_attempts, prior_success_rate, recent_success_rate


def load_model_package(path: str | Path) -> Optional[Dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def fallback_success_probability(student: Dict, question: Dict) -> float:
    topic = question.get("topic", "variables")
    mastery = float(student.get("mastery", {}).get(topic, 0.35))
    difficulty = int(question.get("difficulty", 2))
    diff_penalty = (difficulty - 3) * 0.09
    hint_penalty = min(0.12, student.get("hint_count", 0) / max(10, student.get("attempts", 1)) * 0.08)
    recent = recent_success_rate(student)
    logit = -0.35 + 2.35 * mastery + 0.85 * (recent - 0.5) - diff_penalty - hint_penalty
    return float(1.0 / (1.0 + math.exp(-logit)))


def model_success_probability(student: Dict, question: Dict, model_package: Optional[Dict]) -> float:
    if model_package is None:
        return fallback_success_probability(student, question)
    try:
        model = model_package["model"]
        difficulty = int(question.get("difficulty", 2))
        q_num = int(str(question.get("question_id", "JAVA_001")).split("_")[-1])
        row = app_feature_row(
            difficulty=difficulty,
            prior_items=int(student.get("attempts", 0)),
            prior_success_rate=prior_success_rate(student),
            prior_mean_attempts=prior_mean_attempts(student),
            recent_success_rate=recent_success_rate(student),
            current_streak=int(student.get("current_streak", 0)),
            phase_code=1,
            assignment_order_norm=min(1.0, difficulty / 5),
            problem_order_norm=(q_num % 60) / 60,
            prior_compile_success_rate=0.5,
        )
        proba = float(model.predict_proba(row)[:, 1][0])
        # Blend historical model with live mastery because generated app questions are
        # not the same item set as CSEDM problem IDs.
        return float(0.62 * proba + 0.38 * fallback_success_probability(student, question))
    except Exception:
        return fallback_success_probability(student, question)


def diagnostic_topics(question_bank: List[Dict], count: int = 8) -> List[Dict]:
    core = [
        "variables",
        "conditionals",
        "loops",
        "arrays",
        "methods",
        "classes_objects",
        "debugging",
        "code_tracing",
    ]
    selected = []
    used = set()
    for topic in core[:count]:
        candidates = [q for q in question_bank if q.get("topic") == topic and q.get("difficulty", 1) <= 3]
        if candidates:
            q = sorted(candidates, key=lambda x: x.get("difficulty", 1))[0]
            selected.append(q)
            used.add(q.get("question_id"))
    if len(selected) < count:
        for q in question_bank:
            if q.get("question_id") not in used:
                selected.append(q)
                used.add(q.get("question_id"))
            if len(selected) == count:
                break
    return selected


def choose_next_question(
    question_bank: List[Dict],
    student: Dict,
    model_package: Optional[Dict] = None,
    target_range: Tuple[float, float] = (0.65, 0.80),
) -> Dict:
    answered = set(student.get("answered_question_ids", []))
    available = [q for q in question_bank if q.get("question_id") not in answered]
    if not available:
        available = question_bank[:]

    if student.get("phase") == "diagnostic":
        diagnostic = diagnostic_topics(question_bank, student.get("diagnostic_remaining", 8))
        for q in diagnostic:
            if q.get("question_id") not in answered:
                return q

    mastery = student.get("mastery", {})
    weak_topics = sorted(mastery, key=lambda t: mastery.get(t, 0.5))[:5]
    topic_candidates = [q for q in available if q.get("topic") in weak_topics]
    if not topic_candidates:
        topic_candidates = available

    scored = []
    low, high = target_range
    for q in topic_candidates:
        topic = q.get("topic", "variables")
        target_diff = mastery_to_difficulty(mastery.get(topic, 0.35))
        p = model_success_probability(student, q, model_package)
        target_penalty = 0 if low <= p <= high else min(abs(p - low), abs(p - high))
        difficulty_penalty = abs(int(q.get("difficulty", 1)) - target_diff) * 0.06
        review_bonus = -0.03 if topic == student.get("recommended_next_topic") else 0
        score = target_penalty + difficulty_penalty + review_bonus + random.random() * 0.01
        scored.append((score, p, q))
    scored.sort(key=lambda x: x[0])
    return scored[0][2]


def get_hint(question: Dict, hint_map: Dict, hint_level: int) -> str:
    """Return the best hint for a question.

    Question-specific hints are preferred so each item can give targeted help.
    The misconception hint map is kept as a fallback for older question banks
    or questions that do not define their own `hints` list.

    `hint_level` comes from the Streamlit app as 1, 2, or 3, so convert it
    to a zero-based list index before reading the hint sequence.
    """
    try:
        idx = max(0, int(hint_level) - 1)
    except Exception:
        idx = 0

    # Preferred path: use the hints stored directly on the selected question.
    # This enables personalized, question-specific hints.
    question_hints = question.get("hints", []) or []
    if question_hints:
        return str(question_hints[min(idx, len(question_hints) - 1)])

    # Fallback path: use misconception-level hints for older/generated banks.
    tags = question.get("misconception_tags", [])
    for tag in tags:
        item = hint_map.get(tag)
        if not item:
            continue
        hints = item.get("hint_sequence", []) or []
        if hints:
            chosen = hints[min(idx, len(hints) - 1)]
            if isinstance(chosen, dict):
                return str(chosen.get("text", "Try breaking the problem into smaller steps."))
            return str(chosen)

    return "Focus on the relevant Java rule, trace the code slowly, and test one example input."


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().replace(";", "").split())


def check_answer(question: Dict, user_answer: str) -> bool:
    correct = question.get("correct_answer", "")
    qtype = question.get("question_type", "multiple_choice")
    normalized_user = normalize_answer(user_answer)

    if qtype == "multiple_choice":
        return normalized_user == normalize_answer(correct)

    accepted = [correct] + question.get("accepted_answers", [])
    if any(normalized_user == normalize_answer(a) for a in accepted):
        return True

    # Short coding questions: use conservative keyword matching when exact match is too strict.
    patterns = question.get("answer_patterns", [])
    if patterns:
        return all(p.lower() in str(user_answer).lower() for p in patterns)
    return False
