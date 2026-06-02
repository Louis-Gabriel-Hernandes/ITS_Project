from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

JAVA_TOPICS = [
    "variables",
    "primitive_types",
    "strings",
    "operators",
    "conditionals",
    "loops",
    "arrays",
    "methods",
    "classes_objects",
    "constructors",
    "encapsulation",
    "inheritance",
    "polymorphism",
    "interfaces",
    "exceptions",
    "recursion",
    "arraylist",
    "algorithms",
    "debugging",
    "code_tracing",
]


def new_student_model() -> Dict:
    return {
        "is_new_student": True,
        "phase": "adaptive",
        "diagnostic_remaining": 0,
        "mastery": {topic: 0.20 for topic in JAVA_TOPICS},
        "attempts": 0,
        "correct": 0,
        "incorrect": 0,
        "hint_count": 0,
        "hint_usage_by_topic": {topic: 0 for topic in JAVA_TOPICS},
        "attempts_by_topic": {topic: 0 for topic in JAVA_TOPICS},
        "correct_by_topic": {topic: 0 for topic in JAVA_TOPICS},
        "misconceptions": {},
        "answered_question_ids": [],
        "recent_history": [],
        "current_streak": 0,
        "current_difficulty": 1,
        "confidence": 0.20,
        "recommended_next_topic": "variables",
    }


def mastery_to_difficulty(mastery: float) -> int:
    if mastery < 0.25:
        return 1
    if mastery < 0.45:
        return 2
    if mastery < 0.65:
        return 3
    if mastery < 0.82:
        return 4
    return 5


def update_after_answer(
    student: Dict,
    question: Dict,
    is_correct: bool,
    hints_used: int,
    max_attempts_reached: bool = False,
) -> Dict:
    student = deepcopy(student)
    topic = question.get("topic", "variables")
    old = float(student["mastery"].get(topic, 0.35))

    student["attempts"] += 1
    student["hint_count"] += int(hints_used)
    student["attempts_by_topic"][topic] = student["attempts_by_topic"].get(topic, 0) + 1
    student["hint_usage_by_topic"][topic] = student["hint_usage_by_topic"].get(topic, 0) + int(hints_used)

    if is_correct:
        student["correct"] += 1
        student["correct_by_topic"][topic] = student["correct_by_topic"].get(topic, 0) + 1
        gain = 0.095 if hints_used == 0 else 0.045
        if max_attempts_reached:
            gain *= 0.5
        student["mastery"][topic] = min(0.98, old + gain * (1.05 - old))
        student["current_streak"] = max(1, student["current_streak"] + 1)
    else:
        student["incorrect"] += 1
        penalty = 0.075 + 0.025 * hints_used
        student["mastery"][topic] = max(0.02, old - penalty * old)
        student["current_streak"] = min(-1, student["current_streak"] - 1)
        for tag in question.get("misconception_tags", []):
            student["misconceptions"][tag] = student["misconceptions"].get(tag, 0) + 1

    if question.get("question_id") not in student["answered_question_ids"]:
        student["answered_question_ids"].append(question.get("question_id"))

    student["recent_history"].append(
        {
            "question_id": question.get("question_id"),
            "topic": topic,
            "difficulty": question.get("difficulty", 1),
            "correct": bool(is_correct),
            "hints_used": int(hints_used),
        }
    )
    student["recent_history"] = student["recent_history"][-20:]

    weak_topics = sorted(student["mastery"], key=lambda t: student["mastery"][t])
    student["recommended_next_topic"] = weak_topics[0]
    student["current_difficulty"] = mastery_to_difficulty(student["mastery"][topic])
    student["confidence"] = sum(student["mastery"].values()) / max(1, len(student["mastery"]))

    if student["phase"] == "diagnostic":
        student["diagnostic_remaining"] = max(0, student["diagnostic_remaining"] - 1)
        if student["diagnostic_remaining"] == 0:
            student["phase"] = "adaptive"
            student["is_new_student"] = False

    return student


def recent_success_rate(student: Dict, window: int = 5) -> float:
    history = student.get("recent_history", [])[-window:]
    if not history:
        return 0.5
    return sum(1 for h in history if h.get("correct")) / len(history)


def prior_success_rate(student: Dict) -> float:
    if student.get("attempts", 0) == 0:
        return 0.5
    return student.get("correct", 0) / max(1, student.get("attempts", 1))


def prior_mean_attempts(student: Dict) -> float:
    # The app uses one submitted answer per question, plus hints as effort signal.
    if student.get("attempts", 0) == 0:
        return 3.0
    return 1.0 + student.get("hint_count", 0) / max(1, student.get("attempts", 1))
