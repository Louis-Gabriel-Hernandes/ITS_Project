from __future__ import annotations

import hashlib
import html
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.adaptive_engine import (
    check_answer,
    choose_next_question,
    get_hint,
    load_model_package,
    model_success_probability,
)
from src.content_generation import generate_all_content
from src.local_user_store import (
    authenticate_user,
    create_user,
    load_progress,
    reset_progress as reset_saved_progress,
    save_progress,
)
from src.student_model import (
    JAVA_TOPICS,
    new_student_model,
    update_after_answer,
)
from src.utils import project_root, read_json

ROOT = project_root()
DATA_DIR = ROOT / "data" / "generated"
MODEL_PATH = ROOT / "models" / "its_model.pkl"
SESSION_LENGTH = 20
MIXED_TOPIC = "__mixed__"

st.set_page_config(
    page_title="Java Learning",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --bg: #f6f8fc;
  --surface: #ffffff;
  --surface-soft: #f8fafc;
  --ink: #111827;
  --muted: #64748b;
  --line: #e2e8f0;
  --primary: #6366f1;
  --primary-dark: #4f46e5;
  --accent: #8b5cf6;
  --success: #16a34a;
  --danger: #dc2626;
  --warning: #f59e0b;
  --shadow-sm: 0 4px 14px rgba(15, 23, 42, 0.08);
  --shadow-md: 0 18px 45px rgba(15, 23, 42, 0.12);
  --radius: 22px;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes softPulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.035); }
}

@keyframes gradientDrift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes shimmer {
  from { transform: translateX(-120%); }
  to { transform: translateX(120%); }
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(99, 102, 241, 0.13), transparent 34rem),
    radial-gradient(circle at top right, rgba(139, 92, 246, 0.12), transparent 30rem),
    var(--bg);
  color: var(--ink);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1.2rem; }

.block-container {
  max-width: 1220px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
  background: #0f172a;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
  color: #e5e7eb;
}

.sidebar-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.25rem;
}

.sidebar-icon {
  width: 2.7rem;
  height: 2.7rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 1rem;
  background: rgba(99, 102, 241, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.12);
  font-size: 1.35rem;
}

.session-info {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(99,102,241,0.95), rgba(139,92,246,0.95));
  color: white;
  padding: 1.25rem;
  border-radius: 1.35rem;
  margin: 1.25rem 0;
  box-shadow: 0 18px 40px rgba(2, 6, 23, 0.28);
}

.session-info::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent, rgba(255,255,255,0.18), transparent);
  transform: translateX(-120%);
  animation: shimmer 5s ease-in-out infinite;
}

.session-info h3,
.session-info .info-item {
  position: relative;
  z-index: 1;
}

.session-info h3 {
  margin: 0 0 1rem 0;
  font-size: 1.05rem;
  font-weight: 800;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.72rem 0;
  border-top: 1px solid rgba(255, 255, 255, 0.17);
}

.info-label {
  font-weight: 600;
  opacity: 0.9;
}

.info-value {
  font-size: 1.25rem;
  font-weight: 900;
}

.hero-card {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #6366f1, #8b5cf6, #6366f1);
  background-size: 220% 220%;
  animation: gradientDrift 9s ease infinite, fadeUp .45s ease both;
  color: white;
  padding: clamp(1.35rem, 3vw, 2.1rem);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  margin-bottom: 1.3rem;
}

.hero-card::before {
  content: "";
  position: absolute;
  width: 18rem;
  height: 18rem;
  right: -6rem;
  top: -8rem;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.14);
}

.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  padding: .35rem .7rem;
  border-radius: 999px;
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.2);
  font-size: .84rem;
  font-weight: 700;
  margin-bottom: .85rem;
}

.hero-card h1 {
  position: relative;
  z-index: 1;
  margin: 0;
  font-size: clamp(1.85rem, 4vw, 3rem);
  line-height: 1.05;
  font-weight: 900;
  letter-spacing: -0.04em;
}

.hero-card p {
  position: relative;
  z-index: 1;
  max-width: 46rem;
  margin: .75rem 0 0 0;
  font-size: 1.02rem;
  line-height: 1.6;
  opacity: .96;
}

.stat-box {
  animation: fadeUp .45s ease both;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: 1.2rem;
  padding: 1.05rem;
  box-shadow: var(--shadow-sm);
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}

.stat-box:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-md);
  border-color: rgba(99, 102, 241, 0.28);
}

.stat-label {
  display: flex;
  align-items: center;
  gap: .45rem;
  color: var(--muted);
  font-size: .78rem;
  font-weight: 850;
  text-transform: uppercase;
  letter-spacing: .07em;
  margin-bottom: .5rem;
}

.stat-value {
  color: var(--ink);
  font-size: 2rem;
  font-weight: 950;
  letter-spacing: -0.04em;
}

.progress-card,
.question-card,
.feedback-correct,
.feedback-incorrect,
.explanation-box,
.hint-box,
.empty-state {
  animation: fadeUp .45s ease both;
}

.progress-card {
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(226, 232, 240, 0.95);
  box-shadow: var(--shadow-sm);
  border-radius: 1.25rem;
  padding: 1rem 1.1rem;
  margin: 1rem 0 1.4rem 0;
}

.progress-topline {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: .65rem;
  color: var(--muted);
  font-size: .92rem;
  font-weight: 750;
}

.progress-track {
  height: .7rem;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--primary), var(--accent));
  transition: width .55s ease;
}

.question-card {
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: var(--radius);
  padding: clamp(1.1rem, 2.4vw, 2rem);
  box-shadow: var(--shadow-md);
  margin-top: .8rem;
}

.question-metadata {
  display: flex;
  align-items: center;
  gap: .65rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: .35rem;
  background: #eef2ff;
  color: #4338ca;
  border: 1px solid #c7d2fe;
  padding: .45rem .8rem;
  border-radius: 999px;
  font-size: .84rem;
  font-weight: 800;
}

.badge-soft {
  background: #f8fafc;
  color: #475569;
  border-color: #e2e8f0;
}

.question-text {
  color: var(--ink);
  font-size: clamp(1.23rem, 2.2vw, 1.55rem);
  font-weight: 850;
  line-height: 1.55;
  letter-spacing: -0.018em;
  margin: .75rem 0 1.25rem 0;
  white-space: pre-wrap;
}

.small-note {
  color: var(--muted);
  font-size: .92rem;
  line-height: 1.55;
  margin: -.2rem 0 1rem 0;
}

.feedback-correct,
.feedback-incorrect,
.explanation-box,
.hint-box,
.info-box {
  border-radius: 1.1rem;
  padding: 1rem 1.1rem;
  margin: 1rem 0;
  line-height: 1.6;
  border: 1px solid transparent;
  box-shadow: var(--shadow-sm);
}

.feedback-correct {
  background: #f0fdf4;
  border-color: #bbf7d0;
  color: #14532d;
}

.feedback-incorrect {
  background: #fef2f2;
  border-color: #fecaca;
  color: #7f1d1d;
}

.feedback-correct strong,
.feedback-incorrect strong,
.explanation-box strong,
.hint-box strong {
  font-weight: 900;
}

.explanation-box {
  background: #eff6ff;
  border-color: #bfdbfe;
  color: #1e3a8a;
}

.hint-box {
  background: #fffbeb;
  border-color: #fde68a;
  color: #78350f;
}

.info-box {
  background: #f8fafc;
  border-color: #e2e8f0;
  color: #334155;
}

.empty-state {
  text-align: center;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid #e2e8f0;
  border-radius: var(--radius);
  padding: 3rem 1.5rem;
  box-shadow: var(--shadow-md);
}

.empty-state .trophy {
  display: inline-flex;
  width: 5rem;
  height: 5rem;
  align-items: center;
  justify-content: center;
  border-radius: 1.5rem;
  background: #fef3c7;
  font-size: 3rem;
  margin-bottom: 1rem;
  animation: softPulse 2.6s ease-in-out infinite;
}

.stButton > button {
  min-height: 3rem;
  border-radius: .95rem !important;
  font-weight: 850 !important;
  border: 1px solid rgba(148, 163, 184, .38) !important;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 24px rgba(15, 23, 42, .12) !important;
  border-color: rgba(99, 102, 241, .55) !important;
}

.stButton > button:active {
  transform: translateY(0);
}

.stButton button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  color: white !important;
  border: 0 !important;
}

.stRadio [role="radiogroup"] {
  gap: .7rem;
}

.stRadio [role="radio"] {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 1rem;
  padding: .75rem .85rem;
  min-height: 3.05rem;
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}

.stRadio [role="radio"]:hover {
  border-color: #a5b4fc;
  box-shadow: 0 8px 18px rgba(99, 102, 241, .08);
  transform: translateY(-1px);
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
  border-radius: 1rem !important;
  border-color: #cbd5e1 !important;
  box-shadow: none !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 .2rem rgba(99,102,241,.12) !important;
}

/* High-contrast widget overrides.
   These keep Streamlit readable when the user's global theme is dark
   but this app design uses light cards. */
[data-testid="stAppViewContainer"] .stRadio,
[data-testid="stAppViewContainer"] .stRadio *,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"],
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] *,
[data-testid="stAppViewContainer"] [data-testid="stTextArea"],
[data-testid="stAppViewContainer"] [data-testid="stTextArea"] * {
  color: var(--ink) !important;
  opacity: 1 !important;
}

[data-testid="stAppViewContainer"] .stRadio [role="radio"] {
  background: #ffffff !important;
  color: var(--ink) !important;
  border: 1px solid #dbe3ef !important;
}

[data-testid="stAppViewContainer"] .stRadio [role="radio"][aria-checked="true"] {
  background: #eef2ff !important;
  border-color: var(--primary) !important;
  box-shadow: 0 8px 20px rgba(99, 102, 241, .14) !important;
}

[data-testid="stAppViewContainer"] [data-testid="stTextArea"] textarea,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] input {
  background: #ffffff !important;
  color: var(--ink) !important;
  caret-color: var(--primary) !important;
}

[data-testid="stAppViewContainer"] [data-testid="stTextArea"] textarea::placeholder,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] input::placeholder {
  color: #94a3b8 !important;
  opacity: 1 !important;
}

[data-testid="stAppViewContainer"] .stButton > button {
  background: #ffffff !important;
  color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] .stButton > button *,
[data-testid="stAppViewContainer"] .stButton > button p,
[data-testid="stAppViewContainer"] .stButton > button span {
  color: inherit !important;
  opacity: 1 !important;
}

[data-testid="stAppViewContainer"] .stButton button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  color: white !important;
}

.hero-card,
.hero-card *,
.session-info,
.session-info * {
  color: white !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}

[data-testid="stSidebar"] .stButton > button {
  background: rgba(255, 255, 255, 0.08) !important;
  color: #ffffff !important;
  border-color: rgba(255, 255, 255, 0.18) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--radius) !important;
  border-color: #e2e8f0 !important;
  box-shadow: var(--shadow-md);
  background: rgba(255, 255, 255, .94);
}

.mastery-caption {
  color: #cbd5e1;
  font-size: .9rem;
  line-height: 1.45;
  margin-bottom: .75rem;
}


/* Extra contrast fixes for Streamlit theme collisions. */
[data-testid="stAppViewContainer"] {
  color: var(--ink) !important;
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
  text-shadow: none;
}

/* Keep custom light cards readable even if Streamlit is running a dark theme. */
.stat-box,
.progress-card,
.question-card,
.empty-state,
.info-box,
[data-testid="stVerticalBlockBorderWrapper"] {
  background-color: rgba(255, 255, 255, .96) !important;
  color: var(--ink) !important;
}

.stat-box *,
.progress-card *,
.question-card *,
.empty-state *,
.info-box *,
[data-testid="stVerticalBlockBorderWrapper"] * {
  color: inherit;
}

.stat-label,
.progress-topline,
.small-note,
.empty-state p {
  color: var(--muted) !important;
}

.stat-value,
.question-text,
.empty-state h2 {
  color: var(--ink) !important;
}

.badge {
  background: #eef2ff !important;
  color: #3730a3 !important;
  border-color: #c7d2fe !important;
}

.badge-soft {
  background: #f8fafc !important;
  color: #334155 !important;
  border-color: #dbe3ef !important;
}

.feedback-correct,
.feedback-correct * {
  background-color: #f0fdf4 !important;
  color: #14532d !important;
}

.feedback-incorrect,
.feedback-incorrect * {
  background-color: #fef2f2 !important;
  color: #7f1d1d !important;
}

.explanation-box,
.explanation-box * {
  background-color: #eff6ff !important;
  color: #1e3a8a !important;
}

.hint-box,
.hint-box * {
  background-color: #fffbeb !important;
  color: #78350f !important;
}

/* Inputs, select boxes, and text areas: readable text + visible disabled states. */
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] [data-baseweb="select"] > div,
[data-testid="stAppViewContainer"] [data-baseweb="base-input"] {
  background-color: #ffffff !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  border-color: #cbd5e1 !important;
}

[data-testid="stAppViewContainer"] input:disabled,
[data-testid="stAppViewContainer"] textarea:disabled,
[data-testid="stAppViewContainer"] [aria-disabled="true"] {
  opacity: 1 !important;
  color: #475569 !important;
  -webkit-text-fill-color: #475569 !important;
  background-color: #f1f5f9 !important;
}

[data-testid="stAppViewContainer"] [data-baseweb="select"] *,
[data-testid="stAppViewContainer"] [data-baseweb="base-input"] * {
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
}

/* Streamlit popovers/dropdowns can render outside the sidebar, so style both modes explicitly. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"],
[data-baseweb="menu"] li {
  background-color: #ffffff !important;
  color: var(--ink) !important;
}

[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] li * {
  color: var(--ink) !important;
}

[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover {
  background-color: #eef2ff !important;
  color: #3730a3 !important;
}

/* Sidebar controls live on a dark background, so give their own dark surfaces. */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  background-color: #1e293b !important;
  color: #f8fafc !important;
  -webkit-text-fill-color: #f8fafc !important;
  border-color: rgba(255, 255, 255, .22) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="base-input"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
  color: #f8fafc !important;
  -webkit-text-fill-color: #f8fafc !important;
}

[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label * {
  color: #e5e7eb !important;
  opacity: 1 !important;
}

[data-testid="stSidebar"] .stButton > button:disabled,
[data-testid="stSidebar"] .stButton > button[disabled] {
  background-color: rgba(148, 163, 184, .18) !important;
  color: #cbd5e1 !important;
  border-color: rgba(203, 213, 225, .22) !important;
  opacity: 1 !important;
}

[data-testid="stAppViewContainer"] .stButton > button:disabled,
[data-testid="stAppViewContainer"] .stButton > button[disabled] {
  background-color: #f1f5f9 !important;
  color: #475569 !important;
  border-color: #cbd5e1 !important;
  opacity: 1 !important;
}

/* Tabs and status messages sometimes inherit low-contrast theme colors. */
[data-testid="stTabs"] button,
[data-testid="stTabs"] button * {
  color: var(--ink) !important;
  opacity: 1 !important;
}

[data-testid="stAlert"],
[data-testid="stAlert"] * {
  color: var(--ink) !important;
  opacity: 1 !important;
}

/* Make focused controls obvious for keyboard users. */
.stButton > button:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus,
[data-baseweb="select"] > div:focus-within {
  outline: 3px solid rgba(99, 102, 241, .22) !important;
  outline-offset: 2px !important;
}

@media (max-width: 768px) {
  .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 1rem;
  }

  .hero-card,
  .question-card {
    border-radius: 1.25rem;
  }

  .progress-topline {
    align-items: flex-start;
    flex-direction: column;
    gap: .35rem;
  }
}

/* Login/register visibility hotfix.
   Streamlit/BaseWeb can inherit low-contrast tab and password icon colors from
   the active theme. These rules intentionally target tabs and password inputs
   with higher specificity and are kept at the end of the stylesheet. */

/* Make BOTH inactive and active auth tabs readable on the light page background. */
[data-testid="stTabs"] [role="tablist"],
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: transparent !important;
}

[data-testid="stTabs"] [role="tab"],
[data-testid="stTabs"] [data-baseweb="tab"],
[data-baseweb="tab-list"] [role="tab"],
[data-baseweb="tab-list"] [data-baseweb="tab"] {
  background: transparent !important;
  color: #334155 !important;
  -webkit-text-fill-color: #334155 !important;
  opacity: 1 !important;
  font-weight: 750 !important;
  text-shadow: none !important;
}

[data-testid="stTabs"] [role="tab"] *,
[data-testid="stTabs"] [data-baseweb="tab"] *,
[data-baseweb="tab-list"] [role="tab"] *,
[data-baseweb="tab-list"] [data-baseweb="tab"] * {
  color: #334155 !important;
  -webkit-text-fill-color: #334155 !important;
  opacity: 1 !important;
  text-shadow: none !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
[data-baseweb="tab-list"] [role="tab"][aria-selected="true"],
[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {
  color: #ef4444 !important;
  -webkit-text-fill-color: #ef4444 !important;
  border-bottom-color: #ef4444 !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] *,
[data-baseweb="tab-list"] [role="tab"][aria-selected="true"] *,
[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] * {
  color: #ef4444 !important;
  -webkit-text-fill-color: #ef4444 !important;
  opacity: 1 !important;
  font-weight: 850 !important;
}

[data-testid="stTabs"] [role="tab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[data-baseweb="tab-list"] [role="tab"]:hover,
[data-baseweb="tab-list"] [data-baseweb="tab"]:hover {
  color: #dc2626 !important;
  -webkit-text-fill-color: #dc2626 !important;
}

[data-testid="stTabs"] [role="tab"]:hover *,
[data-testid="stTabs"] [data-baseweb="tab"]:hover *,
[data-baseweb="tab-list"] [role="tab"]:hover *,
[data-baseweb="tab-list"] [data-baseweb="tab"]:hover * {
  color: #dc2626 !important;
  -webkit-text-fill-color: #dc2626 !important;
}

/* Fix password reveal icon area: keep the icon button on a white input surface. */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="base-input"] {
  background: #ffffff !important;
  color: var(--ink) !important;
  border-color: #cbd5e1 !important;
  border-radius: 1rem !important;
  overflow: hidden !important;
}

[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stTextInput"] [data-baseweb="base-input"] > div {
  background: #ffffff !important;
  color: var(--ink) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input[type="text"],
[data-testid="stTextInput"] input[type="password"] {
  background: #ffffff !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  caret-color: var(--primary) !important;
}

[data-testid="stTextInput"]:has(input[type="password"]) button,
[data-testid="stTextInput"]:has(input[type="password"]) [role="button"],
[data-testid="stTextInput"]:has(input[type="password"]) [aria-label*="password" i] {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  border: 0 !important;
  box-shadow: none !important;
  opacity: 1 !important;
}

[data-testid="stTextInput"]:has(input[type="password"]) button:hover,
[data-testid="stTextInput"]:has(input[type="password"]) [role="button"]:hover,
[data-testid="stTextInput"]:has(input[type="password"]) [aria-label*="password" i]:hover {
  background: #f8fafc !important;
  color: #111827 !important;
}

[data-testid="stTextInput"]:has(input[type="password"]) button svg,
[data-testid="stTextInput"]:has(input[type="password"]) [role="button"] svg,
[data-testid="stTextInput"]:has(input[type="password"]) [aria-label*="password" i] svg,
[data-testid="stTextInput"]:has(input[type="password"]) svg {
  color: #111827 !important;
  fill: currentColor !important;
  stroke: currentColor !important;
  opacity: 1 !important;
}

/* Fallback for older DOM shapes where the reveal icon is a trailing div instead of a button. */
[data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="input"] > div:last-child,
[data-testid="stTextInput"]:has(input[type="password"]) [data-baseweb="base-input"] > div:last-child {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# SMALL HELPERS
# ============================================================================
def safe_html(value: Any) -> str:
    """Escape user/generated content before rendering inside custom HTML."""
    return html.escape(str(value or ""))


def stable_question_id(question: dict[str, Any]) -> str:
    """Return a stable id even if generated content is missing question_id."""
    explicit_id = question.get("question_id") or question.get("id")
    if explicit_id:
        return str(explicit_id)

    raw = f"{question.get('topic', '')}|{question.get('question_type', '')}|{question.get('prompt', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def st_version_tuple() -> tuple[int, int, int]:
    parts = []
    for piece in st.__version__.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple((parts + [0, 0, 0])[:3])


def supports_empty_radio() -> bool:
    return st_version_tuple() >= (1, 27, 0)


def make_container(border: bool = False):
    """Use bordered containers when supported, fallback gracefully on older Streamlit."""
    try:
        return st.container(border=border)
    except TypeError:
        return st.container()


def get_topic_display_name(topic_key: str) -> str:
    topic_names = {
        MIXED_TOPIC: "Random mixed practice",
        "variables": "Variables & Data Types",
        "primitive_types": "Primitive Types",
        "strings": "Strings",
        "operators": "Operators",
        "conditionals": "If / Else",
        "loops": "Loops",
        "arrays": "Arrays",
        "methods": "Methods & Functions",
        "classes_objects": "Classes & Objects",
        "debugging": "Debugging",
        "code_tracing": "Code Tracing",
        "constructors": "Constructors",
        "encapsulation": "Encapsulation",
        "inheritance": "Inheritance",
        "polymorphism": "Polymorphism",
        "interfaces": "Interfaces",
        "exceptions": "Exceptions",
        "recursion": "Recursion",
        "arraylist": "ArrayList",
        "algorithms": "Algorithms",
    }
    return topic_names.get(topic_key, str(topic_key).replace("_", " ").title())


def friendly_question_type(qtype: str) -> str:
    labels = {
        "multiple_choice": "Multiple choice",
        "fill_blank": "Fill in the blank",
        "code_tracing": "Code tracing",
        "short_code": "Code answer",
        "debugging": "Debugging",
        "free_response": "Written answer",
    }
    return labels.get(qtype, str(qtype).replace("_", " ").title())


def difficulty_label(level: int) -> str:
    labels = {
        1: "1 · Beginner warm-up",
        2: "2 · Basic practice",
        3: "3 · Growing challenge",
        4: "4 · Confident challenge",
        5: "5 · Advanced beginner",
    }
    return labels.get(level, str(level))


def challenge_label(probability: float | None) -> str:
    """Show a friendly label instead of exposing model probability jargon."""
    if probability is None:
        return "Personalized practice"
    if probability >= 0.82:
        return "Warm-up"
    if probability >= 0.62:
        return "Good challenge"
    if probability >= 0.42:
        return "Stretch question"
    return "Growth zone"


# ============================================================================
# DATA LOADING
# ============================================================================
def ensure_generated_content() -> None:
    if not (DATA_DIR / "question_bank.json").exists():
        generate_all_content(DATA_DIR, min_questions=1100)


@st.cache_data(show_spinner=False)
def load_content() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    ensure_generated_content()
    question_bank = read_json(DATA_DIR / "question_bank.json", default=[])
    hint_map = read_json(DATA_DIR / "misconception_hint_map.json", default={})
    policy = read_json(DATA_DIR / "pedagogical_policy.json", default={})

    if not isinstance(question_bank, list):
        question_bank = []
    if not isinstance(hint_map, dict):
        hint_map = {}
    if not isinstance(policy, dict):
        policy = {}

    return question_bank, hint_map, policy


@st.cache_resource(show_spinner=False)
def cached_model_package():
    return load_model_package(MODEL_PATH)


def available_topics(question_bank: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "variables",
        "primitive_types",
        "strings",
        "operators",
        "conditionals",
        "loops",
        "arrays",
        "methods",
        "classes_objects",
        "debugging",
        "code_tracing",
    ]
    present = {q.get("topic") for q in question_bank}
    ordered = [t for t in preferred if t in present]
    ordered.extend(sorted(t for t in present if t and t not in ordered))
    return ordered


# ============================================================================
# AUTHENTICATION & SAVED PROGRESS
# ============================================================================
def progress_payload() -> dict[str, Any]:
    student = dict(st.session_state.get("student", new_student_model()))
    answered_ids = sorted(set(st.session_state.get("answered_question_ids", set())))
    student["answered_question_ids"] = answered_ids
    return {
        "student": student,
        "answered_question_ids": answered_ids,
        "topic_filter": st.session_state.get("active_topic_filter", MIXED_TOPIC),
        "session_number": int(st.session_state.get("session_number", 1)),
        "session_difficulty": int(st.session_state.get("session_difficulty", student.get("current_difficulty", 1))),
        "session_answered_ids": sorted(set(st.session_state.get("session_answered_ids", set()))),
        "session_results": list(st.session_state.get("session_results", [])),
        "session_adjusted": bool(st.session_state.get("session_adjusted", False)),
        "current_question": st.session_state.get("current_question"),
        "hint_level": int(st.session_state.get("hint_level", 0)),
        "feedback": st.session_state.get("feedback"),
        "show_explanation": bool(st.session_state.get("show_explanation", False)),
        "answer_nonce": int(st.session_state.get("answer_nonce", 0)),
        "best_streak": int(st.session_state.get("best_streak", 0)),
    }


def persist_progress() -> None:
    user = st.session_state.get("auth_user")
    if user:
        save_progress(int(user["id"]), progress_payload())


def apply_loaded_progress(progress: dict[str, Any]) -> None:
    student = progress.get("student") or new_student_model()
    answered_ids = set(progress.get("answered_question_ids", [])) | set(student.get("answered_question_ids", []))
    student["answered_question_ids"] = sorted(answered_ids)

    st.session_state.student = student
    st.session_state.answered_question_ids = answered_ids
    st.session_state.active_topic_filter = progress.get("topic_filter", MIXED_TOPIC)
    st.session_state.pending_topic_filter = st.session_state.active_topic_filter
    st.session_state.session_number = int(progress.get("session_number", 1))
    st.session_state.session_difficulty = int(progress.get("session_difficulty", student.get("current_difficulty", 1)))
    st.session_state.session_answered_ids = set(progress.get("session_answered_ids", []))
    st.session_state.session_results = list(progress.get("session_results", []))
    st.session_state.session_adjusted = bool(progress.get("session_adjusted", False))
    st.session_state.current_question = progress.get("current_question")
    st.session_state.hint_level = int(progress.get("hint_level", 0))
    st.session_state.feedback = progress.get("feedback")
    st.session_state.show_explanation = bool(progress.get("show_explanation", False))
    st.session_state.answer_nonce = int(progress.get("answer_nonce", 0))
    st.session_state.best_streak = int(progress.get("best_streak", student.get("current_streak", 0)))


def load_user_into_session(user: dict[str, Any]) -> None:
    st.session_state.auth_user = user
    apply_loaded_progress(load_progress(int(user["id"])))


def clear_auth_state() -> None:
    for key in [
        "auth_user",
        "student",
        "answered_question_ids",
        "active_topic_filter",
        "pending_topic_filter",
        "session_number",
        "session_difficulty",
        "session_answered_ids",
        "session_results",
        "session_adjusted",
        "current_question",
        "hint_level",
        "feedback",
        "show_explanation",
        "answer_nonce",
        "best_streak",
    ]:
        st.session_state.pop(key, None)


def render_auth_screen() -> dict[str, Any]:
    user = st.session_state.get("auth_user")
    if user:
        return user

    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-eyebrow">🔐 Hosted account login</div>
          <h1>Log in to continue your Java practice.</h1>
          <p>Your progress, answered questions, current difficulty, and student model are saved to the database attached to your account.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            username_or_email = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")
            if submitted:
                user = authenticate_user(username_or_email, password)
                if user is None:
                    st.error("Invalid username/email or password.")
                else:
                    load_user_into_session(user)
                    st.success("Logged in successfully.")
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            display_name = st.text_input("Display name", placeholder="Example: Ana")
            username = st.text_input("Username", placeholder="at least 3 characters")
            email = st.text_input("Email optional")
            password = st.text_input("Password", type="password", placeholder="at least 6 characters")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True, type="primary")
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        user = create_user(username, password, display_name=display_name, email=email)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        load_user_into_session(user)
                        st.success("Account created. Your first session starts at difficulty 1.")
                        st.rerun()

    st.info("Hosting mode: on Streamlit Cloud, accounts and progress are saved in Supabase. Offline local testing falls back to SQLite if Supabase secrets are not configured.")
    st.stop()


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================
def start_new_learning_session(topic_filter: str | None = None, increment_session_number: bool = True) -> None:
    student = st.session_state.get("student", new_student_model())
    difficulty = max(1, min(5, int(student.get("current_difficulty", 1) or 1)))
    student["current_difficulty"] = difficulty

    st.session_state.student = student
    st.session_state.active_topic_filter = topic_filter or st.session_state.get("pending_topic_filter", MIXED_TOPIC)
    st.session_state.pending_topic_filter = st.session_state.active_topic_filter
    st.session_state.session_difficulty = difficulty
    st.session_state.session_answered_ids = set()
    st.session_state.session_results = []
    st.session_state.session_adjusted = False
    st.session_state.current_question = None
    st.session_state.hint_level = 0
    st.session_state.feedback = None
    st.session_state.show_explanation = False
    st.session_state.answer_nonce = st.session_state.get("answer_nonce", 0) + 1
    if increment_session_number:
        st.session_state.session_number = int(st.session_state.get("session_number", 1)) + 1
    persist_progress()


def reset_learning_progress_for_current_user() -> None:
    user = st.session_state.get("auth_user")
    if not user:
        return
    progress = reset_saved_progress(int(user["id"]))
    apply_loaded_progress(progress)
    persist_progress()


def completed_this_session() -> int:
    return len(st.session_state.get("session_results", []))


def session_is_complete() -> bool:
    return completed_this_session() >= SESSION_LENGTH


def adjust_difficulty_after_session() -> tuple[int, int, float, str]:
    """Adjust only once, after a 20-question session is complete."""
    results = st.session_state.get("session_results", [])
    old = int(st.session_state.get("session_difficulty", 1))
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / max(1, len(results))

    if st.session_state.get("session_adjusted", False):
        new = int(st.session_state.student.get("current_difficulty", old))
        return old, new, accuracy, "already adjusted"

    if correct >= 17:
        new = min(5, old + 1)
        reason = "level up"
    elif correct < 12:
        new = max(1, old - 1)
        reason = "level down / remediation"
    else:
        new = old
        reason = "stay"

    st.session_state.student["current_difficulty"] = new
    st.session_state.session_adjusted = True
    persist_progress()
    return old, new, accuracy, reason


# ============================================================================
# DATA FOR UI
# ============================================================================
def accuracy(student: dict[str, Any]) -> float:
    attempts = student.get("attempts", 0)
    if not attempts:
        return 0.0
    return (student.get("correct", 0) / attempts) * 100


def session_accuracy() -> float:
    results = st.session_state.get("session_results", [])
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct")) / len(results) * 100


def progress_counts() -> tuple[int, int, float]:
    completed = completed_this_session()
    pct = min(100, round((completed / SESSION_LENGTH) * 100))
    return completed, SESSION_LENGTH, pct


def safe_success_probability(question: dict[str, Any], student: dict[str, Any], model_package: Any) -> float | None:
    try:
        return float(model_success_probability(student, question, model_package))
    except Exception:
        return None


def topic_success_rate(student: dict[str, Any], topic: str) -> float:
    attempts = student.get("attempts_by_topic", {}).get(topic, 0)
    if not attempts:
        return 0.0
    correct = student.get("correct_by_topic", {}).get(topic, 0)
    return correct / max(1, attempts)


def recent_topic_success_rate(student: dict[str, Any], topic: str, window: int = 20) -> float:
    history = [h for h in student.get("recent_history", [])[-window:] if h.get("topic") == topic]
    if not history:
        return 0.0
    return sum(1 for h in history if h.get("correct")) / len(history)


def mastery_rows(student: dict[str, Any], topics: list[str]) -> list[dict[str, Any]]:
    rows = []
    for topic in topics:
        rows.append(
            {
                "Topic": get_topic_display_name(topic),
                "Long-term": round(topic_success_rate(student, topic) * 100),
                "Recent": round(recent_topic_success_rate(student, topic) * 100),
            }
        )
    return rows


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar(user: dict[str, Any], student: dict[str, Any], question_bank: list[dict[str, Any]]) -> None:
    topics = available_topics(question_bank)
    option_values = [MIXED_TOPIC, *topics]
    option_labels = [get_topic_display_name(t) for t in option_values]
    current_filter = st.session_state.get("active_topic_filter", MIXED_TOPIC)
    if current_filter not in option_values:
        current_filter = MIXED_TOPIC

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-title">
              <span class="sidebar-icon">☕</span>
              <div>
                <div style="font-size:1.05rem;font-weight:900;">Java Coach</div>
                <div style="font-size:.85rem;color:#cbd5e1;">Signed in as {safe_html(user.get('display_name') or user.get('username'))}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚪 Log out", use_container_width=True):
            persist_progress()
            clear_auth_state()
            st.rerun()

        st.divider()
        st.markdown("**Practice settings**")
        selected_label = st.selectbox(
            "Question category",
            option_labels,
            index=option_values.index(current_filter),
            help="Choose one topic or mixed practice for the next 20-question session.",
        )
        selected_topic = option_values[option_labels.index(selected_label)]
        st.session_state.pending_topic_filter = selected_topic

        if st.button("▶️ Start new 20-question session", use_container_width=True, type="primary"):
            start_new_learning_session(selected_topic, increment_session_number=True)
            st.rerun()

        reset_confirm = st.checkbox("I want to reset all saved progress")
        if st.button("🧹 Reset saved progress", use_container_width=True, disabled=not reset_confirm):
            reset_learning_progress_for_current_user()
            st.rerun()

        completed, total, _pct = progress_counts()
        session_correct = sum(1 for r in st.session_state.get("session_results", []) if r.get("correct"))
        st.markdown(
            f"""
            <div class="session-info">
              <h3>📊 Your progress</h3>
              <div class="info-item">
                <span class="info-label">Difficulty</span>
                <span class="info-value">{st.session_state.get('session_difficulty', 1)}</span>
              </div>
              <div class="info-item">
                <span class="info-label">Topic</span>
                <span class="info-value" style="font-size:.95rem;">{safe_html(get_topic_display_name(current_filter))}</span>
              </div>
              <div class="info-item">
                <span class="info-label">This session</span>
                <span class="info-value">{completed}/{total}</span>
              </div>
              <div class="info-item">
                <span class="info-label">Session correct</span>
                <span class="info-value">{session_correct}</span>
              </div>
              <div class="info-item">
                <span class="info-label">All-time correct</span>
                <span class="info-value">{student.get('correct', 0)}</span>
              </div>
              <div class="info-item">
                <span class="info-label">All-time attempts</span>
                <span class="info-value">{student.get('attempts', 0)}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='mastery-caption'>Topic confidence updates as you answer questions.</div>", unsafe_allow_html=True)
        rows = mastery_rows(student, topics)
        if rows:
            df = pd.DataFrame(rows)
            fig = px.bar(
                df,
                x="Recent",
                y="Topic",
                orientation="h",
                range_x=[0, 100],
                labels={"Recent": "Recent mastery %", "Topic": ""},
                height=max(260, min(560, len(df) * 34)),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=8, t=8, b=0),
                xaxis=dict(showgrid=False, tickfont=dict(color="#cbd5e1")),
                yaxis=dict(tickfont=dict(color="#e5e7eb")),
                font=dict(color="#e5e7eb"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# MAIN CONTENT
# ============================================================================
def render_header(user: dict[str, Any]) -> None:
    student = st.session_state.student
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-eyebrow">✨ Adaptive Java practice</div>
          <h1>Welcome back, {safe_html(user.get('display_name') or user.get('username'))}.</h1>
          <p>Each session has 20 non-repeating questions at your current difficulty: {safe_html(difficulty_label(st.session_state.get('session_difficulty', student.get('current_difficulty', 1))))}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(student: dict[str, Any]) -> None:
    stats = [
        ("✅", "All-time correct", student.get("correct", 0)),
        ("📝", "All-time attempts", student.get("attempts", 0)),
        ("🔥", "Streak", student.get("current_streak", 0)),
        ("🎯", "All-time accuracy", f"{accuracy(student):.0f}%"),
    ]

    columns = st.columns(4)
    for col, (icon, label, value) in zip(columns, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-box">
                  <div class="stat-label"><span>{icon}</span>{label}</div>
                  <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_progress_bar() -> None:
    completed, total, pct = progress_counts()
    st.markdown(
        f"""
        <div class="progress-card">
          <div class="progress-topline">
            <span>20-question session progress</span>
            <span>{completed} of {total} questions completed · {pct}% · session accuracy {session_accuracy():.0f}%</span>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:{pct}%;"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filtered_available_questions(question_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answered = set(st.session_state.get("answered_question_ids", set())) | set(st.session_state.get("session_answered_ids", set()))
    topic_filter = st.session_state.get("active_topic_filter", MIXED_TOPIC)
    difficulty = int(st.session_state.get("session_difficulty", 1))

    questions = [
        q
        for q in question_bank
        if int(q.get("difficulty", 1)) == difficulty
        and stable_question_id(q) not in answered
        and (topic_filter == MIXED_TOPIC or q.get("topic") == topic_filter)
    ]
    return questions


def current_or_next_question(
    question_bank: list[dict[str, Any]],
    student: dict[str, Any],
    model_package: Any,
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    if session_is_complete():
        return None

    if st.session_state.get("current_question") is not None:
        return st.session_state.current_question

    available = filtered_available_questions(question_bank)
    if not available:
        st.session_state.current_question = None
        return None

    target_cfg = policy.get("target_success_probability", {"lower": 0.65, "upper": 0.80})
    try:
        selected = choose_next_question(
            available,
            student,
            model_package=model_package,
            target_range=(target_cfg.get("lower", 0.65), target_cfg.get("upper", 0.80)),
        )
    except Exception:
        selected = None

    st.session_state.current_question = selected or available[0]
    persist_progress()
    return st.session_state.current_question


def question_header_html(question: dict[str, Any], student: dict[str, Any], model_package: Any) -> str:
    topic = question.get("topic", "")
    qtype = question.get("question_type", "multiple_choice")
    probability = safe_success_probability(question, student, model_package)

    return f"""
    <div class="question-card">
      <div class="question-metadata">
        <span class="badge">📚 {safe_html(get_topic_display_name(topic))}</span>
        <span class="badge badge-soft">Level {safe_html(question.get('difficulty', st.session_state.get('session_difficulty', 1)))}</span>
        <span class="badge badge-soft">{safe_html(friendly_question_type(qtype))}</span>
        <span class="badge badge-soft">⚡ {safe_html(challenge_label(probability))}</span>
      </div>
      <div class="question-text">{safe_html(question.get('prompt', ''))}</div>
      <div class="small-note">Take your best guess. Hints are here to help, not to penalize you.</div>
    </div>
    """


def render_answer_input(question: dict[str, Any], disabled: bool = False) -> Any:
    qtype = question.get("question_type", "multiple_choice")
    qid = stable_question_id(question)
    key = f"answer_{qid}_{st.session_state.get('answer_nonce', 0)}"

    if qtype == "multiple_choice":
        choices = question.get("choices") or []
        if not choices:
            return st.text_input(
                "Your answer",
                key=key,
                placeholder="Type your answer...",
                disabled=disabled,
            )

        if supports_empty_radio():
            return st.radio(
                "Choose one answer:",
                choices,
                index=None,
                key=key,
                disabled=disabled,
            )

        placeholder = "— Select an answer —"
        selected = st.radio(
            "Choose one answer:",
            [placeholder, *choices],
            index=0,
            key=f"{key}_fallback",
            disabled=disabled,
        )
        return "" if selected == placeholder else selected

    if qtype in {"short_code", "debugging"}:
        return st.text_area(
            "Your answer:",
            key=key,
            height=150,
            placeholder="Type your code or explanation here...",
            disabled=disabled,
        )

    return st.text_input(
        "Your answer:",
        key=key,
        placeholder="Type your answer...",
        disabled=disabled,
    )


def render_hint_panel(question: dict[str, Any], hint_map: dict[str, Any]) -> None:
    hint_level = st.session_state.get("hint_level", 0)
    if hint_level <= 0:
        return

    try:
        hint_text = get_hint(question, hint_map, hint_level=hint_level)
    except Exception:
        hint_text = "Look for the key Java concept in the question and eliminate choices that do not match it."

    st.markdown(
        f"""
        <div class="hint-box">
          <strong>💡 Hint {hint_level}:</strong> {safe_html(hint_text)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_actions(question: dict[str, Any], answer: Any, hint_map: dict[str, Any]) -> None:
    submit_col, hint_col = st.columns([1.25, 1])

    with submit_col:
        if st.button("✅ Check my answer", use_container_width=True, type="primary", key="submit_btn"):
            if answer is None or not str(answer).strip():
                st.warning("Choose or type an answer before submitting.")
                return

            try:
                is_correct = check_answer(question, answer)
            except Exception as exc:
                st.error(f"I couldn't check that answer yet: {exc}")
                return

            fixed_session_difficulty = int(st.session_state.get("session_difficulty", 1))
            updated_student = update_after_answer(
                st.session_state.student,
                question,
                is_correct=is_correct,
                hints_used=st.session_state.get("hint_level", 0),
            )
            # Keep the current session fixed at one difficulty. The next session level is adjusted after 20 answers.
            updated_student["current_difficulty"] = fixed_session_difficulty
            updated_student["phase"] = "adaptive"
            updated_student["diagnostic_remaining"] = 0
            st.session_state.student = updated_student
            st.session_state.best_streak = max(
                st.session_state.get("best_streak", 0),
                st.session_state.student.get("current_streak", 0),
            )

            qid = stable_question_id(question)
            answered_ids = set(st.session_state.get("answered_question_ids", set()))
            answered_ids.add(qid)
            st.session_state.answered_question_ids = answered_ids
            st.session_state.student["answered_question_ids"] = sorted(answered_ids)

            session_answered_ids = set(st.session_state.get("session_answered_ids", set()))
            session_answered_ids.add(qid)
            st.session_state.session_answered_ids = session_answered_ids
            st.session_state.session_results = [
                *st.session_state.get("session_results", []),
                {
                    "question_id": qid,
                    "topic": question.get("topic"),
                    "difficulty": fixed_session_difficulty,
                    "correct": bool(is_correct),
                    "hints_used": int(st.session_state.get("hint_level", 0)),
                },
            ]

            st.session_state.feedback = {"correct": is_correct, "answer": answer}
            st.session_state.show_explanation = True
            persist_progress()

            if is_correct:
                st.balloons()
            st.rerun()

    with hint_col:
        if st.session_state.get("hint_level", 0) < 3:
            label = "💡 Show a hint" if st.session_state.get("hint_level", 0) == 0 else "💡 Show another hint"
            if st.button(label, use_container_width=True, key="hint_btn"):
                st.session_state.hint_level = st.session_state.get("hint_level", 0) + 1
                persist_progress()
                st.rerun()
        else:
            st.button("💡 All hints shown", use_container_width=True, disabled=True)


def render_feedback(question: dict[str, Any]) -> None:
    feedback = st.session_state.get("feedback")
    if not feedback:
        return

    if feedback["correct"]:
        st.markdown(
            """
            <div class="feedback-correct">
              <strong>🎉 Nice work — that is correct.</strong><br>
              You are strengthening the right Java concept. Review the explanation, then keep the momentum going.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="feedback-incorrect">
              <strong>Not quite — but this is useful practice.</strong><br>
              Read the explanation below, then try the next question with that idea fresh in mind.
            </div>
            """,
            unsafe_allow_html=True,
        )

    explanation = question.get("explanation") or "Review the concept and try another question."
    st.markdown(
        f"""
        <div class="explanation-box">
          <strong>📖 Explanation:</strong> {safe_html(explanation)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_label = "Finish session →" if session_is_complete() else "Next question →"
    if st.button(next_label, use_container_width=True, type="primary", key="next_btn"):
        st.session_state.current_question = None
        st.session_state.hint_level = 0
        st.session_state.feedback = None
        st.session_state.show_explanation = False
        st.session_state.answer_nonce = st.session_state.get("answer_nonce", 0) + 1
        persist_progress()
        st.rerun()


def render_question_card(
    question: dict[str, Any],
    student: dict[str, Any],
    model_package: Any,
    hint_map: dict[str, Any],
) -> None:
    st.markdown(question_header_html(question, student, model_package), unsafe_allow_html=True)

    with make_container(border=True):
        answered = bool(st.session_state.get("show_explanation"))
        answer = render_answer_input(question, disabled=answered)
        render_hint_panel(question, hint_map)
        if not answered:
            render_answer_actions(question, answer, hint_map)
        render_feedback(question)


def render_completion_screen(student: dict[str, Any]) -> None:
    old, new, final_accuracy, reason = adjust_difficulty_after_session()
    results = st.session_state.get("session_results", [])
    correct = sum(1 for r in results if r.get("correct"))
    best_streak = max(st.session_state.get("best_streak", 0), student.get("current_streak", 0))

    if new > old:
        next_message = "Great performance. The next session will move up one difficulty level."
    elif new < old:
        next_message = "The next session will step down for remediation and confidence building."
    else:
        next_message = "The next session will stay at this difficulty for more practice."

    st.markdown(
        f"""
        <div class="empty-state">
          <div class="trophy">🏆</div>
          <h2 style="margin:.2rem 0 .35rem 0;">20-question session complete!</h2>
          <p style="color:#64748b;font-size:1.05rem;margin:0 0 1.25rem 0;">
            {safe_html(next_message)}
          </p>
          <div class="info-box" style="display:inline-block;text-align:left;min-width:min(100%,28rem);">
            <strong>Session score:</strong> {correct}/{len(results)} ({final_accuracy * 100:.0f}%)<br>
            <strong>Best streak:</strong> 🔥 {best_streak}<br>
            <strong>Completed difficulty:</strong> {old}<br>
            <strong>Next session difficulty:</strong> {new}<br>
            <strong>Adjustment rule:</strong> {safe_html(reason)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Start the next 20-question session", use_container_width=True, type="primary"):
        start_new_learning_session(st.session_state.get("active_topic_filter", MIXED_TOPIC), increment_session_number=True)
        st.rerun()


def render_no_questions_screen(question_bank: list[dict[str, Any]]) -> None:
    topic_filter = st.session_state.get("active_topic_filter", MIXED_TOPIC)
    difficulty = st.session_state.get("session_difficulty", 1)
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="trophy">📚</div>
          <h2 style="margin:.2rem 0 .35rem 0;">No unused questions left here.</h2>
          <p style="color:#64748b;font-size:1.05rem;margin:0 0 1.25rem 0;">
            You have already answered all available questions for {safe_html(get_topic_display_name(topic_filter))} at difficulty {difficulty}.
            Choose another topic, choose mixed practice, or reset saved progress if you want to restart.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Question bank currently has {len(question_bank)} total questions.")


# ============================================================================
# MAIN APP
# ============================================================================
def main() -> None:
    user = render_auth_screen()

    with st.spinner("Preparing your Java practice session..."):
        question_bank, hint_map, policy = load_content()
        model_package = cached_model_package()

    # Fill missing learning state if an old browser session had only auth info.
    if "student" not in st.session_state:
        apply_loaded_progress(load_progress(int(user["id"])))

    render_sidebar(user, st.session_state.student, question_bank)
    render_header(user)
    render_stats(st.session_state.student)
    render_progress_bar()

    if not question_bank:
        st.error("No questions are available yet. Check that generated content was created successfully.")
        return

    question = current_or_next_question(question_bank, st.session_state.student, model_package, policy)
    if question is None:
        if session_is_complete():
            render_completion_screen(st.session_state.student)
        else:
            render_no_questions_screen(question_bank)
    else:
        render_question_card(question, st.session_state.student, model_package, hint_map)


if __name__ == "__main__":
    main()
