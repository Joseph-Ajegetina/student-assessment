"""
Constants and configuration for the Ashesi Student Success Prediction project.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data file paths
DATA_FILES = {
    'application': DATA_DIR / 'application.csv',
    'transcript': DATA_DIR / 'transcript.csv',
    'cgpa': DATA_DIR / 'cgpa.csv',
    'wassce': DATA_DIR / 'WASSCE.csv',
    'ib': DATA_DIR / 'IB.csv',
    'o_a_level': DATA_DIR / 'o_a_level.csv',
    'hsdiploma': DATA_DIR / 'HSDiploma.csv',
    'french': DATA_DIR / 'FRENCH.csv',
    'other': DATA_DIR / 'Other.csv',
    'ajc': DATA_DIR / 'AJC.csv',
}

# Academic policies
PROBATION_THRESHOLD = 2.0  # CGPA below this triggers probation
DEANS_LIST_THRESHOLD = 3.5  # GPA above this for Dean's List
MAJOR_SUCCESS_THRESHOLD = 3.0  # CGPA for "success" in major
STANDARD_SEMESTERS = 8  # Normal graduation time

# High school exam score normalization mappings
WASSCE_GRADE_MAP = {
    'A1': 90, 'B2': 80, 'B3': 70, 'C4': 60, 'C5': 55,
    'C6': 50, 'D7': 45, 'E8': 40, 'F9': 0
}

IB_MAX_SCORE = 7  # IB scores are 1-7

A_LEVEL_GRADE_MAP = {
    'A*': 100, 'A': 90, 'B': 80, 'C': 70, 'D': 60, 'E': 50, 'U': 0,
    # O-Level variations
    '1': 90, '2': 80, '3': 70, '4': 60, '5': 55, '6': 50, '7': 45, '8': 40, '9': 0
}

# Exam type categories
EXAM_TYPES = {
    'wassce': ['WASSCE', 'WASSCE - Ghana', 'WASSCE - Nigeria', 'WASSCE - Sierra Leone', 'WASSCE - Gambia', 'WASSCE - Liberia'],
    'ib': ['IB Diploma', 'International Baccalaureate'],
    'a_level': ['IGCSE A Level', "IGCSE 'A' Level", 'A-Level', 'O-Level', 'K.C.S.E', 'M.S.C.E'],
    'french': ['Baccalauréat', 'Baccalauréat 1', 'French Baccalaureate'],
    'hsdiploma': ['High School Diploma'],
    'other': ['Other Official Exam']
}

# Math course codes for track detection
MATH_TRACK_COURSES = {
    'calculus': ['MATH142', 'MATH142_A', 'Calculus I', 'Calculus 1'],
    'precalculus': ['MATH141', 'MATH141_A', 'MATH141_B', 'Pre-Calculus', 'Pre Calculus'],
    'college_algebra': ['MATH140', 'MATH140_A', 'College Algebra', 'Algebra']
}

# Core subjects for exam normalization
CORE_SUBJECTS = {
    'math': ['Mathematics', 'Maths', 'Math', 'Core Mathematics', 'Additional Mathematics', 'Elective Math', 'Further Mathematics'],
    'english': ['English', 'English Language', 'English A', 'English Literature'],
    'science': ['Physics', 'Chemistry', 'Biology', 'Integrated Science', 'Physical Science', 'Life Science']
}

# Temporal validation splits
TEMPORAL_SPLITS = {
    'train': (2011, 2019),
    'validation': (2020, 2021),
    'test': (2022, 2025)
}

# Model names
MODEL_NAMES = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'XGBoost'
]

# Research questions
RESEARCH_QUESTIONS = {
    1: "Predict first-year academic struggle from admissions data",
    2: "Predict AJC cases from admissions data",
    3: "Predict major success from admissions + Year 1 data",
    4: "Predict major failure/change from admissions + Year 1 data",
    5: "Predict major success from admissions + Year 1-2 data",
    6: "Predict major failure/change from admissions + Year 1-2 data",
    7: "Compare performance across math tracks",
    8: "Can college algebra track students succeed in CS major?",
    9: "Early prediction of extended graduation time"
}
