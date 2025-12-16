"""
High School Exam Score Normalizer for the Ashesi Student Success Prediction project.
Converts various exam scoring systems to a unified 0-100 scale.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional, Union, Dict, List
import warnings


# Score normalization mappings
WASSCE_GRADE_MAP = {
    'A1': 90, 'B2': 80, 'B3': 70, 'C4': 60, 'C5': 55,
    'C6': 50, 'D7': 45, 'E8': 40, 'F9': 0,
    # Handle variations
    'A': 90, 'B': 75, 'C': 55, 'D': 45, 'E': 40, 'F': 0,
    # Numeric variations
    '1': 90, '2': 80, '3': 70, '4': 60, '5': 55, '6': 50, '7': 45, '8': 40, '9': 0
}

A_LEVEL_GRADE_MAP = {
    # A-Level grades
    'A*': 100, 'A': 90, 'B': 80, 'C': 70, 'D': 60, 'E': 50, 'U': 0,
    # O-Level / IGCSE numeric grades (1-9 scale, 1 is best)
    '1': 90, '2': 80, '3': 70, '4': 60, '5': 55, '6': 50, '7': 45, '8': 40, '9': 0,
    # KCSE grades (Kenya)
    'A-': 85, 'B+': 78, 'B-': 72, 'C+': 65, 'C-': 55, 'D+': 48, 'D-': 42,
    # Handle plain letters
    'A': 90, 'B': 75, 'C': 60, 'D': 45, 'E': 35, 'F': 0
}

IB_MAX_SCORE = 7  # IB scores are 1-7

# Core subject column name patterns
MATH_PATTERNS = [
    'mathematics', 'maths', 'math', 'elective_math', 'additional_math',
    'further_math', 'calculus', 'algebra', 'math_studies', 'math_sl', 'math_hl'
]

ENGLISH_PATTERNS = [
    'english', 'english_language', 'english_a', 'english_lit', 'literature_english'
]

SCIENCE_PATTERNS = [
    'physics', 'chemistry', 'biology', 'integrated_science', 'physical_science',
    'life_science', 'general_science', 'science'
]


def normalize_wassce_grade(grade: Union[str, float, None]) -> float:
    """
    Convert WASSCE grade to 0-100 scale.

    Parameters
    ----------
    grade : str, float, or None
        WASSCE grade (e.g., 'A1', 'B2', 'C4')

    Returns
    -------
    float
        Normalized score (0-100), or NaN if invalid
    """
    if pd.isna(grade):
        return np.nan

    grade_str = str(grade).strip().upper()

    # Direct mapping
    if grade_str in WASSCE_GRADE_MAP:
        return WASSCE_GRADE_MAP[grade_str]

    # Try to extract letter+number pattern (e.g., "A1", "B2")
    match = re.match(r'^([A-F])(\d)$', grade_str)
    if match:
        combined = match.group(1) + match.group(2)
        if combined in WASSCE_GRADE_MAP:
            return WASSCE_GRADE_MAP[combined]

    # Try numeric only
    try:
        num = int(float(grade_str))
        if 1 <= num <= 9:
            return WASSCE_GRADE_MAP.get(str(num), np.nan)
    except:
        pass

    return np.nan


def normalize_ib_score(score: Union[int, float, str, None]) -> float:
    """
    Convert IB score (1-7) to 0-100 scale.

    Parameters
    ----------
    score : int, float, str, or None
        IB score (1-7)

    Returns
    -------
    float
        Normalized score (0-100), or NaN if invalid
    """
    if pd.isna(score):
        return np.nan

    try:
        numeric = float(str(score).strip())
        if 1 <= numeric <= 7:
            return (numeric / IB_MAX_SCORE) * 100
        return np.nan
    except:
        return np.nan


def normalize_a_level_grade(grade: Union[str, float, None]) -> float:
    """
    Convert A-Level/O-Level/KCSE grade to 0-100 scale.

    Parameters
    ----------
    grade : str, float, or None
        A-Level grade (e.g., 'A*', 'A', 'B', 'C', or numeric 1-9)

    Returns
    -------
    float
        Normalized score (0-100), or NaN if invalid
    """
    if pd.isna(grade):
        return np.nan

    grade_str = str(grade).strip().upper()

    # Direct mapping
    if grade_str in A_LEVEL_GRADE_MAP:
        return A_LEVEL_GRADE_MAP[grade_str]

    # Handle A* variations
    if grade_str in ['A STAR', 'ASTAR', 'A-STAR']:
        return 100

    # Try numeric
    try:
        num = int(float(grade_str))
        if 1 <= num <= 9:
            return A_LEVEL_GRADE_MAP.get(str(num), np.nan)
    except:
        pass

    return np.nan


def normalize_french_bac_score(score: Union[str, float, None]) -> float:
    """
    Convert French Baccalaureate score to 0-100 scale.
    French Bac uses /20 scale typically.

    Parameters
    ----------
    score : str, float, or None
        French Bac score (e.g., '14/20', '14.15/20', or just '14')

    Returns
    -------
    float
        Normalized score (0-100), or NaN if invalid
    """
    if pd.isna(score):
        return np.nan

    score_str = str(score).strip()

    # Handle "X/20" format
    if '/20' in score_str:
        try:
            num_part = score_str.replace('/20', '').strip()
            numeric = float(num_part)
            return (numeric / 20) * 100
        except:
            pass

    # Handle "X/100" format
    if '/100' in score_str:
        try:
            num_part = score_str.replace('/100', '').strip()
            return float(num_part)
        except:
            pass

    # Try plain numeric (assume /20 if <= 20, /100 otherwise)
    try:
        numeric = float(score_str)
        if numeric <= 20:
            return (numeric / 20) * 100
        elif numeric <= 100:
            return numeric
        return np.nan
    except:
        return np.nan


def normalize_hsdiploma_grade(grade: Union[str, float, None]) -> float:
    """
    Convert High School Diploma grade to 0-100 scale.
    Handles both letter grades and percentages.

    Parameters
    ----------
    grade : str, float, or None
        HS Diploma grade (e.g., 'A', 'B+', '85', '85%')

    Returns
    -------
    float
        Normalized score (0-100), or NaN if invalid
    """
    if pd.isna(grade):
        return np.nan

    grade_str = str(grade).strip().upper()

    # Remove % sign
    grade_str = grade_str.replace('%', '').strip()

    # Letter grade mapping (US system)
    letter_map = {
        'A+': 97, 'A': 93, 'A-': 90,
        'B+': 87, 'B': 83, 'B-': 80,
        'C+': 77, 'C': 73, 'C-': 70,
        'D+': 67, 'D': 63, 'D-': 60,
        'F': 50, 'E': 50
    }

    if grade_str in letter_map:
        return letter_map[grade_str]

    # Try numeric (assume percentage)
    try:
        numeric = float(grade_str)
        if 0 <= numeric <= 100:
            return numeric
        return np.nan
    except:
        return np.nan


def normalize_exam_score(score: Union[str, float, None],
                         exam_type: str) -> float:
    """
    Normalize a score based on the exam type.

    Parameters
    ----------
    score : str, float, or None
        The raw score/grade
    exam_type : str
        One of: 'wassce', 'ib', 'a_level', 'french', 'hsdiploma', 'other'

    Returns
    -------
    float
        Normalized score (0-100), or NaN if invalid
    """
    exam_type = exam_type.lower().strip()

    if exam_type in ['wassce', 'wassce - ghana', 'wassce - nigeria']:
        return normalize_wassce_grade(score)
    elif exam_type in ['ib', 'ib diploma', 'international baccalaureate']:
        return normalize_ib_score(score)
    elif exam_type in ['a_level', 'o_a_level', 'a-level', 'igcse', "igcse 'a' level", 'k.c.s.e', 'm.s.c.e']:
        return normalize_a_level_grade(score)
    elif exam_type in ['french', 'baccalauréat', 'baccalauréat 1', 'french baccalaureate']:
        return normalize_french_bac_score(score)
    elif exam_type in ['hsdiploma', 'high school diploma']:
        return normalize_hsdiploma_grade(score)
    else:
        # For 'other', try each normalizer and take first non-NaN
        for normalizer in [normalize_hsdiploma_grade, normalize_a_level_grade, normalize_wassce_grade]:
            result = normalizer(score)
            if not pd.isna(result):
                return result
        return np.nan


def find_subject_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """
    Find columns matching subject patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search
    patterns : List[str]
        Patterns to match (case-insensitive)

    Returns
    -------
    List[str]
        Matching column names
    """
    matching_cols = []
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        for pattern in patterns:
            if pattern in col_lower:
                matching_cols.append(col)
                break
    return matching_cols


def extract_core_subjects(df: pd.DataFrame,
                          exam_type: str) -> pd.DataFrame:
    """
    Extract and normalize core subject scores (Math, English, Science).

    Parameters
    ----------
    df : pd.DataFrame
        Exam DataFrame with subject columns
    exam_type : str
        Exam type for score normalization

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized core subject scores
    """
    result = df[['StudentRef']].copy() if 'StudentRef' in df.columns else pd.DataFrame()

    # Find and normalize Math
    math_cols = find_subject_columns(df, MATH_PATTERNS)
    if math_cols:
        # Take the first match (most likely core math)
        result['hs_math_score'] = df[math_cols[0]].apply(
            lambda x: normalize_exam_score(x, exam_type)
        )

    # Find and normalize English
    english_cols = find_subject_columns(df, ENGLISH_PATTERNS)
    if english_cols:
        result['hs_english_score'] = df[english_cols[0]].apply(
            lambda x: normalize_exam_score(x, exam_type)
        )

    # Find and normalize Science (best of physics/chemistry/biology)
    science_cols = find_subject_columns(df, SCIENCE_PATTERNS)
    if science_cols:
        # Normalize all science columns
        science_scores = pd.DataFrame()
        for col in science_cols[:3]:  # Max 3 science subjects
            science_scores[col] = df[col].apply(
                lambda x: normalize_exam_score(x, exam_type)
            )
        # Take the best science score
        result['hs_science_score'] = science_scores.max(axis=1)

    return result


def normalize_exam_dataframe(df: pd.DataFrame,
                              exam_type: str,
                              subject_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize all exam scores in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Exam DataFrame
    exam_type : str
        Type of exam for score normalization
    subject_cols : List[str], optional
        Specific columns to normalize. If None, auto-detect.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized scores
    """
    result = df.copy()

    # Auto-detect subject columns if not provided
    if subject_cols is None:
        # Skip non-subject columns
        skip_patterns = ['studentref', 'yeargroup', 'proposed', 'high_school',
                        'exam_type', 'exam_year', 'exam_month', 'elective_subjects',
                        'points', 'other', 'total_aggregate']
        subject_cols = [col for col in df.columns
                       if not any(p in col.lower() for p in skip_patterns)]

    # Normalize each subject column
    for col in subject_cols:
        if col in result.columns:
            new_col = f"{col}_normalized"
            result[new_col] = result[col].apply(
                lambda x: normalize_exam_score(x, exam_type)
            )

    return result


def compute_aggregate_scores(df: pd.DataFrame,
                             normalized_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute aggregate scores from normalized subject scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with normalized scores
    normalized_cols : List[str], optional
        Columns to aggregate. If None, auto-detect '_normalized' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional aggregate columns
    """
    result = df.copy()

    if normalized_cols is None:
        normalized_cols = [col for col in df.columns if '_normalized' in col.lower()]

    if not normalized_cols:
        return result

    # Compute aggregates
    normalized_data = df[normalized_cols]

    result['hs_mean_score'] = normalized_data.mean(axis=1)
    result['hs_max_score'] = normalized_data.max(axis=1)
    result['hs_min_score'] = normalized_data.min(axis=1)
    result['hs_std_score'] = normalized_data.std(axis=1)
    result['hs_subjects_count'] = normalized_data.notna().sum(axis=1)

    # Best 3 subjects
    result['hs_best_3_avg'] = normalized_data.apply(
        lambda row: row.nlargest(3).mean() if row.notna().sum() >= 3 else row.mean(),
        axis=1
    )

    return result


if __name__ == "__main__":
    # Test normalizers
    print("Testing WASSCE normalizer:")
    for grade in ['A1', 'B2', 'C4', 'D7', 'F9', 'A', 'invalid']:
        print(f"  {grade} -> {normalize_wassce_grade(grade)}")

    print("\nTesting IB normalizer:")
    for score in [7, 6, 5, 4, 3, 2, 1, 'invalid']:
        print(f"  {score} -> {normalize_ib_score(score)}")

    print("\nTesting A-Level normalizer:")
    for grade in ['A*', 'A', 'B', 'C', 'D', 'E', 'U', '1', '5', 'A-']:
        print(f"  {grade} -> {normalize_a_level_grade(grade)}")

    print("\nTesting French Bac normalizer:")
    for score in ['14/20', '14.15/20', '10', '75', 'invalid']:
        print(f"  {score} -> {normalize_french_bac_score(score)}")
