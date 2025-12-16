"""
Data loading utilities for the Ashesi Student Success Prediction project.
Handles loading and initial cleaning of all 10 datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

# Import constants
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.constants import DATA_DIR, DATA_FILES


def load_all_datasets(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load all 10 datasets from the data directory.

    Parameters
    ----------
    verbose : bool
        Whether to print loading progress

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with dataset names as keys and DataFrames as values
    """
    datasets = {}

    for name, path in DATA_FILES.items():
        if verbose:
            print(f"Loading {name}...", end=" ")

        try:
            # Try UTF-8 first, fall back to latin-1 for encoding issues
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin-1')

            if name == 'cgpa':
                df = df.rename(columns={'Student Ref': 'StudentRef'})

            datasets[name] = df

            if verbose:
                print(f"OK ({len(df):,} rows, {len(df.columns)} columns)")

        except FileNotFoundError:
            if verbose:
                print(f"NOT FOUND at {path}")
            datasets[name] = None
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            datasets[name] = None

    return datasets


def load_dataset(name: str) -> Optional[pd.DataFrame]:
    """
    Load a single dataset by name.

    Parameters
    ----------
    name : str
        One of: 'application', 'transcript', 'cgpa', 'wassce', 'ib',
        'o_a_level', 'hsdiploma', 'french', 'other', 'ajc'

    Returns
    -------
    pd.DataFrame or None
        The loaded DataFrame, or None if not found
    """
    if name not in DATA_FILES:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATA_FILES.keys())}")

    path = DATA_FILES[name]

    try:
        try:
            return pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin-1')
    except FileNotFoundError:
        warnings.warn(f"Dataset not found: {path}")
        return None


def get_dataset_info(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get summary information about all loaded datasets.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of loaded datasets

    Returns
    -------
    pd.DataFrame
        Summary table with rows, columns, memory usage, and missing values
    """
    info = []

    for name, df in datasets.items():
        if df is not None:
            info.append({
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory (MB)': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'Missing Values': df.isnull().sum().sum(),
                'Missing %': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
            })
        else:
            info.append({
                'Dataset': name,
                'Rows': 'N/A',
                'Columns': 'N/A',
                'Memory (MB)': 'N/A',
                'Missing Values': 'N/A',
                'Missing %': 'N/A'
            })

    return pd.DataFrame(info)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, replace spaces with underscores,
    remove special characters.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names
    """
    df = df.copy()

    # Clean column names
    new_cols = []
    for col in df.columns:
        # Convert to string, lowercase
        new_col = str(col).lower()
        # Replace spaces and special chars with underscore
        new_col = new_col.replace(' ', '_').replace('-', '_').replace('.', '_')
        new_col = new_col.replace('/', '_').replace('(', '').replace(')', '')
        new_col = new_col.replace('?', '').replace(':', '').replace(',', '')
        # Remove multiple underscores
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        new_cols.append(new_col)

    df.columns = new_cols
    return df


def get_common_students(datasets: Dict[str, pd.DataFrame],
                        student_col: str = 'StudentRef') -> set:
    """
    Find students that appear across multiple datasets.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of loaded datasets
    student_col : str
        Column name containing student identifier

    Returns
    -------
    set
        Set of student IDs appearing in multiple datasets
    """
    # Get student IDs from each dataset
    student_sets = {}

    for name, df in datasets.items():
        if df is not None:
            # Try different column name variations
            for col in [student_col, 'studentref', 'StudentRef', 'student_ref']:
                if col in df.columns:
                    student_sets[name] = set(df[col].dropna().unique())
                    break

    if not student_sets:
        return set()

    # Find intersection
    common = set.intersection(*student_sets.values())
    return common


def merge_exam_data(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all high school exam datasets into a single DataFrame.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary containing exam datasets (wassce, ib, o_a_level, etc.)

    Returns
    -------
    pd.DataFrame
        Combined exam data with exam_type indicator
    """
    exam_names = ['wassce', 'ib', 'o_a_level', 'hsdiploma', 'french', 'other']
    exam_dfs = []

    for name in exam_names:
        if name in datasets and datasets[name] is not None:
            df = datasets[name].copy()
            df['exam_source'] = name
            exam_dfs.append(df)

    if not exam_dfs:
        return pd.DataFrame()

    # Concatenate with outer join (keeps all columns)
    combined = pd.concat(exam_dfs, axis=0, ignore_index=True, sort=False)

    return combined


def extract_admission_year(df: pd.DataFrame,
                           year_col: str = 'Admission Year') -> pd.DataFrame:
    """
    Extract numeric admission year from various formats.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with admission year column
    year_col : str
        Column name containing admission year

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned admission_year column
    """
    df = df.copy()

    if year_col not in df.columns:
        return df

    def parse_year(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        # Handle formats like "2013-2014"
        if '-' in val_str:
            try:
                return int(val_str.split('-')[0])
            except:
                return np.nan
        # Handle plain year
        try:
            return int(float(val_str))
        except:
            return np.nan

    df['admission_year_numeric'] = df[year_col].apply(parse_year)
    return df


def get_student_overview(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create an overview of each unique student across all datasets.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of loaded datasets

    Returns
    -------
    pd.DataFrame
        One row per student with presence flags in each dataset
    """
    student_col = 'StudentRef'

    # Collect all unique students
    all_students = set()
    for name, df in datasets.items():
        if df is not None and student_col in df.columns:
            all_students.update(df[student_col].dropna().unique())

    # Create overview DataFrame
    overview = pd.DataFrame({'StudentRef': list(all_students)})

    # Add presence flags
    for name, df in datasets.items():
        if df is not None and student_col in df.columns:
            students_in_dataset = set(df[student_col].dropna().unique())
            overview[f'in_{name}'] = overview['StudentRef'].isin(students_in_dataset)

    return overview


# Convenience function for quick exploration
def quick_explore(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print a quick summary of a DataFrame for exploration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to explore
    name : str
        Name to display in header
    """
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values (top 10):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10))
    print(f"\nSample (first 3 rows):")
    print(df.head(3).T)


if __name__ == "__main__":
    # Quick test of data loading
    print("Testing data loader...")
    datasets = load_all_datasets()
    print("\nDataset Summary:")
    print(get_dataset_info(datasets))
