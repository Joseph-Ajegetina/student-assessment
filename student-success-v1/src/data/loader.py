"""
Data Loading and Integration Module for Ashesi Student Success Prediction
Properly handles unique students across all datasets with correct math track detection.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Load and integrate all datasets with proper student ID handling."""

    def __init__(self, data_path: str = 'data/'):
        self.data_path = Path(data_path)
        self.datasets = {}
        self.master_df = None
        self.semester_records = None

    def load_all_datasets(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Load all datasets from CSV files."""
        files = {
            'application': 'application.csv',
            'cgpa': 'cgpa.csv',
            'transcript': 'transcript.csv',
            'ajc': 'AJC.csv',
            'wassce': 'WASSCE.csv',
            'oa_level': 'o_a_level.csv',
            'hsdiploma': 'HSDiploma.csv',
            'french': 'FRENCH.csv',
            'ib': 'IB.csv',
            'other': 'Other.csv'
        }

        for name, filename in files.items():
            filepath = self.data_path / filename
            if filepath.exists():
                try:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)

                    # Standardize student ID column
                    df = self._standardize_student_id(df)
                    self.datasets[name] = df

                    if verbose:
                        print(f"  ✓ Loaded {name}: {len(df):,} rows, {df['student_id'].nunique():,} unique students")
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Error loading {name}: {e}")
            else:
                if verbose:
                    print(f"  ✗ File not found: {filepath}")

        return self.datasets

    def _standardize_student_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize student ID column across datasets."""
        possible_cols = ['StudentRef', 'Student Ref', 'studentref', 'student_ref']

        for col in possible_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'student_id'})
                df['student_id'] = df['student_id'].astype(str).str.strip()
                break

        return df

    def get_unique_student_summary(self) -> pd.DataFrame:
        """Get count of UNIQUE students per dataset."""
        summary = []
        for name, df in self.datasets.items():
            if 'student_id' in df.columns:
                unique_students = df['student_id'].nunique()
                total_rows = len(df)
                summary.append({
                    'Dataset': name,
                    'Total Rows': total_rows,
                    'Unique Students': unique_students,
                    'Rows per Student': round(total_rows / unique_students, 2) if unique_students > 0 else 0
                })
        return pd.DataFrame(summary)

    def get_all_unique_students(self) -> set:
        """Get set of all unique student IDs across all datasets."""
        all_students = set()
        for df in self.datasets.values():
            if 'student_id' in df.columns:
                all_students.update(df['student_id'].dropna().unique())
        return all_students

    def get_student_data_coverage(self) -> pd.DataFrame:
        """Get which datasets each student appears in."""
        all_students = self.get_all_unique_students()
        coverage = pd.DataFrame({'student_id': list(all_students)})

        for name, df in self.datasets.items():
            if 'student_id' in df.columns:
                students_in_dataset = set(df['student_id'].dropna().unique())
                coverage[f'in_{name}'] = coverage['student_id'].isin(students_in_dataset)

        # Count datasets per student
        data_cols = [col for col in coverage.columns if col.startswith('in_')]
        coverage['datasets_count'] = coverage[data_cols].sum(axis=1)

        return coverage

    def create_master_student_table(self) -> pd.DataFrame:
        """
        Create master table with ONE ROW per UNIQUE student.
        Starts from CGPA data (actual enrolled students only).
        """
        print("\n  Creating master student table (one row per unique student)...")

        if 'cgpa' not in self.datasets:
            raise ValueError("CGPA data required for master table (contains actual enrolled students)")

        # Start with unique students from CGPA (actual enrolled students)
        cgpa = self.datasets['cgpa'].copy()

        # Get unique students with their latest status
        cgpa_sorted = cgpa.sort_values(['student_id', 'Academic Year', 'Semester/Year'])
        master = cgpa_sorted.groupby('student_id').last().reset_index()

        initial_count = len(master)
        print(f"    Starting with {initial_count:,} unique ENROLLED students from CGPA data")

        # === Extract Application Features ===
        master = self._extract_application_features(master)

        # === Add High School Exam Data ===
        master = self._merge_hs_exam_data(master)

        # === Add Math Track (PROPERLY from actual placement columns) ===
        master = self._add_math_track(master)

        # === Add Academic Performance ===
        master = self._merge_academic_performance(master)

        # === Add AJC Data ===
        master = self._merge_ajc_data(master)

        # === Create derived features ===
        master = self._create_derived_features(master)

        self.master_df = master
        print(f"    ✓ Master table created: {len(master):,} unique students, {len(master.columns)} features")

        return master

    def _extract_application_features(self, master: pd.DataFrame) -> pd.DataFrame:
        """Merge and extract features from application data."""

        # Master already has some columns from CGPA (Gender, Nationality, etc.)
        # Extract what we have from master first
        if 'Gender' in master.columns:
            master['is_female'] = (master['Gender'].str.upper().isin(['F', 'FEMALE'])).astype(int)

        if 'Nationality' in master.columns:
            master['is_international'] = (~master['Nationality'].isin(['Country0', 'Ghana'])).astype(int)

        # Merge additional features from application data if available
        if 'application' in self.datasets:
            app = self.datasets['application'].copy()
            app = app.drop_duplicates('student_id')

            # Select columns to merge
            app_features = ['student_id']

            # Financial aid
            aid_col = 'Extra question: Do you Need Financial Aid?'
            if aid_col in app.columns:
                app['needs_financial_aid'] = (app[aid_col].str.lower() == 'yes').astype(int)
                app_features.append('needs_financial_aid')

            # Disadvantaged background
            if 'Disadvantaged background' in app.columns:
                app['is_disadvantaged'] = app['Disadvantaged background'].notna().astype(int)
                app_features.append('is_disadvantaged')

            # Previous application
            prev_col = 'Extra question: Have you applied to Ashesi before? If "yes" indicate the year.'
            if prev_col in app.columns:
                app['has_previous_application'] = (~app[prev_col].isin(['No', 'NaN', np.nan, ''])).astype(int)
                app_features.append('has_previous_application')

            # Intended major
            if 'Offer course name' in app.columns:
                major = app['Offer course name'].fillna('')
                app['intended_cs'] = major.str.contains('Computer Science', case=False, na=False).astype(int)
                app['intended_engineering'] = major.str.contains('Engineering', case=False, na=False).astype(int)
                app['intended_business'] = major.str.contains('Business', case=False, na=False).astype(int)
                app['intended_mis'] = major.str.contains('MIS|Information Systems', case=False, na=False).astype(int)
                app_features.extend(['intended_cs', 'intended_engineering', 'intended_business', 'intended_mis'])

            # Exam type from application
            exam_col = 'Extra question: Type of Exam'
            if exam_col in app.columns:
                app['exam_type_application'] = app[exam_col].fillna('Unknown')
                app_features.append('exam_type_application')

            # Merge
            if len(app_features) > 1:
                master = master.merge(app[app_features], on='student_id', how='left')
                print(f"    Merged application features for {master['student_id'].nunique():,} students")

        return master

    def _merge_hs_exam_data(self, master: pd.DataFrame) -> pd.DataFrame:
        """Merge high school exam data - one row per student."""
        exam_datasets = ['wassce', 'oa_level', 'hsdiploma', 'french', 'ib', 'other']
        hs_records = []

        for exam_name in exam_datasets:
            if exam_name not in self.datasets:
                continue

            df = self.datasets[exam_name].copy()
            if 'student_id' not in df.columns:
                continue

            # One row per student from each exam type
            df_agg = df.groupby('student_id').first().reset_index()
            df_agg['exam_source'] = exam_name

            # Extract standardized scores
            df_agg = self._extract_exam_scores(df_agg, exam_name)
            hs_records.append(df_agg[['student_id', 'exam_source', 'hs_math_score',
                                      'hs_english_score', 'hs_science_score', 'hs_aggregate']].copy())

        if hs_records:
            hs_combined = pd.concat(hs_records, ignore_index=True)
            # Take first available record per student (prefer WASSCE if available)
            hs_combined = hs_combined.sort_values('exam_source').groupby('student_id').first().reset_index()
            master = master.merge(hs_combined, on='student_id', how='left')
            print(f"    Merged HS exam data for {hs_combined['student_id'].nunique():,} students")

        return master

    def _extract_exam_scores(self, df: pd.DataFrame, exam_type: str) -> pd.DataFrame:
        """Extract standardized scores from each exam type."""
        df = df.copy()
        df['hs_math_score'] = np.nan
        df['hs_english_score'] = np.nan
        df['hs_science_score'] = np.nan
        df['hs_aggregate'] = np.nan

        if exam_type == 'wassce':
            # WASSCE grades (A1-F9, lower is better)
            if 'Mathematics' in df.columns:
                df['hs_math_score'] = df['Mathematics'].apply(self._convert_wassce_to_score)
            if 'English Language' in df.columns:
                df['hs_english_score'] = df['English Language'].apply(self._convert_wassce_to_score)
            if 'Integrated Science' in df.columns:
                df['hs_science_score'] = df['Integrated Science'].apply(self._convert_wassce_to_score)
            if 'Total Aggregate' in df.columns:
                df['hs_aggregate'] = pd.to_numeric(df['Total Aggregate'], errors='coerce')

        elif exam_type == 'ib':
            # IB scores (1-7, higher is better)
            math_cols = [col for col in df.columns if 'math' in col.lower()]
            if math_cols:
                df['hs_math_score'] = df[math_cols[0]].apply(lambda x: self._convert_ib_to_score(x))

            eng_cols = [col for col in df.columns if 'english' in col.lower()]
            if eng_cols:
                df['hs_english_score'] = df[eng_cols[0]].apply(lambda x: self._convert_ib_to_score(x))

            if 'Points' in df.columns:
                df['hs_aggregate'] = pd.to_numeric(df['Points'], errors='coerce')

        elif exam_type == 'oa_level':
            # O/A Level grades
            if 'Mathematics' in df.columns:
                df['hs_math_score'] = df['Mathematics'].apply(self._convert_alevel_to_score)
            if 'English' in df.columns:
                df['hs_english_score'] = df['English'].apply(self._convert_alevel_to_score)
            if 'Points' in df.columns:
                df['hs_aggregate'] = pd.to_numeric(df['Points'], errors='coerce')

        elif exam_type == 'french':
            # French Bac (scores out of 20)
            if 'Mathematics' in df.columns:
                df['hs_math_score'] = df['Mathematics'].apply(self._convert_french_to_score)
            if 'Points' in df.columns:
                df['hs_aggregate'] = df['Points'].apply(self._parse_french_points)

        return df

    def _convert_wassce_to_score(self, grade) -> float:
        """Convert WASSCE grade to 0-100 score (higher is better)."""
        if pd.isna(grade):
            return np.nan

        grade_map = {
            'A1': 95, 'B2': 85, 'B3': 75, 'C4': 65, 'C5': 60,
            'C6': 55, 'D7': 45, 'E8': 35, 'F9': 20
        }
        grade_str = str(grade).strip().upper()
        return grade_map.get(grade_str, np.nan)

    def _convert_ib_to_score(self, score) -> float:
        """Convert IB score (1-7) to 0-100."""
        if pd.isna(score):
            return np.nan
        try:
            num = float(score)
            if 1 <= num <= 7:
                return (num / 7) * 100
        except:
            pass
        return np.nan

    def _convert_alevel_to_score(self, grade) -> float:
        """Convert A-Level grade to 0-100."""
        if pd.isna(grade):
            return np.nan

        grade_map = {
            'A*': 100, 'A': 90, 'B': 80, 'C': 70, 'D': 60, 'E': 50, 'U': 20,
            '1': 95, '2': 85, '3': 75, '4': 65, '5': 55, '6': 45, '7': 35, '8': 25, '9': 15
        }
        grade_str = str(grade).strip().upper()
        return grade_map.get(grade_str, np.nan)

    def _convert_french_to_score(self, score) -> float:
        """Convert French Bac score to 0-100."""
        if pd.isna(score):
            return np.nan
        try:
            score_str = str(score)
            if '/' in score_str:
                parts = score_str.split('/')
                return (float(parts[0]) / float(parts[1])) * 100
            return float(score_str) * 5  # Assume out of 20
        except:
            return np.nan

    def _parse_french_points(self, score) -> float:
        """Parse French Bac total points."""
        if pd.isna(score):
            return np.nan
        try:
            score_str = str(score)
            if '/' in score_str:
                parts = score_str.split('/')
                return float(parts[0])
            return float(score_str)
        except:
            return np.nan

    def _add_math_track(self, master: pd.DataFrame) -> pd.DataFrame:
        """
        Add math track detected from transcript data (actual course enrollment).
        Looks at which math course the student first enrolled in.
        """
        print("    Adding math track (from transcript course enrollment)...")

        # Initialize
        master['math_track'] = 'Unknown'
        master['math_track_encoded'] = np.nan

        if 'transcript' not in self.datasets:
            print("      No transcript data available for math track detection")
            return master

        transcript = self.datasets['transcript'].copy()
        if 'Course Name' not in transcript.columns or 'student_id' not in transcript.columns:
            print("      Required columns not found in transcript")
            return master

        # Define math track course patterns
        calculus_patterns = ['Calculus', 'MATH142']
        precalc_patterns = ['Pre-Calculus', 'Pre Calculus', 'PreCalculus', 'MATH141']
        algebra_patterns = ['College Algebra', 'Algebra', 'MATH140']

        def identify_track(course_name):
            course = str(course_name)
            # Check pre-calc first (to avoid matching 'Calculus' in 'Pre-Calculus')
            for pattern in precalc_patterns:
                if pattern.lower() in course.lower():
                    return 'Pre-Calculus'
            for pattern in calculus_patterns:
                if pattern.lower() in course.lower():
                    return 'Calculus'
            for pattern in algebra_patterns:
                if pattern.lower() in course.lower():
                    return 'College Algebra'
            return None

        # Detect math track from course names
        transcript['math_track_detected'] = transcript['Course Name'].apply(identify_track)

        # Filter to math courses only
        math_courses = transcript[transcript['math_track_detected'].notna()].copy()

        if len(math_courses) == 0:
            print("      No math courses found in transcript")
            return master

        # Extract semester number for sorting
        if 'Semester/Year' in math_courses.columns:
            math_courses['semester_num'] = math_courses['Semester/Year'].str.extract(r'(\d+)').astype(float)
        else:
            math_courses['semester_num'] = 1

        # Get the first (earliest) math course per student
        first_math = math_courses.sort_values('semester_num').groupby('student_id').first().reset_index()
        first_math = first_math[['student_id', 'math_track_detected']].rename(
            columns={'math_track_detected': 'detected_track'}
        )

        # Merge with master
        master = master.merge(first_math, on='student_id', how='left')
        master.loc[master['detected_track'].notna(), 'math_track'] = master['detected_track']
        master = master.drop(columns=['detected_track'], errors='ignore')

        detected_count = (master['math_track'] != 'Unknown').sum()
        print(f"      Detected from transcript: {detected_count:,} students")

        # Encode math track
        track_encoding = {'Calculus': 3, 'Pre-Calculus': 2, 'College Algebra': 1, 'Unknown': 0}
        master['math_track_encoded'] = master['math_track'].map(track_encoding)

        # Summary
        print(f"      Math track distribution:")
        print(master['math_track'].value_counts().to_string())

        return master

    def _merge_academic_performance(self, master: pd.DataFrame) -> pd.DataFrame:
        """Merge academic performance data - aggregated per student."""
        if 'cgpa' not in self.datasets:
            return master

        cgpa = self.datasets['cgpa'].copy()
        if 'student_id' not in cgpa.columns:
            return master

        # Parse semester info
        if 'Semester/Year' in cgpa.columns:
            cgpa['semester_num'] = cgpa['Semester/Year'].str.extract(r'Semester (\d)').astype(float)
            # Flag regular semesters (1 and 2) vs summer (3)
            cgpa['is_regular_semester'] = cgpa['semester_num'].isin([1, 2])

        if 'Academic Year' in cgpa.columns and 'Admission Year' in cgpa.columns:
            cgpa['academic_start'] = cgpa['Academic Year'].str.extract(r'(\d{4})').astype(float)
            cgpa['admission_start'] = cgpa['Admission Year'].str.extract(r'(\d{4})').astype(float)
            cgpa['year_num'] = cgpa['academic_start'] - cgpa['admission_start'] + 1

        # Sort for proper ordering
        cgpa = cgpa.sort_values(['student_id', 'academic_start', 'semester_num'])

        # Aggregate per student
        agg_dict = {}
        if 'CGPA' in cgpa.columns:
            agg_dict['CGPA'] = ['last', 'min', 'max', 'mean']
        if 'GPA' in cgpa.columns:
            agg_dict['GPA'] = ['mean', 'std', 'min', 'max']
        if 'Program' in cgpa.columns:
            agg_dict['Program'] = 'last'
        if 'Student Status' in cgpa.columns:
            agg_dict['Student Status'] = 'last'

        if agg_dict:
            perf = cgpa.groupby('student_id').agg(agg_dict)
            perf.columns = ['_'.join(col).strip('_') for col in perf.columns]
            perf = perf.reset_index()

            # Rename for clarity
            rename_map = {
                'CGPA_last': 'final_cgpa', 'CGPA_min': 'min_cgpa', 'CGPA_max': 'max_cgpa', 'CGPA_mean': 'avg_cgpa',
                'GPA_mean': 'avg_gpa', 'GPA_std': 'gpa_std', 'GPA_min': 'min_gpa', 'GPA_max': 'max_gpa',
                'Program_last': 'final_program',
                'Student Status_last': 'student_status'
            }
            perf = perf.rename(columns=rename_map)

            master = master.merge(perf, on='student_id', how='left')
            print(f"    Merged academic performance for {perf['student_id'].nunique():,} students")

        # Count semesters separately: total, regular only, summer only
        semester_counts = cgpa.groupby('student_id').agg({
            'semester_num': 'count',  # Total semesters
            'is_regular_semester': 'sum',  # Regular semesters only (for extended grad calc)
            'year_num': 'max'  # Number of academic years
        }).reset_index()
        semester_counts.columns = ['student_id', 'total_semesters', 'regular_semesters', 'academic_years']
        semester_counts['summer_semesters'] = semester_counts['total_semesters'] - semester_counts['regular_semesters']

        master = master.merge(semester_counts, on='student_id', how='left')
        print(f"    Added semester counts (regular vs summer)")

        # Year 1 specific features
        if 'year_num' in cgpa.columns:
            y1 = cgpa[cgpa['year_num'] == 1]
            if len(y1) > 0:
                y1_agg = y1.groupby('student_id').agg({
                    'GPA': ['mean', 'min', 'max'],
                    'CGPA': 'last'
                })
                y1_agg.columns = ['y1_gpa_mean', 'y1_gpa_min', 'y1_gpa_max', 'y1_cgpa_end']
                y1_agg = y1_agg.reset_index()
                master = master.merge(y1_agg, on='student_id', how='left')
                print(f"    Added Year 1 features for {y1_agg['student_id'].nunique():,} students")

        # Save semester records
        self.semester_records = cgpa

        return master

    def _merge_ajc_data(self, master: pd.DataFrame) -> pd.DataFrame:
        """Merge AJC (misconduct) data."""
        if 'ajc' not in self.datasets:
            master['has_ajc_case'] = 0
            master['ajc_guilty'] = 0
            return master

        ajc = self.datasets['ajc'].copy()
        if 'student_id' not in ajc.columns:
            master['has_ajc_case'] = 0
            master['ajc_guilty'] = 0
            return master

        ajc_agg = ajc.groupby('student_id').agg({
            'Type of Misconduct': 'count',
            'Verdict': lambda x: (x == 'Guilty').sum()
        }).reset_index()
        ajc_agg.columns = ['student_id', 'ajc_case_count', 'ajc_guilty_count']

        master = master.merge(ajc_agg, on='student_id', how='left')
        master['ajc_case_count'] = master['ajc_case_count'].fillna(0).astype(int)
        master['ajc_guilty_count'] = master['ajc_guilty_count'].fillna(0).astype(int)
        master['has_ajc_case'] = (master['ajc_case_count'] > 0).astype(int)
        master['ajc_guilty'] = (master['ajc_guilty_count'] > 0).astype(int)

        print(f"    Merged AJC data: {ajc_agg['student_id'].nunique():,} students with cases")

        return master

    def _create_derived_features(self, master: pd.DataFrame) -> pd.DataFrame:
        """Create derived features and target variables."""
        # Student status flags
        if 'student_status' in master.columns:
            status = master['student_status'].str.lower().fillna('')
            master['is_graduated'] = status.str.contains('graduat').astype(int)
            master['is_active'] = status.str.contains('active').astype(int)
            master['is_dismissed'] = status.str.contains('dismiss').astype(int)
            master['is_withdrawn'] = status.str.contains('withdraw').astype(int)

        # Probation indicators
        if 'min_cgpa' in master.columns:
            master['ever_on_probation'] = (master['min_cgpa'] < 2.0).astype(int)

        # Target: First year struggle
        if 'y1_cgpa_end' in master.columns:
            master['first_year_struggle'] = (master['y1_cgpa_end'] < 2.0).astype(int)

        # Target: Major success (graduated with CGPA >= 3.0)
        if 'final_cgpa' in master.columns and 'is_graduated' in master.columns:
            master['major_success'] = ((master['final_cgpa'] >= 3.0) & (master['is_graduated'] == 1)).astype(int)

        # Target: Extended graduation (> 4 academic years)
        # Uses academic_years, NOT total_semesters (summer semesters don't count)
        if 'academic_years' in master.columns:
            master['extended_graduation'] = (master['academic_years'] > 4).astype(int)
        elif 'regular_semesters' in master.columns:
            # Fallback: > 8 regular semesters
            master['extended_graduation'] = (master['regular_semesters'] > 8).astype(int)

        # CS major flag
        if 'final_program' in master.columns:
            master['is_cs_major'] = master['final_program'].str.contains(
                'Computer Science|Computer Engineering', case=False, na=False
            ).astype(int)
        elif 'intended_cs' in master.columns:
            master['is_cs_major'] = master['intended_cs']

        return master

    def save_processed_data(self, output_dir: str = 'data/processed/'):
        """Save processed datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.master_df is not None:
            self.master_df.to_csv(output_path / 'master_student_data.csv', index=False)
            print(f"  Saved master_student_data.csv ({len(self.master_df):,} students)")

        if self.semester_records is not None:
            self.semester_records.to_csv(output_path / 'semester_records.csv', index=False)
            print(f"  Saved semester_records.csv ({len(self.semester_records):,} records)")


def quick_load_master_data(data_path: str = 'data/') -> pd.DataFrame:
    """Convenience function to load or create master student data."""
    processed_path = Path(data_path) / 'processed' / 'master_student_data.csv'

    if processed_path.exists():
        print("Loading existing master data...")
        return pd.read_csv(processed_path)
    else:
        print("Creating master data from raw files...")
        loader = DataLoader(data_path)
        loader.load_all_datasets()
        master = loader.create_master_student_table()
        loader.save_processed_data()
        return master


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)

    loader = DataLoader('data/')
    datasets = loader.load_all_datasets()

    print("\n" + "=" * 60)
    print("Unique Student Summary")
    print("=" * 60)
    print(loader.get_unique_student_summary())

    print("\n" + "=" * 60)
    print("Creating Master Table")
    print("=" * 60)
    master = loader.create_master_student_table()

    print("\n" + "=" * 60)
    print("Master Table Summary")
    print("=" * 60)
    print(f"Total unique students: {len(master):,}")
    print(f"Total features: {len(master.columns)}")
    print(f"\nMath track distribution:\n{master['math_track'].value_counts()}")
