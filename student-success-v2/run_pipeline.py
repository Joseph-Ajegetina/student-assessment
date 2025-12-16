"""
Ashesi Student Success Prediction Pipeline
Updated based on actual data samples
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import re

warnings.filterwarnings('ignore')

# Create necessary directories
DIRECTORIES = [
    'data/raw', 'data/processed', 'data/features',
    'models', 'reports/figures', 'reports', 'logs'
]
for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

class DataLoader:
    """Load all datasets with proper handling"""
    
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.datasets = {}
        
    def load_all_datasets(self):
        """Load all datasets from CSV files"""
        
        print("  Loading datasets...")
        
        # Define all expected files
        files = {
            'application': 'application.csv',
            'cgpa': 'cgpa.csv',
            'ajc': 'ajc.csv',
            'wasce': 'wasce.csv',
            'oa_level': 'oa_level.csv',
            'hsdiploma': 'hsdiploma.csv',
            'french': 'french.csv',
            'ib': 'ib.csv',
            'other': 'other.csv'
        }
        
        for name, filename in files.items():
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                try:
                    self.datasets[name] = pd.read_csv(filepath, low_memory=False)
                    print(f"    ✓ Loaded {name}: {len(self.datasets[name])} rows, {len(self.datasets[name].columns)} cols")
                except Exception as e:
                    print(f"    ✗ Error loading {name}: {e}")
            else:
                print(f"    ✗ File not found: {filepath}")
        
        return self.datasets
    
    def get_summary(self):
        """Get summary of loaded datasets"""
        summary = []
        for name, df in self.datasets.items():
            summary.append({
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Missing %': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 1)
            })
        return pd.DataFrame(summary)


# ============================================================================
# SECTION 2: DATA INTEGRATION
# ============================================================================

class DataIntegrator:
    """Integrate all datasets with proper ID standardization"""
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.master_df = None
        self.high_school_combined = None
        self.semester_records = None
        
    def standardize_student_ids(self):
        """Standardize student ID column names"""
        
        print("  Standardizing student IDs...")
        
        # Map of dataset name to ID column name
        id_mappings = {
            'application': 'StudentRef',
            'cgpa': 'StudentRef',  # Note: Sample showed 'StudentRef', not 'Student Ref'
            'ajc': 'StudentRef',
            'wasce': 'StudentRef',
            'oa_level': 'StudentRef',
            'hsdiploma': 'StudentRef',
            'french': 'StudentRef',
            'ib': 'StudentRef',
            'other': 'StudentRef'
        }
        
        for dataset_name, id_col in id_mappings.items():
            if dataset_name in self.datasets:
                df = self.datasets[dataset_name]
                
                # Find the ID column (handle variations)
                possible_cols = ['StudentRef', 'Student Ref', 'studentref', 'student_ref']
                found_col = None
                for col in possible_cols:
                    if col in df.columns:
                        found_col = col
                        break
                
                if found_col:
                    df = df.rename(columns={found_col: 'student_id'})
                    df['student_id'] = df['student_id'].astype(str).str.strip()
                    self.datasets[dataset_name] = df
                    
        return self.datasets
    
    def combine_high_school_exams(self):
        """Combine all high school exam datasets with proper grade handling"""
        
        print("  Combining high school exam data...")
        
        exam_datasets = ['wasce', 'oa_level', 'hsdiploma', 'french', 'ib', 'other']
        combined_records = []
        
        for exam_name in exam_datasets:
            if exam_name not in self.datasets:
                continue
                
            df = self.datasets[exam_name].copy()
            
            # Extract key information
            record = {
                'student_id': df.get('student_id', df.get('StudentRef')),
                'yeargroup': df.get('Yeargroup'),
                'proposed_major': df.get('Proposed Major'),
                'high_school': df.get('High School'),
                'exam_type': df.get('Exam Type'),
                'exam_year': df.get('Exam Year'),
                'exam_source': exam_name
            }
            
            # Extract and standardize scores based on exam type
            if exam_name == 'wasce':
                record['total_score'] = df.get('Total Aggregate')
                record['math_score'] = self._convert_wassce_grade(df.get('Mathematics'))
                record['english_score'] = self._convert_wassce_grade(df.get('English Language'))
                record['science_score'] = self._convert_wassce_grade(df.get('Integrated Science'))
                
            elif exam_name == 'ib':
                record['total_score'] = df.get('Points')
                record['math_score'] = df.get('Maths (SL)', df.get('Mathematics SL in English'))
                record['english_score'] = df.get('English A: Literature HL', df.get('English A: Lang & Lit. SL'))
                
            elif exam_name == 'french':
                # Parse French Bac scores (e.g., "14.15/20")
                record['total_score'] = df.get('Points').apply(self._parse_french_score) if 'Points' in df.columns else None
                record['math_score'] = df.get('Mathematics').apply(self._parse_french_subject_score) if 'Mathematics' in df.columns else None
                
            else:
                # Generic handling for other exam types
                record['total_score'] = df.get('Points')
                record['math_score'] = df.get('Mathematics', df.get('Maths'))
                record['english_score'] = df.get('English', df.get('English Language'))
            
            # Create DataFrame from this exam type
            exam_df = pd.DataFrame(record)
            exam_df['exam_source'] = exam_name
            combined_records.append(exam_df)
        
        if combined_records:
            self.high_school_combined = pd.concat(combined_records, ignore_index=True)
            print(f"    ✓ Combined {len(self.high_school_combined)} exam records from {len(combined_records)} sources")
        else:
            self.high_school_combined = pd.DataFrame()
            
        return self.high_school_combined
    
    def _convert_wassce_grade(self, series):
        """Convert WASSCE grades (A1-F9) to numeric scores"""
        if series is None:
            return None
            
        grade_map = {
            'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5,
            'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9
        }
        
        if isinstance(series, pd.Series):
            return series.map(grade_map)
        else:
            return grade_map.get(series)
    
    def _parse_french_score(self, score_str):
        """Parse French Bac score format (e.g., '14.15/20')"""
        if pd.isna(score_str):
            return np.nan
        
        score_str = str(score_str)
        match = re.match(r'([\d.]+)/(\d+)', score_str)
        if match:
            score = float(match.group(1))
            max_score = float(match.group(2))
            # Normalize to percentage
            return (score / max_score) * 100
        
        try:
            return float(score_str)
        except:
            return np.nan
    
    def _parse_french_subject_score(self, score_str):
        """Parse French subject scores (various formats)"""
        if pd.isna(score_str):
            return np.nan
        
        score_str = str(score_str)
        
        # Handle "35/100" format
        match = re.match(r'([\d.]+)/(\d+)', score_str)
        if match:
            return float(match.group(1))
        
        try:
            return float(score_str)
        except:
            return np.nan
    
    def create_semester_records(self):
        """Create longitudinal semester records from CGPA data"""
        
        print("  Creating semester records...")
        
        if 'cgpa' not in self.datasets:
            print("    ✗ CGPA data not available")
            return None
            
        cgpa = self.datasets['cgpa'].copy()
        
        # Parse semester information from 'Semester/Year' column
        # Format: "Semester 1", "Semester 2"
        if 'Semester/Year' in cgpa.columns:
            cgpa['semester_num'] = cgpa['Semester/Year'].str.extract(r'Semester (\d)').astype(float)
        
        # Calculate year number from Academic Year and Admission Year
        # Academic Year format: "2015-2016", Admission Year format: "2013-2014"
        if 'Academic Year' in cgpa.columns and 'Admission Year' in cgpa.columns:
            def extract_start_year(year_str):
                if pd.isna(year_str):
                    return np.nan
                match = re.match(r'(\d{4})', str(year_str))
                return int(match.group(1)) if match else np.nan
            
            cgpa['academic_start_year'] = cgpa['Academic Year'].apply(extract_start_year)
            cgpa['admission_start_year'] = cgpa['Admission Year'].apply(extract_start_year)
            cgpa['year_num'] = cgpa['academic_start_year'] - cgpa['admission_start_year'] + 1
        
        # Create semester order (for sorting and trajectory analysis)
        if 'year_num' in cgpa.columns and 'semester_num' in cgpa.columns:
            cgpa['semester_order'] = (cgpa['year_num'] - 1) * 2 + cgpa['semester_num']
        
        # Sort by student and semester
        sort_cols = ['student_id']
        if 'semester_order' in cgpa.columns:
            sort_cols.append('semester_order')
        cgpa = cgpa.sort_values(sort_cols)
        
        # Calculate performance indicators
        if 'GPA' in cgpa.columns:
            cgpa['gpa_change'] = cgpa.groupby('student_id')['GPA'].diff()
        
        if 'CGPA' in cgpa.columns:
            cgpa['on_probation'] = (cgpa['CGPA'] < 2.0).astype(int)
            cgpa['deans_list'] = (cgpa['GPA'] >= 3.5).astype(int) if 'GPA' in cgpa.columns else 0
        
        # Identify consecutive probation (dismissal risk)
        if 'on_probation' in cgpa.columns:
            def check_consecutive_probation(group):
                group = group.sort_values('semester_order') if 'semester_order' in group.columns else group
                group['consecutive_probation'] = (
                    (group['on_probation'] == 1) & 
                    (group['on_probation'].shift(1) == 1)
                )
                return group
            
            cgpa = cgpa.groupby('student_id', group_keys=False).apply(check_consecutive_probation)
        
        self.semester_records = cgpa
        print(f"    ✓ Created {len(self.semester_records)} semester records")
        
        return cgpa
    
    def create_master_table(self):
        """Create master student table with all features"""
        
        print("  Creating master student table...")
        
        if 'application' not in self.datasets:
            print("    ✗ Application data not available")
            return None
        
        # Start with application data
        master = self.datasets['application'].copy()
        
        # === Merge High School Scores ===
        if self.high_school_combined is not None and len(self.high_school_combined) > 0:
            # Aggregate to one row per student (take first/best scores)
            hs_agg = self.high_school_combined.groupby('student_id').agg({
                'total_score': 'first',
                'math_score': 'first',
                'english_score': 'first',
                'science_score': 'first' if 'science_score' in self.high_school_combined.columns else 'first',
                'exam_type': 'first',
                'exam_source': 'first',
                'high_school': 'first'
            }).reset_index()
            
            master = master.merge(hs_agg, on='student_id', how='left')
        
        # === Merge AJC Information ===
        if 'ajc' in self.datasets:
            ajc = self.datasets['ajc'].copy()
            
            ajc_agg = ajc.groupby('student_id').agg({
                'Type of Misconduct': 'count',
                'Verdict': lambda x: (x == 'Guilty').sum()
            }).reset_index()
            ajc_agg.columns = ['student_id', 'ajc_cases_total', 'ajc_guilty_count']
            
            master = master.merge(ajc_agg, on='student_id', how='left')
            master['ajc_cases_total'] = master['ajc_cases_total'].fillna(0)
            master['ajc_guilty_count'] = master['ajc_guilty_count'].fillna(0)
            master['has_ajc_case'] = (master['ajc_cases_total'] > 0).astype(int)
        
        # === Merge CGPA/Performance Data ===
        if self.semester_records is not None and len(self.semester_records) > 0:
            # Get final performance metrics per student
            perf_agg = self.semester_records.groupby('student_id').agg({
                'CGPA': 'last',
                'GPA': ['mean', 'std', 'min', 'max'],
                'Program': 'last',
                'Student Status': 'last',
                'Yeargroup': 'first',
                'on_probation': 'sum' if 'on_probation' in self.semester_records.columns else 'first',
                'deans_list': 'sum' if 'deans_list' in self.semester_records.columns else 'first',
                'semester_order': 'max'
            })
            
            # Flatten column names
            perf_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                               for col in perf_agg.columns]
            perf_agg = perf_agg.reset_index()
            
            # Rename for clarity
            perf_agg = perf_agg.rename(columns={
                'CGPA_last': 'final_cgpa',
                'CGPA': 'final_cgpa',
                'GPA_mean': 'avg_gpa',
                'GPA_std': 'gpa_std',
                'GPA_min': 'min_gpa',
                'GPA_max': 'max_gpa',
                'Program_last': 'final_program',
                'Program': 'final_program',
                'Student Status_last': 'student_status',
                'Student Status': 'student_status',
                'Yeargroup_first': 'yeargroup',
                'Yeargroup': 'yeargroup',
                'on_probation_sum': 'probation_count',
                'on_probation': 'probation_count',
                'deans_list_sum': 'deans_list_count',
                'deans_list': 'deans_list_count',
                'semester_order_max': 'total_semesters',
                'semester_order': 'total_semesters'
            })
            
            master = master.merge(perf_agg, on='student_id', how='left')
        
        # === Get Math Placement from WASCE data ===
        if 'wasce' in self.datasets:
            wasce = self.datasets['wasce']
            math_placement_cols = ['Auto WASSCE Ashesi Math Placement', 'Ashesi Actual Math Placement']
            
            for col in math_placement_cols:
                if col in wasce.columns:
                    placement = wasce[['student_id', col]].dropna()
                    placement = placement.rename(columns={col: 'math_track'})
                    master = master.merge(placement, on='student_id', how='left')
                    break
        
        self.master_df = master
        print(f"    ✓ Created master table: {len(master)} students, {len(master.columns)} features")
        
        return master


# ============================================================================
# SECTION 3: DATA CLEANING
# ============================================================================

class DataCleaner:
    """Clean and prepare data for analysis"""
    
    def __init__(self, master_df, semester_records):
        self.master_df = master_df.copy() if master_df is not None else pd.DataFrame()
        self.semester_records = semester_records.copy() if semester_records is not None else pd.DataFrame()
        
    def clean_all(self):
        """Run all cleaning steps"""
        
        if len(self.master_df) == 0:
            return self.master_df, self.semester_records
            
        print("  Cleaning data...")
        
        self._clean_dates()
        self._clean_categorical()
        self._clean_numeric()
        self._handle_missing()
        
        print(f"    ✓ Cleaning complete")
        
        return self.master_df, self.semester_records
    
    def _clean_dates(self):
        """Parse and clean date columns"""
        
        # Application dates (format: "15/01/2018 5:20")
        date_cols = ['Created date', 'Submitted date']
        
        for col in date_cols:
            if col in self.master_df.columns:
                self.master_df[col] = pd.to_datetime(
                    self.master_df[col], 
                    format='%d/%m/%Y %H:%M',
                    errors='coerce'
                )
                # Extract useful features
                self.master_df[f'{col}_year'] = self.master_df[col].dt.year
                self.master_df[f'{col}_month'] = self.master_df[col].dt.month
    
    def _clean_categorical(self):
        """Standardize categorical variables"""
        
        # Gender
        if 'Gender' in self.master_df.columns:
            gender_map = {
                'M': 'Male', 'F': 'Female', 
                'Male': 'Male', 'Female': 'Female',
                'm': 'Male', 'f': 'Female'
            }
            self.master_df['Gender'] = self.master_df['Gender'].map(gender_map).fillna('Unknown')
        
        # Offer type (admission status)
        if 'Offer type' in self.master_df.columns:
            self.master_df['was_admitted'] = (
                ~self.master_df['Offer type'].isin(['Failed', 'Rejected', 'Declined'])
            ).astype(int)
        
        # Program/Major standardization
        program_cols = ['Offer course name', 'final_program', 'Program']
        for col in program_cols:
            if col in self.master_df.columns:
                self.master_df[col] = self.master_df[col].astype(str).str.strip()
                # Remove [B.Sc.] prefix
                self.master_df[col] = self.master_df[col].str.replace(r'\[B\.Sc\.\]\s*', '', regex=True)
        
        # Nationality regions
        if 'Nationality' in self.master_df.columns:
            # Since data is anonymized as Country0, Country1, etc., we'll keep as-is
            # In real scenario, we'd map to regions
            self.master_df['nationality_region'] = self.master_df['Nationality']
    
    def _clean_numeric(self):
        """Clean numeric columns"""
        
        # CGPA/GPA should be 0-4
        gpa_cols = [col for col in self.master_df.columns 
                   if 'cgpa' in col.lower() or 'gpa' in col.lower()]
        
        for col in gpa_cols:
            if col in self.master_df.columns:
                self.master_df[col] = pd.to_numeric(self.master_df[col], errors='coerce')
                self.master_df[col] = self.master_df[col].clip(0, 4)
        
        # Total semesters should be reasonable (1-16)
        if 'total_semesters' in self.master_df.columns:
            self.master_df['total_semesters'] = self.master_df['total_semesters'].clip(1, 16)
    
    def _handle_missing(self):
        """Handle missing values"""
        
        # Fill numeric with median
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.master_df[col].isnull().any():
                self.master_df[col] = self.master_df[col].fillna(self.master_df[col].median())
        
        # Fill categorical with 'Unknown'
        cat_cols = self.master_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.master_df[col] = self.master_df[col].fillna('Unknown')


# ============================================================================
# SECTION 4: SCORE STANDARDIZATION
# ============================================================================

class ScoreStandardizer:
    """Standardize scores across different exam systems"""
    
    def __init__(self):
        # Define grading scales
        self.grade_scales = {
            'wassce': {'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9},
            'letter_plus_minus': {'A+': 12, 'A': 11, 'A-': 10, 'B+': 9, 'B': 8, 'B-': 7, 
                                  'C+': 6, 'C': 5, 'C-': 4, 'D+': 3, 'D': 2, 'D-': 1, 'F': 0},
            'ib': list(range(1, 8))  # 1-7 scale
        }
    
    def standardize_all_scores(self, df):
        """Standardize all exam scores to 0-100 percentile scale"""
        
        if 'exam_source' not in df.columns:
            return df
        
        df = df.copy()
        
        def standardize_row(row):
            source = row.get('exam_source', '')
            total = row.get('total_score')
            
            if pd.isna(total):
                return np.nan
            
            try:
                total = float(str(total).split('/')[0])  # Handle "14.15/20" format
            except:
                return np.nan
            
            if source == 'wasce':
                # WASSCE: Lower is better, typical range 6-54 for 6 subjects
                # Best possible: 6 (6 A1s), Worst: 54 (6 F9s)
                return max(0, 100 - ((total - 6) / 48) * 100)
            
            elif source == 'ib':
                # IB: 0-45 scale, higher is better
                return (total / 45) * 100
            
            elif source == 'french':
                # French Bac: 0-20 scale, higher is better
                if total <= 20:
                    return (total / 20) * 100
                else:
                    return min(100, total)  # Already percentage
            
            elif source in ['oa_level', 'hsdiploma', 'other']:
                # Various scales - normalize based on apparent range
                if total <= 12:  # Likely letter grade converted
                    return (total / 12) * 100
                elif total <= 45:  # Likely IB-style
                    return (total / 45) * 100
                elif total <= 100:  # Likely percentage
                    return total
                else:
                    return min(100, (total / 200) * 100)
            
            return np.nan
        
        df['standardized_score'] = df.apply(standardize_row, axis=1)
        
        return df
    
    def create_performance_tiers(self, df, score_col='standardized_score'):
        """Create performance tier categories"""
        
        if score_col not in df.columns:
            df['performance_tier'] = 'Unknown'
            return df
        
        def assign_tier(score):
            if pd.isna(score):
                return 'Unknown'
            elif score >= 85:
                return 'Excellent'
            elif score >= 70:
                return 'Good'
            elif score >= 55:
                return 'Average'
            elif score >= 40:
                return 'Below Average'
            else:
                return 'At Risk'
        
        df['performance_tier'] = df[score_col].apply(assign_tier)
        
        return df


# ============================================================================
# SECTION 5: TARGET VARIABLE CREATION (UPDATED)
# ============================================================================

class TargetCreator:
    """Create target variables with proper handling of student status"""
    
    def __init__(self, master_df, semester_records):
        self.master_df = master_df.copy() if master_df is not None else pd.DataFrame()
        self.semester_records = semester_records.copy() if semester_records is not None else pd.DataFrame()
        self._identify_student_status()
    
    def _identify_student_status(self):
        """Identify and flag student status for proper filtering"""
        
        print("  Identifying student status...")
        
        # Find status column
        status_col = None
        for col in ['student_status', 'Student Status']:
            if col in self.master_df.columns:
                status_col = col
                break
        
        if status_col:
            status = self.master_df[status_col].astype(str).str.lower().str.strip()
            
            # Create status flags
            self.master_df['is_graduated'] = status.str.contains('graduat', na=False)
            self.master_df['is_active'] = status.str.contains('active', na=False)
            self.master_df['is_withdrawn'] = status.str.contains('withdraw|left|dropped|quit', na=False)
            self.master_df['is_dismissed'] = status.str.contains('dismiss|expel|suspend', na=False)
            
            # Students with known final outcomes (not active)
            self.master_df['has_final_outcome'] = ~self.master_df['is_active']
            
            # Print summary
            print(f"    Student Status Distribution:")
            print(f"      ├─ Graduated:  {self.master_df['is_graduated'].sum():,}")
            print(f"      ├─ Active:     {self.master_df['is_active'].sum():,}")
            print(f"      ├─ Withdrawn:  {self.master_df['is_withdrawn'].sum():,}")
            print(f"      ├─ Dismissed:  {self.master_df['is_dismissed'].sum():,}")
            print(f"      └─ Other:      {(~(self.master_df['is_graduated'] | self.master_df['is_active'] | self.master_df['is_withdrawn'] | self.master_df['is_dismissed'])).sum():,}")
            
        else:
            print("    ⚠️ Student status column not found - assuming all graduated")
            self.master_df['is_graduated'] = True
            self.master_df['is_active'] = False
            self.master_df['is_withdrawn'] = False
            self.master_df['is_dismissed'] = False
            self.master_df['has_final_outcome'] = True
    
    def create_all_targets(self):
        """Create all target variables with proper filtering"""
        
        if len(self.master_df) == 0:
            return self.master_df
        
        print("  Creating target variables...")
        
        self._create_first_year_struggle()
        self._create_ajc_target()
        self._create_major_success()
        self._create_extended_graduation()
        self._create_completion_target()
        self._create_retention_target()
        self._create_math_track()
        
        # Generate validity summary
        self._print_target_summary()
        
        return self.master_df
    
    def _create_first_year_struggle(self):
        """
        Q1: First year academic struggle
        
        Valid for: ALL students who completed Year 1 (including active, withdrawn)
        Definition: CGPA < 2.0 OR was on probation during Year 1
        """
        
        # Get Year 1 data from semester records
        if len(self.semester_records) > 0 and 'year_num' in self.semester_records.columns:
            y1_data = self.semester_records[self.semester_records['year_num'] == 1]
            
            if len(y1_data) > 0:
                # Aggregate Year 1 performance
                agg_dict = {'student_id': 'first'}
                
                if 'CGPA' in y1_data.columns:
                    agg_dict['CGPA'] = 'last'
                if 'GPA' in y1_data.columns:
                    agg_dict['GPA'] = 'mean'
                if 'on_probation' in y1_data.columns:
                    agg_dict['on_probation'] = 'max'
                
                y1_agg = y1_data.groupby('student_id').agg(agg_dict).reset_index(drop=True)
                
                # Rename columns
                col_rename = {
                    'CGPA': 'y1_cgpa',
                    'GPA': 'y1_avg_gpa',
                    'on_probation': 'y1_was_on_probation'
                }
                y1_agg = y1_agg.rename(columns=col_rename)
                
                # Merge with master
                self.master_df = self.master_df.merge(
                    y1_agg, on='student_id', how='left', suffixes=('', '_y1')
                )
                
                # Mark students who completed Year 1
                self.master_df['completed_year1'] = self.master_df['y1_cgpa'].notna()
        else:
            self.master_df['completed_year1'] = False
        
        # Create target variable
        if 'y1_cgpa' in self.master_df.columns:
            # Only valid for students who completed Year 1
            conditions = [
                self.master_df['completed_year1'] == True
            ]
            
            struggle_condition = (
                (self.master_df['y1_cgpa'] < 2.0) |
                (self.master_df.get('y1_was_on_probation', 0) == 1)
            )
            
            self.master_df['first_year_struggle'] = np.where(
                self.master_df['completed_year1'],
                struggle_condition.astype(int),
                np.nan
            )
        else:
            self.master_df['first_year_struggle'] = np.nan
    
    def _create_ajc_target(self):
        """
        Q2: AJC case prediction
        
        Valid for: ALL students (graduated, active, withdrawn, dismissed)
        Definition: Had at least one AJC case
        """
        
        if 'ajc_cases_total' in self.master_df.columns:
            self.master_df['has_ajc_case'] = (
                self.master_df['ajc_cases_total'].fillna(0) > 0
            ).astype(int)
        else:
            self.master_df['has_ajc_case'] = 0
        
        if 'ajc_guilty_count' in self.master_df.columns:
            self.master_df['found_guilty'] = (
                self.master_df['ajc_guilty_count'].fillna(0) > 0
            ).astype(int)
        else:
            self.master_df['found_guilty'] = 0
    
    def _create_major_success(self):
        """
        Q3-Q6: Major success/failure
        
        Valid for: ONLY GRADUATED students (known final outcome)
        Definition: 
            - Success: Final CGPA >= 3.0
            - Excellence: Final CGPA >= 3.5
            - Struggle: Final CGPA < 2.0 OR was dismissed
        """
        
        if 'final_cgpa' not in self.master_df.columns:
            self.master_df['major_success'] = np.nan
            self.master_df['major_excellence'] = np.nan
            self.master_df['major_struggle'] = np.nan
            return
        
        graduated_mask = self.master_df['is_graduated'] == True
        dismissed_mask = self.master_df['is_dismissed'] == True
        
        # Major Success: Graduated with CGPA >= 3.0
        self.master_df['major_success'] = np.where(
            graduated_mask,
            (self.master_df['final_cgpa'] >= 3.0).astype(int),
            np.nan
        )
        
        # Major Excellence: Graduated with CGPA >= 3.5 (Dean's List level)
        self.master_df['major_excellence'] = np.where(
            graduated_mask,
            (self.master_df['final_cgpa'] >= 3.5).astype(int),
            np.nan
        )
        
        # Major Struggle: Dismissed OR graduated with low CGPA
        self.master_df['major_struggle'] = np.where(
            graduated_mask | dismissed_mask,
            (
                dismissed_mask |
                ((graduated_mask) & (self.master_df['final_cgpa'] < 2.0))
            ).astype(int),
            np.nan
        )
    
    def _create_extended_graduation(self):
        """
        Q9: Extended graduation (>8 semesters)
        
        Valid for: ONLY GRADUATED students
        Definition: Took more than 8 semesters (4 years) to graduate
        """
        
        if 'total_semesters' not in self.master_df.columns:
            self.master_df['extended_graduation'] = np.nan
            return
        
        graduated_mask = self.master_df['is_graduated'] == True
        
        self.master_df['extended_graduation'] = np.where(
            graduated_mask,
            (self.master_df['total_semesters'] > 8).astype(int),
            np.nan
        )
        
        # Also create graduation speed categories
        def categorize_graduation(row):
            if not row.get('is_graduated', False):
                return np.nan
            semesters = row.get('total_semesters', np.nan)
            if pd.isna(semesters):
                return np.nan
            elif semesters <= 8:
                return 'On Time'
            elif semesters <= 10:
                return 'Slightly Extended'
            else:
                return 'Significantly Extended'
        
        self.master_df['graduation_speed'] = self.master_df.apply(categorize_graduation, axis=1)
    
    def _create_completion_target(self):
        """
        NEW: Did the student complete their degree?
        
        Valid for: All NON-ACTIVE students (have final outcome)
        Definition: 
            - 1 = Graduated successfully
            - 0 = Withdrawn or Dismissed (did not complete)
        """
        
        active_mask = self.master_df['is_active'] == True
        graduated_mask = self.master_df['is_graduated'] == True
        
        self.master_df['completed_degree'] = np.where(
            ~active_mask,  # Has final outcome
            graduated_mask.astype(int),
            np.nan
        )
    
    def _create_retention_target(self):
        """
        NEW: Was the student retained after Year 1?
        
        Valid for: All students who completed Year 1
        Definition: Student continued past Year 1 (not withdrawn/dismissed in Year 1)
        """
        
        if len(self.semester_records) == 0:
            self.master_df['retained_after_y1'] = np.nan
            return
        
        # Check if student has Year 2 data
        if 'year_num' in self.semester_records.columns:
            students_with_y2 = self.semester_records[
                self.semester_records['year_num'] >= 2
            ]['student_id'].unique()
            
            self.master_df['has_year2_data'] = self.master_df['student_id'].isin(students_with_y2)
            
            # Retained = has Year 2 data OR is currently active in Year 1 OR graduated
            self.master_df['retained_after_y1'] = np.where(
                self.master_df['completed_year1'],
                (
                    self.master_df['has_year2_data'] |
                    self.master_df['is_graduated'] |
                    self.master_df['is_active']
                ).astype(int),
                np.nan
            )
        else:
            self.master_df['retained_after_y1'] = np.nan
    
    def _create_math_track(self):
        """Q7-Q8: Math track indicators"""
        
        # Look for existing math track column
        math_track_cols = [col for col in self.master_df.columns 
                          if 'math' in col.lower() and 'placement' in col.lower()]
        
        if math_track_cols:
            self.master_df['math_track'] = self.master_df[math_track_cols[0]]
        elif 'math_score' in self.master_df.columns:
            self.master_df['math_track'] = self.master_df['math_score'].apply(
                self._infer_math_track
            )
        else:
            self.master_df['math_track'] = 'Unknown'
        
        # CS major indicator
        program_col = None
        for col in ['Offer course name', 'final_program', 'Program']:
            if col in self.master_df.columns:
                program_col = col
                break
        
        if program_col:
            self.master_df['is_cs_major'] = self.master_df[program_col].astype(str).str.contains(
                'Computer Science|Computer Engineering|CS',
                case=False,
                na=False
            ).astype(int)
        else:
            self.master_df['is_cs_major'] = 0
    
    def _infer_math_track(self, score):
        """Infer math track from score"""
        if pd.isna(score):
            return 'Unknown'
        
        try:
            score = float(score)
        except (ValueError, TypeError):
            # Handle letter grades (WASSCE)
            wassce_map = {
                'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 
                'C5': 5, 'C6': 6, 'D7': 7, 'E8': 8, 'F9': 9
            }
            score = wassce_map.get(str(score).upper(), 5)
        
        # For WASSCE-style (1-9, lower is better)
        if score <= 9:
            if score <= 2:
                return 'Calculus'
            elif score <= 4:
                return 'Pre-Calculus'
            else:
                return 'College Algebra'
        # For percentage-style (0-100, higher is better)
        else:
            if score >= 80:
                return 'Calculus'
            elif score >= 60:
                return 'Pre-Calculus'
            else:
                return 'College Algebra'
    
    def _print_target_summary(self):
        """Print summary of target variable validity"""
        
        print("\n    Target Variable Summary:")
        print("    " + "-" * 50)
        
        targets = [
            ('first_year_struggle', 'All who completed Y1'),
            ('has_ajc_case', 'All students'),
            ('major_success', 'Graduated only'),
            ('major_struggle', 'Graduated + Dismissed'),
            ('extended_graduation', 'Graduated only'),
            ('completed_degree', 'Non-active students'),
            ('retained_after_y1', 'All who completed Y1')
        ]
        
        for target, valid_for in targets:
            if target in self.master_df.columns:
                valid = self.master_df[target].notna().sum()
                total = len(self.master_df)
                
                if valid > 0:
                    positive = (self.master_df[target] == 1).sum()
                    pos_rate = positive / valid * 100
                    print(f"    {target}:")
                    print(f"      Valid: {valid}/{total} ({valid/total*100:.1f}%)")
                    print(f"      Positive rate: {pos_rate:.1f}%")
                    print(f"      Use for: {valid_for}")
                else:
                    print(f"    {target}: No valid cases")
        
        print("    " + "-" * 50)


# ============================================================================
# SECTION 6: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Engineer features for modeling"""
    
    def __init__(self, master_df, semester_records):
        self.master_df = master_df.copy() if master_df is not None else pd.DataFrame()
        self.semester_records = semester_records.copy() if semester_records is not None else pd.DataFrame()
    
    def engineer_all_features(self):
        """Create all engineered features"""
        
        if len(self.master_df) == 0:
            return self.master_df
        
        print("  Engineering features...")
        
        self._academic_features()
        self._demographic_features()
        self._application_features()
        self._family_features()
        
        print(f"    ✓ Feature engineering complete")
        
        return self.master_df
    
    def _academic_features(self):
        """Create academic performance features"""
        
        # Subject score aggregates
        score_cols = ['math_score', 'english_score', 'science_score']
        existing_scores = [c for c in score_cols if c in self.master_df.columns]
        
        if existing_scores:
            # Convert any remaining letter grades to numeric
            for col in existing_scores:
                self.master_df[col] = pd.to_numeric(self.master_df[col], errors='coerce')
            
            self.master_df['avg_core_subjects'] = self.master_df[existing_scores].mean(axis=1)
            self.master_df['score_variance'] = self.master_df[existing_scores].var(axis=1)
    
    def _demographic_features(self):
        """Create demographic features"""
        
        # Gender
        if 'Gender' in self.master_df.columns:
            self.master_df['is_female'] = (self.master_df['Gender'] == 'Female').astype(int)
        
        # Disadvantaged background
        if 'Disadvantaged background' in self.master_df.columns:
            self.master_df['is_disadvantaged'] = self.master_df['Disadvantaged background'].isin(
                ['Yes', 'TRUE', True, 1, 'yes', 'Y']
            ).astype(int)
        
        # Financial aid
        aid_col = 'Extra question: Do you Need Financial Aid?'
        if aid_col in self.master_df.columns:
            self.master_df['needs_financial_aid'] = self.master_df[aid_col].isin(
                ['Yes', 'TRUE', True, 1, 'yes', 'Y']
            ).astype(int)
        
        # International (non-Country0 assuming Country0 is Ghana)
        if 'Nationality' in self.master_df.columns:
            self.master_df['is_international'] = (
                self.master_df['Nationality'] != 'Country0'
            ).astype(int)
    
    def _application_features(self):
        """Create application-based features"""
        
        # Previous application
        prev_app_col = 'Extra question: Have you applied to Ashesi before? If "yes" indicate the year.'
        if prev_app_col in self.master_df.columns:
            self.master_df['has_previous_application'] = (
                ~self.master_df[prev_app_col].isin(['No', 'no', 'N/A', np.nan, ''])
            ).astype(int)
        
        # Ashesi event attendance
        event_col = 'Extra question: Have you ever attended any Ashesi sponsored high school event? If "yes" please state event and year of attendance.'
        if event_col in self.master_df.columns:
            self.master_df['attended_ashesi_event'] = (
                ~self.master_df[event_col].isin(['No', 'no', 'N/A', np.nan, ''])
            ).astype(int)
    
    def _family_features(self):
        """Create family-related features"""
        
        # Family connection to Ashesi
        family_col = 'Extra question: Have any of your family members gained admission to Ashesi University?'
        if family_col in self.master_df.columns:
            self.master_df['has_family_connection'] = self.master_df[family_col].isin(
                ['Yes', 'yes', 'Y', True, 1]
            ).astype(int)
        
        # Family education level (from multiple columns)
        edu_cols = [c for c in self.master_df.columns if 'Level of education' in c]
        
        edu_map = {
            'PhD': 5, 'Doctorate': 5, 'phd': 5,
            'Masters': 4, "Master's": 4, 'masters': 4, 'MSc': 4, 'MA': 4, 'MBA': 4,
            'Bachelors': 3, "Bachelor's": 3, 'bachelors': 3, 'BSc': 3, 'BA': 3,
            'Diploma': 2, 'HND': 2, 'diploma': 2,
            'Secondary': 1, 'High School': 1, 'SE': 1, 'secondary': 1,
            'Primary': 0, 'primary': 0, 'None': 0
        }
        
        if edu_cols:
            def get_max_education(row):
                max_level = 0
                for col in edu_cols:
                    val = str(row.get(col, '')).strip()
                    for key, level in edu_map.items():
                        if key.lower() in val.lower():
                            max_level = max(max_level, level)
                            break
                return max_level
            
            self.master_df['max_parent_education'] = self.master_df.apply(get_max_education, axis=1)
    
    def create_year_features(self, year):
        """Create features from specific year's data"""
        
        if len(self.semester_records) == 0 or 'year_num' not in self.semester_records.columns:
            return self.master_df
        
        yr_data = self.semester_records[self.semester_records['year_num'] <= year]
        
        if len(yr_data) == 0:
            return self.master_df
        
        agg_funcs = {}
        if 'GPA' in yr_data.columns:
            agg_funcs['GPA'] = ['mean', 'std', 'min', 'max']
        if 'CGPA' in yr_data.columns:
            agg_funcs['CGPA'] = 'last'
        if 'on_probation' in yr_data.columns:
            agg_funcs['on_probation'] = 'sum'
        if 'deans_list' in yr_data.columns:
            agg_funcs['deans_list'] = 'sum'
        
        if not agg_funcs:
            return self.master_df
        
        yr_agg = yr_data.groupby('student_id').agg(agg_funcs)
        
        # Flatten columns
        new_cols = []
        for col in yr_agg.columns:
            if isinstance(col, tuple):
                new_cols.append(f'y{year}_{col[0]}_{col[1]}')
            else:
                new_cols.append(f'y{year}_{col}')
        yr_agg.columns = new_cols
        yr_agg = yr_agg.reset_index()
        
        self.master_df = self.master_df.merge(yr_agg, on='student_id', how='left')
        
        return self.master_df
    
    def get_feature_sets(self):
        """Define feature sets for different prediction tasks"""
        
        # Features available at admission time
        admission_features = [
            'standardized_score', 'math_score', 'english_score', 'science_score',
            'avg_core_subjects', 'score_variance', 'performance_tier',
            'is_female', 'is_disadvantaged', 'needs_financial_aid', 'is_international',
            'has_previous_application', 'attended_ashesi_event', 'has_family_connection',
            'max_parent_education', 'math_track', 'exam_source'
        ]
        
        # Features after Year 1
        year1_features = admission_features + [
            'y1_GPA_mean', 'y1_GPA_std', 'y1_GPA_min', 'y1_GPA_max',
            'y1_CGPA_last', 'y1_on_probation_sum', 'y1_deans_list_sum'
        ]
        
        # Features after Year 2
        year2_features = year1_features + [
            'y2_GPA_mean', 'y2_GPA_std', 'y2_GPA_min', 'y2_GPA_max',
            'y2_CGPA_last', 'y2_on_probation_sum', 'y2_deans_list_sum'
        ]
        
        # Filter to existing columns
        return {
            'admission': [f for f in admission_features if f in self.master_df.columns],
            'year1': [f for f in year1_features if f in self.master_df.columns],
            'year2': [f for f in year2_features if f in self.master_df.columns]
        }

# ============================================================================
# SECTION 6B: UNSUPERVISED LEARNING
# ============================================================================

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

class UnsupervisedAnalyzer:
    """
    Unsupervised learning for student segmentation and pattern discovery
    
    Goals:
    1. Identify natural student clusters/segments
    2. Discover patterns in admission and performance data
    3. Find at-risk student profiles
    4. Support targeted intervention strategies
    """
    
    def __init__(self, master_df, feature_sets, output_dir='reports/figures/'):
        self.master_df = master_df.copy()
        self.feature_sets = feature_sets
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_scaled = None
        self.features_used = None
        self.cluster_labels = {}
        self.cluster_profiles = {}
        self.results = {}
        
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def prepare_data(self, feature_type='admission'):
        """Prepare data for clustering"""
        
        print("  Preparing data for clustering...")
        
        features = self.feature_sets.get(feature_type, [])
        
        # Exclude non-predictive columns
        exclude_cols = [
            'student_id', 'is_graduated', 'is_active', 'is_withdrawn', 
            'is_dismissed', 'has_final_outcome', 'completed_year1',
            'student_status', 'Student Status'
        ]
        
        features = [f for f in features if f in self.master_df.columns and f not in exclude_cols]
        
        if len(features) < 3:
            print(f"    ✗ Insufficient features ({len(features)})")
            return None
        
        # Get data
        df = self.master_df[features].copy()
        
        # Encode categorical variables
        self.encoders = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Scale features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(df)
        self.features_used = features
        self.scaler = scaler
        
        print(f"    ✓ Prepared {len(self.X_scaled)} samples with {len(features)} features")
        
        return self.X_scaled
    
    def find_optimal_clusters(self, max_k=10, method='kmeans'):
        """
        Find optimal number of clusters using multiple metrics
        
        Methods:
        - Elbow method (inertia)
        - Silhouette score
        - Calinski-Harabasz index
        - Davies-Bouldin index
        """
        
        if self.X_scaled is None:
            print("    ✗ Data not prepared. Call prepare_data() first.")
            return None
        
        print(f"\n  Finding optimal clusters (max_k={max_k})...")
        
        results = {
            'k': list(range(2, max_k + 1)),
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for k in range(2, max_k + 1):
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif method == 'gmm':
                model = GaussianMixture(n_components=k, random_state=42)
            
            labels = model.fit_predict(self.X_scaled)
            
            # Metrics
            if hasattr(model, 'inertia_'):
                results['inertia'].append(model.inertia_)
            else:
                results['inertia'].append(None)
            
            results['silhouette'].append(silhouette_score(self.X_scaled, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(self.X_scaled, labels))
            results['davies_bouldin'].append(davies_bouldin_score(self.X_scaled, labels))
        
        # Find optimal k
        # Silhouette: higher is better
        optimal_silhouette = results['k'][np.argmax(results['silhouette'])]
        # Davies-Bouldin: lower is better
        optimal_db = results['k'][np.argmin(results['davies_bouldin'])]
        
        print(f"    Optimal k by silhouette: {optimal_silhouette}")
        print(f"    Optimal k by Davies-Bouldin: {optimal_db}")
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Elbow plot
        if results['inertia'][0] is not None:
            axes[0, 0].plot(results['k'], results['inertia'], 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Number of Clusters (k)')
            axes[0, 0].set_ylabel('Inertia')
            axes[0, 0].set_title('Elbow Method')
            axes[0, 0].grid(True)
        
        # Silhouette
        axes[0, 1].plot(results['k'], results['silhouette'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].axvline(x=optimal_silhouette, color='r', linestyle='--', label=f'Optimal: {optimal_silhouette}')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (higher is better)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Calinski-Harabasz
        axes[1, 0].plot(results['k'], results['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Index')
        axes[1, 0].set_title('Calinski-Harabasz Index (higher is better)')
        axes[1, 0].grid(True)
        
        # Davies-Bouldin
        axes[1, 1].plot(results['k'], results['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
        axes[1, 1].axvline(x=optimal_db, color='g', linestyle='--', label=f'Optimal: {optimal_db}')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Davies-Bouldin Index')
        axes[1, 1].set_title('Davies-Bouldin Index (lower is better)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}cluster_optimization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.results['cluster_optimization'] = results
        
        # Recommend k (average of methods, rounded)
        recommended_k = int(round((optimal_silhouette + optimal_db) / 2))
        print(f"    Recommended k: {recommended_k}")
        
        return recommended_k, results
    
    def perform_clustering(self, n_clusters, method='kmeans'):
        """
        Perform clustering with specified method
        
        Methods:
        - kmeans: K-Means clustering
        - gmm: Gaussian Mixture Model
        - hierarchical: Agglomerative clustering
        - dbscan: DBSCAN (density-based)
        """
        
        if self.X_scaled is None:
            print("    ✗ Data not prepared")
            return None
        
        print(f"\n  Performing {method} clustering with k={n_clusters}...")
        
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(self.X_scaled)
            
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(self.X_scaled)
            
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(self.X_scaled)
            
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(self.X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"    DBSCAN found {n_clusters} clusters")
        
        self.cluster_labels[method] = labels
        self.master_df[f'cluster_{method}'] = labels
        
        # Calculate metrics
        if len(set(labels)) > 1:
            metrics = {
                'n_clusters': n_clusters,
                'silhouette': silhouette_score(self.X_scaled, labels),
                'calinski_harabasz': calinski_harabasz_score(self.X_scaled, labels),
                'davies_bouldin': davies_bouldin_score(self.X_scaled, labels)
            }
            print(f"    Silhouette: {metrics['silhouette']:.3f}")
        else:
            metrics = {'n_clusters': n_clusters}
        
        self.results[f'{method}_metrics'] = metrics
        
        return labels
    
    def analyze_clusters(self, cluster_col='cluster_kmeans'):
        """
        Analyze characteristics of each cluster
        """
        
        if cluster_col not in self.master_df.columns:
            print(f"    ✗ Cluster column '{cluster_col}' not found")
            return None
        
        print(f"\n  Analyzing clusters from '{cluster_col}'...")
        
        # Define metrics to analyze
        outcome_cols = [
            'first_year_struggle', 'has_ajc_case', 'major_success',
            'major_struggle', 'extended_graduation', 'completed_degree'
        ]
        
        performance_cols = [
            'final_cgpa', 'y1_cgpa', 'standardized_score', 'avg_core_subjects'
        ]
        
        demographic_cols = [
            'is_female', 'is_international', 'needs_financial_aid', 'is_disadvantaged'
        ]
        
        # Build aggregation
        agg_dict = {'student_id': 'count'}
        
        for col in outcome_cols + performance_cols + demographic_cols:
            if col in self.master_df.columns:
                agg_dict[col] = 'mean'
        
        cluster_summary = self.master_df.groupby(cluster_col).agg(agg_dict).round(3)
        cluster_summary = cluster_summary.rename(columns={'student_id': 'count'})
        
        # Calculate percentages
        total = cluster_summary['count'].sum()
        cluster_summary['pct_of_total'] = (cluster_summary['count'] / total * 100).round(1)
        
        # Create cluster profiles/names
        def create_profile_name(row):
            profiles = []
            
            # Academic performance
            if 'final_cgpa' in row and pd.notna(row.get('final_cgpa')):
                if row['final_cgpa'] >= 3.5:
                    profiles.append('High Achievers')
                elif row['final_cgpa'] >= 3.0:
                    profiles.append('Solid Performers')
                elif row['final_cgpa'] < 2.5:
                    profiles.append('Struggling')
            
            # Risk indicators
            if 'first_year_struggle' in row and row.get('first_year_struggle', 0) > 0.4:
                profiles.append('At-Risk')
            
            if 'has_ajc_case' in row and row.get('has_ajc_case', 0) > 0.2:
                profiles.append('Conduct Issues')
            
            # Demographics
            if 'is_international' in row and row.get('is_international', 0) > 0.5:
                profiles.append('International')
            
            if 'needs_financial_aid' in row and row.get('needs_financial_aid', 0) > 0.7:
                profiles.append('Financial Need')
            
            if not profiles:
                profiles.append('Average')
            
            return ' / '.join(profiles[:2])  # Max 2 descriptors
        
        cluster_summary['profile'] = cluster_summary.apply(create_profile_name, axis=1)
        
        self.cluster_profiles[cluster_col] = cluster_summary
        
        print("\n    Cluster Profiles:")
        print("    " + "-" * 60)
        
        for idx, row in cluster_summary.iterrows():
            print(f"    Cluster {idx}: {row['profile']}")
            print(f"      Size: {int(row['count'])} ({row['pct_of_total']}%)")
            if 'final_cgpa' in row:
                print(f"      Avg CGPA: {row.get('final_cgpa', 'N/A'):.2f}" if pd.notna(row.get('final_cgpa')) else "      Avg CGPA: N/A")
            if 'first_year_struggle' in row:
                print(f"      Struggle Rate: {row.get('first_year_struggle', 0)*100:.1f}%")
            print()
        
        return cluster_summary
    
    def visualize_clusters(self, cluster_col='cluster_kmeans', method='pca'):
        """
        Visualize clusters using dimensionality reduction
        
        Methods:
        - pca: Principal Component Analysis
        - tsne: t-SNE
        - both: Both PCA and t-SNE
        """
        
        if self.X_scaled is None or cluster_col not in self.master_df.columns:
            print("    ✗ Data or clusters not available")
            return None
        
        print(f"\n  Visualizing clusters using {method}...")
        
        labels = self.master_df[cluster_col].values
        
        fig, axes = plt.subplots(1, 2 if method == 'both' else 1, 
                                  figsize=(14 if method == 'both' else 8, 6))
        
        if method != 'both':
            axes = [axes]
        
        # Color palette
        n_clusters = len(set(labels))
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        # PCA
        if method in ['pca', 'both']:
            ax = axes[0]
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X_scaled)
            
            scatter = ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=labels, cmap='viridis', alpha=0.6, s=50
            )
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title(f'PCA Visualization\n(Total variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)')
            
            # Add cluster centers
            for cluster_id in set(labels):
                mask = labels == cluster_id
                center = X_pca[mask].mean(axis=0)
                ax.annotate(
                    f'C{cluster_id}',
                    center,
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # t-SNE
        if method in ['tsne', 'both']:
            ax = axes[-1]
            
            # t-SNE is slow, use subset if data is large
            if len(self.X_scaled) > 5000:
                print("    Using subset for t-SNE (5000 samples)...")
                idx = np.random.choice(len(self.X_scaled), 5000, replace=False)
                X_subset = self.X_scaled[idx]
                labels_subset = labels[idx]
            else:
                X_subset = self.X_scaled
                labels_subset = labels
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_subset)
            
            scatter = ax.scatter(
                X_tsne[:, 0], X_tsne[:, 1],
                c=labels_subset, cmap='viridis', alpha=0.6, s=50
            )
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('t-SNE Visualization')
            
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}cluster_visualization_{method}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {self.output_dir}cluster_visualization_{method}.png")
        
        return fig
    
    def cluster_feature_importance(self, cluster_col='cluster_kmeans'):
        """
        Identify which features are most important for distinguishing clusters
        """
        
        if cluster_col not in self.master_df.columns:
            return None
        
        print("\n  Analyzing feature importance for clusters...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Use cluster labels as target
        y = self.master_df[cluster_col].values
        
        # Train random forest to predict cluster membership
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_scaled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.features_used,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_n = min(15, len(importance))
        top_features = importance.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))[::-1]
        
        ax.barh(top_features['feature'], top_features['importance'], color=colors)
        ax.set_xlabel('Importance')
        ax.set_title('Features Most Important for Cluster Separation')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}cluster_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Top 5 distinguishing features:")
        for i, row in importance.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        return importance
    
    def identify_risk_clusters(self, cluster_col='cluster_kmeans'):
        """
        Identify which clusters represent at-risk students
        """
        
        if cluster_col not in self.master_df.columns:
            return None
        
        print("\n  Identifying risk clusters...")
        
        risk_indicators = [
            'first_year_struggle', 'has_ajc_case', 'major_struggle',
            'extended_graduation'
        ]
        
        available_indicators = [r for r in risk_indicators if r in self.master_df.columns]
        
        if not available_indicators:
            print("    ✗ No risk indicators available")
            return None
        
        # Calculate composite risk score per cluster
        risk_scores = self.master_df.groupby(cluster_col)[available_indicators].mean()
        risk_scores['composite_risk'] = risk_scores.mean(axis=1)
        risk_scores['cluster_size'] = self.master_df.groupby(cluster_col).size()
        
        # Classify clusters
        def classify_risk(score):
            if score > 0.5:
                return 'HIGH RISK'
            elif score > 0.3:
                return 'MODERATE RISK'
            elif score > 0.15:
                return 'LOW RISK'
            else:
                return 'MINIMAL RISK'
        
        risk_scores['risk_level'] = risk_scores['composite_risk'].apply(classify_risk)
        
        # Sort by risk
        risk_scores = risk_scores.sort_values('composite_risk', ascending=False)
        
        print("\n    Risk Assessment by Cluster:")
        print("    " + "-" * 60)
        
        for cluster_id, row in risk_scores.iterrows():
            print(f"    Cluster {cluster_id}: {row['risk_level']}")
            print(f"      Composite Risk Score: {row['composite_risk']:.3f}")
            print(f"      Size: {int(row['cluster_size'])} students")
            for indicator in available_indicators:
                print(f"      {indicator}: {row[indicator]*100:.1f}%")
            print()
        
        self.results['risk_clusters'] = risk_scores
        
        return risk_scores
    
    def cluster_outcome_analysis(self, cluster_col='cluster_kmeans'):
        """
        Analyze academic outcomes by cluster
        """
        
        print("\n  Analyzing outcomes by cluster...")
        
        outcome_cols = ['final_cgpa', 'major_success', 'extended_graduation', 'completed_degree']
        available_outcomes = [c for c in outcome_cols if c in self.master_df.columns]
        
        if not available_outcomes:
            print("    ✗ No outcome columns available")
            return None
        
        # Create visualization
        n_outcomes = len(available_outcomes)
        fig, axes = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 5))
        
        if n_outcomes == 1:
            axes = [axes]
        
        for ax, outcome in zip(axes, available_outcomes):
            if outcome == 'final_cgpa':
                # Box plot for continuous outcome
                data_to_plot = []
                labels_to_plot = []
                
                for cluster in sorted(self.master_df[cluster_col].unique()):
                    cluster_data = self.master_df[
                        self.master_df[cluster_col] == cluster
                    ][outcome].dropna()
                    if len(cluster_data) > 0:
                        data_to_plot.append(cluster_data)
                        labels_to_plot.append(f'C{cluster}')
                
                ax.boxplot(data_to_plot, labels=labels_to_plot)
                ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=3.0, color='g', linestyle='--', alpha=0.5)
                ax.set_ylabel('CGPA')
                
            else:
                # Bar plot for binary outcomes
                outcome_by_cluster = self.master_df.groupby(cluster_col)[outcome].mean()
                bars = ax.bar(
                    [f'C{c}' for c in outcome_by_cluster.index],
                    outcome_by_cluster.values,
                    color=plt.cm.RdYlGn(outcome_by_cluster.values) if 'success' in outcome or 'completed' in outcome
                          else plt.cm.RdYlGn_r(outcome_by_cluster.values)
                )
                ax.set_ylabel('Rate')
                ax.set_ylim(0, 1)
            
            ax.set_xlabel('Cluster')
            ax.set_title(outcome.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}cluster_outcomes.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {self.output_dir}cluster_outcomes.png")
    
    def run_full_analysis(self, feature_type='admission', n_clusters=None):
        """
        Run complete unsupervised analysis pipeline
        """
        
        print("\n" + "=" * 60)
        print("UNSUPERVISED LEARNING ANALYSIS")
        print("=" * 60)
        
        # Step 1: Prepare data
        self.prepare_data(feature_type)
        
        if self.X_scaled is None:
            return None
        
        # Step 2: Find optimal clusters if not specified
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(max_k=8)
        
        # Step 3: Perform clustering
        self.perform_clustering(n_clusters, method='kmeans')
        
        # Also try GMM for comparison
        self.perform_clustering(n_clusters, method='gmm')
        
        # Step 4: Analyze clusters
        cluster_summary = self.analyze_clusters('cluster_kmeans')
        
        # Step 5: Visualize
        self.visualize_clusters('cluster_kmeans', method='both')
        
        # Step 6: Feature importance
        self.cluster_feature_importance('cluster_kmeans')
        
        # Step 7: Risk identification
        self.identify_risk_clusters('cluster_kmeans')
        
        # Step 8: Outcome analysis
        self.cluster_outcome_analysis('cluster_kmeans')
        
        print("\n" + "=" * 60)
        print("UNSUPERVISED ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            'cluster_summary': cluster_summary,
            'risk_clusters': self.results.get('risk_clusters'),
            'master_df_with_clusters': self.master_df
        }
    
    def generate_cluster_report(self, output_path='reports/cluster_report.txt'):
        """Generate text report of clustering results"""
        
        lines = [
            "=" * 70,
            "STUDENT SEGMENTATION REPORT",
            "Unsupervised Learning Analysis",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            ""
        ]
        
        # Clustering metrics
        if 'kmeans_metrics' in self.results:
            metrics = self.results['kmeans_metrics']
            lines.extend([
                "CLUSTERING METRICS",
                "-" * 40,
                f"Number of clusters: {metrics.get('n_clusters', 'N/A')}",
                f"Silhouette score: {metrics.get('silhouette', 'N/A'):.3f}",
                f"Calinski-Harabasz index: {metrics.get('calinski_harabasz', 'N/A'):.1f}",
                f"Davies-Bouldin index: {metrics.get('davies_bouldin', 'N/A'):.3f}",
                ""
            ])
        
        # Cluster profiles
        if 'cluster_kmeans' in self.cluster_profiles:
            profiles = self.cluster_profiles['cluster_kmeans']
            lines.extend([
                "CLUSTER PROFILES",
                "-" * 40
            ])
            
            for idx, row in profiles.iterrows():
                lines.append(f"\nCluster {idx}: {row['profile']}")
                lines.append(f"  Size: {int(row['count'])} ({row['pct_of_total']}%)")
                
                for col in row.index:
                    if col not in ['count', 'pct_of_total', 'profile']:
                        val = row[col]
                        if pd.notna(val):
                            if isinstance(val, float):
                                lines.append(f"  {col}: {val:.3f}")
            
            lines.append("")
        
        # Risk clusters
        if 'risk_clusters' in self.results:
            risk = self.results['risk_clusters']
            lines.extend([
                "RISK ASSESSMENT",
                "-" * 40
            ])
            
            for cluster_id, row in risk.iterrows():
                lines.append(f"\nCluster {cluster_id}: {row['risk_level']}")
                lines.append(f"  Composite Risk: {row['composite_risk']:.3f}")
                lines.append(f"  Size: {int(row['cluster_size'])}")
        
        lines.extend([
            "",
            "=" * 70,
            "RECOMMENDATIONS FOR INTERVENTION",
            "=" * 70,
            ""
        ])
        
        # Generate recommendations based on clusters
        if 'risk_clusters' in self.results:
            high_risk = self.results['risk_clusters'][
                self.results['risk_clusters']['risk_level'] == 'HIGH RISK'
            ]
            
            if len(high_risk) > 0:
                total_at_risk = high_risk['cluster_size'].sum()
                lines.append(f"• HIGH PRIORITY: {int(total_at_risk)} students in high-risk clusters")
                lines.append("  Recommended: Immediate academic intervention and support")
            
            moderate_risk = self.results['risk_clusters'][
                self.results['risk_clusters']['risk_level'] == 'MODERATE RISK'
            ]
            
            if len(moderate_risk) > 0:
                total_moderate = moderate_risk['cluster_size'].sum()
                lines.append(f"• MODERATE PRIORITY: {int(total_moderate)} students need monitoring")
                lines.append("  Recommended: Regular check-ins and preventive support")
        
        lines.extend([
            "",
            "=" * 70
        ])
        
        report = "\n".join(lines)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"    ✓ Saved: {output_path}")
        
        return report


# ============================================================================
# SECTION 7: MODEL TRAINING (UPDATED WITH TEMPORAL SPLIT)
# ============================================================================

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate models with multiple split strategies"""
    
    def __init__(self, master_df, feature_sets, semester_records=None):
        self.master_df = master_df.copy()
        self.feature_sets = feature_sets
        self.semester_records = semester_records
        self.results = {}
        self.best_models = {}
        self.scalers = {}
        self.encoders = {}
        self.split_info = {}
    
    def prepare_data(self, feature_type, target, split_method='random', 
                     test_size=0.2, test_years=None):
        """
        Prepare data with flexible split strategies
        
        Parameters:
        -----------
        feature_type : str
            'admission', 'year1', or 'year2'
        target : str
            Target variable name
        split_method : str
            'random' - Random stratified split (default)
            'temporal' - Split by year group
        test_size : float
            Proportion for test set (for random split)
        test_years : list
            Specific years to use as test set (for temporal split)
        """
        
        features = self.feature_sets.get(feature_type, [])
        
        if not features or target not in self.master_df.columns:
            print(f"    ✗ Missing features or target: {target}")
            return None
        
        # Get data with NON-NULL target (filters out invalid cases automatically)
        df = self.master_df[self.master_df[target].notna()].copy()
        
        print(f"      Valid samples for '{target}': {len(df)}")
        
        # Filter to available features (exclude status flags)
        exclude_cols = [
            'is_graduated', 'is_active', 'is_withdrawn', 'is_dismissed',
            'has_final_outcome', 'completed_year1', 'student_status', 
            'Student Status', 'student_id'
        ]
        available_features = [f for f in features 
                             if f in df.columns and f not in exclude_cols]
        
        if len(available_features) < 3:
            print(f"    ✗ Insufficient features ({len(available_features)})")
            return None
        
        print(f"      Using {len(available_features)} features")
        
        # Choose split method
        if split_method == 'temporal':
            return self._temporal_split(df, available_features, target, test_years)
        else:
            return self._random_split(df, available_features, target, test_size)
    
    def _random_split(self, df, features, target, test_size=0.2):
        """Standard random stratified split"""
        
        X = df[features].copy()
        y = df[target].astype(int).copy()
        
        # Encode categorical features
        X = self._encode_features(X, f"random_{target}")
        
        # Check minimum samples
        if len(X) < 50:
            print(f"    ✗ Insufficient samples ({len(X)})")
            return None
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"      Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            print(f"    ✗ Only one class present")
            return None
        
        if class_counts.min() < 5:
            print(f"    ⚠️ Minority class very small ({class_counts.min()})")
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
        except ValueError as e:
            print(f"    ✗ Split failed: {e}")
            return None
        
        print(f"      Split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Scale
        X_train_scaled, X_test_scaled, scaler = self._scale_features(
            X_train, X_test, features, f"random_{target}"
        )
        
        # Handle imbalance
        X_train_scaled, y_train = self._handle_imbalance(X_train_scaled, y_train)
        
        self.split_info[target] = {
            'method': 'random',
            'test_size': test_size,
            'train_size': len(X_train_scaled),
            'test_size_actual': len(X_test)
        }
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'features': features
        }
    
    def _temporal_split(self, df, features, target, test_years=None):
        """
        Temporal split - train on older cohorts, test on newer cohorts
        
        This is more realistic for predicting future student outcomes
        """
        
        # Find year column
        yg_col = None
        for col in ['Yeargroup', 'yeargroup', 'Year Group']:
            if col in df.columns:
                yg_col = col
                break
        
        if yg_col is None:
            print("    ⚠️ No year column found, falling back to random split")
            return self._random_split(df, features, target)
        
        # Get years sorted
        df[yg_col] = pd.to_numeric(df[yg_col], errors='coerce')
        df = df.dropna(subset=[yg_col])
        
        years = sorted(df[yg_col].unique())
        print(f"      Available years: {years}")
        
        if len(years) < 2:
            print("    ⚠️ Only one year available, falling back to random split")
            return self._random_split(df, features, target)
        
        # Determine test years
        if test_years is None:
            # Use most recent 20-25% of years as test
            n_test_years = max(1, len(years) // 4)
            test_years = years[-n_test_years:]
        
        train_years = [y for y in years if y not in test_years]
        
        print(f"      Train years: {train_years}")
        print(f"      Test years: {test_years}")
        
        # Split data
        train_mask = df[yg_col].isin(train_years)
        test_mask = df[yg_col].isin(test_years)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 30 or len(test_df) < 10:
            print(f"    ⚠️ Insufficient data for temporal split, falling back to random")
            return self._random_split(df, features, target)
        
        X_train = train_df[features].copy()
        y_train = train_df[target].astype(int).copy()
        X_test = test_df[features].copy()
        y_test = test_df[target].astype(int).copy()
        
        print(f"      Split: Train={len(X_train)} ({train_years}), Test={len(X_test)} ({test_years})")
        
        # Encode (fit on train, transform both)
        X_train = self._encode_features(X_train, f"temporal_{target}", fit=True)
        X_test = self._encode_features(X_test, f"temporal_{target}", fit=False)
        
        # Check class distribution in both sets
        print(f"      Train class dist: {dict(y_train.value_counts())}")
        print(f"      Test class dist: {dict(y_test.value_counts())}")
        
        # Scale
        X_train_scaled, X_test_scaled, scaler = self._scale_features(
            X_train, X_test, features, f"temporal_{target}"
        )
        
        # Handle imbalance (only on training data)
        X_train_scaled, y_train = self._handle_imbalance(X_train_scaled, y_train)
        
        self.split_info[target] = {
            'method': 'temporal',
            'train_years': train_years,
            'test_years': test_years,
            'train_size': len(X_train_scaled),
            'test_size': len(X_test)
        }
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'features': features,
            'train_years': train_years,
            'test_years': test_years
        }
    
    def _encode_features(self, X, key_prefix, fit=True):
        """Encode categorical features"""
        
        X = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('Unknown')
                
                encoder_key = f"{key_prefix}_{col}"
                
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[encoder_key] = le
                else:
                    if encoder_key in self.encoders:
                        le = self.encoders[encoder_key]
                        # Handle unseen categories
                        X[col] = X[col].apply(
                            lambda x: x if x in le.classes_ else 'Unknown'
                        )
                        X[col] = le.transform(X[col].astype(str))
                    else:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def _scale_features(self, X_train, X_test, features, key):
        """Scale features"""
        
        scaler = StandardScaler()
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=features,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=features,
            index=X_test.index
        )
        
        self.scalers[key] = scaler
        
        return X_train_scaled, X_test_scaled, scaler
    
    def _handle_imbalance(self, X_train, y_train):
        """Handle class imbalance with SMOTE"""
        
        class_counts = y_train.value_counts()
        class_ratio = class_counts.min() / class_counts.max()
        
        if class_ratio < 0.3 and len(y_train) > 100:
            print(f"      Applying SMOTE (ratio: {class_ratio:.2f})")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"      After SMOTE: {len(X_resampled)} samples")
                return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
            except ImportError:
                print("      ⚠️ imbalanced-learn not installed")
            except Exception as e:
                print(f"      ⚠️ SMOTE failed: {e}")
        
        return X_train, y_train
    
    def get_models(self):
        """Define models to evaluate"""
        
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
        }
    
    def train_task(self, feature_type, target, split_method='temporal'):
        """Train models for a specific prediction task"""
        
        task_key = f"{feature_type}_{target}"
        print(f"\n    Training: {task_key} (split: {split_method})")
        
        data = self.prepare_data(feature_type, target, split_method=split_method)
        
        if data is None:
            return None
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        features = data['features']
        
        models = self.get_models()
        
        best_auc = 0
        best_model_name = None
        task_results = {}
        
        for name, model in models.items():
            try:
                # Cross-validation on training data
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                # Fit on full training set
                model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std()
                }
                
                task_results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"      {name}: AUC={metrics['roc_auc']:.3f}, F1={metrics['f1']:.3f}")
                
                if metrics['roc_auc'] > best_auc:
                    best_auc = metrics['roc_auc']
                    best_model_name = name
                    
            except Exception as e:
                print(f"      {name}: FAILED - {e}")
        
        # Store best model
        if best_model_name:
            self.best_models[task_key] = {
                'name': best_model_name,
                'model': task_results[best_model_name]['model'],
                'metrics': task_results[best_model_name]['metrics'],
                'features': features,
                'split_info': self.split_info.get(target, {}),
                'confusion_matrix': task_results[best_model_name]['confusion_matrix']
            }
            print(f"      ✓ Best: {best_model_name} (AUC={best_auc:.3f})")
        
        self.results[task_key] = task_results
        
        return task_results
    
    def train_all_tasks(self, split_method='temporal'):
        """Train models for all research questions"""
        
        print(f"\n  Training all models (split method: {split_method})...")
        
        # Define tasks with appropriate targets
        tasks = [
            # Q1: First year struggle - can include active students who completed Y1
            ('admission', 'first_year_struggle'),
            
            # Q2: AJC case - can include all students
            ('admission', 'has_ajc_case'),
            
            # Q3-Q4: Major success/struggle with Y1 data - graduated only
            ('year1', 'major_success'),
            ('year1', 'major_struggle'),
            
            # Q9: Extended graduation - graduated only
            ('year1', 'extended_graduation'),
            
            # NEW: Completion - non-active students
            ('admission', 'completed_degree'),
            
            # NEW: Retention after Y1
            ('admission', 'retained_after_y1'),
        ]
        
        for feature_type, target in tasks:
            if target in self.master_df.columns:
                # Check if we have valid cases
                valid_count = self.master_df[target].notna().sum()
                if valid_count >= 50:
                    self.train_task(feature_type, target, split_method=split_method)
                else:
                    print(f"\n    Skipping {target}: Only {valid_count} valid cases")
            else:
                print(f"\n    Skipping {target}: Column not found")
        
        return self.results
    
    def get_feature_importance(self, task_key):
        """Get feature importance from best model"""
        
        if task_key not in self.best_models:
            return None
        
        model_info = self.best_models[task_key]
        model = model_info['model']
        features = model_info['features']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_summary(self):
        """Get summary of all results"""
        
        summary = []
        
        for task_key, model_info in self.best_models.items():
            metrics = model_info['metrics']
            split_info = model_info.get('split_info', {})
            
            summary.append({
                'Task': task_key,
                'Best Model': model_info['name'],
                'ROC AUC': round(metrics['roc_auc'], 3),
                'Accuracy': round(metrics['accuracy'], 3),
                'F1 Score': round(metrics['f1'], 3),
                'CV AUC': f"{metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}",
                'Split': split_info.get('method', 'unknown'),
                'Train Size': split_info.get('train_size', 'N/A'),
                'Test Size': split_info.get('test_size', split_info.get('test_size_actual', 'N/A'))
            })
        
        return pd.DataFrame(summary)
    
    def save_models(self, output_dir='models/'):
        """Save trained models"""
        
        import joblib
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        for task_key, model_info in self.best_models.items():
            # Save model
            model_path = f"{output_dir}{task_key}_model.joblib"
            joblib.dump(model_info['model'], model_path)
            
            # Save scaler if exists
            for scaler_key, scaler in self.scalers.items():
                if task_key.split('_')[-1] in scaler_key:
                    scaler_path = f"{output_dir}{task_key}_scaler.joblib"
                    joblib.dump(scaler, scaler_path)
                    break
            
            # Save features
            features_path = f"{output_dir}{task_key}_features.json"
            with open(features_path, 'w') as f:
                json.dump(model_info['features'], f)
            
            # Save split info
            split_path = f"{output_dir}{task_key}_split_info.json"
            with open(split_path, 'w') as f:
                # Convert any numpy types to native Python types
                split_info = model_info.get('split_info', {})
                split_info_clean = {}
                for k, v in split_info.items():
                    if isinstance(v, (np.integer, np.floating)):
                        split_info_clean[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        split_info_clean[k] = v.tolist()
                    elif isinstance(v, list):
                        split_info_clean[k] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
                    else:
                        split_info_clean[k] = v
                json.dump(split_info_clean, f)
            
            print(f"    Saved: {task_key}")


# ============================================================================
# SECTION 8: STATISTICAL ANALYSIS
# ============================================================================

from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal

class StatisticalAnalyzer:
    """Statistical tests for research questions"""
    
    def __init__(self, master_df, semester_records):
        self.master_df = master_df.copy()
        self.semester_records = semester_records.copy() if semester_records is not None else pd.DataFrame()
        self.results = {}
    
    def run_all_analyses(self):
        """Run all statistical analyses"""
        
        print("\n  Running statistical analyses...")
        
        self._analyze_math_track()      # Q7
        self._analyze_cs_college_algebra()  # Q8
        self._analyze_risk_factors()
        
        return self.results
    
    def _analyze_math_track(self):
        """Q7: Compare performance across math tracks"""
        
        if 'math_track' not in self.master_df.columns:
            print("    ✗ Math track data not available")
            return
        
        df = self.master_df[
            (self.master_df['math_track'].notna()) & 
            (self.master_df['math_track'] != 'Unknown') &
            (self.master_df['final_cgpa'].notna())
        ]
        
        if len(df) < 30:
            print("    ✗ Insufficient data for math track analysis")
            return
        
        print("    Q7: Math Track Analysis")
        
        # Summary stats
        summary = df.groupby('math_track').agg({
            'final_cgpa': ['count', 'mean', 'std', 'median'],
            'total_semesters': 'mean' if 'total_semesters' in df.columns else 'count',
            'first_year_struggle': 'mean' if 'first_year_struggle' in df.columns else 'count'
        })
        
        print(f"       Summary:\n{summary}")
        
        # Kruskal-Wallis test
        tracks = df['math_track'].unique()
        groups = [df[df['math_track'] == t]['final_cgpa'].dropna() for t in tracks]
        groups = [g for g in groups if len(g) >= 5]
        
        if len(groups) >= 2:
            stat, p_value = kruskal(*groups)
            print(f"       Kruskal-Wallis: H={stat:.3f}, p={p_value:.4f}")
            print(f"       Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
            
            self.results['math_track'] = {
                'summary': summary,
                'kruskal_h': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    def _analyze_cs_college_algebra(self):
        """Q8: Can College Algebra students succeed in CS?"""
        
        if 'is_cs_major' not in self.master_df.columns:
            # Try to identify CS students
            if 'final_program' in self.master_df.columns:
                self.master_df['is_cs_major'] = self.master_df['final_program'].str.contains(
                    'Computer', case=False, na=False
                ).astype(int)
            else:
                print("    ✗ Cannot identify CS majors")
                return
        
        cs_df = self.master_df[self.master_df['is_cs_major'] == 1]
        
        if len(cs_df) < 10:
            print("    ✗ Insufficient CS students for analysis")
            return
        
        print(f"    Q8: CS + College Algebra Analysis (n={len(cs_df)})")
        
        if 'math_track' not in cs_df.columns or cs_df['math_track'].isna().all():
            print("       ✗ Math track not available for CS students")
            return
        
        # Summary by math track
        cs_summary = cs_df.groupby('math_track').agg({
            'student_id': 'count',
            'final_cgpa': 'mean',
            'major_success': 'mean' if 'major_success' in cs_df.columns else 'count'
        })
        cs_summary.columns = ['Count', 'Avg CGPA', 'Success Rate']
        
        print(f"       {cs_summary}")
        
        # Specific analysis for College Algebra in CS
        ca_cs = cs_df[cs_df['math_track'] == 'College Algebra']
        other_cs = cs_df[cs_df['math_track'] != 'College Algebra']
        
        if len(ca_cs) >= 5 and len(other_cs) >= 5:
            # Mann-Whitney U test
            stat, p = mannwhitneyu(
                ca_cs['final_cgpa'].dropna(),
                other_cs['final_cgpa'].dropna(),
                alternative='two-sided'
            )
            
            ca_success = ca_cs['major_success'].mean() if 'major_success' in ca_cs.columns else 'N/A'
            
            print(f"       College Algebra in CS: n={len(ca_cs)}, Success Rate={ca_success:.1%}" if isinstance(ca_success, float) else f"       College Algebra in CS: n={len(ca_cs)}")
            print(f"       Mann-Whitney U: p={p:.4f}")
            
            self.results['cs_college_algebra'] = {
                'summary': cs_summary,
                'ca_count': len(ca_cs),
                'ca_success_rate': ca_success if isinstance(ca_success, float) else None,
                'mann_whitney_p': p
            }
    
    def _analyze_risk_factors(self):
        """Analyze key risk factors"""
        
        print("    Risk Factor Analysis")
        
        target_col = 'first_year_struggle'
        
        if target_col not in self.master_df.columns:
            return
        
        # Categorical factors
        cat_factors = ['Gender', 'math_track', 'exam_source', 'performance_tier']
        
        for factor in cat_factors:
            if factor in self.master_df.columns:
                try:
                    contingency = pd.crosstab(
                        self.master_df[factor], 
                        self.master_df[target_col]
                    )
                    
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        chi2, p, dof, expected = chi2_contingency(contingency)
                        print(f"       {factor}: χ²={chi2:.2f}, p={p:.4f}")
                except:
                    pass
    
    def generate_report(self):
        """Generate text report"""
        
        lines = [
            "=" * 60,
            "STATISTICAL ANALYSIS REPORT",
            "=" * 60,
            ""
        ]
        
        if 'math_track' in self.results:
            r = self.results['math_track']
            lines.extend([
                "Q7: MATH TRACK PERFORMANCE",
                "-" * 40,
                f"Kruskal-Wallis H: {r['kruskal_h']:.3f}",
                f"p-value: {r['p_value']:.4f}",
                f"Significant: {'Yes' if r['significant'] else 'No'}",
                ""
            ])
        
        if 'cs_college_algebra' in self.results:
            r = self.results['cs_college_algebra']
            lines.extend([
                "Q8: CS + COLLEGE ALGEBRA",
                "-" * 40,
                f"College Algebra CS students: {r['ca_count']}",
                f"Success rate: {r['ca_success_rate']:.1%}" if r['ca_success_rate'] else "Success rate: N/A",
                f"Mann-Whitney p-value: {r['mann_whitney_p']:.4f}",
                ""
            ])
        
        return "\n".join(lines)


# ============================================================================
# SECTION 9: EDA VISUALIZATION
# ============================================================================

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class EDAGenerator:
    """Generate EDA visualizations"""
    
    def __init__(self, master_df, output_dir='reports/figures/'):
        self.master_df = master_df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def generate_all(self):
        """Generate all plots"""
        
        print("\n  Generating visualizations...")
        
        self._plot_distributions()
        self._plot_performance()
        self._plot_risk_factors()
        self._plot_correlations()
        
        print(f"    ✓ Saved to {self.output_dir}")
    
    def _plot_distributions(self):
        """Plot target variable distributions"""
        
        targets = ['first_year_struggle', 'has_ajc_case', 'major_success', 'extended_graduation']
        available = [t for t in targets if t in self.master_df.columns]
        
        if not available:
            return
        
        n_plots = len(available)
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, target in zip(axes, available):
            counts = self.master_df[target].value_counts()
            colors = ['#27ae60', '#e74c3c']
            ax.bar([str(i) for i in counts.index], counts.values, color=colors[:len(counts)])
            ax.set_title(target.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            
            # Add percentages
            total = counts.sum()
            for i, v in enumerate(counts.values):
                ax.text(i, v + total*0.01, f'{v/total:.1%}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}target_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance(self):
        """Plot performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. CGPA distribution
        if 'final_cgpa' in self.master_df.columns:
            ax = axes[0, 0]
            self.master_df['final_cgpa'].hist(bins=20, ax=ax, color='steelblue', edgecolor='white')
            ax.axvline(x=2.0, color='red', linestyle='--', label='Probation (2.0)')
            ax.axvline(x=3.0, color='orange', linestyle='--', label='Success (3.0)')
            ax.axvline(x=3.5, color='green', linestyle='--', label="Dean's List (3.5)")
            ax.set_title('Final CGPA Distribution')
            ax.set_xlabel('CGPA')
            ax.legend()
        
        # 2. Performance by major
        if 'final_program' in self.master_df.columns and 'final_cgpa' in self.master_df.columns:
            ax = axes[0, 1]
            major_perf = self.master_df.groupby('final_program')['final_cgpa'].agg(['mean', 'count'])
            major_perf = major_perf[major_perf['count'] >= 10].sort_values('mean')
            
            if len(major_perf) > 0:
                ax.barh(major_perf.index, major_perf['mean'], color='teal')
                ax.axvline(x=3.0, color='green', linestyle='--')
                ax.set_title('Average CGPA by Major')
                ax.set_xlabel('CGPA')
        
        # 3. Math track performance
        if 'math_track' in self.master_df.columns and 'final_cgpa' in self.master_df.columns:
            ax = axes[1, 0]
            df = self.master_df[self.master_df['math_track'] != 'Unknown']
            if len(df) > 0:
                df.boxplot(column='final_cgpa', by='math_track', ax=ax)
                ax.axhline(y=2.0, color='red', linestyle='--')
                ax.axhline(y=3.0, color='green', linestyle='--')
                ax.set_title('CGPA by Math Track')
                ax.set_xlabel('Math Track')
                ax.set_ylabel('Final CGPA')
                plt.suptitle('')
        
        # 4. Struggle rate by performance tier
        if 'performance_tier' in self.master_df.columns and 'first_year_struggle' in self.master_df.columns:
            ax = axes[1, 1]
            tier_struggle = self.master_df.groupby('performance_tier')['first_year_struggle'].mean()
            tier_order = ['Excellent', 'Good', 'Average', 'Below Average', 'At Risk']
            tier_struggle = tier_struggle.reindex([t for t in tier_order if t in tier_struggle.index])
            
            if len(tier_struggle) > 0:
                ax.bar(tier_struggle.index, tier_struggle.values, color='coral')
                ax.set_title('First Year Struggle Rate by HS Performance')
                ax.set_xlabel('High School Performance Tier')
                ax.set_ylabel('Struggle Rate')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}performance_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_factors(self):
        """Plot risk factor analysis"""
        
        if 'first_year_struggle' not in self.master_df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. By gender
        if 'Gender' in self.master_df.columns:
            ax = axes[0, 0]
            gender_risk = self.master_df.groupby('Gender')['first_year_struggle'].mean()
            ax.bar(gender_risk.index, gender_risk.values, color=['#3498db', '#e74c3c', '#95a5a6'])
            ax.set_title('First Year Struggle Rate by Gender')
            ax.set_ylabel('Struggle Rate')
        
        # 2. By financial aid need
        if 'needs_financial_aid' in self.master_df.columns:
            ax = axes[0, 1]
            aid_risk = self.master_df.groupby('needs_financial_aid')['first_year_struggle'].mean()
            ax.bar(['No Aid', 'Needs Aid'], aid_risk.values, color=['#2ecc71', '#e74c3c'])
            ax.set_title('Struggle Rate by Financial Aid Need')
            ax.set_ylabel('Struggle Rate')
        
        # 3. By exam type
        if 'exam_source' in self.master_df.columns:
            ax = axes[1, 0]
            exam_risk = self.master_df.groupby('exam_source')['first_year_struggle'].mean()
            exam_risk = exam_risk.sort_values()
            ax.barh(exam_risk.index, exam_risk.values, color='steelblue')
            ax.set_title('Struggle Rate by Exam Type')
            ax.set_xlabel('Struggle Rate')
        
        # 4. By international status
        if 'is_international' in self.master_df.columns:
            ax = axes[1, 1]
            intl_risk = self.master_df.groupby('is_international')['first_year_struggle'].mean()
            ax.bar(['Domestic', 'International'], intl_risk.values, color=['#3498db', '#9b59b6'])
            ax.set_title('Struggle Rate: Domestic vs International')
            ax.set_ylabel('Struggle Rate')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}risk_factors.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_correlations(self):
        """Plot correlation matrix"""
        
        # Select numeric columns related to outcomes
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns
        
        key_patterns = ['cgpa', 'gpa', 'score', 'struggle', 'success', 'ajc', 'probation', 'semester']
        key_cols = [c for c in numeric_cols if any(p in c.lower() for p in key_patterns)]
        
        if len(key_cols) < 3:
            return
        
        # Limit to top 15 columns
        key_cols = key_cols[:15]
        
        corr = self.master_df[key_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title('Correlation Matrix of Key Variables')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}correlations.png', dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# SECTION 10: REPORT GENERATION (CONTINUED)
# ============================================================================

class ReportGenerator:
    """Generate final reports"""
    
    def __init__(self, master_df, model_results, stats_results):
        self.master_df = master_df
        self.model_results = model_results or {}
        self.stats_results = stats_results or {}
    
    def generate_executive_summary(self, output_path='reports/executive_summary.txt'):
        """Generate executive summary"""
        
        lines = [
            "=" * 80,
            "ASHESI STUDENT SUCCESS PREDICTION",
            "EXECUTIVE SUMMARY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "1. OVERVIEW",
            "-" * 40
        ]
        
        # Data summary
        if self.master_df is not None and len(self.master_df) > 0:
            lines.append(f"Total Students Analyzed: {len(self.master_df)}")
            
            if 'first_year_struggle' in self.master_df.columns:
                rate = self.master_df['first_year_struggle'].mean() * 100
                lines.append(f"First Year Struggle Rate: {rate:.1f}%")
            
            if 'major_success' in self.master_df.columns:
                rate = self.master_df['major_success'].mean() * 100
                lines.append(f"Major Success Rate: {rate:.1f}%")
            
            if 'has_ajc_case' in self.master_df.columns:
                rate = self.master_df['has_ajc_case'].mean() * 100
                lines.append(f"AJC Case Rate: {rate:.1f}%")
            
            if 'extended_graduation' in self.master_df.columns:
                rate = self.master_df['extended_graduation'].mean() * 100
                lines.append(f"Extended Graduation Rate: {rate:.1f}%")
        
        lines.extend(["", "2. MODEL PERFORMANCE", "-" * 40])
        
        # Model results
        if self.model_results:
            for task, info in self.model_results.items():
                if 'metrics' in info:
                    m = info['metrics']
                    lines.append(f"\n{task}:")
                    lines.append(f"  Best Model: {info.get('name', 'N/A')}")
                    lines.append(f"  ROC AUC: {m.get('roc_auc', 0):.3f}")
                    lines.append(f"  Accuracy: {m.get('accuracy', 0):.3f}")
                    lines.append(f"  F1 Score: {m.get('f1', 0):.3f}")
        else:
            lines.append("  No model results available")
        
        lines.extend(["", "3. KEY FINDINGS", "-" * 40])
        
        # Key findings based on analysis
        findings = self._generate_key_findings()
        for i, finding in enumerate(findings, 1):
            lines.append(f"  {i}. {finding}")
        
        lines.extend(["", "4. RESEARCH QUESTIONS ADDRESSED", "-" * 40])
        
        # Research questions summary
        questions = [
            ("Q1", "Predict first-year academic struggle", "admission_first_year_struggle"),
            ("Q2", "Predict AJC cases", "admission_has_ajc_case"),
            ("Q3", "Predict major success (Y1 data)", "year1_major_success"),
            ("Q4", "Predict major struggle (Y1 data)", "year1_major_struggle"),
            ("Q5", "Predict major success (Y1+Y2 data)", "year2_major_success"),
            ("Q6", "Predict major struggle (Y1+Y2 data)", "year2_major_struggle"),
            ("Q7", "Math track performance differences", "statistical_test"),
            ("Q8", "College Algebra success in CS", "statistical_test"),
            ("Q9", "Predict extended graduation", "year1_extended_graduation")
        ]
        
        for q_id, q_desc, task_key in questions:
            if task_key in self.model_results:
                auc = self.model_results[task_key]['metrics'].get('roc_auc', 0)
                status = f"✓ Achieved (AUC: {auc:.3f})"
            elif task_key == "statistical_test":
                if q_id == "Q7" and 'math_track' in self.stats_results:
                    p = self.stats_results['math_track'].get('p_value', 1)
                    sig = "Significant" if p < 0.05 else "Not significant"
                    status = f"✓ Analyzed (p={p:.4f}, {sig})"
                elif q_id == "Q8" and 'cs_college_algebra' in self.stats_results:
                    status = "✓ Analyzed"
                else:
                    status = "○ Not analyzed"
            else:
                status = "○ Not trained"
            
            lines.append(f"  {q_id}: {q_desc}")
            lines.append(f"      Status: {status}")
        
        lines.extend(["", "5. STATISTICAL ANALYSIS RESULTS", "-" * 40])
        
        # Math track analysis (Q7)
        if 'math_track' in self.stats_results:
            r = self.stats_results['math_track']
            lines.append("\n  Q7: Math Track Performance Comparison")
            lines.append(f"      Kruskal-Wallis H-statistic: {r.get('kruskal_h', 'N/A'):.3f}")
            lines.append(f"      p-value: {r.get('p_value', 'N/A'):.4f}")
            lines.append(f"      Conclusion: {'Significant difference exists' if r.get('significant') else 'No significant difference'}")
        
        # CS + College Algebra analysis (Q8)
        if 'cs_college_algebra' in self.stats_results:
            r = self.stats_results['cs_college_algebra']
            lines.append("\n  Q8: College Algebra Track in Computer Science")
            lines.append(f"      College Algebra CS students: {r.get('ca_count', 'N/A')}")
            if r.get('ca_success_rate') is not None:
                lines.append(f"      Success rate: {r['ca_success_rate']:.1%}")
            lines.append(f"      Conclusion: College Algebra students {'CAN' if r.get('ca_success_rate', 0) >= 0.5 else 'face challenges to'} succeed in CS")
        
        lines.extend(["", "6. RECOMMENDATIONS", "-" * 40])
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.extend(["", "7. IMPLEMENTATION PRIORITIES", "-" * 40])
        
        priorities = [
            "HIGH PRIORITY:",
            "  • Deploy early warning system for incoming students",
            "  • Implement first-semester intervention protocols",
            "  • Create math track support programs",
            "",
            "MEDIUM PRIORITY:",
            "  • Develop major-specific mentoring programs",
            "  • Build academic integrity awareness campaigns",
            "  • Enhance financial aid advising",
            "",
            "LOW PRIORITY (Long-term):",
            "  • Integrate with student information system",
            "  • Create personalized learning pathways",
            "  • Build predictive analytics dashboard"
        ]
        
        lines.extend(priorities)
        
        lines.extend([
            "",
            "8. DATA QUALITY NOTES",
            "-" * 40
        ])
        
        # Data quality summary
        if self.master_df is not None and len(self.master_df) > 0:
            total_cols = len(self.master_df.columns)
            missing_pct = (self.master_df.isnull().sum().sum() / 
                          (len(self.master_df) * total_cols)) * 100
            
            lines.append(f"  Total features: {total_cols}")
            lines.append(f"  Overall missing data: {missing_pct:.1f}%")
            lines.append(f"  Students with complete records: {len(self.master_df.dropna())}")
            
            # Exam type distribution
            if 'exam_source' in self.master_df.columns:
                lines.append("\n  Exam Type Distribution:")
                exam_dist = self.master_df['exam_source'].value_counts()
                for exam, count in exam_dist.items():
                    lines.append(f"    • {exam}: {count} ({count/len(self.master_df)*100:.1f}%)")
        
        lines.extend([
            "",
            "=" * 80,
            "END OF EXECUTIVE SUMMARY",
            "=" * 80
        ])
        
        # Write to file
        report_text = "\n".join(lines)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"    ✓ Saved: {output_path}")
        
        return report_text
    
    def _generate_key_findings(self):
        """Generate key findings based on analysis results"""
        
        findings = []
        
        # Finding 1: Early prediction capability
        if 'admission_first_year_struggle' in self.model_results:
            auc = self.model_results['admission_first_year_struggle']['metrics'].get('roc_auc', 0)
            if auc > 0.7:
                findings.append(
                    f"Early prediction of academic struggle is achievable with good accuracy "
                    f"(ROC AUC: {auc:.3f}) using only admission data."
                )
            elif auc > 0.6:
                findings.append(
                    f"Moderate prediction of first-year struggle is possible (ROC AUC: {auc:.3f}), "
                    f"but additional features may improve accuracy."
                )
        
        # Finding 2: Math track impact
        if 'math_track' in self.stats_results:
            if self.stats_results['math_track'].get('significant'):
                findings.append(
                    "Math track placement significantly impacts academic performance. "
                    "Students on different tracks show statistically different outcomes."
                )
        
        # Finding 3: CS + College Algebra
        if 'cs_college_algebra' in self.stats_results:
            r = self.stats_results['cs_college_algebra']
            if r.get('ca_success_rate') is not None:
                if r['ca_success_rate'] >= 0.5:
                    findings.append(
                        f"College Algebra students CAN succeed in Computer Science "
                        f"(Success rate: {r['ca_success_rate']:.1%}), though additional support is recommended."
                    )
                else:
                    findings.append(
                        f"College Algebra students face challenges in Computer Science "
                        f"(Success rate: {r['ca_success_rate']:.1%}). Targeted support programs are needed."
                    )
        
        # Finding 4: First semester importance
        if self.master_df is not None and 'y1_GPA_mean' in self.master_df.columns:
            findings.append(
                "First semester/year GPA is a strong predictor of overall academic success. "
                "Early intervention during Year 1 is critical."
            )
        
        # Finding 5: AJC prediction
        if 'admission_has_ajc_case' in self.model_results:
            auc = self.model_results['admission_has_ajc_case']['metrics'].get('roc_auc', 0)
            findings.append(
                f"Academic misconduct risk can be partially predicted from admission data "
                f"(ROC AUC: {auc:.3f}). Proactive integrity education is recommended."
            )
        
        # Finding 6: Extended graduation
        if 'year1_extended_graduation' in self.model_results:
            auc = self.model_results['year1_extended_graduation']['metrics'].get('roc_auc', 0)
            findings.append(
                f"Students at risk of extended graduation (>8 semesters) can be identified "
                f"after Year 1 (ROC AUC: {auc:.3f}). Course planning support should be offered."
            )
        
        # Default finding if none generated
        if not findings:
            findings.append(
                "Analysis completed. Further data collection and model refinement recommended."
            )
        
        return findings
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Recommendation 1: Early Warning System
        if 'admission_first_year_struggle' in self.model_results:
            auc = self.model_results['admission_first_year_struggle']['metrics'].get('roc_auc', 0)
            if auc > 0.65:
                recommendations.append(
                    "EARLY WARNING SYSTEM: Deploy the first-year struggle prediction model "
                    "to identify at-risk students during admission. Flag students with >50% "
                    "risk probability for immediate academic support enrollment."
                )
        
        # Recommendation 2: Math Track Support
        if 'math_track' in self.stats_results:
            recommendations.append(
                "MATH SUPPORT PROGRAM: Create differentiated support programs for each math track. "
                "College Algebra students pursuing quantitative majors (CS, Engineering, MIS) "
                "should receive supplementary math tutoring and extended office hours."
            )
        
        # Recommendation 3: First Semester Intervention
        recommendations.append(
            "FIRST SEMESTER INTERVENTION: Implement mandatory check-ins for all students "
            "after week 6 of first semester. Students with GPA < 2.5 should be enrolled "
            "in academic success workshops and peer tutoring programs."
        )
        
        # Recommendation 4: AJC Prevention
        if 'admission_has_ajc_case' in self.model_results:
            recommendations.append(
                "ACADEMIC INTEGRITY PROGRAM: Conduct proactive academic integrity workshops "
                "during orientation. Students identified as higher risk should receive "
                "additional ethics training and citation skills workshops."
            )
        
        # Recommendation 5: Graduation Planning
        recommendations.append(
            "GRADUATION TIMELINE ADVISING: Students predicted to need >8 semesters should "
            "receive enhanced academic advising including optimized course sequencing, "
            "summer course recommendations, and workload management support."
        )
        
        # Recommendation 6: Major-Specific Support
        recommendations.append(
            "MAJOR-SPECIFIC MENTORING: Pair at-risk students with successful upper-year "
            "students in their intended major. Focus on majors with higher struggle rates."
        )
        
        # Recommendation 7: Dashboard Deployment
        recommendations.append(
            "ANALYTICS DASHBOARD: Deploy the interactive dashboard for admissions officers "
            "and academic advisors to access real-time risk predictions and student insights."
        )
        
        return recommendations
    
    def generate_model_report(self, trainer, output_path='reports/model_report.txt'):
        """Generate detailed model performance report"""
        
        lines = [
            "=" * 80,
            "PREDICTIVE MODEL PERFORMANCE REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]
        
        for task_key, model_info in trainer.best_models.items():
            lines.append(f"\n{'='*60}")
            lines.append(f"TASK: {task_key}")
            lines.append(f"{'='*60}")
            
            lines.append(f"\nBest Model: {model_info['name']}")
            lines.append(f"Features Used: {len(model_info['features'])}")
            
            # Metrics
            metrics = model_info['metrics']
            lines.append("\nPerformance Metrics:")
            lines.append(f"  • ROC AUC: {metrics['roc_auc']:.4f}")
            lines.append(f"  • Accuracy: {metrics['accuracy']:.4f}")
            lines.append(f"  • Precision: {metrics['precision']:.4f}")
            lines.append(f"  • Recall: {metrics['recall']:.4f}")
            lines.append(f"  • F1 Score: {metrics['f1']:.4f}")
            lines.append(f"  • CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
            
            # Feature importance
            importance = trainer.get_feature_importance(task_key)
            if importance is not None and len(importance) > 0:
                lines.append("\nTop 10 Important Features:")
                for i, row in importance.head(10).iterrows():
                    lines.append(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            lines.append("")
        
        # Write to file
        report_text = "\n".join(lines)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"    ✓ Saved: {output_path}")
        
        return report_text
    
    def generate_statistical_report(self, output_path='reports/statistical_report.txt'):
        """Generate statistical analysis report"""
        
        lines = [
            "=" * 80,
            "STATISTICAL ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]
        
        # Q7: Math Track Analysis
        lines.append("Q7: MATH TRACK PERFORMANCE COMPARISON")
        lines.append("-" * 50)
        
        if 'math_track' in self.stats_results:
            r = self.stats_results['math_track']
            lines.append(f"\nStatistical Test: Kruskal-Wallis H-test")
            lines.append(f"H-statistic: {r.get('kruskal_h', 'N/A'):.4f}")
            lines.append(f"p-value: {r.get('p_value', 'N/A'):.6f}")
            lines.append(f"Alpha level: 0.05")
            lines.append(f"\nConclusion: {'REJECT' if r.get('significant') else 'FAIL TO REJECT'} null hypothesis")
            
            if r.get('significant'):
                lines.append("Interpretation: There IS a statistically significant difference in ")
                lines.append("academic performance across different math tracks.")
            else:
                lines.append("Interpretation: There is NO statistically significant difference in ")
                lines.append("academic performance across different math tracks.")
        else:
            lines.append("Analysis not performed - insufficient data")
        
        lines.append("")
        
        # Q8: CS + College Algebra
        lines.append("\nQ8: COLLEGE ALGEBRA TRACK SUCCESS IN COMPUTER SCIENCE")
        lines.append("-" * 50)
        
        if 'cs_college_algebra' in self.stats_results:
            r = self.stats_results['cs_college_algebra']
            lines.append(f"\nTotal CS Students Analyzed: {r.get('ca_count', 'N/A') + r.get('other_count', 0)}")
            lines.append(f"College Algebra CS Students: {r.get('ca_count', 'N/A')}")
            
            if r.get('ca_success_rate') is not None:
                lines.append(f"College Algebra Success Rate: {r['ca_success_rate']:.1%}")
            
            if r.get('mann_whitney_p') is not None:
                lines.append(f"\nMann-Whitney U Test p-value: {r['mann_whitney_p']:.6f}")
            
            lines.append("\nConclusion:")
            if r.get('ca_success_rate', 0) >= 0.5:
                lines.append("  College Algebra students CAN succeed in Computer Science.")
                lines.append("  However, additional support is recommended to maximize success rates.")
            else:
                lines.append("  College Algebra students face significant challenges in Computer Science.")
                lines.append("  Strong intervention programs are recommended for these students.")
        else:
            lines.append("Analysis not performed - insufficient data")
        
        lines.append("")
        
        # Academic Policy Thresholds
        lines.append("\nACADEMIC POLICY REFERENCE")
        lines.append("-" * 50)
        lines.append("• Academic Probation: CGPA < 2.0 at end of any regular semester")
        lines.append("• Dismissal: Two consecutive semesters on probation without GPA ≥ 2.0")
        lines.append("• Dean's List: Semester GPA ≥ 3.5")
        lines.append("• Standard Graduation: 8 semesters (4 years)")
        
        lines.append("")
        lines.append("=" * 80)
        
        # Write to file
        report_text = "\n".join(lines)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"    ✓ Saved: {output_path}")
        
        return report_text


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_full_pipeline(data_path='data/raw/', split_method='temporal', verbose=True):
    """
    Run the complete analysis pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to raw data files
    split_method : str
        'temporal' - Train on older cohorts, test on newer (recommended)
        'random' - Random stratified split
    verbose : bool
        Print detailed output
    """
    
    print("\n" + "=" * 70)
    print("ASHESI STUDENT SUCCESS PREDICTION PIPELINE")
    print(f"Split Method: {split_method.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # =========== PHASE 1: DATA LOADING ===========
    print("\n[PHASE 1] DATA LOADING")
    print("-" * 50)
    
    loader = DataLoader(data_path)
    datasets = loader.load_all_datasets()
    
    if not datasets:
        print("ERROR: No datasets loaded!")
        return None
    
    # =========== PHASE 2: DATA INTEGRATION ===========
    print("\n[PHASE 2] DATA INTEGRATION")
    print("-" * 50)
    
    integrator = DataIntegrator(datasets)
    integrator.standardize_student_ids()
    integrator.combine_high_school_exams()
    integrator.create_semester_records()
    master_df = integrator.create_master_table()
    semester_records = integrator.semester_records
    
    if master_df is None or len(master_df) == 0:
        print("ERROR: Failed to create master table!")
        return None
    
    # =========== PHASE 3: DATA CLEANING ===========
    print("\n[PHASE 3] DATA CLEANING")
    print("-" * 50)
    
    cleaner = DataCleaner(master_df, semester_records)
    master_df, semester_records = cleaner.clean_all()
    
    # =========== PHASE 4: SCORE STANDARDIZATION ===========
    print("\n[PHASE 4] SCORE STANDARDIZATION")
    print("-" * 50)
    
    standardizer = ScoreStandardizer()
    master_df = standardizer.standardize_all_scores(master_df)
    master_df = standardizer.create_performance_tiers(master_df)
    
    # =========== PHASE 5: TARGET CREATION (UPDATED) ===========
    print("\n[PHASE 5] TARGET VARIABLE CREATION")
    print("-" * 50)
    
    target_creator = TargetCreator(master_df, semester_records)
    master_df = target_creator.create_all_targets()
    
    # =========== PHASE 6: FEATURE ENGINEERING ===========
    print("\n[PHASE 6] FEATURE ENGINEERING")
    print("-" * 50)
    
    engineer = FeatureEngineer(master_df, semester_records)
    master_df = engineer.engineer_all_features()
    master_df = engineer.create_year_features(1)
    master_df = engineer.create_year_features(2)
    feature_sets = engineer.get_feature_sets()
    
    # Save processed data
    print("\n  Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    master_df.to_csv('data/processed/master_student_data.csv', index=False)
    if semester_records is not None:
        semester_records.to_csv('data/processed/semester_records.csv', index=False)
    
    # =========== PHASE 6B: UNSUPERVISED LEARNING (NEW) ===========
    print("\n[PHASE 6B] UNSUPERVISED LEARNING")
    print("-" * 50)
    
    unsupervised = UnsupervisedAnalyzer(master_df, feature_sets)
    unsupervised_results = unsupervised.run_full_analysis(feature_type='admission')
    
    if unsupervised_results:
        master_df = unsupervised_results['master_df_with_clusters']
        unsupervised.generate_cluster_report()
    
    # =========== PHASE 7: MODEL TRAINING (UPDATED) ===========
    print("\n[PHASE 7] MODEL TRAINING")
    print("-" * 50)
    
    trainer = ModelTrainer(master_df, feature_sets, semester_records)
    trainer.train_all_tasks(split_method=split_method)
    
    if trainer.best_models:
        print("\n  Model Performance Summary:")
        print(trainer.get_summary().to_string(index=False))
        
        print("\n  Saving models...")
        trainer.save_models()
    
    # =========== PHASE 8: STATISTICAL ANALYSIS ===========
    print("\n[PHASE 8] STATISTICAL ANALYSIS")
    print("-" * 50)
    
    stats_analyzer = StatisticalAnalyzer(master_df, semester_records)
    stats_results = stats_analyzer.run_all_analyses()
    
    # =========== PHASE 9: VISUALIZATION ===========
    print("\n[PHASE 9] VISUALIZATION")
    print("-" * 50)
    
    eda = EDAGenerator(master_df)
    eda.generate_all()
    
    # =========== PHASE 10: REPORT GENERATION ===========
    print("\n[PHASE 10] REPORT GENERATION")
    print("-" * 50)
    
    reporter = ReportGenerator(master_df, trainer.best_models, stats_results)
    reporter.generate_executive_summary()
    reporter.generate_model_report(trainer)
    reporter.generate_statistical_report()
    
    
    # =========== COMPLETION ===========
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\nOutputs generated:")
    print("  📁 data/processed/")
    print("     • master_student_data.csv")
    print("     • semester_records.csv")
    print("  📁 models/")
    print("     • *_model.joblib")
    print("     • *_scaler.joblib")
    print("     • *_features.json")
    print("  reports/")
    print("    ├── executive_summary.txt")
    print("    ├── model_report.txt")
    print("    ├── statistical_report.txt")
    print("    └── cluster_report.txt")
    print("  reports/figures/")
    print("    ├── target_distributions.png")
    print("    ├── performance_analysis.png")
    print("    ├── risk_factors.png")
    print("    ├── correlations.png")
    print("    ├── cluster_optimization.png")
    print("    ├── cluster_visualization_both.png")
    print("    ├── cluster_feature_importance.png")
    print("    └── cluster_outcomes.png")
    
    print("\n📊 Analysis Summary:")
    print(f"  • Total students analyzed: {len(master_df):,}")
    print(f"  • Graduated: {master_df['is_graduated'].sum():,}")
    print(f"  • Active: {master_df['is_active'].sum():,}")
    print(f"  • Models trained: {len(trainer.best_models)}")
    print(f"  • Student clusters identified: {master_df['cluster_kmeans'].nunique() if 'cluster_kmeans' in master_df.columns else 'N/A'}")
    
    print("\n🚀 Next steps:")
    print("  1. Review reports in 'reports/' folder")
    print("  2. Launch dashboard: streamlit run dashboard/app.py")
    print("  3. Present findings to stakeholders")
    
    return {
        'master_df': master_df,
        'semester_records': semester_records,
        'feature_sets': feature_sets,
        'trainer': trainer,
        'stats_results': stats_results,
        'unsupervised_results': unsupervised_results
    }


def main():
    """Main entry point with command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Ashesi Student Success Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with temporal split (recommended)
  python run_pipeline.py --mode full --split temporal
  
  # Run full pipeline with random split
  python run_pipeline.py --mode full --split random
  
  # Run only data processing
  python run_pipeline.py --mode data
  
  # Run only unsupervised learning
  python run_pipeline.py --mode unsupervised
  
  # Launch dashboard
  python run_pipeline.py --mode dashboard
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'data', 'train', 'unsupervised', 'analysis', 'eda', 'report', 'dashboard'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--split',
        choices=['temporal', 'random'],
        default='temporal',
        help='Data split method for training (default: temporal)'
    )
    
    parser.add_argument(
        '--data-path',
        default='data/raw/',
        help='Path to raw data directory (default: data/raw/)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters for unsupervised learning (default: auto-detect)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Dashboard mode
    if args.mode == 'dashboard':
        print("\n🚀 Launching Streamlit Dashboard...")
        print("   Access at: http://localhost:8501")
        print("   Press Ctrl+C to stop\n")
        os.system('streamlit run dashboard/app.py')
        return
    
    # Full pipeline
    if args.mode == 'full':
        results = run_full_pipeline(
            data_path=args.data_path,
            split_method=args.split,
            verbose=args.verbose
        )
        return results
    
    # Partial pipeline modes
    if args.mode == 'data':
        print("\n[DATA PROCESSING MODE]")
        print("=" * 50)
        
        loader = DataLoader(args.data_path)
        datasets = loader.load_all_datasets()
        
        integrator = DataIntegrator(datasets)
        integrator.standardize_student_ids()
        integrator.combine_high_school_exams()
        integrator.create_semester_records()
        master_df = integrator.create_master_table()
        
        cleaner = DataCleaner(master_df, integrator.semester_records)
        master_df, semester_records = cleaner.clean_all()
        
        standardizer = ScoreStandardizer()
        master_df = standardizer.standardize_all_scores(master_df)
        master_df = standardizer.create_performance_tiers(master_df)
        
        target_creator = TargetCreator(master_df, semester_records)
        master_df = target_creator.create_all_targets()
        
        engineer = FeatureEngineer(master_df, semester_records)
        master_df = engineer.engineer_all_features()
        master_df = engineer.create_year_features(1)
        master_df = engineer.create_year_features(2)
        
        os.makedirs('data/processed', exist_ok=True)
        master_df.to_csv('data/processed/master_student_data.csv', index=False)
        semester_records.to_csv('data/processed/semester_records.csv', index=False)
        
        print("\n✓ Data processing complete")
        print(f"  Saved to data/processed/")
        print(f"  Total students: {len(master_df):,}")
        
    elif args.mode == 'unsupervised':
        print("\n[UNSUPERVISED LEARNING MODE]")
        print("=" * 50)
        
        try:
            master_df = pd.read_csv('data/processed/master_student_data.csv')
            print(f"  Loaded {len(master_df):,} students")
            
            engineer = FeatureEngineer(master_df, None)
            feature_sets = engineer.get_feature_sets()
            
            unsupervised = UnsupervisedAnalyzer(master_df, feature_sets)
            results = unsupervised.run_full_analysis(
                feature_type='admission',
                n_clusters=args.n_clusters
            )
            
            if results:
                # Save updated master_df with cluster labels
                results['master_df_with_clusters'].to_csv(
                    'data/processed/master_student_data.csv', index=False
                )
                unsupervised.generate_cluster_report()
            
            print("\n✓ Unsupervised learning complete")
            
        except FileNotFoundError:
            print("ERROR: Processed data not found. Run with --mode data first.")
    
    elif args.mode == 'train':
        print("\n[MODEL TRAINING MODE]")
        print("=" * 50)
        
        try:
            master_df = pd.read_csv('data/processed/master_student_data.csv')
            semester_records = pd.read_csv('data/processed/semester_records.csv')
            print(f"  Loaded {len(master_df):,} students")
            
            engineer = FeatureEngineer(master_df, semester_records)
            feature_sets = engineer.get_feature_sets()
            
            trainer = ModelTrainer(master_df, feature_sets, semester_records)
            trainer.train_all_tasks(split_method=args.split)
            trainer.save_models()
            
            print("\n✓ Model training complete")
            print("\nModel Performance Summary:")
            print(trainer.get_summary().to_string(index=False))
            
        except FileNotFoundError:
            print("ERROR: Processed data not found. Run with --mode data first.")
    
    elif args.mode == 'analysis':
        print("\n[STATISTICAL ANALYSIS MODE]")
        print("=" * 50)
        
        try:
            master_df = pd.read_csv('data/processed/master_student_data.csv')
            semester_records = pd.read_csv('data/processed/semester_records.csv')
            
            analyzer = StatisticalAnalyzer(master_df, semester_records)
            analyzer.run_all_analyses()
            
            print("\n✓ Statistical analysis complete")
            
        except FileNotFoundError:
            print("ERROR: Processed data not found. Run with --mode data first.")
    
    elif args.mode == 'eda':
        print("\n[EDA VISUALIZATION MODE]")
        print("=" * 50)
        
        try:
            master_df = pd.read_csv('data/processed/master_student_data.csv')
            
            eda = EDAGenerator(master_df)
            eda.generate_all()
            
            print("\n✓ EDA visualizations generated")
            
        except FileNotFoundError:
            print("ERROR: Processed data not found. Run with --mode data first.")
    
    elif args.mode == 'report':
        print("\n[REPORT GENERATION MODE]")
        print("=" * 50)
        
        try:
            master_df = pd.read_csv('data/processed/master_student_data.csv')
            
            # Try to load model results
            model_results = {}
            if os.path.exists('models/'):
                import joblib
                for f in os.listdir('models/'):
                    if f.endswith('_model.joblib'):
                        task_key = f.replace('_model.joblib', '')
                        model_results[task_key] = {
                            'name': 'Loaded',
                            'metrics': {}
                        }
            
            reporter = ReportGenerator(master_df, model_results, {})
            reporter.generate_executive_summary()
            
            print("\n✓ Reports generated")
            
        except FileNotFoundError:
            print("ERROR: Processed data not found. Run with --mode data first.")


if __name__ == "__main__":
    main()
