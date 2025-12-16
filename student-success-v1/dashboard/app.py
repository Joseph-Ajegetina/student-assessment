# dashboard/app.py
# Comprehensive Streamlit Dashboard for Ashesi Student Success Prediction
# Integrated from existing project with ethics section

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Get the project root
DASHBOARD_DIR = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'

# Page configuration
st.set_page_config(
    page_title="Ashesi Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .insight-box {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef5e7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
    .ethics-section {
        background-color: #e8f4f8;
        border-left: 4px solid #1f4e79;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_master_data():
    """Load master student data"""
    try:
        path = PROCESSED_DIR / 'master_student_data.csv'
        if path.exists():
            return pd.read_csv(path)
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_semester_data():
    """Load semester records"""
    try:
        path = PROCESSED_DIR / 'semester_records.csv'
        if path.exists():
            return pd.read_csv(path)
        return None
    except:
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    if not MODELS_DIR.exists():
        return models
    try:
        import joblib
        for model_file in MODELS_DIR.glob('*_model.joblib'):
            task_key = model_file.stem.replace('_model', '')
            try:
                model_data = {'model': joblib.load(model_file)}
                features_path = MODELS_DIR / f'{task_key}_features.json'
                if features_path.exists():
                    with open(features_path, 'r') as f:
                        model_data['features'] = json.load(f)
                models[task_key] = model_data
            except:
                pass
    except ImportError:
        pass
    return models


# ============================================================================
# DASHBOARD CLASS
# ============================================================================

class DashboardApp:
    """Streamlit dashboard for student success prediction"""

    def __init__(self):
        self.master_df = load_master_data()
        self.semester_records = load_semester_data()
        self.models = load_models()
        self._show_loading_status()

    def _show_loading_status(self):
        """Display data loading status in sidebar"""
        st.sidebar.markdown("### 📊 Data Status")
        if self.master_df is not None:
            st.sidebar.success(f"✓ {len(self.master_df):,} students loaded")
            if 'is_graduated' in self.master_df.columns:
                graduated = int(self.master_df['is_graduated'].sum())
                st.sidebar.caption(f"  📗 Graduated: {graduated:,}")
            if 'is_active' in self.master_df.columns:
                active = int(self.master_df['is_active'].sum())
                st.sidebar.caption(f"  📘 Active: {active:,}")
        else:
            st.sidebar.error("❌ Data not loaded")
        if self.semester_records is not None:
            st.sidebar.info(f"✓ {len(self.semester_records):,} semester records")
        if self.models:
            st.sidebar.info(f"✓ {len(self.models)} models loaded")

    def run(self):
        """Run the dashboard"""
        st.sidebar.title("🎓 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            [
                "📊 Overview",
                "🔍 Student Lookup",
                "⚠️ Risk Analysis",
                "🎯 Cluster Analysis",
                "📅 Cohort Analysis",
                "🎓 Major Analysis",
                "📐 Math Track Analysis",
                "🔬 Research Questions",
                "🤖 Model Performance",
                "🔮 Predictions",
                "⚖️ Ethics & Fairness"
            ]
        )

        if self.master_df is None or len(self.master_df) == 0:
            st.error("""
            ## ❌ No Data Loaded

            Please run the EDA notebook first to generate processed data:
            ```bash
            cd notebooks && jupyter notebook 01_data_exploration.ipynb
            ```
            Or run the data loader directly in Python.
            """)
            return

        # Route pages
        if "Overview" in page:
            self.show_overview()
        elif "Student Lookup" in page:
            self.show_student_lookup()
        elif "Risk Analysis" in page:
            self.show_risk_analysis()
        elif "Cluster Analysis" in page:
            self.show_cluster_analysis()
        elif "Cohort Analysis" in page:
            self.show_cohort_analysis()
        elif "Major Analysis" in page:
            self.show_major_analysis()
        elif "Math Track" in page:
            self.show_math_track_analysis()
        elif "Research Questions" in page:
            self.show_research_questions()
        elif "Model Performance" in page:
            self.show_model_performance()
        elif "Predictions" in page:
            self.show_predictions()
        elif "Ethics" in page:
            self.show_ethics()

    def get_program_column(self):
        for col in ['final_program', 'Program', 'Offer course name']:
            if col in self.master_df.columns:
                return col
        return None

    def get_yeargroup_column(self):
        for col in ['Yeargroup', 'yeargroup', 'Year Group']:
            if col in self.master_df.columns:
                return col
        return None

    # ========================================================================
    # OVERVIEW PAGE
    # ========================================================================
    def show_overview(self):
        st.markdown('<h1 class="main-header">🎓 Ashesi Student Success Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Students", f"{len(self.master_df):,}")
        with col2:
            if 'is_graduated' in self.master_df.columns:
                st.metric("Graduated", f"{int(self.master_df['is_graduated'].sum()):,}")
            else:
                st.metric("Graduated", "N/A")
        with col3:
            if 'is_active' in self.master_df.columns:
                st.metric("Active", f"{int(self.master_df['is_active'].sum()):,}")
            else:
                st.metric("Active", "N/A")
        with col4:
            if 'first_year_struggle' in self.master_df.columns:
                valid = self.master_df['first_year_struggle'].notna()
                if valid.sum() > 0:
                    rate = self.master_df.loc[valid, 'first_year_struggle'].mean() * 100
                    st.metric("Struggle Rate", f"{rate:.1f}%")
                else:
                    st.metric("Struggle Rate", "N/A")
        with col5:
            if 'major_success' in self.master_df.columns:
                valid = self.master_df['major_success'].notna()
                if valid.sum() > 0:
                    rate = self.master_df.loc[valid, 'major_success'].mean() * 100
                    st.metric("Success Rate", f"{rate:.1f}%")
                else:
                    st.metric("Success Rate", "N/A")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👥 Student Status Distribution")
            status_counts = {}
            for status in ['is_graduated', 'is_active', 'is_withdrawn', 'is_dismissed']:
                if status in self.master_df.columns:
                    status_counts[status.replace('is_', '').title()] = int(self.master_df[status].sum())
            if status_counts:
                fig = px.pie(values=list(status_counts.values()), names=list(status_counts.keys()),
                            color_discrete_sequence=['#27ae60', '#3498db', '#f39c12', '#e74c3c'], hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 CGPA Distribution")
            if 'final_cgpa' in self.master_df.columns:
                cgpa_data = self.master_df['final_cgpa'].dropna()
                if len(cgpa_data) > 0:
                    fig = px.histogram(cgpa_data, nbins=30, color_discrete_sequence=['#3498db'])
                    fig.add_vline(x=2.0, line_dash="dash", line_color="red", annotation_text="Probation")
                    fig.add_vline(x=3.0, line_dash="dash", line_color="orange", annotation_text="Success")
                    fig.add_vline(x=3.5, line_dash="dash", line_color="green", annotation_text="Dean's List")
                    fig.update_layout(xaxis_title="Final CGPA", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

        # Math Track Overview
        if 'math_track' in self.master_df.columns:
            st.markdown("---")
            st.subheader("📐 Math Track Distribution")
            track_counts = self.master_df['math_track'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=track_counts.values, names=track_counts.index,
                            color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Track Summary:**")
                for track, count in track_counts.items():
                    st.write(f"- **{track}**: {count:,} ({count/len(self.master_df)*100:.1f}%)")

        # Academic Policy Reference
        st.markdown("---")
        st.subheader("📋 Academic Policy Reference")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="warning-box"><b>⚠️ Probation</b><br>CGPA &lt; 2.0</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="warning-box"><b>🚫 Dismissal</b><br>2 consecutive semesters on probation</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="insight-box"><b>⭐ Dean\'s List</b><br>Semester GPA ≥ 3.5</div>', unsafe_allow_html=True)

    # ========================================================================
    # STUDENT LOOKUP PAGE
    # ========================================================================
    def show_student_lookup(self):
        st.header("🔍 Student Lookup")
        search_method = st.radio("Search by:", ["Student ID", "Year Group"], horizontal=True)
        student = None

        if search_method == "Student ID":
            student_id = st.text_input("Enter Student ID:")
            if student_id:
                matches = self.master_df[self.master_df['student_id'].astype(str).str.upper().str.contains(student_id.upper(), na=False)]
                if len(matches) == 0:
                    st.warning(f"No student found matching '{student_id}'")
                elif len(matches) == 1:
                    student = matches.iloc[0]
                else:
                    selected_id = st.selectbox(f"Found {len(matches)} matches:", matches['student_id'].tolist())
                    student = matches[matches['student_id'] == selected_id].iloc[0]
        else:
            yg_col = self.get_yeargroup_column()
            if yg_col:
                years = sorted(self.master_df[yg_col].dropna().unique())
                selected_year = st.selectbox("Year Group:", years)
                year_students = self.master_df[self.master_df[yg_col] == selected_year]
                selected_id = st.selectbox(f"Students in {selected_year}:", year_students['student_id'].tolist())
                student = year_students[year_students['student_id'] == selected_id].iloc[0]

        if student is not None:
            st.markdown("---")
            status = "🟢 Graduated" if student.get('is_graduated', False) else "🔵 Active" if student.get('is_active', False) else "⚪ Unknown"
            st.subheader(f"📋 {student['student_id']} {status}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Demographics**")
                st.write(f"Gender: {student.get('Gender', 'N/A')}")
                st.write(f"International: {'Yes' if student.get('is_international', 0) else 'No'}")
            with col2:
                st.markdown("**Academic**")
                st.write(f"Program: {student.get(self.get_program_column(), 'N/A') if self.get_program_column() else 'N/A'}")
                st.write(f"Math Track: {student.get('math_track', 'N/A')}")
            with col3:
                st.markdown("**Performance**")
                if pd.notna(student.get('final_cgpa')):
                    st.write(f"Final CGPA: {student['final_cgpa']:.2f}")
                if pd.notna(student.get('total_semesters')):
                    st.write(f"Semesters: {int(student['total_semesters'])}")

            # Risk Assessment
            st.markdown("---")
            st.subheader("⚠️ Risk Assessment")
            col1, col2, col3, col4 = st.columns(4)
            for (risk_col, label), col in zip([('first_year_struggle', 'Y1 Struggle'), ('has_ajc_case', 'AJC Case'),
                                                ('major_success', 'Major Success'), ('extended_graduation', 'Extended Grad')],
                                               [col1, col2, col3, col4]):
                with col:
                    st.markdown(f"**{label}**")
                    if risk_col in student.index and pd.notna(student[risk_col]):
                        st.markdown("🔴 **Yes**" if int(student[risk_col]) == 1 else "🟢 No")
                    else:
                        st.markdown("⚪ N/A")

    # ========================================================================
    # RISK ANALYSIS PAGE
    # ========================================================================
    def show_risk_analysis(self):
        st.header("⚠️ Risk Analysis")
        risk_options = {"First Year Struggle": "first_year_struggle", "AJC Case": "has_ajc_case",
                       "Major Success": "major_success", "Extended Graduation": "extended_graduation"}
        available_risks = {k: v for k, v in risk_options.items() if v in self.master_df.columns}

        if not available_risks:
            st.warning("No risk indicators available")
            return

        selected_risk = st.selectbox("Risk Type", list(available_risks.keys()))
        risk_col = available_risks[selected_risk]
        filtered_df = self.master_df[self.master_df[risk_col].notna()].copy()

        if len(filtered_df) == 0:
            st.warning("No valid data")
            return

        col1, col2, col3, col4 = st.columns(4)
        total = len(filtered_df)
        at_risk = int(filtered_df[risk_col].sum())
        with col1:
            st.metric("Valid Students", f"{total:,}")
        with col2:
            st.metric("At Risk", f"{at_risk:,}")
        with col3:
            st.metric("Risk Rate", f"{at_risk/total*100:.1f}%")
        with col4:
            st.metric("Not At Risk", f"{total - at_risk:,}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if 'math_track' in filtered_df.columns:
                st.subheader("Risk by Math Track")
                valid_tracks = filtered_df[filtered_df['math_track'] != 'Unknown']
                if len(valid_tracks) > 0:
                    risk_by_track = valid_tracks.groupby('math_track')[risk_col].mean() * 100
                    fig = px.bar(x=risk_by_track.index, y=risk_by_track.values, color=risk_by_track.values,
                                color_continuous_scale='RdYlGn_r', labels={'x': 'Math Track', 'y': 'Risk Rate (%)'})
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'Gender' in filtered_df.columns:
                st.subheader("Risk by Gender")
                risk_by_gender = filtered_df.groupby('Gender')[risk_col].mean() * 100
                fig = px.bar(x=risk_by_gender.index, y=risk_by_gender.values, color=risk_by_gender.values,
                            color_continuous_scale='RdYlGn_r', labels={'x': 'Gender', 'y': 'Risk Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # CLUSTER ANALYSIS PAGE
    # ========================================================================
    def show_cluster_analysis(self):
        st.header("🎯 Cluster Analysis")
        cluster_cols = [col for col in self.master_df.columns if col.startswith('cluster_')]
        if not cluster_cols:
            st.warning("⚠️ No cluster data available. Run unsupervised learning notebook first.")
            return
        cluster_col = st.selectbox("Clustering Method", cluster_cols)
        n_clusters = self.master_df[cluster_col].nunique()
        st.info(f"📊 **{n_clusters} clusters** identified")

        agg_dict = {'student_id': 'count'}
        for col in ['final_cgpa', 'first_year_struggle', 'has_ajc_case']:
            if col in self.master_df.columns:
                agg_dict[col] = 'mean'
        summary = self.master_df.groupby(cluster_col).agg(agg_dict).round(3)
        summary = summary.rename(columns={'student_id': 'Count'})
        st.dataframe(summary, use_container_width=True)

    # ========================================================================
    # COHORT ANALYSIS PAGE
    # ========================================================================
    def show_cohort_analysis(self):
        st.header("📅 Cohort Analysis")
        yg_col = self.get_yeargroup_column()
        if not yg_col:
            st.warning("Year group data not available")
            return
        cohorts = sorted(self.master_df[yg_col].dropna().unique())
        selected_cohorts = st.multiselect("Select Cohorts", cohorts, default=cohorts[-3:] if len(cohorts) >= 3 else cohorts)
        if not selected_cohorts:
            return
        cohort_df = self.master_df[self.master_df[yg_col].isin(selected_cohorts)]
        agg_dict = {'student_id': 'count'}
        for col in ['final_cgpa', 'first_year_struggle', 'is_graduated']:
            if col in cohort_df.columns:
                agg_dict[col] = 'mean'
        summary = cohort_df.groupby(yg_col).agg(agg_dict).round(3).rename(columns={'student_id': 'Total'})
        st.dataframe(summary, use_container_width=True)

    # ========================================================================
    # MAJOR ANALYSIS PAGE
    # ========================================================================
    def show_major_analysis(self):
        st.header("🎓 Major Analysis")
        program_col = self.get_program_column()
        if not program_col:
            st.warning("Program data not available")
            return
        major_counts = self.master_df[program_col].value_counts()
        valid_majors = major_counts[major_counts >= 10].index.tolist()
        if not valid_majors:
            st.warning("No majors with sufficient data")
            return
        selected_major = st.selectbox("Select Major", valid_majors)
        major_df = self.master_df[self.master_df[program_col] == selected_major]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students", len(major_df))
        with col2:
            if 'final_cgpa' in major_df.columns:
                st.metric("Avg CGPA", f"{major_df['final_cgpa'].mean():.2f}")
        with col3:
            if 'major_success' in major_df.columns:
                valid = major_df['major_success'].notna()
                if valid.sum() > 0:
                    st.metric("Success Rate", f"{major_df.loc[valid, 'major_success'].mean()*100:.1f}%")

    # ========================================================================
    # MATH TRACK ANALYSIS PAGE (RQ7 & RQ8)
    # ========================================================================
    def show_math_track_analysis(self):
        st.header("📐 Math Track Analysis")
        st.markdown("**RQ7**: Performance difference across math tracks? **RQ8**: Can College Algebra students succeed in CS?")

        if 'math_track' not in self.master_df.columns:
            st.warning("Math track data not available")
            return

        df = self.master_df[(self.master_df['math_track'].notna()) & (self.master_df['math_track'] != 'Unknown')]
        if len(df) < 30:
            st.warning("Insufficient data")
            return

        if 'final_cgpa' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(df, x='math_track', y='final_cgpa', color='math_track', title="CGPA by Math Track")
                fig.add_hline(y=2.0, line_dash="dash", line_color="red")
                fig.add_hline(y=3.0, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                summary = df.groupby('math_track')['final_cgpa'].agg(['count', 'mean', 'std']).round(2)
                st.dataframe(summary, use_container_width=True)

        # RQ8: CS + College Algebra
        st.markdown("---")
        st.subheader("💻 RQ8: College Algebra in CS")
        program_col = self.get_program_column()
        if program_col:
            cs_df = self.master_df[self.master_df[program_col].astype(str).str.contains('Computer', case=False, na=False)]
            if len(cs_df) >= 10 and 'math_track' in cs_df.columns:
                cs_summary = cs_df.groupby('math_track').agg({'student_id': 'count', 'final_cgpa': 'mean'}).round(2)
                cs_summary.columns = ['Count', 'Avg CGPA']
                st.dataframe(cs_summary, use_container_width=True)

    # ========================================================================
    # RESEARCH QUESTIONS PAGE
    # ========================================================================
    def show_research_questions(self):
        st.header("🔬 Research Questions Dashboard")
        tabs = st.tabs(["RQ1-2", "RQ3-6", "RQ7", "RQ8", "RQ9"])

        with tabs[0]:
            st.subheader("RQ1: First-Year Struggle | RQ2: AJC Cases")
            col1, col2 = st.columns(2)
            with col1:
                if 'first_year_struggle' in self.master_df.columns:
                    valid = self.master_df['first_year_struggle'].notna()
                    rate = self.master_df.loc[valid, 'first_year_struggle'].mean() * 100 if valid.sum() > 0 else 0
                    st.metric("First-Year Struggle Rate", f"{rate:.1f}%")
            with col2:
                if 'has_ajc_case' in self.master_df.columns:
                    rate = self.master_df['has_ajc_case'].mean() * 100
                    st.metric("AJC Case Rate", f"{rate:.2f}%")
            st.markdown('<div class="insight-box"><b>Key Finding:</b> HS math scores are the strongest predictors of first-year success.</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.subheader("RQ3-6: Major Success/Failure Prediction")
            st.markdown("Adding Year 1 academic data significantly improves prediction accuracy.")

        with tabs[2]:
            st.subheader("RQ7: Math Track Performance")
            if 'math_track' in self.master_df.columns and 'final_cgpa' in self.master_df.columns:
                track_stats = self.master_df.groupby('math_track')['final_cgpa'].agg(['mean', 'std', 'count']).round(2)
                st.dataframe(track_stats, use_container_width=True)

        with tabs[3]:
            st.subheader("RQ8: College Algebra in CS")
            st.markdown('<div class="warning-box"><b>Finding:</b> College algebra students CAN succeed in CS but may need additional math support.</div>', unsafe_allow_html=True)

        with tabs[4]:
            st.subheader("RQ9: Extended Graduation")
            if 'extended_graduation' in self.master_df.columns:
                valid = self.master_df['extended_graduation'].notna()
                rate = self.master_df.loc[valid, 'extended_graduation'].mean() * 100 if valid.sum() > 0 else 0
                st.metric("Extended Graduation Rate", f"{rate:.1f}%")

    # ========================================================================
    # MODEL PERFORMANCE PAGE
    # ========================================================================
    def show_model_performance(self):
        st.header("🤖 Model Performance")
        if not self.models:
            st.warning("No models loaded. Run supervised learning notebooks first.")
            return
        st.success(f"✓ {len(self.models)} models loaded")
        model_summary = [{'Task': k, 'Model': type(v['model']).__name__, 'Features': len(v.get('features', []))} for k, v in self.models.items()]
        st.dataframe(pd.DataFrame(model_summary), use_container_width=True, hide_index=True)

    # ========================================================================
    # PREDICTIONS PAGE
    # ========================================================================
    def show_predictions(self):
        st.header("🔮 Risk Prediction")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                hs_score = st.slider("HS Score (0-100)", 0, 100, 70)
                math_score = st.slider("Math Score", 0, 100, 70)
            with col2:
                math_track = st.selectbox("Math Track", ["Calculus", "Pre-Calculus", "College Algebra"])
                major = st.selectbox("Intended Major", ["CS", "Business", "Engineering", "Other"])
            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            base_risk = (100 - hs_score) / 100 * 0.4 + (100 - math_score) / 100 * 0.4
            if math_track == "College Algebra":
                base_risk += 0.15
            if major in ["CS", "Engineering"] and math_track == "College Algebra":
                base_risk += 0.1
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Struggle Risk", f"{min(base_risk * 100, 95):.0f}%")
            with col2:
                st.metric("Extended Grad Risk", f"{min(base_risk * 70, 85):.0f}%")

    # ========================================================================
    # ETHICS PAGE
    # ========================================================================
    def show_ethics(self):
        st.header("⚖️ Ethics & Fairness Analysis")
        st.markdown("""
        This page examines the ethical implications of predictive models for student success.
        """)

        st.subheader("🎯 Core Principles")
        st.markdown("""
        <div class="ethics-section">
        <ol>
        <li><b>Beneficence</b>: Help students succeed, don't punish them</li>
        <li><b>Non-maleficence</b>: Avoid self-fulfilling prophecies</li>
        <li><b>Autonomy</b>: Students have agency over their journey</li>
        <li><b>Justice</b>: Don't amplify existing inequalities</li>
        <li><b>Transparency</b>: Explain how predictions are made</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Fairness Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Gender Parity")
            if 'Gender' in self.master_df.columns and 'first_year_struggle' in self.master_df.columns:
                gender_stats = self.master_df.groupby('Gender').agg({
                    'student_id': 'count', 'first_year_struggle': 'mean', 'final_cgpa': 'mean'
                }).round(3)
                gender_stats.columns = ['Count', 'Struggle Rate', 'Avg CGPA']
                st.dataframe(gender_stats, use_container_width=True)

        with col2:
            st.markdown("#### Math Track Equity")
            if 'math_track' in self.master_df.columns and 'first_year_struggle' in self.master_df.columns:
                track_stats = self.master_df[self.master_df['math_track'] != 'Unknown'].groupby('math_track').agg({
                    'student_id': 'count', 'first_year_struggle': 'mean'
                }).round(3)
                track_stats.columns = ['Count', 'Struggle Rate']
                st.dataframe(track_stats, use_container_width=True)

        st.markdown("---")
        st.subheader("⚠️ Potential Harms & Mitigations")
        harms = [
            ("Self-fulfilling Prophecy", "Students labeled 'at-risk' may internalize this", "Frame as support opportunity"),
            ("Stigmatization", "Different treatment based on predictions", "Limit access; train advisors"),
            ("Bias Perpetuation", "Historical data may reflect past discrimination", "Regular fairness audits"),
            ("Privacy Concerns", "Sensitive student data used", "Data minimization; strong access controls")
        ]
        for harm, desc, mitigation in harms:
            with st.expander(f"🔴 {harm}"):
                st.write(f"**Risk**: {desc}")
                st.write(f"**Mitigation**: {mitigation}")

        st.markdown("---")
        st.subheader("✅ Responsible Use Guidelines")
        st.markdown("""
        <div class="ethics-section">
        <h4>DO:</h4>
        <ul>
        <li>Use predictions to <b>offer support</b>, not restrict opportunities</li>
        <li>Combine model predictions with human judgment</li>
        <li>Regularly audit for fairness across groups</li>
        </ul>
        <h4>DON'T:</h4>
        <ul>
        <li>Use for admissions or scholarship decisions</li>
        <li>Share individual risk scores without context</li>
        <li>Assume the model is always correct</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main()
