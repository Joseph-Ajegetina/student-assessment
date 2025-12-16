# dashboard/app.py
# Comprehensive Streamlit Dashboard for Ashesi Student Success Prediction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
RESULTS_DIR = PROJECT_ROOT / 'results'
REPORTS_DIR = RESULTS_DIR / 'reports'
FIGURES_DIR = RESULTS_DIR / 'figures'

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
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .insight-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .ethics-section {
        background-color: #e8f4f8;
        border-left: 4px solid #1f4e79;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .improvement-positive {
        color: #27ae60;
        font-weight: bold;
    }
    .improvement-negative {
        color: #e74c3c;
        font-weight: bold;
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
def load_full_features():
    """Load full features data"""
    try:
        path = PROCESSED_DIR / 'full_features.csv'
        if path.exists():
            return pd.read_csv(path)
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def load_model_results():
    """Load all model results from CSV files"""
    results = {}
    if REPORTS_DIR.exists():
        for csv_file in REPORTS_DIR.glob('*.csv'):
            try:
                results[csv_file.stem] = pd.read_csv(csv_file)
            except:
                pass
    return results

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
        self.full_features = load_full_features()
        self.model_results = load_model_results()
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

        if self.model_results:
            st.sidebar.info(f"✓ {len(self.model_results)} result files loaded")
        if self.models:
            st.sidebar.info(f"✓ {len(self.models)} models loaded")

    def run(self):
        """Run the dashboard"""
        st.sidebar.title("🎓 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            [
                "📊 Overview",
                "📈 Model Results",
                "🔬 Research Questions",
                "📐 Math Track Analysis",
                "🔍 Student Lookup",
                "⚠️ Risk Analysis",
                "🎯 Cluster Analysis",
                "📅 Cohort Analysis",
                "🎓 Major Analysis",
                "🔮 Predictions",
                "⚖️ Ethics & Fairness"
            ]
        )

        if self.master_df is None or len(self.master_df) == 0:
            st.error("""
            ## ❌ No Data Loaded

            Please run the notebooks first to generate processed data:
            ```bash
            cd notebooks && jupyter notebook 03_feature_engineering.ipynb
            ```
            """)
            return

        # Route pages
        if "Overview" in page:
            self.show_overview()
        elif "Model Results" in page:
            self.show_model_results()
        elif "Research Questions" in page:
            self.show_research_questions()
        elif "Math Track" in page:
            self.show_math_track_analysis()
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

        st.markdown("""
        <div class="info-box">
        <b>Project Overview:</b> This dashboard presents machine learning models for predicting student success at Ashesi University.
        The models use admissions data, high school performance, and early academic indicators to identify at-risk students for early intervention.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Students", f"{len(self.master_df):,}")
        with col2:
            if 'is_graduated' in self.master_df.columns:
                st.metric("Graduated", f"{int(self.master_df['is_graduated'].sum()):,}")
        with col3:
            if 'is_active' in self.master_df.columns:
                st.metric("Active", f"{int(self.master_df['is_active'].sum()):,}")
        with col4:
            if 'first_year_struggle' in self.master_df.columns:
                valid = self.master_df['first_year_struggle'].notna()
                if valid.sum() > 0:
                    rate = self.master_df.loc[valid, 'first_year_struggle'].mean() * 100
                    st.metric("Struggle Rate", f"{rate:.1f}%")
        with col5:
            if 'major_success' in self.master_df.columns:
                valid = self.master_df['major_success'].notna()
                if valid.sum() > 0:
                    rate = self.master_df.loc[valid, 'major_success'].mean() * 100
                    st.metric("Success Rate", f"{rate:.1f}%")

        st.markdown("---")

        # Research Questions Summary
        st.markdown('<p class="section-header">🔬 Research Questions</p>', unsafe_allow_html=True)

        rq_data = [
            ("RQ1", "First-Year Struggle Prediction", "Predict academic struggle from admissions data"),
            ("RQ2", "AJC Case Prediction", "Predict academic integrity violations"),
            ("RQ3", "Major Success (Adm + Y1)", "Predict success using admissions + Year 1 data"),
            ("RQ4", "Probation (Adm + Y1)", "Predict probation using admissions + Year 1 data"),
            ("RQ5", "Major Success (Adm + Y1 + Y2)", "Predict success with Year 2 data added"),
            ("RQ6", "Probation (Adm + Y1 + Y2)", "Predict probation with Year 2 data added"),
            ("RQ7", "Math Track Comparison", "Statistical comparison of math track outcomes"),
            ("RQ8", "College Algebra in CS", "Can College Algebra students succeed in CS?"),
            ("RQ9", "Extended Graduation", "Predict >4 year graduation time"),
        ]

        cols = st.columns(3)
        for i, (rq, title, desc) in enumerate(rq_data):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                <b>{rq}: {title}</b><br>
                <small>{desc}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Visualizations
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
                fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 CGPA Distribution")
            if 'final_cgpa' in self.master_df.columns:
                cgpa_data = self.master_df['final_cgpa'].dropna()
                if len(cgpa_data) > 0:
                    fig = px.histogram(cgpa_data, nbins=30, color_discrete_sequence=['#3498db'])
                    fig.add_vline(x=2.0, line_dash="dash", line_color="red", annotation_text="Probation (2.0)")
                    fig.add_vline(x=3.0, line_dash="dash", line_color="green", annotation_text="Success (3.0)")
                    fig.update_layout(xaxis_title="Final CGPA", yaxis_title="Count", margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # MODEL RESULTS PAGE
    # ========================================================================
    def show_model_results(self):
        st.header("📈 Model Performance Results")

        st.markdown("""
        <div class="info-box">
        <b>Evaluation Method:</b> Temporal Split - Models trained on older cohorts, tested on recent cohorts.
        This simulates real-world deployment where we predict outcomes for new students.
        </div>
        """, unsafe_allow_html=True)

        if not self.model_results:
            st.warning("No model results found. Run the supervised learning notebooks first.")
            return

        # Tabs for different result sets
        result_keys = list(self.model_results.keys())

        if 'rq3_rq5_results' in result_keys:
            st.markdown('<p class="section-header">📊 Major Success Prediction (RQ3 & RQ5)</p>', unsafe_allow_html=True)
            self._show_feature_comparison('rq3_rq5_results', 'Major Success')

        if 'rq4_rq6_results' in result_keys:
            st.markdown('<p class="section-header">📊 Probation Prediction (RQ4 & RQ6)</p>', unsafe_allow_html=True)
            self._show_feature_comparison('rq4_rq6_results', 'Probation')

        if 'rq1_results' in result_keys:
            st.markdown('<p class="section-header">📊 First-Year Struggle (RQ1)</p>', unsafe_allow_html=True)
            self._show_simple_results('rq1_results')

        if 'rq2_results' in result_keys:
            st.markdown('<p class="section-header">📊 AJC Case Prediction (RQ2)</p>', unsafe_allow_html=True)
            self._show_simple_results('rq2_results')

        if 'rq9_results' in result_keys:
            st.markdown('<p class="section-header">📊 Extended Graduation (RQ9)</p>', unsafe_allow_html=True)
            self._show_simple_results('rq9_results')

    def _show_feature_comparison(self, result_key, title):
        """Show comparison across feature sets"""
        df = self.model_results.get(result_key)
        if df is None or len(df) == 0:
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart comparing feature sets
            if 'Feature Set' in df.columns:
                fig = px.bar(df, x='Model', y='F1', color='Feature Set',
                            barmode='group', title=f'{title} - F1 Score by Feature Set',
                            color_discrete_sequence=['#e74c3c', '#f39c12', '#27ae60'])
                fig.update_layout(yaxis_range=[0, 1], legend=dict(orientation='h', y=1.15))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Best Results by Feature Set:**")
            if 'Feature Set' in df.columns:
                for fs in df['Feature Set'].unique():
                    fs_df = df[df['Feature Set'] == fs]
                    best = fs_df.loc[fs_df['F1'].idxmax()]
                    st.markdown(f"""
                    <div class="metric-card">
                    <b>{fs}</b><br>
                    Best: {best['Model']}<br>
                    F1: {best['F1']:.3f} | AUC: {best['AUC']:.3f}
                    </div>
                    """, unsafe_allow_html=True)

        # Show improvement table
        st.markdown("**Detailed Results:**")
        display_cols = ['Feature Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].round(3), use_container_width=True, hide_index=True)

        # Calculate improvements
        if 'Feature Set' in df.columns:
            st.markdown("**Improvement Analysis:**")
            feature_sets = df['Feature Set'].unique()
            if len(feature_sets) >= 2:
                improvements = []
                for model in df['Model'].unique():
                    model_df = df[df['Model'] == model]
                    if len(model_df) >= 2:
                        f1_by_fs = model_df.set_index('Feature Set')['F1']
                        for i in range(len(feature_sets) - 1):
                            if feature_sets[i] in f1_by_fs.index and feature_sets[i+1] in f1_by_fs.index:
                                imp = f1_by_fs[feature_sets[i+1]] - f1_by_fs[feature_sets[i]]
                                improvements.append({
                                    'Model': model,
                                    'From': feature_sets[i],
                                    'To': feature_sets[i+1],
                                    'F1 Improvement': f"{imp*100:+.1f}%"
                                })
                if improvements:
                    st.dataframe(pd.DataFrame(improvements), use_container_width=True, hide_index=True)

    def _show_simple_results(self, result_key):
        """Show simple results table"""
        df = self.model_results.get(result_key)
        if df is None or len(df) == 0:
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            if 'F1' in df.columns:
                fig = px.bar(df, x='Model', y='F1', color='F1',
                            color_continuous_scale='Greens',
                            title='F1 Score by Model')
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            best = df.loc[df['F1'].idxmax()] if 'F1' in df.columns else df.iloc[0]
            st.markdown(f"""
            <div class="model-card">
            <h3>🏆 Best Model</h3>
            <b>{best.get('Model', 'N/A')}</b><br>
            F1: {best.get('F1', 0):.3f}<br>
            AUC: {best.get('AUC', 0):.3f}<br>
            Accuracy: {best.get('Accuracy', 0):.3f}
            </div>
            """, unsafe_allow_html=True)

        display_cols = [c for c in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'] if c in df.columns]
        st.dataframe(df[display_cols].round(3), use_container_width=True, hide_index=True)

    # ========================================================================
    # RESEARCH QUESTIONS PAGE
    # ========================================================================
    def show_research_questions(self):
        st.header("🔬 Research Questions Analysis")

        tabs = st.tabs(["RQ1-2: Early Prediction", "RQ3-6: Feature Comparison", "RQ7-8: Math Tracks", "RQ9: Extended Graduation"])

        with tabs[0]:
            self._show_rq1_rq2()

        with tabs[1]:
            self._show_rq3_rq6()

        with tabs[2]:
            self._show_rq7_rq8()

        with tabs[3]:
            self._show_rq9()

    def _show_rq1_rq2(self):
        st.subheader("RQ1: First-Year Struggle Prediction")
        st.markdown("""
        **Question:** Can we predict first-year academic struggle from admissions data alone?

        **Target:** CGPA < 2.0 in Year 1 (Academic Probation)
        """)

        if 'first_year_struggle' in self.master_df.columns:
            valid = self.master_df['first_year_struggle'].notna()
            rate = self.master_df.loc[valid, 'first_year_struggle'].mean() * 100 if valid.sum() > 0 else 0
            col1, col2 = st.columns(2)
            with col1:
                st.metric("First-Year Struggle Rate", f"{rate:.1f}%")
            with col2:
                st.metric("Students Analyzed", f"{valid.sum():,}")

        # Show figure if exists
        fig_path = FIGURES_DIR / 'rq1_model_comparison.png'
        if fig_path.exists():
            st.image(str(fig_path), caption="RQ1 Model Comparison")

        st.markdown("---")
        st.subheader("RQ2: AJC Case Prediction")
        st.markdown("""
        **Question:** Can we predict Academic Judicial Committee (AJC) cases?

        **Target:** Student has an AJC case record
        """)

        if 'has_ajc_case' in self.master_df.columns:
            rate = self.master_df['has_ajc_case'].mean() * 100
            st.metric("AJC Case Rate", f"{rate:.2f}%")

        st.markdown("""
        <div class="warning-box">
        <b>Note:</b> AJC cases are rare events (~3-4%), making prediction challenging.
        The model prioritizes recall to identify at-risk students for intervention.
        </div>
        """, unsafe_allow_html=True)

    def _show_rq3_rq6(self):
        st.subheader("RQ3-6: Impact of Adding Academic Data")
        st.markdown("""
        **Key Question:** Does adding Year 1 and Year 2 academic data improve predictions?

        - **RQ3/RQ4:** Use Admissions + Year 1 data
        - **RQ5/RQ6:** Use Admissions + Year 1 + Year 2 data
        """)

        # Show comparison figures
        col1, col2 = st.columns(2)

        with col1:
            fig_path = FIGURES_DIR / 'rq3_rq5_comparison.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="Major Success Prediction Comparison")

        with col2:
            fig_path = FIGURES_DIR / 'rq4_rq6_comparison.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="Probation Prediction Comparison")

        st.markdown("""
        <div class="insight-box">
        <b>Key Findings:</b>
        <ul>
        <li>Adding Year 1 data improves F1 by ~20-25% over admissions alone</li>
        <li>Adding Year 2 data provides additional ~8-10% improvement</li>
        <li>Best model achieves ~94% F1 score with full data (Adm + Y1 + Y2)</li>
        <li>GPA trajectory and failure count are the strongest predictors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    def _show_rq7_rq8(self):
        st.subheader("RQ7: Math Track Performance Comparison")
        st.markdown("""
        **Question:** Is there a significant difference in academic outcomes between math tracks?

        **Tracks:** Calculus, Pre-Calculus, College Algebra
        """)

        fig_path = FIGURES_DIR / 'rq7_math_track_comparison.png'
        if fig_path.exists():
            st.image(str(fig_path), caption="Math Track Performance Comparison")

        if 'math_track' in self.master_df.columns and 'final_cgpa' in self.master_df.columns:
            df = self.master_df[self.master_df['math_track'].notna()]
            summary = df.groupby('math_track').agg({
                'final_cgpa': ['count', 'mean', 'std'],
                'first_year_struggle': 'mean'
            }).round(3)
            summary.columns = ['Count', 'Mean CGPA', 'Std CGPA', 'Struggle Rate']
            st.dataframe(summary, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>Statistical Finding:</b> ANOVA test shows SIGNIFICANT difference between math tracks (p < 0.001).
        Calculus track students have higher average CGPAs, but all tracks can achieve success.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("RQ8: College Algebra Students in CS")
        st.markdown("**Question:** Can College Algebra track students succeed in Computer Science?")

        fig_path = FIGURES_DIR / 'rq8_cs_math_track.png'
        if fig_path.exists():
            st.image(str(fig_path), caption="CS Success by Math Track")

        st.markdown("""
        <div class="warning-box">
        <b>Finding:</b> College Algebra students CAN succeed in CS (60%+ graduation rate),
        but may benefit from additional math support. Pre-Calculus track shows lower success rates,
        suggesting the issue is not just starting math level.
        </div>
        """, unsafe_allow_html=True)

    def _show_rq9(self):
        st.subheader("RQ9: Extended Graduation Prediction")
        st.markdown("""
        **Question:** Can we predict which students will take longer than 4 years to graduate?

        **Target:** Graduated in > 4 academic years (8+ regular semesters)
        """)

        if 'extended_graduation' in self.master_df.columns:
            valid = self.master_df['extended_graduation'].notna()
            if valid.sum() > 0:
                rate = self.master_df.loc[valid, 'extended_graduation'].mean() * 100
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Extended Graduation Rate", f"{rate:.1f}%")
                with col2:
                    st.metric("Graduated Students Analyzed", f"{valid.sum():,}")

        fig_path = FIGURES_DIR / 'rq9_feature_importance.png'
        if fig_path.exists():
            st.image(str(fig_path), caption="Top Predictors of Extended Graduation")

        st.markdown("""
        <div class="insight-box">
        <b>Key Predictors:</b>
        <ul>
        <li>Year 1-2 GPA and CGPA</li>
        <li>Course failure count and rate</li>
        <li>Math track performance</li>
        <li>GPA trajectory (improvement or decline)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # MATH TRACK ANALYSIS PAGE
    # ========================================================================
    def show_math_track_analysis(self):
        st.header("📐 Math Track Analysis")

        if 'math_track' not in self.master_df.columns:
            st.warning("Math track data not available")
            return

        df = self.master_df[(self.master_df['math_track'].notna()) & (self.master_df['math_track'] != 'Unknown')]

        # Overview
        st.subheader("Math Track Distribution")
        col1, col2 = st.columns(2)

        with col1:
            track_counts = df['math_track'].value_counts()
            fig = px.pie(values=track_counts.values, names=track_counts.index,
                        color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'],
                        title="Students by Math Track")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'final_cgpa' in df.columns:
                fig = px.box(df, x='math_track', y='final_cgpa', color='math_track',
                            title="CGPA Distribution by Math Track",
                            color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'])
                fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Probation")
                fig.add_hline(y=3.0, line_dash="dash", line_color="green", annotation_text="Success")
                st.plotly_chart(fig, use_container_width=True)

        # Detailed stats
        st.subheader("Detailed Statistics")
        agg_dict = {'student_id': 'count'}
        for col in ['final_cgpa', 'first_year_struggle', 'major_success', 'extended_graduation']:
            if col in df.columns:
                agg_dict[col] = 'mean'

        summary = df.groupby('math_track').agg(agg_dict).round(3)
        summary.columns = ['Count'] + [c.replace('_', ' ').title() for c in list(agg_dict.keys())[1:]]
        st.dataframe(summary, use_container_width=True)

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

    # ========================================================================
    # RISK ANALYSIS PAGE
    # ========================================================================
    def show_risk_analysis(self):
        st.header("⚠️ Risk Analysis")

        risk_options = {
            "First Year Struggle": "first_year_struggle",
            "AJC Case": "has_ajc_case",
            "Major Success": "major_success",
            "Extended Graduation": "extended_graduation"
        }
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
                    fig = px.bar(x=risk_by_track.index, y=risk_by_track.values,
                                color=risk_by_track.values, color_continuous_scale='RdYlGn_r',
                                labels={'x': 'Math Track', 'y': 'Risk Rate (%)'})
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'Gender' in filtered_df.columns:
                st.subheader("Risk by Gender")
                risk_by_gender = filtered_df.groupby('Gender')[risk_col].mean() * 100
                fig = px.bar(x=risk_by_gender.index, y=risk_by_gender.values,
                            color=risk_by_gender.values, color_continuous_scale='RdYlGn_r',
                            labels={'x': 'Gender', 'y': 'Risk Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # CLUSTER ANALYSIS PAGE
    # ========================================================================
    def show_cluster_analysis(self):
        st.header("🎯 Cluster Analysis")

        cluster_cols = [col for col in self.master_df.columns if col.startswith('cluster_')]
        if not cluster_cols:
            st.warning("No cluster data available. Run unsupervised learning notebook first.")
            # Show figure if exists
            fig_path = FIGURES_DIR / 'cluster_profiles.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="Student Cluster Profiles")
            return

        cluster_col = st.selectbox("Clustering Method", cluster_cols)
        n_clusters = self.master_df[cluster_col].nunique()
        st.info(f"📊 **{n_clusters} clusters** identified")

        agg_dict = {'student_id': 'count'}
        for col in ['final_cgpa', 'first_year_struggle', 'has_ajc_case', 'major_success']:
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
        selected_cohorts = st.multiselect("Select Cohorts", cohorts,
                                          default=cohorts[-5:] if len(cohorts) >= 5 else cohorts)

        if not selected_cohorts:
            return

        cohort_df = self.master_df[self.master_df[yg_col].isin(selected_cohorts)]

        agg_dict = {'student_id': 'count'}
        for col in ['final_cgpa', 'first_year_struggle', 'is_graduated', 'major_success']:
            if col in cohort_df.columns:
                agg_dict[col] = 'mean'

        summary = cohort_df.groupby(yg_col).agg(agg_dict).round(3)
        summary = summary.rename(columns={'student_id': 'Total'})

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(summary, use_container_width=True)

        with col2:
            if 'final_cgpa' in agg_dict:
                fig = px.line(summary.reset_index(), x=yg_col, y='final_cgpa',
                             title="Average CGPA by Cohort", markers=True)
                st.plotly_chart(fig, use_container_width=True)

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

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(major_df))
        with col2:
            if 'final_cgpa' in major_df.columns:
                st.metric("Avg CGPA", f"{major_df['final_cgpa'].mean():.2f}")
        with col3:
            if 'is_graduated' in major_df.columns:
                grad_rate = major_df['is_graduated'].mean() * 100
                st.metric("Graduation Rate", f"{grad_rate:.1f}%")
        with col4:
            if 'major_success' in major_df.columns:
                valid = major_df['major_success'].notna()
                if valid.sum() > 0:
                    success_rate = major_df.loc[valid, 'major_success'].mean() * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")

    # ========================================================================
    # PREDICTIONS PAGE
    # ========================================================================
    def show_predictions(self):
        st.header("🔮 Risk Prediction Tool")

        st.markdown("""
        <div class="warning-box">
        <b>Disclaimer:</b> This is a simplified prediction tool for demonstration.
        Actual predictions should use the trained models with proper feature engineering.
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.subheader("Student Information")
            col1, col2 = st.columns(2)

            with col1:
                hs_score = st.slider("High School Aggregate Score (0-100)", 0, 100, 75)
                math_score = st.slider("Math Score (0-100)", 0, 100, 70)
                y1_gpa = st.slider("Year 1 GPA (if available)", 0.0, 4.0, 3.0, 0.1)

            with col2:
                math_track = st.selectbox("Math Track", ["Calculus", "Pre-Calculus", "College Algebra"])
                major = st.selectbox("Intended Major", ["Computer Science", "Business Administration",
                                                        "Engineering", "Management Information Systems"])
                has_y1_data = st.checkbox("Has Year 1 Data", value=True)

            submitted = st.form_submit_button("🔮 Calculate Risk")

        if submitted:
            # Simple risk calculation
            base_risk = (100 - hs_score) / 100 * 0.3 + (100 - math_score) / 100 * 0.3

            if has_y1_data:
                gpa_risk = max(0, (3.0 - y1_gpa) / 3.0) * 0.4
                base_risk = base_risk * 0.4 + gpa_risk

            if math_track == "College Algebra":
                base_risk += 0.1
            elif math_track == "Pre-Calculus":
                base_risk += 0.05

            if major in ["Computer Science", "Engineering"] and math_track == "College Algebra":
                base_risk += 0.05

            struggle_risk = min(base_risk * 100, 95)
            success_prob = max(100 - struggle_risk * 1.2, 5)

            st.markdown("---")
            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                color = "🔴" if struggle_risk > 50 else "🟡" if struggle_risk > 25 else "🟢"
                st.metric(f"{color} Struggle Risk", f"{struggle_risk:.0f}%")
            with col2:
                color = "🟢" if success_prob > 70 else "🟡" if success_prob > 50 else "🔴"
                st.metric(f"{color} Success Probability", f"{success_prob:.0f}%")
            with col3:
                ext_risk = min(struggle_risk * 0.7, 85)
                color = "🔴" if ext_risk > 40 else "🟡" if ext_risk > 20 else "🟢"
                st.metric(f"{color} Extended Grad Risk", f"{ext_risk:.0f}%")

            if struggle_risk > 40:
                st.markdown("""
                <div class="warning-box">
                <b>Recommendation:</b> Consider early intervention - academic advising,
                tutoring support, and regular check-ins with faculty advisor.
                </div>
                """, unsafe_allow_html=True)

    # ========================================================================
    # ETHICS PAGE
    # ========================================================================
    def show_ethics(self):
        st.header("⚖️ Ethics & Fairness Analysis")

        st.markdown("""
        <div class="ethics-section">
        <h3>Ethical Framework</h3>
        This predictive system is designed to <b>support students</b>, not to label or restrict them.
        All predictions should be used to offer additional resources, not to make admissions or enrollment decisions.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🎯 Core Ethical Principles")

        principles = [
            ("Beneficence", "Use predictions to help students succeed, not to punish them", "✅"),
            ("Non-maleficence", "Avoid creating self-fulfilling prophecies", "⚠️"),
            ("Autonomy", "Students have agency over their academic journey", "🔑"),
            ("Justice", "Ensure the system doesn't amplify existing inequalities", "⚖️"),
            ("Transparency", "Explain how predictions are made", "🔍")
        ]

        for principle, desc, icon in principles:
            st.markdown(f"""
            <div class="metric-card">
            <b>{icon} {principle}</b><br>
            {desc}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Fairness Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Gender Parity")
            if 'Gender' in self.master_df.columns:
                cols_to_agg = ['student_id']
                agg_funcs = {'student_id': 'count'}
                for col in ['first_year_struggle', 'final_cgpa', 'major_success']:
                    if col in self.master_df.columns:
                        agg_funcs[col] = 'mean'

                gender_stats = self.master_df.groupby('Gender').agg(agg_funcs).round(3)
                gender_stats.columns = ['Count'] + [c.replace('_', ' ').title() for c in list(agg_funcs.keys())[1:]]
                st.dataframe(gender_stats, use_container_width=True)

        with col2:
            st.markdown("#### Math Track Equity")
            if 'math_track' in self.master_df.columns:
                agg_funcs = {'student_id': 'count'}
                for col in ['first_year_struggle', 'major_success']:
                    if col in self.master_df.columns:
                        agg_funcs[col] = 'mean'

                track_stats = self.master_df[self.master_df['math_track'] != 'Unknown'].groupby('math_track').agg(agg_funcs).round(3)
                track_stats.columns = ['Count'] + [c.replace('_', ' ').title() for c in list(agg_funcs.keys())[1:]]
                st.dataframe(track_stats, use_container_width=True)

        st.markdown("---")
        st.subheader("⚠️ Potential Harms & Mitigations")

        harms = [
            ("Self-fulfilling Prophecy", "Students labeled 'at-risk' may internalize this and perform worse",
             "Frame as support opportunity, not a label. Focus on growth potential."),
            ("Stigmatization", "Different treatment based on predictions may stigmatize students",
             "Limit access to predictions. Train advisors on supportive intervention."),
            ("Bias Perpetuation", "Historical data may reflect past discrimination",
             "Regular fairness audits. Monitor outcomes by demographic groups."),
            ("Privacy Concerns", "Sensitive student data is used for predictions",
             "Data minimization. Strong access controls. Clear consent processes.")
        ]

        for harm, desc, mitigation in harms:
            with st.expander(f"🔴 {harm}"):
                st.markdown(f"**Risk:** {desc}")
                st.markdown(f"**Mitigation:** {mitigation}")

        st.markdown("---")
        st.subheader("✅ Responsible Use Guidelines")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>✅ DO:</h4>
            <ul>
            <li>Use predictions to <b>offer support</b></li>
            <li>Combine with human judgment</li>
            <li>Regularly audit for fairness</li>
            <li>Be transparent with students</li>
            <li>Focus on actionable interventions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>❌ DON'T:</h4>
            <ul>
            <li>Use for admissions decisions</li>
            <li>Share individual scores without context</li>
            <li>Assume predictions are always correct</li>
            <li>Ignore student agency</li>
            <li>Use as punitive measures</li>
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
