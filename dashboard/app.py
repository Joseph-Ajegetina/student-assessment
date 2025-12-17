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
def load_cluster_data():
    """Load student cluster assignments from unsupervised learning"""
    try:
        path = PROCESSED_DIR / 'student_clusters.csv'
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
def load_trained_models():
    """Load trained ML models for predictions"""
    models = {}

    try:
        import joblib

        # Load RQ1 model (First-Year Struggle)
        rq1_model_path = MODELS_DIR / 'rq1_y1_struggle_model.joblib'
        rq1_preprocessor_path = MODELS_DIR / 'rq1_preprocessor.joblib'
        rq1_metadata_path = MODELS_DIR / 'rq1_metadata.json'

        if rq1_model_path.exists():
            models['rq1_struggle'] = {
                'model': joblib.load(rq1_model_path),
                'preprocessor': joblib.load(rq1_preprocessor_path) if rq1_preprocessor_path.exists() else None,
                'metadata': json.load(open(rq1_metadata_path)) if rq1_metadata_path.exists() else {}
            }

        # Load RQ2 model (AJC Case)
        rq2_model_path = MODELS_DIR / 'rq2_ajc_case_model.joblib'
        rq2_preprocessor_path = MODELS_DIR / 'rq2_preprocessor.joblib'
        rq2_metadata_path = MODELS_DIR / 'rq2_metadata.json'

        if rq2_model_path.exists():
            models['rq2_ajc'] = {
                'model': joblib.load(rq2_model_path),
                'preprocessor': joblib.load(rq2_preprocessor_path) if rq2_preprocessor_path.exists() else None,
                'metadata': json.load(open(rq2_metadata_path)) if rq2_metadata_path.exists() else {}
            }

        # Load RQ3 model (Major Success)
        rq3_model_path = MODELS_DIR / 'rq3_major_success_model.joblib'
        rq3_preprocessor_path = MODELS_DIR / 'rq3_preprocessor.joblib'
        rq3_metadata_path = MODELS_DIR / 'rq3_metadata.json'

        if rq3_model_path.exists():
            models['rq3_success'] = {
                'model': joblib.load(rq3_model_path),
                'preprocessor': joblib.load(rq3_preprocessor_path) if rq3_preprocessor_path.exists() else None,
                'metadata': json.load(open(rq3_metadata_path)) if rq3_metadata_path.exists() else {}
            }

        # Load RQ4 model (Probation)
        rq4_model_path = MODELS_DIR / 'rq4_probation_model.joblib'
        rq4_preprocessor_path = MODELS_DIR / 'rq4_preprocessor.joblib'
        rq4_metadata_path = MODELS_DIR / 'rq4_metadata.json'

        if rq4_model_path.exists():
            models['rq4_probation'] = {
                'model': joblib.load(rq4_model_path),
                'preprocessor': joblib.load(rq4_preprocessor_path) if rq4_preprocessor_path.exists() else None,
                'metadata': json.load(open(rq4_metadata_path)) if rq4_metadata_path.exists() else {}
            }

        # Load RQ9 model (Extended Graduation)
        rq9_model_path = MODELS_DIR / 'rq9_extended_graduation_model.joblib'
        rq9_preprocessor_path = MODELS_DIR / 'rq9_preprocessor.joblib'
        rq9_metadata_path = MODELS_DIR / 'rq9_metadata.json'

        if rq9_model_path.exists():
            models['rq9_extended'] = {
                'model': joblib.load(rq9_model_path),
                'preprocessor': joblib.load(rq9_preprocessor_path) if rq9_preprocessor_path.exists() else None,
                'metadata': json.load(open(rq9_metadata_path)) if rq9_metadata_path.exists() else {}
            }

    except ImportError:
        pass
    except Exception as e:
        pass

    return models

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
        self.cluster_data = load_cluster_data()
        self.model_results = load_model_results()
        self.models = load_models()
        self.trained_models = load_trained_models()
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

        # if self.cluster_data is not None:
        #     st.sidebar.info(f"✓ {len(self.cluster_data):,} students with clusters")
        # if self.model_results:
        #     st.sidebar.info(f"✓ {len(self.model_results)} result files loaded")
        # if self.trained_models:
        #     st.sidebar.success(f"✓ {len(self.trained_models)} ML models ready")
        # if self.models:
        #     st.sidebar.info(f"✓ {len(self.models)} models loaded")

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
        st.header("🔍 Student Lookup & Browser")

        # Filters in sidebar-style columns
        st.subheader("🔎 Filter Students")

        col1, col2, col3, col4 = st.columns(4)

        # Year Group Filter
        yg_col = self.get_yeargroup_column()
        with col1:
            if yg_col and yg_col in self.master_df.columns:
                years = ['All'] + sorted([int(y) for y in self.master_df[yg_col].dropna().unique()])
                selected_year = st.selectbox("Year Group", years)
            else:
                selected_year = 'All'

        # Program Filter
        program_col = self.get_program_column()
        with col2:
            if program_col and program_col in self.master_df.columns:
                programs = ['All'] + sorted(self.master_df[program_col].dropna().unique().tolist())
                selected_program = st.selectbox("Program", programs)
            else:
                selected_program = 'All'

        # Math Track Filter
        with col3:
            if 'math_track' in self.master_df.columns:
                tracks = ['All'] + sorted(self.master_df['math_track'].dropna().unique().tolist())
                selected_track = st.selectbox("Math Track", tracks)
            else:
                selected_track = 'All'

        # Status Filter
        with col4:
            status_options = ['All', 'Graduated', 'Active', 'Other']
            selected_status = st.selectbox("Status", status_options)

        # Apply filters
        filtered_df = self.master_df.copy()

        if selected_year != 'All' and yg_col:
            filtered_df = filtered_df[filtered_df[yg_col] == selected_year]

        if selected_program != 'All' and program_col:
            filtered_df = filtered_df[filtered_df[program_col] == selected_program]

        if selected_track != 'All' and 'math_track' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['math_track'] == selected_track]

        if selected_status != 'All':
            if selected_status == 'Graduated' and 'is_graduated' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['is_graduated'] == 1]
            elif selected_status == 'Active' and 'is_active' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['is_active'] == 1]
            elif selected_status == 'Other':
                if 'is_graduated' in filtered_df.columns and 'is_active' in filtered_df.columns:
                    filtered_df = filtered_df[(filtered_df['is_graduated'] != 1) & (filtered_df['is_active'] != 1)]

        # Search box
        search_term = st.text_input("🔍 Search by Student ID:", placeholder="Enter partial or full student ID...")
        if search_term:
            filtered_df = filtered_df[filtered_df['student_id'].astype(str).str.upper().str.contains(search_term.upper(), na=False)]

        st.markdown(f"**Showing {len(filtered_df):,} students** (out of {len(self.master_df):,} total)")

        # Student table
        st.markdown("---")
        st.subheader("📋 Student List")

        # Select columns to display
        display_cols = ['student_id']
        if yg_col and yg_col in filtered_df.columns:
            display_cols.append(yg_col)
        if program_col and program_col in filtered_df.columns:
            display_cols.append(program_col)
        for col in ['Gender', 'math_track', 'final_cgpa', 'is_graduated', 'is_active']:
            if col in filtered_df.columns:
                display_cols.append(col)

        # Show paginated table
        if len(filtered_df) > 0:
            # Limit display for performance
            max_display = 100
            if len(filtered_df) > max_display:
                st.info(f"Showing first {max_display} students. Use filters to narrow down.")
                display_df = filtered_df[display_cols].head(max_display)
            else:
                display_df = filtered_df[display_cols]

            # Round CGPA for display
            if 'final_cgpa' in display_df.columns:
                display_df = display_df.copy()
                display_df['final_cgpa'] = display_df['final_cgpa'].round(2)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Student detail selector
            st.markdown("---")
            st.subheader("👤 Student Details")

            student_ids = filtered_df['student_id'].tolist()
            if len(student_ids) > 0:
                selected_id = st.selectbox(
                    "Select a student to view details:",
                    student_ids,
                    key="student_detail_select"
                )

                if selected_id:
                    student = filtered_df[filtered_df['student_id'] == selected_id].iloc[0]

                    # Status badge
                    if student.get('is_graduated', 0) == 1:
                        status = "🟢 Graduated"
                    elif student.get('is_active', 0) == 1:
                        status = "🔵 Active"
                    elif student.get('is_withdrawn', 0) == 1:
                        status = "🟡 Withdrawn"
                    elif student.get('is_dismissed', 0) == 1:
                        status = "🔴 Dismissed"
                    else:
                        status = "⚪ Unknown"

                    st.markdown(f"### {selected_id} {status}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**📝 Demographics**")
                        st.write(f"• Gender: {student.get('Gender', 'N/A')}")
                        st.write(f"• International: {'Yes' if student.get('is_international', 0) == 1 else 'No'}")
                        if yg_col and pd.notna(student.get(yg_col)):
                            st.write(f"• Year Group: {int(student[yg_col])}")

                    with col2:
                        st.markdown("**🎓 Academic Info**")
                        if program_col:
                            st.write(f"• Program: {student.get(program_col, 'N/A')}")
                        st.write(f"• Math Track: {student.get('math_track', 'N/A')}")
                        if pd.notna(student.get('first_math_grade')):
                            st.write(f"• First Math Grade: {student.get('first_math_grade', 'N/A')}")

                    with col3:
                        st.markdown("**📊 Performance**")
                        if pd.notna(student.get('final_cgpa')):
                            cgpa = student['final_cgpa']
                            cgpa_color = "🟢" if cgpa >= 3.0 else "🟡" if cgpa >= 2.0 else "🔴"
                            st.write(f"• Final CGPA: {cgpa_color} {cgpa:.2f}")
                        if pd.notna(student.get('total_semesters')):
                            st.write(f"• Total Semesters: {int(student['total_semesters'])}")
                        if pd.notna(student.get('academic_years')):
                            st.write(f"• Academic Years: {int(student['academic_years'])}")

                    # Risk indicators
                    st.markdown("---")
                    st.markdown("**⚠️ Risk Indicators**")
                    risk_cols = st.columns(5)

                    risk_indicators = [
                        ('first_year_struggle', 'Y1 Struggle'),
                        ('has_ajc_case', 'AJC Case'),
                        ('major_success', 'Major Success'),
                        ('extended_graduation', 'Extended Grad'),
                        ('ever_on_probation', 'Ever Probation')
                    ]

                    for i, (col_name, label) in enumerate(risk_indicators):
                        with risk_cols[i]:
                            if col_name in student.index and pd.notna(student[col_name]):
                                val = int(student[col_name])
                                if col_name == 'major_success':
                                    icon = "🟢 Yes" if val == 1 else "🔴 No"
                                else:
                                    icon = "🔴 Yes" if val == 1 else "🟢 No"
                                st.metric(label, icon)
                            else:
                                st.metric(label, "N/A")
        else:
            st.warning("No students match the selected filters.")

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

        if self.cluster_data is None or len(self.cluster_data) == 0:
            st.warning("No cluster data available. Run unsupervised learning notebook first.")
            # Show figure if exists
            fig_path = FIGURES_DIR / 'cluster_profiles.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="Student Cluster Profiles")
            return

        # Merge cluster data with master data for analysis
        cluster_df = self.cluster_data.copy()
        if self.master_df is not None:
            cluster_df = cluster_df.merge(
                self.master_df[['student_id'] + [c for c in self.master_df.columns
                                                  if c in ['final_cgpa', 'first_year_struggle',
                                                          'has_ajc_case', 'major_success',
                                                          'extended_graduation', 'Gender', 'math_track']]],
                on='student_id', how='left'
            )

        n_clusters = cluster_df['cluster'].nunique()
        cluster_names = cluster_df['cluster_name'].unique().tolist()

        st.info(f"📊 **{n_clusters} student clusters** identified: {', '.join(cluster_names)}")

        # Cluster distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Distribution")
            cluster_counts = cluster_df['cluster_name'].value_counts()
            fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                        color_discrete_sequence=['#27ae60', '#f39c12', '#e74c3c', '#3498db'],
                        title="Students by Cluster")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cluster Sizes")
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        color=cluster_counts.values, color_continuous_scale='Blues',
                        labels={'x': 'Cluster', 'y': 'Number of Students'},
                        title="Students per Cluster")
            st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles with outcomes
        st.markdown("---")
        st.subheader("Cluster Profiles")

        agg_dict = {'student_id': 'count'}
        for col in ['final_cgpa', 'first_year_struggle', 'has_ajc_case', 'major_success', 'extended_graduation']:
            if col in cluster_df.columns:
                agg_dict[col] = 'mean'

        summary = cluster_df.groupby('cluster_name').agg(agg_dict).round(3)
        summary = summary.rename(columns={
            'student_id': 'Count',
            'final_cgpa': 'Avg CGPA',
            'first_year_struggle': 'Struggle Rate',
            'has_ajc_case': 'AJC Rate',
            'major_success': 'Success Rate',
            'extended_graduation': 'Extended Grad Rate'
        })
        st.dataframe(summary, use_container_width=True)

        # PCA visualization if available
        if 'pca_1' in cluster_df.columns and 'pca_2' in cluster_df.columns:
            st.markdown("---")
            st.subheader("PCA Cluster Visualization")

            fig = px.scatter(cluster_df, x='pca_1', y='pca_2', color='cluster_name',
                           title="Student Clusters in PCA Space",
                           labels={'pca_1': 'Principal Component 1', 'pca_2': 'Principal Component 2'},
                           color_discrete_sequence=['#27ae60', '#f39c12', '#e74c3c', '#3498db'],
                           opacity=0.6)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Show saved figures
        st.markdown("---")
        st.subheader("Cluster Analysis Figures")

        col1, col2 = st.columns(2)
        with col1:
            fig_path = FIGURES_DIR / 'cluster_profiles.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="Cluster Feature Profiles (Heatmap)")

        with col2:
            fig_path = FIGURES_DIR / 'outcomes_by_cluster.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="Outcome Rates by Cluster")

        col1, col2 = st.columns(2)
        with col1:
            fig_path = FIGURES_DIR / 'pca_visualization.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="PCA Visualization")

        with col2:
            fig_path = FIGURES_DIR / 'tsne_visualization.png'
            if fig_path.exists():
                st.image(str(fig_path), caption="t-SNE Visualization")

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
    def _display_risk_gauge(self, title, value, thresholds=(0.30, 0.60)):
        """Display a risk gauge visualization"""
        low, high = thresholds

        if value > high:
            color, level = "#e74c3c", "HIGH"
        elif value > low:
            color, level = "#f39c12", "MEDIUM"
        else:
            color, level = "#27ae60", "LOW"

        st.markdown(f"### {title}")
        st.markdown(f"**{level}**: {value*100:.0f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, low*100], 'color': '#d4edda'},
                    {'range': [low*100, high*100], 'color': '#fff3cd'},
                    {'range': [high*100, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': value * 100
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(t=20, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    def _create_feature_vector(self, gender, is_international, needs_aid, intended_major,
                                 math_track, hs_math, hs_english, hs_science, hs_aggregate):
        """Create feature vector matching the trained model's expected input"""
        # Map categorical values to binary features matching training data
        feature_dict = {
            'gender_male': 1 if gender == 'Male' else 0,
            'is_international': 1 if is_international else 0,
            'needs_financial_aid': 1 if needs_aid else 0,
            'disadvantaged_background': 0,  # Default
            'intended_cs': 1 if intended_major == 'Computer Science' else 0,
            'intended_engineering': 1 if intended_major == 'Engineering' else 0,
            'intended_business': 1 if intended_major == 'Business Administration' else 0,
            'intended_mis': 1 if intended_major == 'Management Information Systems' else 0,
            'exam_wassce': 1,  # Default to WASSCE
            'exam_ib': 0,
            'exam_alevel': 0,
            'has_previous_application': 0,
            'hs_mathematics': hs_math,
            'hs_english_language': hs_english,
            'hs_best_science': hs_science,
            'hs_aggregate_score': hs_aggregate,
            'has_elective_math': 1 if math_track == 'Calculus' else 0
        }
        return feature_dict

    def show_predictions(self):
        st.header("🔮 Risk Prediction Tool")

        # Check if real models are available
        has_real_models = bool(self.trained_models)

        # Toggle between ML models and heuristics
        col_toggle1, col_toggle2 = st.columns([3, 1])
        with col_toggle1:
            st.markdown("**Prediction Method:**")
        with col_toggle2:
            use_heuristics = st.toggle("Use Demo Mode", value=True, help="Toggle between ML models and rule-based predictions")

        if use_heuristics:
            st.markdown("""
            <div class="info-box">
            <b>📊 Demo Mode (Rule-Based):</b> Predictions use simplified heuristic rules based on
            research findings about student success factors.
            </div>
            """, unsafe_allow_html=True)
        elif has_real_models:
            st.markdown("""
            <div class="insight-box">
            <b>🤖 ML Model Mode:</b> Predictions are made using actual machine learning models
            trained on historical Ashesi student data.
            </div>
            """, unsafe_allow_html=True)

            # Show model info
            with st.expander("📊 Model Information"):
                for model_key, model_info in self.trained_models.items():
                    metadata = model_info.get('metadata', {})
                    st.markdown(f"**{model_key.upper()}**: {metadata.get('model_name', 'Unknown')}")
                    if 'metrics' in metadata:
                        metrics = metadata['metrics']
                        st.write(f"  - Recall: {metrics.get('recall', 0):.3f}")
                        st.write(f"  - F2 Score: {metrics.get('f2', 0):.3f}")
                        st.write(f"  - AUC: {metrics.get('auc', 0):.3f}")
        else:
            st.markdown("""
            <div class="warning-box">
            <b>Demo Mode:</b> No trained models found. Using rule-based predictions.
            Run the supervised learning notebooks to train and save ML models.
            </div>
            """, unsafe_allow_html=True)
            use_heuristics = True  # Force heuristics if no models

        with st.form("prediction_form"):
            st.subheader("Student Information")
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                is_international = st.checkbox("International Student")
                needs_aid = st.checkbox("Needs Financial Aid")

            with col2:
                math_track = st.selectbox("Math Track", ["Calculus", "Pre-Calculus", "College Algebra"])
                intended_major = st.selectbox("Intended Major", [
                    "Computer Science", "Business Administration",
                    "Engineering", "Management Information Systems", "Other"
                ])

            st.subheader("High School Scores (0-100)")
            col3, col4, col5, col6 = st.columns(4)

            with col3:
                hs_math = st.slider("Math Score", 0, 100, 70)
            with col4:
                hs_english = st.slider("English Score", 0, 100, 70)
            with col5:
                hs_science = st.slider("Best Science", 0, 100, 70)
            with col6:
                hs_aggregate = st.slider("Aggregate Score", 0, 100, 75)

            submitted = st.form_submit_button("🔮 Calculate Risk", type="primary")

        if submitted:
            # Create feature dictionary
            features = self._create_feature_vector(
                gender, is_international, needs_aid, intended_major,
                math_track, hs_math, hs_english, hs_science, hs_aggregate
            )

            # Use heuristics if toggle is on, or if no models available
            if use_heuristics or not has_real_models:
                # Rule-based heuristic predictions
                base_risk = (100 - hs_math) / 100 * 0.3 + (100 - hs_aggregate) / 100 * 0.3

                if math_track == "College Algebra":
                    base_risk += 0.1
                elif math_track == "Pre-Calculus":
                    base_risk += 0.05

                if intended_major in ["Computer Science", "Engineering"] and math_track == "College Algebra":
                    base_risk += 0.05

                if needs_aid:
                    base_risk += 0.03

                struggle_risk = min(base_risk, 0.95)
                ajc_risk = min(base_risk * 0.25, 0.40)
                success_prob = max(1 - struggle_risk * 1.2, 0.05)
                ext_risk = min(struggle_risk * 0.7, 0.85)
                prediction_method = "Demo (Rule-Based)"

            elif 'rq1_struggle' in self.trained_models:
                # Use ML models
                feature_df = pd.DataFrame([features])

                # Get model components
                rq1_info = self.trained_models['rq1_struggle']
                model = rq1_info['model']
                preprocessor = rq1_info['preprocessor']
                expected_features = rq1_info['metadata'].get('features', list(features.keys()))

                # Ensure correct feature order
                feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

                # Preprocess
                if preprocessor:
                    X_processed = preprocessor.transform(feature_df)
                else:
                    X_processed = feature_df.values

                # Predict
                struggle_prob = model.predict_proba(X_processed)[0, 1]
                struggle_risk = float(struggle_prob)

                # AJC prediction if model available
                if 'rq2_ajc' in self.trained_models:
                    rq2_info = self.trained_models['rq2_ajc']
                    model2 = rq2_info['model']
                    preprocessor2 = rq2_info['preprocessor']

                    if preprocessor2:
                        X_processed2 = preprocessor2.transform(feature_df)
                    else:
                        X_processed2 = feature_df.values

                    ajc_risk = float(model2.predict_proba(X_processed2)[0, 1])
                else:
                    ajc_risk = struggle_risk * 0.25

                # Success probability - derive from struggle risk
                # (RQ3 model requires Y1+Y2 data which we don't have in admission form)
                # Use inverse relationship: low struggle = high success
                success_prob = max(1 - struggle_risk * 1.3, 0.10)

                # Adjust based on math track and scores
                if math_track == "Calculus":
                    success_prob = min(success_prob + 0.10, 0.95)
                elif math_track == "College Algebra":
                    success_prob = max(success_prob - 0.08, 0.10)

                if hs_math >= 80:
                    success_prob = min(success_prob + 0.05, 0.95)
                elif hs_math < 50:
                    success_prob = max(success_prob - 0.10, 0.10)

                # Extended graduation risk - derive from struggle risk
                # (RQ9 model requires Y1+Y2 data which we don't have)
                ext_risk = min(struggle_risk * 0.8, 0.85)

                # Adjust based on factors
                if math_track == "College Algebra":
                    ext_risk = min(ext_risk + 0.05, 0.85)
                if needs_aid:
                    ext_risk = min(ext_risk + 0.03, 0.85)
                if hs_aggregate < 60:
                    ext_risk = min(ext_risk + 0.08, 0.85)

                prediction_method = "ML Model"
            else:
                # Fallback to heuristic
                base_risk = (100 - hs_math) / 100 * 0.3 + (100 - hs_aggregate) / 100 * 0.3
                struggle_risk = min(base_risk, 0.95)
                ajc_risk = min(base_risk * 0.25, 0.40)
                success_prob = max(1 - struggle_risk * 1.2, 0.05)
                ext_risk = min(struggle_risk * 0.7, 0.85)
                prediction_method = "Demo (Rule-Based)"

            st.markdown("---")
            st.subheader("📊 Prediction Results")

            # Show prediction method
            st.caption(f"🤖 Prediction Method: **{prediction_method}**")

            # Display gauges
            col1, col2 = st.columns(2)

            with col1:
                self._display_risk_gauge("First Year Struggle Risk", struggle_risk, (0.25, 0.50))

            with col2:
                self._display_risk_gauge("Extended Graduation Risk", ext_risk, (0.20, 0.40))

            col3, col4 = st.columns(2)

            with col3:
                self._display_risk_gauge("AJC Case Risk", ajc_risk, (0.10, 0.25))

            with col4:
                # Success probability gauge (inverted colors - high is good)
                st.markdown("### Success Probability")
                if success_prob > 0.70:
                    color, level = "#27ae60", "HIGH"
                elif success_prob > 0.50:
                    color, level = "#f39c12", "MEDIUM"
                else:
                    color, level = "#e74c3c", "LOW"

                st.markdown(f"**{level}**: {success_prob*100:.0f}%")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=success_prob * 100,
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': '#f8d7da'},
                            {'range': [50, 70], 'color': '#fff3cd'},
                            {'range': [70, 100], 'color': '#d4edda'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 2},
                            'thickness': 0.75,
                            'value': success_prob * 100
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=20, b=20, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")

            recs = []

            if struggle_risk > 0.40:
                recs.append("🎯 **Academic Support**: Enroll in tutoring and study skills programs")

            if math_track == "College Algebra":
                recs.append("📐 **Math Support**: Consider math tutoring or bridge courses")

            if intended_major in ["Computer Science", "Engineering"] and math_track != "Calculus":
                recs.append("💻 **STEM Mentorship**: Connect with peer mentors in your field")

            if ext_risk > 0.30:
                recs.append("📅 **Academic Planning**: Create detailed graduation timeline with advisor")

            if ajc_risk > 0.15:
                recs.append("📚 **Academic Integrity**: Review honor code and citation practices")

            if not recs:
                recs.append("✅ **Low Risk Profile**: Standard support should be sufficient")

            for rec in recs:
                st.markdown(f"- {rec}")

            # Overall assessment
            overall_risk = (struggle_risk + ext_risk + ajc_risk) / 3

            if overall_risk > 0.40:
                st.markdown("""
                <div class="warning-box">
                <b>⚠️ High Risk Alert:</b> This student profile shows elevated risk indicators.
                Consider early intervention including academic advising, tutoring support,
                and regular check-ins with faculty advisor.
                </div>
                """, unsafe_allow_html=True)
            elif overall_risk > 0.25:
                st.markdown("""
                <div class="info-box">
                <b>ℹ️ Moderate Risk:</b> Some risk factors present.
                Proactive support recommended to ensure student success.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-box">
                <b>✅ Low Risk:</b> This student profile shows favorable indicators.
                Standard academic support should be sufficient.
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
