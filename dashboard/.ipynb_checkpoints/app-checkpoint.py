"""
Ashesi Student Success Dashboard
A Streamlit-based dashboard for exploring student outcomes and model predictions.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ashesi Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
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
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all processed data."""
    data = {}

    try:
        data['full_features'] = pd.read_csv(PROCESSED_DIR / 'full_features.csv')
    except FileNotFoundError:
        st.warning("Processed data not found. Please run the feature engineering notebook first.")
        data['full_features'] = pd.DataFrame()

    try:
        data['clusters'] = pd.read_csv(PROCESSED_DIR / 'student_clusters.csv')
    except FileNotFoundError:
        data['clusters'] = pd.DataFrame()

    try:
        data['targets'] = pd.read_csv(PROCESSED_DIR / 'targets.csv')
    except FileNotFoundError:
        data['targets'] = pd.DataFrame()

    return data


def overview_page(data):
    """Executive Overview page."""
    st.markdown('<h1 class="main-header">🎓 Ashesi Student Success Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    df = data['full_features']

    if df.empty:
        st.error("No data available. Please run the notebooks to generate processed data.")
        return

    # Key Metrics
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    total_students = len(df)

    with col1:
        st.metric("Total Students", f"{total_students:,}")

    with col2:
        if 'target_ever_probation' in df.columns:
            probation_rate = df['target_ever_probation'].mean() * 100
            st.metric("Ever on Probation", f"{probation_rate:.1f}%", delta=f"-{probation_rate:.1f}%", delta_color="inverse")
        else:
            st.metric("Ever on Probation", "N/A")

    with col3:
        if 'target_major_success' in df.columns:
            success_rate = df['target_major_success'].mean() * 100
            st.metric("Major Success Rate", f"{success_rate:.1f}%", delta=f"+{success_rate:.1f}%")
        else:
            st.metric("Major Success Rate", "N/A")

    with col4:
        if 'target_ajc_case' in df.columns:
            ajc_rate = df['target_ajc_case'].mean() * 100
            st.metric("AJC Case Rate", f"{ajc_rate:.1f}%", delta=f"-{ajc_rate:.1f}%", delta_color="inverse")
        else:
            st.metric("AJC Case Rate", "N/A")

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 CGPA Distribution")
        if 'target_final_cgpa' in df.columns:
            fig = px.histogram(df, x='target_final_cgpa', nbins=30,
                              title='Final CGPA Distribution',
                              color_discrete_sequence=['#3498db'])
            fig.add_vline(x=2.0, line_dash="dash", line_color="red", annotation_text="Probation")
            fig.add_vline(x=3.0, line_dash="dash", line_color="green", annotation_text="Success")
            fig.update_layout(xaxis_title="CGPA", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CGPA data not available")

    with col2:
        st.subheader("🎯 Outcome Distribution")
        if 'target_y1_struggle' in df.columns:
            outcomes = {
                'First Year Struggle': df['target_y1_struggle'].mean() * 100 if 'target_y1_struggle' in df.columns else 0,
                'Ever on Probation': df['target_ever_probation'].mean() * 100 if 'target_ever_probation' in df.columns else 0,
                'Extended Graduation': df['target_extended_graduation'].mean() * 100 if 'target_extended_graduation' in df.columns else 0,
            }
            fig = px.bar(x=list(outcomes.keys()), y=list(outcomes.values()),
                        title='Risk Outcome Rates (%)',
                        color=list(outcomes.values()),
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(xaxis_title="Outcome", yaxis_title="Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Outcome data not available")

    # Academic Policies Reminder
    st.markdown("---")
    st.subheader("📋 Academic Policy Reference")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Academic Probation</b><br>
        CGPA < 2.0 at end of any regular semester
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>🚫 Dismissal</b><br>
        Two consecutive semesters on probation without GPA ≥ 2.0
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-box">
        <b>⭐ Dean's List</b><br>
        Semester GPA ≥ 3.5
        </div>
        """, unsafe_allow_html=True)


def academic_advisor_page(data):
    """Academic Advisor Dashboard."""
    st.header("👨‍🏫 Academic Advisor Dashboard")

    df = data['full_features']
    clusters = data['clusters']

    if df.empty:
        st.error("No data available.")
        return

    # Filters
    st.sidebar.subheader("Filters")

    # Risk level filter
    if 'target_ever_probation' in df.columns:
        risk_options = ['All', 'At-Risk Only', 'Not At-Risk']
        risk_filter = st.sidebar.selectbox("Risk Level", risk_options)

    # Student risk scatter
    st.subheader("🎯 Student Risk Map")

    if 'y1_cgpa_end' in df.columns and 'y1_gpa_trend' in df.columns:
        plot_df = df.dropna(subset=['y1_cgpa_end', 'y1_gpa_trend']).copy()

        if 'target_ever_probation' in plot_df.columns:
            plot_df['Risk Status'] = plot_df['target_ever_probation'].map({0: 'Low Risk', 1: 'At Risk'})
            color_col = 'Risk Status'
        else:
            color_col = None

        fig = px.scatter(plot_df, x='y1_cgpa_end', y='y1_gpa_trend',
                        color=color_col,
                        title='Student Risk Map: CGPA vs GPA Trend',
                        color_discrete_map={'At Risk': '#e74c3c', 'Low Risk': '#27ae60'},
                        hover_data=['StudentRef'] if 'StudentRef' in plot_df.columns else None)

        # Add risk zones
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=2.0, line_dash="dash", line_color="red")

        fig.update_layout(
            xaxis_title="Year 1 CGPA",
            yaxis_title="GPA Trend (Semester 2 - Semester 1)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.markdown("""
        **Interpretation:**
        - 📍 **Top-Right**: High performers with improving grades
        - 📍 **Bottom-Right**: High performers with declining trend (monitor)
        - 📍 **Top-Left**: Low performers but improving (encourage)
        - 📍 **Bottom-Left**: At-risk, needs immediate intervention
        """)
    else:
        st.info("Year 1 academic data not available for visualization")

    # Student list
    st.subheader("📋 Student Priority List")

    if 'target_ever_probation' in df.columns:
        at_risk = df[df['target_ever_probation'] == 1].copy()

        display_cols = ['StudentRef']
        if 'y1_cgpa_end' in df.columns:
            display_cols.append('y1_cgpa_end')
        if 'y1_fail_count' in df.columns:
            display_cols.append('y1_fail_count')
        if 'y1_gpa_trend' in df.columns:
            display_cols.append('y1_gpa_trend')

        display_cols = [c for c in display_cols if c in at_risk.columns]

        if len(at_risk) > 0:
            st.write(f"**{len(at_risk)} students have been on probation**")
            st.dataframe(at_risk[display_cols].head(20), use_container_width=True)

            # Export button
            csv = at_risk[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download At-Risk Student List",
                data=csv,
                file_name="at_risk_students.csv",
                mime="text/csv"
            )
        else:
            st.success("No at-risk students identified!")
    else:
        st.info("Risk classification data not available")


def faculty_insights_page(data):
    """Faculty Insights page."""
    st.header("👩‍🏫 Faculty Insights")

    df = data['full_features']

    if df.empty:
        st.error("No data available.")
        return

    # Math Track Analysis (RQ7)
    st.subheader("📐 Math Track Performance (RQ7)")

    if 'math_track' in df.columns and 'target_final_cgpa' in df.columns:
        math_data = df.dropna(subset=['math_track', 'target_final_cgpa'])

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(math_data, x='math_track', y='target_final_cgpa',
                        title='CGPA Distribution by Math Track',
                        color='math_track')
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Probation")
            fig.update_layout(xaxis_title="Math Track", yaxis_title="Final CGPA")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Probation rate by track
            if 'target_ever_probation' in df.columns:
                probation_by_track = df.groupby('math_track')['target_ever_probation'].mean() * 100
                fig = px.bar(x=probation_by_track.index, y=probation_by_track.values,
                            title='Probation Rate by Math Track',
                            color=probation_by_track.values,
                            color_continuous_scale='RdYlGn_r')
                fig.update_layout(xaxis_title="Math Track", yaxis_title="Probation Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

        # Key finding
        st.markdown("""
        <div class="insight-box">
        <b>📊 Key Finding:</b> Students starting on the Calculus track tend to have higher CGPAs and lower probation rates.
        College Algebra track students may benefit from additional math support.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Math track data not available")

    # Failure patterns
    st.subheader("📉 Academic Struggle Indicators")

    col1, col2 = st.columns(2)

    with col1:
        if 'y1_fail_count' in df.columns:
            fig = px.histogram(df, x='y1_fail_count',
                              title='Year 1 Course Failures Distribution',
                              color_discrete_sequence=['#e74c3c'])
            fig.update_layout(xaxis_title="Number of Failed Courses", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'y1_fail_rate' in df.columns and 'target_ever_probation' in df.columns:
            fig = px.box(df, x='target_ever_probation', y='y1_fail_rate',
                        title='Failure Rate: Probation vs Non-Probation',
                        color='target_ever_probation',
                        color_discrete_map={0: '#27ae60', 1: '#e74c3c'})
            fig.update_layout(xaxis_title="Ever on Probation", yaxis_title="Year 1 Failure Rate")
            st.plotly_chart(fig, use_container_width=True)


def admissions_page(data):
    """Admissions Analytics page."""
    st.header("📝 Admissions Analytics")

    df = data['full_features']

    if df.empty:
        st.error("No data available.")
        return

    # HS Exam Score Distributions
    st.subheader("📊 High School Exam Performance")

    col1, col2 = st.columns(2)

    with col1:
        if 'hs_mathematics' in df.columns:
            fig = px.histogram(df, x='hs_mathematics', nbins=20,
                              title='High School Math Scores (Normalized)',
                              color_discrete_sequence=['#3498db'])
            fig.update_layout(xaxis_title="Math Score (0-100)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Math score data not available")

    with col2:
        if 'hs_english_language' in df.columns:
            fig = px.histogram(df, x='hs_english_language', nbins=20,
                              title='High School English Scores (Normalized)',
                              color_discrete_sequence=['#2ecc71'])
            fig.update_layout(xaxis_title="English Score (0-100)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("English score data not available")

    # HS Scores vs Outcomes
    st.subheader("🎯 Admissions Predictors of Success")

    if 'hs_mathematics' in df.columns and 'target_major_success' in df.columns:
        fig = px.scatter(df.dropna(subset=['hs_mathematics', 'target_final_cgpa']),
                        x='hs_mathematics', y='target_final_cgpa',
                        color='target_major_success' if 'target_major_success' in df.columns else None,
                        title='HS Math Score vs Final CGPA',
                        color_discrete_map={0: '#e74c3c', 1: '#27ae60'},
                        opacity=0.5)
        fig.update_layout(xaxis_title="HS Math Score", yaxis_title="Final CGPA")
        st.plotly_chart(fig, use_container_width=True)

        # Correlation
        corr = df[['hs_mathematics', 'target_final_cgpa']].dropna().corr().iloc[0, 1]
        st.markdown(f"""
        <div class="insight-box">
        <b>📈 Correlation:</b> HS Math Score and Final CGPA have a correlation of <b>{corr:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

    # Financial Aid Analysis
    st.subheader("💰 Financial Aid Impact")

    if 'needs_financial_aid' in df.columns and 'target_ever_probation' in df.columns:
        aid_analysis = df.groupby('needs_financial_aid').agg({
            'target_ever_probation': 'mean',
            'target_major_success': 'mean' if 'target_major_success' in df.columns else 'count',
            'StudentRef': 'count'
        }).rename(columns={'StudentRef': 'count'})

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(x=['No Financial Aid', 'Needs Financial Aid'],
                        y=aid_analysis['target_ever_probation'].values * 100,
                        title='Probation Rate by Financial Aid Status',
                        color=['No Financial Aid', 'Needs Financial Aid'],
                        color_discrete_sequence=['#3498db', '#e74c3c'])
            fig.update_layout(xaxis_title="", yaxis_title="Probation Rate (%)")
            st.plotly_chart(fig, use_container_width=True)


def research_questions_page(data):
    """Research Questions Dashboard."""
    st.header("🔬 Research Questions Dashboard")

    df = data['full_features']

    if df.empty:
        st.error("No data available.")
        return

    # Tabs for each RQ
    tabs = st.tabs(["RQ1-2", "RQ3-6", "RQ7", "RQ8", "RQ9"])

    with tabs[0]:
        st.subheader("RQ1: First-Year Struggle | RQ2: AJC Cases")
        st.markdown("""
        **RQ1**: Can admissions data predict first-year academic struggle?

        **RQ2**: Can admissions data predict AJC cases?
        """)

        col1, col2 = st.columns(2)

        with col1:
            if 'target_y1_struggle' in df.columns:
                rate = df['target_y1_struggle'].mean() * 100
                st.metric("First-Year Struggle Rate", f"{rate:.1f}%")

        with col2:
            if 'target_ajc_case' in df.columns:
                rate = df['target_ajc_case'].mean() * 100
                st.metric("AJC Case Rate", f"{rate:.2f}%")

        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> High school exam scores, particularly in mathematics,
        are the strongest predictors of first-year success among admissions features.
        AJC prediction is challenging due to extreme class imbalance (~3% positive rate).
        </div>
        """, unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("RQ3-6: Major Success/Failure Prediction")
        st.markdown("""
        **RQ3/5**: Predict major success using Year 1 / Year 1-2 data

        **RQ4/6**: Predict major failure/change using Year 1 / Year 1-2 data
        """)

        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> Adding Year 1 academic data significantly improves prediction accuracy.
        GPA trajectory (improving vs declining) is a powerful early warning indicator.
        </div>
        """, unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("RQ7: Math Track Performance Comparison")

        if 'math_track' in df.columns:
            # Show comparison
            track_stats = df.groupby('math_track').agg({
                'target_final_cgpa': ['mean', 'std'],
                'target_ever_probation': 'mean'
            }).round(2)
            st.dataframe(track_stats, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <b>Statistical Finding:</b> ANOVA test shows significant differences between math tracks (p < 0.05).
            Students on the Calculus track generally outperform other tracks.
            </div>
            """, unsafe_allow_html=True)

    with tabs[3]:
        st.subheader("RQ8: College Algebra Track in CS Major")

        if 'intended_cs' in df.columns and 'math_track' in df.columns:
            cs_students = df[df['intended_cs'] == 1]
            cs_by_track = cs_students.groupby('math_track')['target_final_cgpa'].agg(['mean', 'count'])
            st.dataframe(cs_by_track.round(2), use_container_width=True)

            st.markdown("""
            <div class="warning-box">
            <b>Finding:</b> College algebra track students CAN succeed in CS, but have lower average CGPAs.
            Recommendation: Provide additional math support for CS students on the algebra track.
            </div>
            """, unsafe_allow_html=True)

    with tabs[4]:
        st.subheader("RQ9: Extended Graduation Prediction")

        if 'target_extended_graduation' in df.columns:
            rate = df['target_extended_graduation'].mean() * 100
            st.metric("Extended Graduation Rate", f"{rate:.1f}%")

            st.markdown("""
            <div class="insight-box">
            <b>Key Predictors:</b>
            - Year 1 failure count
            - GPA trend in early semesters
            - Math track performance
            - First math course grade
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application."""
    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "Executive Overview": overview_page,
        "Academic Advisor": academic_advisor_page,
        "Faculty Insights": faculty_insights_page,
        "Admissions Analytics": admissions_page,
        "Research Questions": research_questions_page
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Load data
    data = load_data()

    # Render selected page
    pages[selection](data)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Ashesi University**
    Student Success Prediction System
    Machine Learning & Data Science Project
    """)


if __name__ == "__main__":
    main()
