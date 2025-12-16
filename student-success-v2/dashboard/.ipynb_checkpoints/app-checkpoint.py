# dashboard/app.py
# Fixed Streamlit Dashboard for Ashesi Student Success Prediction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

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
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS (Outside class to work with caching)
# ============================================================================

@st.cache_data(ttl=3600)
def load_master_data():
    """Load master student data"""
    try:
        df = pd.read_csv('data/processed/master_student_data.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_data(ttl=3600)
def load_semester_data():
    """Load semester records"""
    try:
        df = pd.read_csv('data/processed/semester_records.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_dir = 'models/'
    
    if not os.path.exists(model_dir):
        return models
    
    try:
        import joblib
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            task_key = model_file.replace('_model.joblib', '')
            
            try:
                model_data = {
                    'model': joblib.load(f'{model_dir}{task_key}_model.joblib')
                }
                
                # Load scaler if exists
                scaler_path = f'{model_dir}{task_key}_scaler.joblib'
                if os.path.exists(scaler_path):
                    model_data['scaler'] = joblib.load(scaler_path)
                
                # Load features if exists
                features_path = f'{model_dir}{task_key}_features.json'
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        model_data['features'] = json.load(f)
                
                models[task_key] = model_data
                
            except Exception as e:
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
        # Load data using cached functions
        self.master_df = load_master_data()
        self.semester_records = load_semester_data()
        self.models = load_models()
        
        # Show loading status in sidebar
        if self.master_df is not None:
            st.sidebar.success(f"✓ Loaded {len(self.master_df):,} students")
        else:
            st.sidebar.error("❌ Master data not found")
        
        if self.semester_records is not None:
            st.sidebar.info(f"✓ Loaded {len(self.semester_records):,} semester records")
        
        if self.models:
            st.sidebar.info(f"✓ Loaded {len(self.models)} models")
    
    def run(self):
        """Run the dashboard"""
        
        # Sidebar navigation
        st.sidebar.title("🎓 Navigation")
        
        page = st.sidebar.radio(
            "Select Page",
            [
                "📊 Overview",
                "🔍 Student Lookup",
                "⚠️ Risk Analysis",
                "📅 Cohort Analysis",
                "🎓 Major Analysis",
                "📐 Math Track Analysis",
                "🤖 Model Performance",
                "🔮 Predictions"
            ]
        )
        
        # Check if data is loaded
        if self.master_df is None or len(self.master_df) == 0:
            st.error("""
            ## ❌ No Data Loaded
            
            Please run the data pipeline first:
            ```bash
            python run_pipeline.py --mode full
            ```
            
            Or ensure that `data/processed/master_student_data.csv` exists.
            """)
            return
        
        # Route to appropriate page
        if "Overview" in page:
            self.show_overview()
        elif "Student Lookup" in page:
            self.show_student_lookup()
        elif "Risk Analysis" in page:
            self.show_risk_analysis()
        elif "Cohort Analysis" in page:
            self.show_cohort_analysis()
        elif "Major Analysis" in page:
            self.show_major_analysis()
        elif "Math Track" in page:
            self.show_math_track_analysis()
        elif "Model Performance" in page:
            self.show_model_performance()
        elif "Predictions" in page:
            self.show_predictions()
    
    def get_program_column(self):
        """Get the appropriate program/major column name"""
        for col in ['final_program', 'Program', 'Offer course name']:
            if col in self.master_df.columns:
                return col
        return None
    
    def get_yeargroup_column(self):
        """Get the appropriate year group column name"""
        for col in ['Yeargroup', 'yeargroup', 'Year Group']:
            if col in self.master_df.columns:
                return col
        return None
    
    def show_overview(self):
        """Show overview page"""
        
        st.markdown('<h1 class="main-header">🎓 Ashesi Student Success Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Students",
                f"{len(self.master_df):,}",
                help="Total number of students in the dataset"
            )
        
        with col2:
            if 'first_year_struggle' in self.master_df.columns:
                struggle_rate = self.master_df['first_year_struggle'].mean() * 100
                st.metric(
                    "Struggle Rate",
                    f"{struggle_rate:.1f}%",
                    delta=f"{struggle_rate - 15:.1f}%" if struggle_rate > 15 else None,
                    delta_color="inverse"
                )
            else:
                st.metric("Struggle Rate", "N/A")
        
        with col3:
            if 'major_success' in self.master_df.columns:
                success_rate = self.master_df['major_success'].mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            else:
                st.metric("Success Rate", "N/A")
        
        with col4:
            if 'has_ajc_case' in self.master_df.columns:
                ajc_rate = self.master_df['has_ajc_case'].mean() * 100
                st.metric("AJC Case Rate", f"{ajc_rate:.1f}%")
            else:
                st.metric("AJC Case Rate", "N/A")
        
        with col5:
            if 'extended_graduation' in self.master_df.columns:
                ext_rate = self.master_df['extended_graduation'].mean() * 100
                st.metric("Extended Grad", f"{ext_rate:.1f}%")
            else:
                st.metric("Extended Grad", "N/A")
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 CGPA Distribution")
            
            if 'final_cgpa' in self.master_df.columns:
                cgpa_data = self.master_df['final_cgpa'].dropna()
                
                if len(cgpa_data) > 0:
                    fig = px.histogram(
                        cgpa_data, 
                        nbins=30,
                        color_discrete_sequence=['#3498db']
                    )
                    fig.add_vline(x=2.0, line_dash="dash", line_color="red", 
                                 annotation_text="Probation (2.0)")
                    fig.add_vline(x=3.0, line_dash="dash", line_color="orange",
                                 annotation_text="Success (3.0)")
                    fig.add_vline(x=3.5, line_dash="dash", line_color="green", 
                                 annotation_text="Dean's List (3.5)")
                    fig.update_layout(
                        xaxis_title="Final CGPA",
                        yaxis_title="Count",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No CGPA data available")
            else:
                st.info("CGPA column not found in data")
        
        with col2:
            st.subheader("🎯 Performance by Major")
            
            program_col = self.get_program_column()
            
            if program_col and 'final_cgpa' in self.master_df.columns:
                # Filter valid data
                valid_data = self.master_df[[program_col, 'final_cgpa']].dropna()
                
                if len(valid_data) > 0:
                    major_perf = valid_data.groupby(program_col).agg({
                        'final_cgpa': ['mean', 'count']
                    }).reset_index()
                    major_perf.columns = ['program', 'avg_cgpa', 'count']
                    major_perf = major_perf[major_perf['count'] >= 10]
                    major_perf = major_perf.sort_values('avg_cgpa', ascending=True)
                    
                    if len(major_perf) > 0:
                        fig = px.bar(
                            major_perf,
                            x='avg_cgpa',
                            y='program',
                            orientation='h',
                            color='avg_cgpa',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.add_vline(x=3.0, line_dash="dash", line_color="green")
                        fig.update_layout(
                            xaxis_title="Average CGPA",
                            yaxis_title="",
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data per major (min 10 students)")
                else:
                    st.info("No valid program/CGPA data")
            else:
                st.info("Program or CGPA data not available")
        
        # Risk distribution row
        st.markdown("---")
        st.subheader("📊 Risk Distribution Overview")
        
        col1, col2, col3 = st.columns(3)
        
        risk_configs = [
            ('first_year_struggle', 'First Year Struggle', col1),
            ('has_ajc_case', 'AJC Cases', col2),
            ('extended_graduation', 'Extended Graduation', col3)
        ]
        
        for risk_col, title, col in risk_configs:
            with col:
                if risk_col in self.master_df.columns:
                    risk_data = self.master_df[risk_col].dropna()
                    
                    if len(risk_data) > 0:
                        counts = risk_data.value_counts()
                        
                        fig = px.pie(
                            values=counts.values,
                            names=['No Risk' if i == 0 else 'At Risk' for i in counts.index],
                            title=title,
                            color_discrete_sequence=['#27ae60', '#e74c3c']
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No {title} data")
                else:
                    st.info(f"{title} not available")
        
        # Data summary
        st.markdown("---")
        st.subheader("📋 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Information**")
            st.write(f"- Total records: {len(self.master_df):,}")
            st.write(f"- Total features: {len(self.master_df.columns)}")
            
            missing_cells = self.master_df.isnull().sum().sum()
            total_cells = len(self.master_df) * len(self.master_df.columns)
            missing_pct = (missing_cells / total_cells) * 100
            st.write(f"- Missing data: {missing_pct:.1f}%")
            
            if 'exam_source' in self.master_df.columns:
                st.markdown("**Exam Type Distribution**")
                exam_dist = self.master_df['exam_source'].value_counts()
                for exam, count in exam_dist.head(5).items():
                    st.write(f"- {exam}: {count:,} ({count/len(self.master_df)*100:.1f}%)")
        
        with col2:
            yg_col = self.get_yeargroup_column()
            
            if yg_col:
                st.markdown("**Cohort Distribution**")
                yg_data = self.master_df[yg_col].dropna()
                
                if len(yg_data) > 0:
                    yg_dist = yg_data.value_counts().sort_index()
                    
                    fig = px.bar(
                        x=yg_dist.index.astype(str),
                        y=yg_dist.values,
                        labels={'x': 'Year Group', 'y': 'Count'}
                    )
                    fig.update_layout(xaxis_title="Year Group", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_student_lookup(self):
        """Individual student lookup and analysis"""
        
        st.header("🔍 Student Lookup")
        
        # Search options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_method = st.radio(
                "Search by:",
                ["Student ID", "Year Group"],
                horizontal=True
            )
        
        student = None
        
        if search_method == "Student ID":
            # Get list of student IDs for autocomplete
            student_ids = self.master_df['student_id'].astype(str).tolist()
            
            student_id = st.text_input(
                "Enter Student ID:", 
                placeholder="e.g., S7047A4E6DF5E8CF5"
            )
            
            if student_id:
                # Find matching students
                matches = self.master_df[
                    self.master_df['student_id'].astype(str).str.upper().str.contains(
                        student_id.upper(), na=False
                    )
                ]
                
                if len(matches) == 0:
                    st.warning(f"No student found matching '{student_id}'")
                elif len(matches) == 1:
                    student = matches.iloc[0]
                    st.success(f"Found student: {student['student_id']}")
                else:
                    st.info(f"Found {len(matches)} matching students")
                    selected_id = st.selectbox(
                        "Select student:", 
                        matches['student_id'].tolist()
                    )
                    student = matches[matches['student_id'] == selected_id].iloc[0]
        
        else:  # Year Group
            yg_col = self.get_yeargroup_column()
            
            if yg_col:
                year_groups = sorted(self.master_df[yg_col].dropna().unique())
                
                if len(year_groups) > 0:
                    selected_yg = st.selectbox("Select Year Group:", year_groups)
                    
                    yg_students = self.master_df[self.master_df[yg_col] == selected_yg]
                    
                    if len(yg_students) > 0:
                        selected_id = st.selectbox(
                            f"Select student from {selected_yg} ({len(yg_students)} students):",
                            yg_students['student_id'].tolist()
                        )
                        student = yg_students[yg_students['student_id'] == selected_id].iloc[0]
                else:
                    st.warning("No year group data available")
            else:
                st.warning("Year group column not found in data")
        
        # Display student information
        if student is not None:
            st.markdown("---")
            st.subheader(f"📋 Student Profile: {student['student_id']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                
                gender = student.get('Gender', 'N/A')
                st.write(f"🧑 Gender: {gender if pd.notna(gender) else 'N/A'}")
                
                nationality = student.get('Nationality', student.get('nationality_region', 'N/A'))
                st.write(f"🌍 Nationality: {nationality if pd.notna(nationality) else 'N/A'}")
                
                yg_col = self.get_yeargroup_column()
                yg_val = student.get(yg_col, 'N/A') if yg_col else 'N/A'
                st.write(f"📅 Year Group: {yg_val if pd.notna(yg_val) else 'N/A'}")
            
            with col2:
                st.markdown("**Academic**")
                
                program_col = self.get_program_column()
                program = student.get(program_col, 'N/A') if program_col else 'N/A'
                st.write(f"📚 Program: {program if pd.notna(program) else 'N/A'}")
                
                math_track = student.get('math_track', 'N/A')
                st.write(f"📐 Math Track: {math_track if pd.notna(math_track) else 'N/A'}")
                
                exam_type = student.get('exam_source', student.get('Exam Type', 'N/A'))
                st.write(f"📝 Exam Type: {exam_type if pd.notna(exam_type) else 'N/A'}")
                
                if 'standardized_score' in student.index:
                    score = student['standardized_score']
                    if pd.notna(score):
                        st.write(f"📊 HS Score: {score:.1f}")
            
            with col3:
                st.markdown("**Performance**")
                
                if 'final_cgpa' in student.index:
                    cgpa = student['final_cgpa']
                    if pd.notna(cgpa):
                        st.write(f"🎯 Final CGPA: {cgpa:.2f}")
                    else:
                        st.write("🎯 Final CGPA: N/A")
                
                status = student.get('student_status', student.get('Student Status', 'N/A'))
                st.write(f"📋 Status: {status if pd.notna(status) else 'N/A'}")
                
                if 'total_semesters' in student.index:
                    semesters = student['total_semesters']
                    if pd.notna(semesters):
                        st.write(f"⏱️ Semesters: {int(semesters)}")
            
            # Risk Assessment
            st.markdown("---")
            st.subheader("⚠️ Risk Assessment")
            
            col1, col2, col3, col4 = st.columns(4)
            
            risk_items = [
                ('first_year_struggle', 'First Year Struggle', col1),
                ('has_ajc_case', 'AJC Case', col2),
                ('major_struggle', 'Major Struggle', col3),
                ('extended_graduation', 'Extended Graduation', col4)
            ]
            
            for risk_col, label, col in risk_items:
                with col:
                    st.markdown(f"**{label}**")
                    
                    if risk_col in student.index:
                        val = student[risk_col]
                        if pd.notna(val):
                            is_risk = int(val) == 1
                            if is_risk:
                                st.markdown("🔴 **Yes**")
                            else:
                                st.markdown("🟢 No")
                        else:
                            st.markdown("⚪ N/A")
                    else:
                        st.markdown("⚪ N/A")
            
            # GPA Trajectory
            if self.semester_records is not None and len(self.semester_records) > 0:
                student_semesters = self.semester_records[
                    self.semester_records['student_id'].astype(str) == str(student['student_id'])
                ]
                
                if len(student_semesters) > 0:
                    st.markdown("---")
                    st.subheader("📈 GPA Trajectory")
                    
                    # Sort by semester
                    if 'semester_order' in student_semesters.columns:
                        student_semesters = student_semesters.sort_values('semester_order')
                        x_vals = student_semesters['semester_order'].values
                    else:
                        x_vals = list(range(1, len(student_semesters) + 1))
                    
                    fig = go.Figure()
                    
                    if 'GPA' in student_semesters.columns:
                        gpa_vals = student_semesters['GPA'].values
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=gpa_vals,
                            mode='lines+markers',
                            name='Semester GPA',
                            line=dict(color='#3498db', width=2),
                            marker=dict(size=10)
                        ))
                    
                    if 'CGPA' in student_semesters.columns:
                        cgpa_vals = student_semesters['CGPA'].values
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=cgpa_vals,
                            mode='lines+markers',
                            name='Cumulative GPA',
                            line=dict(color='#9b59b6', width=2),
                            marker=dict(size=10)
                        ))
                    
                    # Reference lines
                    fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                                 annotation_text="Probation (2.0)")
                    fig.add_hline(y=3.5, line_dash="dash", line_color="green",
                                 annotation_text="Dean's List (3.5)")
                    
                    fig.update_layout(
                        xaxis_title="Semester",
                        yaxis_title="GPA",
                        yaxis_range=[0, 4.2],
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show semester details
                    with st.expander("📋 Semester Details"):
                        display_cols = ['Academic Year', 'Semester/Year', 'GPA', 'CGPA', 'Program']
                        display_cols = [c for c in display_cols if c in student_semesters.columns]
                        st.dataframe(student_semesters[display_cols], hide_index=True)
    
    def show_risk_analysis(self):
        """Show risk analysis page"""
        
        st.header("⚠️ Risk Analysis")
        
        # Risk type selection
        risk_options = {
            "First Year Struggle": "first_year_struggle",
            "AJC Case": "has_ajc_case",
            "Major Struggle": "major_struggle",
            "Extended Graduation": "extended_graduation"
        }
        
        available_risks = {k: v for k, v in risk_options.items() 
                          if v in self.master_df.columns}
        
        if not available_risks:
            st.warning("No risk indicators available in the data")
            st.info("Available columns: " + ", ".join(self.master_df.columns.tolist()[:20]))
            return
        
        # Sidebar filters
        st.sidebar.subheader("🔧 Filters")
        
        selected_risk = st.sidebar.selectbox("Risk Type", list(available_risks.keys()))
        risk_col = available_risks[selected_risk]
        
        # Year group filter
        yg_col = self.get_yeargroup_column()
        selected_yg = 'All'
        
        if yg_col:
            year_groups = ['All'] + sorted([
                str(yg) for yg in self.master_df[yg_col].dropna().unique()
            ])
            selected_yg = st.sidebar.selectbox("Year Group", year_groups)
        
        # Program filter
        program_col = self.get_program_column()
        selected_program = 'All'
        
        if program_col:
            programs = ['All'] + sorted([
                str(p) for p in self.master_df[program_col].dropna().unique()
            ])
            selected_program = st.sidebar.selectbox("Program", programs)
        
        # Apply filters
        filtered_df = self.master_df.copy()
        
        if selected_yg != 'All' and yg_col:
            filtered_df = filtered_df[filtered_df[yg_col].astype(str) == selected_yg]
        
        if selected_program != 'All' and program_col:
            filtered_df = filtered_df[filtered_df[program_col].astype(str) == selected_program]
        
        # Remove nulls for risk column
        filtered_df = filtered_df[filtered_df[risk_col].notna()]
        
        if len(filtered_df) == 0:
            st.warning("No data available with current filters")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(filtered_df)
        at_risk = int(filtered_df[risk_col].sum())
        risk_rate = (at_risk / total * 100) if total > 0 else 0
        
        with col1:
            st.metric("Total Students", f"{total:,}")
        with col2:
            st.metric("At Risk", f"{at_risk:,}")
        with col3:
            st.metric("Risk Rate", f"{risk_rate:.1f}%")
        with col4:
            st.metric("Not At Risk", f"{total - at_risk:,}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk by Program")
            
            if program_col:
                risk_by_program = filtered_df.groupby(program_col)[risk_col].agg(['sum', 'count', 'mean'])
                risk_by_program.columns = ['at_risk', 'total', 'rate']
                risk_by_program = risk_by_program[risk_by_program['total'] >= 5]
                risk_by_program = risk_by_program.sort_values('rate', ascending=True)
                
                if len(risk_by_program) > 0:
                    fig = px.bar(
                        risk_by_program.reset_index(),
                        x='rate',
                        y=program_col,
                        orientation='h',
                        color='rate',
                        color_continuous_scale='RdYlGn_r',
                        labels={'rate': 'Risk Rate', program_col: 'Program'}
                    )
                    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data per program")
            else:
                st.info("Program data not available")
        
        with col2:
            st.subheader("Risk by Performance Tier")
            
            if 'performance_tier' in filtered_df.columns:
                risk_by_tier = filtered_df.groupby('performance_tier')[risk_col].mean()
                
                tier_order = ['Excellent', 'Good', 'Average', 'Below Average', 'At Risk', 'Unknown']
                risk_by_tier = risk_by_tier.reindex([t for t in tier_order if t in risk_by_tier.index])
                
                if len(risk_by_tier) > 0:
                    fig = px.bar(
                        x=risk_by_tier.index,
                        y=risk_by_tier.values,
                        color=risk_by_tier.values,
                        color_continuous_scale='RdYlGn_r',
                        labels={'x': 'Performance Tier', 'y': 'Risk Rate'}
                    )
                    fig.update_layout(coloraxis_showscale=False, xaxis_title="Performance Tier")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Performance tier data not available")
        
        # At-risk student table
        st.markdown("---")
        st.subheader("🚨 At-Risk Students")
        
        at_risk_students = filtered_df[filtered_df[risk_col] == 1].copy()
        
        if len(at_risk_students) > 0:
            # Select display columns
            display_cols = ['student_id']
            
            optional_cols = [
                'Gender', yg_col, program_col, 'final_cgpa', 
                'standardized_score', 'math_track', 'exam_source'
            ]
            
            for col in optional_cols:
                if col and col in at_risk_students.columns:
                    display_cols.append(col)
            
            # Remove duplicates while preserving order
            display_cols = list(dict.fromkeys(display_cols))
            
            # Sort by CGPA if available
            if 'final_cgpa' in at_risk_students.columns:
                at_risk_students = at_risk_students.sort_values('final_cgpa')
            
            st.write(f"Showing {min(100, len(at_risk_students))} of {len(at_risk_students)} at-risk students")
            
            st.dataframe(
                at_risk_students[display_cols].head(100),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = at_risk_students[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Full At-Risk List (CSV)",
                data=csv,
                file_name=f"at_risk_{risk_col}_{selected_yg}_{selected_program}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 No at-risk students in the selected group!")
    
    def show_cohort_analysis(self):
        """Show cohort analysis page"""
        
        st.header("📅 Cohort Analysis")
        
        yg_col = self.get_yeargroup_column()
        
        if not yg_col:
            st.warning("Year group data not available in the dataset")
            return
        
        # Get available cohorts
        cohorts = sorted(self.master_df[yg_col].dropna().unique())
        
        if len(cohorts) == 0:
            st.warning("No cohort data available")
            return
        
        # Cohort selection
        default_cohorts = cohorts[-3:] if len(cohorts) >= 3 else cohorts
        
        selected_cohorts = st.multiselect(
            "Select Cohorts to Compare",
            options=cohorts,
            default=default_cohorts
        )
        
        if not selected_cohorts:
            st.warning("Please select at least one cohort")
            return
        
        cohort_df = self.master_df[self.master_df[yg_col].isin(selected_cohorts)].copy()
        
        if len(cohort_df) == 0:
            st.warning("No data for selected cohorts")
            return
        
        # Build aggregation dictionary
        agg_dict = {'student_id': 'count'}
        
        if 'final_cgpa' in cohort_df.columns:
            agg_dict['final_cgpa'] = 'mean'
        if 'first_year_struggle' in cohort_df.columns:
            agg_dict['first_year_struggle'] = 'mean'
        if 'has_ajc_case' in cohort_df.columns:
            agg_dict['has_ajc_case'] = 'mean'
        if 'major_success' in cohort_df.columns:
            agg_dict['major_success'] = 'mean'
        if 'extended_graduation' in cohort_df.columns:
            agg_dict['extended_graduation'] = 'mean'
        
        cohort_metrics = cohort_df.groupby(yg_col).agg(agg_dict).round(3)
        
        # Rename columns
        rename_map = {
            'student_id': 'Students',
            'final_cgpa': 'Avg CGPA',
            'first_year_struggle': 'Struggle Rate',
            'has_ajc_case': 'AJC Rate',
            'major_success': 'Success Rate',
            'extended_graduation': 'Extended Grad Rate'
        }
        cohort_metrics = cohort_metrics.rename(columns=rename_map)
        
        st.subheader("📊 Cohort Comparison")
        st.dataframe(cohort_metrics, use_container_width=True)
        
        # Trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Avg CGPA' in cohort_metrics.columns:
                st.subheader("CGPA Trend")
                
                fig = px.line(
                    cohort_metrics.reset_index(),
                    x=yg_col,
                    y='Avg CGPA',
                    markers=True,
                    labels={yg_col: 'Year Group'}
                )
                fig.add_hline(y=3.0, line_dash="dash", line_color="green",
                             annotation_text="Success (3.0)")
                fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                             annotation_text="Probation (2.0)")
                fig.update_layout(yaxis_range=[1.5, 4])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Trends")
            
            risk_cols = ['Struggle Rate', 'AJC Rate', 'Extended Grad Rate']
            available_risk_cols = [c for c in risk_cols if c in cohort_metrics.columns]
            
            if available_risk_cols:
                fig = go.Figure()
                
                colors = ['#e74c3c', '#f39c12', '#9b59b6']
                
                for i, col in enumerate(available_risk_cols):
                    fig.add_trace(go.Scatter(
                        x=cohort_metrics.index.astype(str),
                        y=cohort_metrics[col] * 100,  # Convert to percentage
                        mode='lines+markers',
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    xaxis_title="Year Group",
                    yaxis_title="Rate (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk metrics available")
        
        # CGPA distribution
        st.markdown("---")
        st.subheader("📈 CGPA Distribution by Cohort")
        
        if 'final_cgpa' in cohort_df.columns:
            cgpa_data = cohort_df.dropna(subset=['final_cgpa'])
            
            if len(cgpa_data) > 0:
                fig = px.violin(
                    cgpa_data,
                    x=yg_col,
                    y='final_cgpa',
                    color=yg_col,
                    box=True,
                    labels={yg_col: 'Year Group', 'final_cgpa': 'Final CGPA'}
                )
                fig.add_hline(y=2.0, line_dash="dash", line_color="red")
                fig.add_hline(y=3.5, line_dash="dash", line_color="green")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    def show_major_analysis(self):
        """Show major-specific analysis"""
        
        st.header("🎓 Major Analysis")
        
        program_col = self.get_program_column()
        
        if not program_col:
            st.warning("Program/Major data not available")
            return
        
        # Get majors with sufficient data
        major_counts = self.master_df[program_col].value_counts()
        valid_majors = major_counts[major_counts >= 10].index.tolist()
        
        if not valid_majors:
            st.warning("No majors with sufficient data (minimum 10 students)")
            return
        
        selected_major = st.selectbox(
            "Select Major",
            options=valid_majors,
            format_func=lambda x: f"{x} ({major_counts[x]} students)"
        )
        
        major_df = self.master_df[self.master_df[program_col] == selected_major].copy()
        
        # Overview metrics
        st.subheader(f"📊 {selected_major}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(major_df))
        
        with col2:
            if 'final_cgpa' in major_df.columns:
                avg_cgpa = major_df['final_cgpa'].mean()
                st.metric("Avg CGPA", f"{avg_cgpa:.2f}" if pd.notna(avg_cgpa) else "N/A")
            else:
                st.metric("Avg CGPA", "N/A")
        
        with col3:
            if 'major_success' in major_df.columns:
                success_rate = major_df['major_success'].mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            else:
                st.metric("Success Rate", "N/A")
        
        with col4:
            if 'first_year_struggle' in major_df.columns:
                struggle_rate = major_df['first_year_struggle'].mean() * 100
                st.metric("Struggle Rate", f"{struggle_rate:.1f}%")
            else:
                st.metric("Struggle Rate", "N/A")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CGPA Distribution")
            
            if 'final_cgpa' in major_df.columns:
                cgpa_data = major_df['final_cgpa'].dropna()
                
                if len(cgpa_data) > 0:
                    fig = px.histogram(
                        cgpa_data,
                        nbins=20,
                        color_discrete_sequence=['#3498db']
                    )
                    fig.add_vline(x=2.0, line_dash="dash", line_color="red")
                    fig.add_vline(x=3.0, line_dash="dash", line_color="orange")
                    fig.add_vline(x=3.5, line_dash="dash", line_color="green")
                    fig.update_layout(xaxis_title="CGPA", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No CGPA data available")
            else:
                st.info("CGPA data not available")
        
        with col2:
            st.subheader("Performance by Math Track")
            
            if 'math_track' in major_df.columns and 'final_cgpa' in major_df.columns:
                valid_data = major_df[
                    (major_df['math_track'].notna()) & 
                    (major_df['math_track'] != 'Unknown') &
                    (major_df['final_cgpa'].notna())
                ]
                
                if len(valid_data) >= 5:
                    track_perf = valid_data.groupby('math_track')['final_cgpa'].agg(['mean', 'count'])
                    track_perf = track_perf[track_perf['count'] >= 3]
                    
                    if len(track_perf) > 0:
                        fig = px.bar(
                            x=track_perf.index,
                            y=track_perf['mean'],
                            color=track_perf['mean'],
                            color_continuous_scale='RdYlGn',
                            labels={'x': 'Math Track', 'y': 'Average CGPA'}
                        )
                        fig.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data by math track")
                else:
                    st.info("Insufficient data for math track analysis")
            else:
                st.info("Math track or CGPA data not available")
        
        # Risk indicators
        st.markdown("---")
        st.subheader("📊 Risk Indicators")
        
        risk_cols = {
            'first_year_struggle': 'First Year Struggle',
            'has_ajc_case': 'AJC Cases',
            'extended_graduation': 'Extended Graduation'
        }
        
        available_risks = {v: major_df[k].mean() * 100 
                          for k, v in risk_cols.items() 
                          if k in major_df.columns}
        
        if available_risks:
            fig = px.bar(
                x=list(available_risks.keys()),
                y=list(available_risks.values()),
                color=list(available_risks.values()),
                color_continuous_scale='RdYlGn_r',
                labels={'x': 'Risk Type', 'y': 'Rate (%)'}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk indicators available")
    
    def show_math_track_analysis(self):
        """Q7 & Q8: Math track analysis"""
        
        st.header("📐 Math Track Analysis")
        
        st.markdown("""
        This analysis addresses two key research questions:
        - **Q7**: Is there a significant difference in performance between students on different math tracks?
        - **Q8**: Can students on the College Algebra track succeed in Computer Science?
        """)
        
        if 'math_track' not in self.master_df.columns:
            st.error("❌ Math track data not available in the dataset")
            st.info("Available columns: " + ", ".join(sorted(self.master_df.columns.tolist())))
            return
        
        # Filter valid math track data
        df = self.master_df[
            (self.master_df['math_track'].notna()) & 
            (self.master_df['math_track'] != 'Unknown')
        ].copy()
        
        if len(df) < 30:
            st.warning(f"Insufficient data for analysis ({len(df)} students with math track info)")
            return
        
        st.info(f"📊 Analyzing {len(df)} students with math track information")
        
        # ==================== Q7 Analysis ====================
        st.markdown("---")
        st.subheader("📊 Q7: Performance by Math Track")
        
        # Summary statistics
        summary_agg = {'student_id': 'count'}
        
        if 'final_cgpa' in df.columns:
            summary_agg['final_cgpa'] = ['mean', 'std', 'median']
        if 'first_year_struggle' in df.columns:
            summary_agg['first_year_struggle'] = 'mean'
        if 'major_success' in df.columns:
            summary_agg['major_success'] = 'mean'
        if 'extended_graduation' in df.columns:
            summary_agg['extended_graduation'] = 'mean'
        
        track_summary = df.groupby('math_track').agg(summary_agg).round(3)
        
        # Flatten multi-level columns
        if isinstance(track_summary.columns, pd.MultiIndex):
            track_summary.columns = ['_'.join(col).strip('_') for col in track_summary.columns]
        
        st.dataframe(track_summary, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'final_cgpa' in df.columns:
                cgpa_data = df.dropna(subset=['final_cgpa'])
                
                if len(cgpa_data) > 0:
                    fig = px.box(
                        cgpa_data,
                        x='math_track',
                        y='final_cgpa',
                        color='math_track',
                        title="CGPA Distribution by Math Track"
                    )
                    fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                                 annotation_text="Probation")
                    fig.add_hline(y=3.0, line_dash="dash", line_color="green",
                                 annotation_text="Success")
                    fig.update_layout(showlegend=False, xaxis_title="Math Track")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'first_year_struggle' in df.columns:
                struggle_by_track = df.groupby('math_track')['first_year_struggle'].mean() * 100
                
                fig = px.bar(
                    x=struggle_by_track.index,
                    y=struggle_by_track.values,
                    color=struggle_by_track.values,
                    color_continuous_scale='RdYlGn_r',
                    title="First Year Struggle Rate by Math Track",
                    labels={'x': 'Math Track', 'y': 'Struggle Rate (%)'}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistical test
        st.markdown("---")
        st.subheader("📈 Statistical Test (Kruskal-Wallis)")
        
        if 'final_cgpa' in df.columns:
            try:
                from scipy.stats import kruskal
                
                tracks = df['math_track'].unique()
                groups = [
                    df[df['math_track'] == t]['final_cgpa'].dropna().values 
                    for t in tracks
                ]
                groups = [g for g in groups if len(g) >= 5]
                
                if len(groups) >= 2:
                    stat, p_value = kruskal(*groups)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("H-Statistic", f"{stat:.3f}")
                    with col2:
                        st.metric("p-value", f"{p_value:.4f}")
                    with col3:
                        significant = p_value < 0.05
                        st.metric("Significant (α=0.05)", "Yes ✓" if significant else "No")
                    
                    if significant:
                        st.success(
                            "✅ **Conclusion**: There IS a statistically significant difference "
                            "in academic performance across different math tracks."
                        )
                    else:
                        st.info(
                            "ℹ️ **Conclusion**: No statistically significant difference was found "
                            "in academic performance across math tracks."
                        )
                else:
                    st.warning("Not enough groups with sufficient data for statistical test")
                    
            except ImportError:
                st.warning("scipy not installed - statistical test not available")
            except Exception as e:
                st.error(f"Error performing statistical test: {e}")
        
        # ==================== Q8 Analysis ====================
        st.markdown("---")
        st.subheader("💻 Q8: College Algebra in Computer Science")
        
        program_col = self.get_program_column()
        
        if not program_col:
            st.warning("Program data not available for CS analysis")
            return
        
        # Find CS students
        cs_mask = self.master_df[program_col].astype(str).str.contains(
            'Computer Science|Computer Engineering|CS', 
            case=False, 
            na=False
        )
        cs_df = self.master_df[cs_mask].copy()
        
        if len(cs_df) < 10:
            st.warning(f"Insufficient CS students ({len(cs_df)}) for analysis")
            return
        
        st.write(f"**Total Computer Science students:** {len(cs_df)}")
        
        # CS students with math track
        cs_with_track = cs_df[
            (cs_df['math_track'].notna()) & 
            (cs_df['math_track'] != 'Unknown')
        ]
        
        if len(cs_with_track) < 5:
            st.warning("Not enough CS students with math track information")
            return
        
        # Summary by track
        cs_agg = {'student_id': 'count'}
        if 'final_cgpa' in cs_with_track.columns:
            cs_agg['final_cgpa'] = 'mean'
        if 'major_success' in cs_with_track.columns:
            cs_agg['major_success'] = 'mean'
        if 'first_year_struggle' in cs_with_track.columns:
            cs_agg['first_year_struggle'] = 'mean'
        
        cs_by_track = cs_with_track.groupby('math_track').agg(cs_agg).round(3)
        cs_by_track.columns = ['Count', 'Avg CGPA', 'Success Rate', 'Struggle Rate'][:len(cs_by_track.columns)]
        
        st.dataframe(cs_by_track, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'major_success' in cs_with_track.columns:
                success_by_track = cs_with_track.groupby('math_track')['major_success'].mean()
                
                fig = px.bar(
                    x=success_by_track.index,
                    y=success_by_track.values,
                    color=success_by_track.values,
                    color_continuous_scale='RdYlGn',
                    title="CS Success Rate by Math Track",
                    labels={'x': 'Math Track', 'y': 'Success Rate'}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'final_cgpa' in cs_with_track.columns:
                cgpa_data = cs_with_track.dropna(subset=['final_cgpa'])
                
                if len(cgpa_data) > 0:
                    fig = px.box(
                        cgpa_data,
                        x='math_track',
                        y='final_cgpa',
                        color='math_track',
                        title="CS Student CGPA by Math Track"
                    )
                    fig.add_hline(y=3.0, line_dash="dash", line_color="green")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Conclusion for Q8
        st.markdown("---")
        st.subheader("📋 Q8 Conclusion")
        
        ca_cs = cs_with_track[cs_with_track['math_track'] == 'College Algebra']
        
        if len(ca_cs) >= 3:
            if 'major_success' in ca_cs.columns:
                ca_success_rate = ca_cs['major_success'].mean()
                ca_avg_cgpa = ca_cs['final_cgpa'].mean() if 'final_cgpa' in ca_cs.columns else None
                
                if ca_success_rate >= 0.5:
                    st.success(f"""
                    ✅ **YES**, College Algebra students CAN succeed in Computer Science!
                    
                    📊 **Statistics:**
                    - Sample size: **{len(ca_cs)}** students
                    - Success rate: **{ca_success_rate*100:.1f}%**
                    {f'- Average CGPA: **{ca_avg_cgpa:.2f}**' if ca_avg_cgpa else ''}
                    
                    💡 **Recommendation:** While success is possible, additional math support 
                    is recommended for optimal outcomes.
                    """)
                else:
                    st.warning(f"""
                    ⚠️ College Algebra students face **challenges** in Computer Science
                    
                    📊 **Statistics:**
                    - Sample size: **{len(ca_cs)}** students
                    - Success rate: **{ca_success_rate*100:.1f}%**
                    {f'- Average CGPA: **{ca_avg_cgpa:.2f}**' if ca_avg_cgpa else ''}
                    
                    💡 **Recommendation:** Implement targeted math support programs and 
                    consider prerequisite courses before core CS classes.
                    """)
            else:
                st.info(f"College Algebra CS students: {len(ca_cs)} (success rate data not available)")
        else:
            st.info(f"""
            ℹ️ **Insufficient data** for definitive conclusion
            
            Only {len(ca_cs)} College Algebra students found in CS program.
            Minimum 3 students required for analysis.
            """)
    
    def show_model_performance(self):
        """Show model performance page"""
        
        st.header("🤖 Model Performance")
        
        # Check for models
        if not self.models:
            st.warning("""
            ⚠️ No trained models found. 
            
            Run the pipeline first:
            ```bash
            python run_pipeline.py --mode full
            ```
            """)
        else:
            st.success(f"✓ {len(self.models)} models loaded")
            
            for task_key, model_info in self.models.items():
                with st.expander(f"📊 {task_key}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Model Type:**")
                        model = model_info.get('model')
                        if model:
                            st.code(type(model).__name__)
                        
                        features = model_info.get('features', [])
                        st.markdown(f"**Features Used:** {len(features)}")
                    
                    with col2:
                        if features:
                            st.markdown("**Top Features:**")
                            for i, feat in enumerate(features[:8], 1):
                                st.write(f"{i}. {feat}")
        
        # Show reports if available
        st.markdown("---")
        st.subheader("📄 Reports")
        
        reports = [
            ('reports/executive_summary.txt', 'Executive Summary'),
            ('reports/model_report.txt', 'Model Report'),
            ('reports/statistical_report.txt', 'Statistical Report')
        ]
        
        for report_path, report_name in reports:
            if os.path.exists(report_path):
                with st.expander(f"📄 {report_name}"):
                    with open(report_path, 'r') as f:
                        st.text(f.read())
            else:
                st.info(f"{report_name} not found at {report_path}")
    
    def show_predictions(self):
        """Make predictions for new students"""
        
        st.header("🔮 Student Risk Prediction")
        
        st.markdown("""
        Enter student information to predict their risk levels. 
        This tool estimates risk based on the patterns found in historical data.
        """)
        
        # Check if models are available
        has_models = bool(self.models)
        
        if not has_models:
            st.info("ℹ️ No trained models loaded. Using simplified estimation based on historical patterns.")
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("📝 Enter Student Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                hs_score = st.slider(
                    "High School Score (Standardized 0-100)",
                    min_value=0,
                    max_value=100,
                    value=70,
                    help="Higher is better. For WASSCE, this is converted from grades."
                )
                
                math_score = st.slider(
                    "Math Score (0-100 or 1-9 for WASSCE)",
                    min_value=0,
                    max_value=100,
                    value=70
                )
                
                english_score = st.slider(
                    "English Score",
                    min_value=0,
                    max_value=100,
                    value=70
                )
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                
                math_track = st.selectbox(
                    "Math Track Placement",
                    ["Calculus", "Pre-Calculus", "College Algebra"],
                    help="Math placement based on entrance assessment"
                )
                
                exam_type = st.selectbox(
                    "High School Exam Type",
                    ["WASSCE", "IB Diploma", "A-Level", "French Bac", "Other"]
                )
            
            col3, col4 = st.columns(2)
            
            with col3:
                financial_aid = st.selectbox(
                    "Needs Financial Aid?",
                    ["No", "Yes"]
                )
                
                intended_major = st.selectbox(
                    "Intended Major",
                    [
                        "Computer Science",
                        "Business Administration",
                        "Management Information Systems",
                        "Engineering",
                        "Economics",
                        "Other"
                    ]
                )
            
            with col4:
                previous_app = st.selectbox(
                    "Applied to Ashesi Before?",
                    ["No", "Yes"]
                )
                
                family_connection = st.selectbox(
                    "Family Member Attended Ashesi?",
                    ["No", "Yes"]
                )
            
            submitted = st.form_submit_button("🔮 Predict Risk Levels", type="primary")
        
        # Process prediction
        if submitted:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Calculate risk scores
            # Base risk from academic performance
            base_risk = (100 - hs_score) / 100 * 0.5 + (100 - math_score) / 100 * 0.3 + (100 - english_score) / 100 * 0.2
            
            # Adjustments based on other factors
            adjustments = 0
            
            # Math track adjustment
            if math_track == "College Algebra":
                adjustments += 0.15
            elif math_track == "Pre-Calculus":
                adjustments += 0.05
            
            # Major + math track interaction
            if intended_major in ["Computer Science", "Engineering"] and math_track == "College Algebra":
                adjustments += 0.10
            
            # Financial aid (slight adjustment)
            if financial_aid == "Yes":
                adjustments += 0.03
            
            # Positive factors
            if previous_app == "Yes":
                adjustments -= 0.05  # Persistence is positive
            
            if family_connection == "Yes":
                adjustments -= 0.03  # Familiarity with institution
            
            # Calculate final risks
            first_year_risk = np.clip(base_risk + adjustments, 0.05, 0.95)
            ajc_risk = np.clip(base_risk * 0.4 + adjustments * 0.5, 0.02, 0.50)
            extended_risk = np.clip(base_risk * 0.8 + adjustments * 0.7, 0.05, 0.85)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self._render_risk_gauge("First Year Struggle", first_year_risk)
            
            with col2:
                self._render_risk_gauge("AJC Risk", ajc_risk, thresholds=(0.15, 0.30))
            
            with col3:
                self._render_risk_gauge("Extended Graduation", extended_risk)
            
            # Overall assessment
            st.markdown("---")
            
            overall_risk = (first_year_risk + ajc_risk * 2 + extended_risk) / 4
            
            if overall_risk < 0.25:
                st.success("""
                ### ✅ Low Risk Profile
                This student shows strong indicators for academic success.
                Standard support should be sufficient.
                """)
            elif overall_risk < 0.45:
                st.info("""
                ### ℹ️ Moderate Risk Profile
                This student may benefit from targeted support in specific areas.
                """)
            else:
                st.warning("""
                ### ⚠️ Higher Risk Profile
                This student would benefit from comprehensive support services.
                Consider early intervention programs.
                """)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            recommendations = []
            
            if first_year_risk > 0.4:
                recommendations.append("🎯 **Academic Support**: Enroll in peer tutoring and academic success workshops")
            
            if math_track == "College Algebra":
                recommendations.append("📐 **Math Preparation**: Consider supplementary math support or bridge courses")
            
            if intended_major in ["Computer Science", "Engineering"] and math_track != "Calculus":
                recommendations.append("💻 **STEM Readiness**: Connect with upper-year mentor in intended major")
            
            if financial_aid == "Yes":
                recommendations.append("💰 **Financial Planning**: Early meeting with financial aid office recommended")
            
            if extended_risk > 0.4:
                recommendations.append("📅 **Course Planning**: Meet with academic advisor for optimal course sequencing")
            
            if ajc_risk > 0.2:
                recommendations.append("📚 **Academic Integrity**: Attend academic integrity orientation workshop")
            
            if not recommendations:
                recommendations.append("✅ No specific interventions needed - standard orientation should suffice")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    def _render_risk_gauge(self, title, risk_value, thresholds=(0.30, 0.60)):
        """Render a risk gauge visualization"""
        
        low_thresh, high_thresh = thresholds
        
        # Determine risk level
        if risk_value > high_thresh:
            level = "HIGH"
            color = "#e74c3c"
            emoji = "🔴"
        elif risk_value > low_thresh:
            level = "MEDIUM"
            color = "#f39c12"
            emoji = "🟡"
        else:
            level = "LOW"
            color = "#27ae60"
            emoji = "🟢"
        
        st.markdown(f"### {title}")
        st.markdown(f"{emoji} **{level}**: {risk_value*100:.1f}%")
        
        # Progress bar
        st.progress(min(risk_value, 1.0))
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_value * 100,
            number={'suffix': '%', 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, low_thresh * 100], 'color': '#d4edda'},
                    {'range': [low_thresh * 100, high_thresh * 100], 'color': '#fff3cd'},
                    {'range': [high_thresh * 100, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': risk_value * 100
                }
            }
        ))
        
        fig.update_layout(
            height=180,
            margin=dict(t=10, b=10, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function to run the dashboard"""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()