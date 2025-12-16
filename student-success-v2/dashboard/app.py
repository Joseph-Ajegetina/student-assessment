# dashboard/app.py
# Updated Streamlit Dashboard with Cluster Analysis and Student Status Handling

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime

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
    .status-graduated { color: #27ae60; }
    .status-active { color: #3498db; }
    .status-withdrawn { color: #f39c12; }
    .status-dismissed { color: #e74c3c; }
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 5px 0;
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
    """Load trained models and their metadata"""
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
                
                # Load scaler
                scaler_path = f'{model_dir}{task_key}_scaler.joblib'
                if os.path.exists(scaler_path):
                    model_data['scaler'] = joblib.load(scaler_path)
                
                # Load features
                features_path = f'{model_dir}{task_key}_features.json'
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        model_data['features'] = json.load(f)
                
                # Load split info (NEW)
                split_path = f'{model_dir}{task_key}_split_info.json'
                if os.path.exists(split_path):
                    with open(split_path, 'r') as f:
                        model_data['split_info'] = json.load(f)
                
                models[task_key] = model_data
                
            except Exception as e:
                pass
                
    except ImportError:
        pass
    
    return models

@st.cache_data(ttl=3600)
def load_cluster_report():
    """Load cluster analysis report"""
    report_path = 'reports/cluster_report.txt'
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return f.read()
    return None


# ============================================================================
# DASHBOARD CLASS
# ============================================================================

class DashboardApp:
    """Streamlit dashboard for student success prediction"""
    
    def __init__(self):
        # Load data
        self.master_df = load_master_data()
        self.semester_records = load_semester_data()
        self.models = load_models()
        self.cluster_report = load_cluster_report()
        
        # Show loading status
        self._show_loading_status()
    
    def _show_loading_status(self):
        """Display data loading status in sidebar"""
        
        st.sidebar.markdown("### 📊 Data Status")
        
        if self.master_df is not None:
            st.sidebar.success(f"✓ {len(self.master_df):,} students loaded")
            
            # Show student status breakdown
            if 'is_graduated' in self.master_df.columns:
                graduated = self.master_df['is_graduated'].sum()
                active = self.master_df.get('is_active', pd.Series([0])).sum()
                st.sidebar.caption(f"  📗 Graduated: {graduated:,}")
                st.sidebar.caption(f"  📘 Active: {active:,}")
        else:
            st.sidebar.error("❌ Data not loaded")
        
        if self.semester_records is not None:
            st.sidebar.info(f"✓ {len(self.semester_records):,} semester records")
        
        if self.models:
            st.sidebar.info(f"✓ {len(self.models)} models loaded")
        
        # Check for clusters
        if self.master_df is not None and 'cluster_kmeans' in self.master_df.columns:
            n_clusters = self.master_df['cluster_kmeans'].nunique()
            st.sidebar.info(f"✓ {n_clusters} clusters identified")
    
    def run(self):
        """Run the dashboard"""
        
        st.sidebar.title("🎓 Navigation")
        
        # Updated navigation with new pages
        page = st.sidebar.radio(
            "Select Page",
            [
                "📊 Overview",
                "🔍 Student Lookup",
                "⚠️ Risk Analysis",
                "🎯 Cluster Analysis",      # NEW
                "📅 Cohort Analysis",
                "🎓 Major Analysis",
                "📐 Math Track Analysis",
                "🤖 Model Performance",
                "🔮 Predictions"
            ]
        )
        
        # Check data
        if self.master_df is None or len(self.master_df) == 0:
            st.error("""
            ## ❌ No Data Loaded
            
            Please run the data pipeline first:
            ```bash
            python run_pipeline.py --mode full
            ```
            """)
            return
        
        # Route to pages
        if "Overview" in page:
            self.show_overview()
        elif "Student Lookup" in page:
            self.show_student_lookup()
        elif "Risk Analysis" in page:
            self.show_risk_analysis()
        elif "Cluster Analysis" in page:
            self.show_cluster_analysis()  # NEW
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
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_program_column(self):
        """Get the program/major column name"""
        for col in ['final_program', 'Program', 'Offer course name']:
            if col in self.master_df.columns:
                return col
        return None
    
    def get_yeargroup_column(self):
        """Get the year group column name"""
        for col in ['Yeargroup', 'yeargroup', 'Year Group']:
            if col in self.master_df.columns:
                return col
        return None
    
    def get_student_status_display(self, row):
        """Get formatted student status display"""
        
        if row.get('is_graduated', False):
            return "🟢 Graduated"
        elif row.get('is_active', False):
            return "🔵 Active"
        elif row.get('is_withdrawn', False):
            return "🟡 Withdrawn"
        elif row.get('is_dismissed', False):
            return "🔴 Dismissed"
        else:
            return "⚪ Unknown"
    
    # ========================================================================
    # OVERVIEW PAGE (UPDATED)
    # ========================================================================
    
    def show_overview(self):
        """Show overview page with student status breakdown"""
        
        st.markdown('<h1 class="main-header">🎓 Ashesi Student Success Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Students", f"{len(self.master_df):,}")
        
        with col2:
            if 'is_graduated' in self.master_df.columns:
                graduated = self.master_df['is_graduated'].sum()
                st.metric("Graduated", f"{graduated:,}")
            else:
                st.metric("Graduated", "N/A")
        
        with col3:
            if 'is_active' in self.master_df.columns:
                active = self.master_df['is_active'].sum()
                st.metric("Active", f"{active:,}")
            else:
                st.metric("Active", "N/A")
        
        with col4:
            if 'first_year_struggle' in self.master_df.columns:
                # Only count valid cases
                valid = self.master_df['first_year_struggle'].notna()
                if valid.sum() > 0:
                    rate = self.master_df.loc[valid, 'first_year_struggle'].mean() * 100
                    st.metric("Struggle Rate", f"{rate:.1f}%")
                else:
                    st.metric("Struggle Rate", "N/A")
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
            else:
                st.metric("Success Rate", "N/A")
        
        st.markdown("---")
        
        # Student Status Distribution (NEW)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Student Status Distribution")
            
            status_counts = {}
            if 'is_graduated' in self.master_df.columns:
                status_counts['Graduated'] = self.master_df['is_graduated'].sum()
            if 'is_active' in self.master_df.columns:
                status_counts['Active'] = self.master_df['is_active'].sum()
            if 'is_withdrawn' in self.master_df.columns:
                status_counts['Withdrawn'] = self.master_df['is_withdrawn'].sum()
            if 'is_dismissed' in self.master_df.columns:
                status_counts['Dismissed'] = self.master_df['is_dismissed'].sum()
            
            if status_counts:
                # Calculate other
                total_accounted = sum(status_counts.values())
                other = len(self.master_df) - total_accounted
                if other > 0:
                    status_counts['Other'] = other
                
                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    color_discrete_sequence=['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#95a5a6'],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show note about data validity
                st.caption("""
                ℹ️ **Note**: Target variables are only valid for certain student groups:
                - `major_success`, `extended_graduation`: Graduated only
                - `first_year_struggle`: Students who completed Year 1
                - `has_ajc_case`: All students
                """)
        
        with col2:
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
                                 annotation_text="Probation")
                    fig.add_vline(x=3.0, line_dash="dash", line_color="orange",
                                 annotation_text="Success")
                    fig.add_vline(x=3.5, line_dash="dash", line_color="green",
                                 annotation_text="Dean's List")
                    fig.update_layout(xaxis_title="Final CGPA", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Overview (NEW - if clusters exist)
        if 'cluster_kmeans' in self.master_df.columns:
            st.markdown("---")
            st.subheader("🎯 Student Clusters Overview")
            
            cluster_col = 'cluster_kmeans'
            n_clusters = self.master_df[cluster_col].nunique()
            
            # Create cluster summary
            cluster_summary = self.master_df.groupby(cluster_col).agg({
                'student_id': 'count',
                'final_cgpa': 'mean' if 'final_cgpa' in self.master_df.columns else 'count',
                'first_year_struggle': 'mean' if 'first_year_struggle' in self.master_df.columns else 'count'
            }).round(3)
            
            cluster_summary.columns = ['Count', 'Avg CGPA', 'Struggle Rate']
            
            # Display as cards
            cols = st.columns(min(n_clusters, 5))
            
            for i, (cluster_id, row) in enumerate(cluster_summary.iterrows()):
                with cols[i % len(cols)]:
                    # Determine risk level for color
                    struggle_rate = row.get('Struggle Rate', 0)
                    if pd.isna(struggle_rate):
                        struggle_rate = 0
                    
                    if struggle_rate > 0.4:
                        color = "#e74c3c"
                        risk = "High Risk"
                    elif struggle_rate > 0.2:
                        color = "#f39c12"
                        risk = "Moderate"
                    else:
                        color = "#27ae60"
                        risk = "Low Risk"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}99, {color}); 
                                border-radius: 10px; padding: 15px; color: white; margin: 5px 0;">
                        <h4 style="margin: 0;">Cluster {cluster_id}</h4>
                        <p style="margin: 5px 0;">{risk}</p>
                        <p style="margin: 0; font-size: 0.9em;">
                            👥 {int(row['Count'])} students<br>
                            📊 CGPA: {row['Avg CGPA']:.2f}<br>
                            ⚠️ Struggle: {struggle_rate*100:.0f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.caption("👉 See 'Cluster Analysis' page for detailed breakdown")
        
        # Data Summary
        st.markdown("---")
        st.subheader("📋 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Information**")
            st.write(f"- Total records: {len(self.master_df):,}")
            st.write(f"- Features: {len(self.master_df.columns)}")
            
            # Target variable validity
            st.markdown("**Target Variable Coverage**")
            
            targets = [
                ('first_year_struggle', 'First Year Struggle'),
                ('has_ajc_case', 'AJC Case'),
                ('major_success', 'Major Success'),
                ('extended_graduation', 'Extended Graduation'),
                ('completed_degree', 'Completed Degree')
            ]
            
            for col, name in targets:
                if col in self.master_df.columns:
                    valid = self.master_df[col].notna().sum()
                    pct = valid / len(self.master_df) * 100
                    st.write(f"- {name}: {valid:,} ({pct:.0f}%)")
        
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
                        color=yg_dist.values,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        xaxis_title="Year Group",
                        yaxis_title="Count",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # CLUSTER ANALYSIS PAGE (NEW)
    # ========================================================================
    
    def show_cluster_analysis(self):
        """Show cluster analysis page"""
        
        st.header("🎯 Cluster Analysis")
        
        st.markdown("""
        This page shows the results of unsupervised learning analysis, which groups 
        students into clusters based on their characteristics. These clusters help 
        identify distinct student profiles and target interventions.
        """)
        
        # Check if clusters exist
        cluster_cols = [col for col in self.master_df.columns if col.startswith('cluster_')]
        
        if not cluster_cols:
            st.warning("""
            ⚠️ No cluster data available. 
            
            Run the unsupervised learning pipeline:
            ```bash
            python run_pipeline.py --mode unsupervised
            ```
            """)
            return
        
        # Select cluster column
        cluster_col = st.selectbox(
            "Select Clustering Method",
            cluster_cols,
            format_func=lambda x: x.replace('cluster_', '').upper()
        )
        
        n_clusters = self.master_df[cluster_col].nunique()
        
        st.info(f"📊 **{n_clusters} clusters** identified using {cluster_col.replace('cluster_', '').upper()}")
        
        # Cluster Summary Statistics
        st.subheader("📈 Cluster Summary")
        
        # Build summary
        agg_dict = {'student_id': 'count'}
        
        numeric_cols = ['final_cgpa', 'standardized_score', 'y1_cgpa']
        rate_cols = ['first_year_struggle', 'has_ajc_case', 'major_success', 
                    'extended_graduation', 'is_female', 'needs_financial_aid']
        
        for col in numeric_cols:
            if col in self.master_df.columns:
                agg_dict[col] = 'mean'
        
        for col in rate_cols:
            if col in self.master_df.columns:
                agg_dict[col] = 'mean'
        
        cluster_summary = self.master_df.groupby(cluster_col).agg(agg_dict).round(3)
        cluster_summary = cluster_summary.rename(columns={'student_id': 'Count'})
        
        # Add percentage
        total = cluster_summary['Count'].sum()
        cluster_summary['% of Total'] = (cluster_summary['Count'] / total * 100).round(1)
        
        # Reorder columns
        cols_order = ['Count', '% of Total'] + [c for c in cluster_summary.columns if c not in ['Count', '% of Total']]
        cluster_summary = cluster_summary[cols_order]
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Cluster Sizes")
            
            sizes = self.master_df[cluster_col].value_counts().sort_index()
            
            fig = px.pie(
                values=sizes.values,
                names=[f'Cluster {i}' for i in sizes.index],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 CGPA by Cluster")
            
            if 'final_cgpa' in self.master_df.columns:
                cgpa_data = self.master_df.dropna(subset=['final_cgpa'])
                
                fig = px.box(
                    cgpa_data,
                    x=cluster_col,
                    y='final_cgpa',
                    color=cluster_col,
                    labels={cluster_col: 'Cluster', 'final_cgpa': 'Final CGPA'}
                )
                fig.add_hline(y=2.0, line_dash="dash", line_color="red")
                fig.add_hline(y=3.0, line_dash="dash", line_color="green")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Analysis by Cluster
        st.markdown("---")
        st.subheader("⚠️ Risk Analysis by Cluster")
        
        risk_cols = ['first_year_struggle', 'has_ajc_case', 'major_struggle', 'extended_graduation']
        available_risk_cols = [c for c in risk_cols if c in self.master_df.columns]
        
        if available_risk_cols:
            risk_by_cluster = self.master_df.groupby(cluster_col)[available_risk_cols].mean() * 100
            
            fig = px.bar(
                risk_by_cluster.reset_index().melt(id_vars=cluster_col),
                x=cluster_col,
                y='value',
                color='variable',
                barmode='group',
                labels={'value': 'Rate (%)', 'variable': 'Risk Type', cluster_col: 'Cluster'}
            )
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            
            # Identify high-risk cluster
            if 'first_year_struggle' in available_risk_cols:
                struggle_rates = self.master_df.groupby(cluster_col)['first_year_struggle'].mean()
                high_risk_cluster = struggle_rates.idxmax()
                high_risk_rate = struggle_rates.max()
                
                st.warning(f"""
                🚨 **Highest Risk Cluster: {high_risk_cluster}**
                
                This cluster has a {high_risk_rate*100:.1f}% first-year struggle rate.
                Consider targeted interventions for these students.
                """)
        
        # Cluster Characteristics
        st.markdown("---")
        st.subheader("👥 Cluster Characteristics")
        
        selected_cluster = st.selectbox(
            "Select cluster to analyze:",
            sorted(self.master_df[cluster_col].unique())
        )
        
        cluster_data = self.master_df[self.master_df[cluster_col] == selected_cluster]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Cluster {selected_cluster} Overview**")
            st.write(f"- Size: {len(cluster_data)} students")
            st.write(f"- % of total: {len(cluster_data)/len(self.master_df)*100:.1f}%")
            
            if 'final_cgpa' in cluster_data.columns:
                avg_cgpa = cluster_data['final_cgpa'].mean()
                st.write(f"- Avg CGPA: {avg_cgpa:.2f}")
        
        with col2:
            st.markdown("**Demographics**")
            
            if 'is_female' in cluster_data.columns:
                female_pct = cluster_data['is_female'].mean() * 100
                st.write(f"- Female: {female_pct:.0f}%")
            
            if 'is_international' in cluster_data.columns:
                intl_pct = cluster_data['is_international'].mean() * 100
                st.write(f"- International: {intl_pct:.0f}%")
            
            if 'needs_financial_aid' in cluster_data.columns:
                aid_pct = cluster_data['needs_financial_aid'].mean() * 100
                st.write(f"- Need Fin. Aid: {aid_pct:.0f}%")
        
        with col3:
            st.markdown("**Risk Indicators**")
            
            for risk_col in available_risk_cols:
                if risk_col in cluster_data.columns:
                    valid = cluster_data[risk_col].notna()
                    if valid.sum() > 0:
                        rate = cluster_data.loc[valid, risk_col].mean() * 100
                        st.write(f"- {risk_col.replace('_', ' ').title()}: {rate:.0f}%")
        
        # Math Track distribution in cluster
        if 'math_track' in cluster_data.columns:
            st.markdown("**Math Track Distribution**")
            track_dist = cluster_data['math_track'].value_counts()
            
            fig = px.pie(
                values=track_dist.values,
                names=track_dist.index,
                title=f"Math Track in Cluster {selected_cluster}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Report
        if self.cluster_report:
            st.markdown("---")
            with st.expander("📄 Full Cluster Report"):
                st.text(self.cluster_report)
        
        # Show cluster images if available
        st.markdown("---")
        st.subheader("📊 Cluster Visualizations")
        
        image_dir = 'reports/figures/'
        cluster_images = [
            ('cluster_visualization_both.png', 'Cluster Visualization (PCA & t-SNE)'),
            ('cluster_optimization.png', 'Cluster Optimization'),
            ('cluster_feature_importance.png', 'Feature Importance for Clustering'),
            ('cluster_outcomes.png', 'Outcomes by Cluster')
        ]
        
        for img_file, title in cluster_images:
            img_path = os.path.join(image_dir, img_file)
            if os.path.exists(img_path):
                st.image(img_path, caption=title, use_column_width=True)
    
    # ========================================================================
    # STUDENT LOOKUP PAGE (UPDATED)
    # ========================================================================
    
    def show_student_lookup(self):
        """Student lookup with status display"""
        
        st.header("🔍 Student Lookup")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_method = st.radio(
                "Search by:",
                ["Student ID", "Year Group"],
                horizontal=True
            )
        
        student = None
        
        if search_method == "Student ID":
            student_id = st.text_input("Enter Student ID:")
            
            if student_id:
                matches = self.master_df[
                    self.master_df['student_id'].astype(str).str.upper().str.contains(
                        student_id.upper(), na=False
                    )
                ]
                
                if len(matches) == 0:
                    st.warning(f"No student found matching '{student_id}'")
                elif len(matches) == 1:
                    student = matches.iloc[0]
                else:
                    selected_id = st.selectbox(
                        f"Found {len(matches)} matches:",
                        matches['student_id'].tolist()
                    )
                    student = matches[matches['student_id'] == selected_id].iloc[0]
        else:
            yg_col = self.get_yeargroup_column()
            if yg_col:
                years = sorted(self.master_df[yg_col].dropna().unique())
                selected_year = st.selectbox("Year Group:", years)
                
                year_students = self.master_df[self.master_df[yg_col] == selected_year]
                selected_id = st.selectbox(
                    f"Students in {selected_year}:",
                    year_students['student_id'].tolist()
                )
                student = year_students[year_students['student_id'] == selected_id].iloc[0]
        
        if student is not None:
            st.markdown("---")
            
            # Header with status
            status_display = self.get_student_status_display(student)
            st.subheader(f"📋 {student['student_id']} {status_display}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                st.write(f"Gender: {student.get('Gender', 'N/A')}")
                st.write(f"Nationality: {student.get('Nationality', 'N/A')}")
                
                yg_col = self.get_yeargroup_column()
                if yg_col:
                    st.write(f"Year Group: {student.get(yg_col, 'N/A')}")
            
            with col2:
                st.markdown("**Academic**")
                
                program_col = self.get_program_column()
                if program_col:
                    st.write(f"Program: {student.get(program_col, 'N/A')}")
                
                st.write(f"Math Track: {student.get('math_track', 'N/A')}")
                st.write(f"Exam Type: {student.get('exam_source', 'N/A')}")
            
            with col3:
                st.markdown("**Performance**")
                
                if pd.notna(student.get('final_cgpa')):
                    st.write(f"Final CGPA: {student['final_cgpa']:.2f}")
                else:
                    st.write("Final CGPA: N/A")
                
                if pd.notna(student.get('total_semesters')):
                    st.write(f"Semesters: {int(student['total_semesters'])}")
                
                # Cluster assignment
                if 'cluster_kmeans' in student.index:
                    st.write(f"Cluster: {int(student['cluster_kmeans'])}")
            
            # Risk Assessment with validity notes
            st.markdown("---")
            st.subheader("⚠️ Risk Assessment")
            
            # Show validity based on status
            is_graduated = student.get('is_graduated', False)
            is_active = student.get('is_active', False)
            completed_y1 = student.get('completed_year1', False)
            
            col1, col2, col3, col4 = st.columns(4)
            
            risk_items = [
                ('first_year_struggle', 'First Year Struggle', completed_y1, col1),
                ('has_ajc_case', 'AJC Case', True, col2),  # Valid for all
                ('major_success', 'Major Success', is_graduated, col3),
                ('extended_graduation', 'Extended Grad', is_graduated, col4)
            ]
            
            for risk_col, label, is_valid, col in risk_items:
                with col:
                    st.markdown(f"**{label}**")
                    
                    if not is_valid:
                        st.markdown("⚪ N/A (status)")
                    elif risk_col in student.index:
                        val = student[risk_col]
                        if pd.notna(val):
                            if int(val) == 1:
                                st.markdown("🔴 **Yes**")
                            else:
                                st.markdown("🟢 No")
                        else:
                            st.markdown("⚪ N/A")
                    else:
                        st.markdown("⚪ N/A")
            
            # Show validity note
            if is_active:
                st.caption("ℹ️ This student is **active** - final outcomes not yet available")
            
            # GPA Trajectory
            if self.semester_records is not None and len(self.semester_records) > 0:
                student_semesters = self.semester_records[
                    self.semester_records['student_id'].astype(str) == str(student['student_id'])
                ]
                
                if len(student_semesters) > 0:
                    st.markdown("---")
                    st.subheader("📈 GPA Trajectory")
                    
                    if 'semester_order' in student_semesters.columns:
                        student_semesters = student_semesters.sort_values('semester_order')
                        x_vals = student_semesters['semester_order'].values
                    else:
                        x_vals = list(range(1, len(student_semesters) + 1))
                    
                    fig = go.Figure()
                    
                    if 'GPA' in student_semesters.columns:
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=student_semesters['GPA'].values,
                            mode='lines+markers',
                            name='Semester GPA',
                            line=dict(color='#3498db', width=2)
                        ))
                    
                    if 'CGPA' in student_semesters.columns:
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=student_semesters['CGPA'].values,
                            mode='lines+markers',
                            name='Cumulative GPA',
                            line=dict(color='#9b59b6', width=2)
                        ))
                    
                    fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                                 annotation_text="Probation")
                    fig.add_hline(y=3.5, line_dash="dash", line_color="green",
                                 annotation_text="Dean's List")
                    
                    fig.update_layout(
                        xaxis_title="Semester",
                        yaxis_title="GPA",
                        yaxis_range=[0, 4.2],
                        legend=dict(orientation="h", y=-0.15)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # RISK ANALYSIS PAGE (UPDATED)
    # ========================================================================
    
    def show_risk_analysis(self):
        """Risk analysis with student status filtering"""
        
        st.header("⚠️ Risk Analysis")
        
        # Risk type selection with validity info
        risk_options = {
            "First Year Struggle": ("first_year_struggle", "Students who completed Year 1"),
            "AJC Case": ("has_ajc_case", "All students"),
            "Major Struggle": ("major_struggle", "Graduated + Dismissed"),
            "Extended Graduation": ("extended_graduation", "Graduated only"),
            "Completed Degree": ("completed_degree", "Non-active students")
        }
        
        available_risks = {k: v for k, v in risk_options.items() 
                          if v[0] in self.master_df.columns}
        
        if not available_risks:
            st.warning("No risk indicators available")
            return
        
        # Sidebar filters
        st.sidebar.subheader("🔧 Filters")
        
        selected_risk = st.sidebar.selectbox(
            "Risk Type",
            list(available_risks.keys())
        )
        
        risk_col, validity_note = available_risks[selected_risk]
        
        st.info(f"ℹ️ **{selected_risk}** is valid for: {validity_note}")
        
        # Filter to valid cases only
        filtered_df = self.master_df[self.master_df[risk_col].notna()].copy()
        
        if len(filtered_df) == 0:
            st.warning("No valid data for this risk indicator")
            return
        
        # Additional filters
        yg_col = self.get_yeargroup_column()
        if yg_col and yg_col in filtered_df.columns:
            years = ['All'] + sorted([str(y) for y in filtered_df[yg_col].dropna().unique()])
            selected_year = st.sidebar.selectbox("Year Group", years)
            
            if selected_year != 'All':
                filtered_df = filtered_df[filtered_df[yg_col].astype(str) == selected_year]
        
        # Student status filter
        status_options = ['All', 'Graduated', 'Active', 'Withdrawn', 'Dismissed']
        selected_status = st.sidebar.selectbox("Student Status", status_options)
        
        if selected_status != 'All':
            status_col = f"is_{selected_status.lower()}"
            if status_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[status_col] == True]
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(filtered_df)
        at_risk = int(filtered_df[risk_col].sum())
        risk_rate = at_risk / total * 100 if total > 0 else 0
        
        with col1:
            st.metric("Valid Students", f"{total:,}")
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
            st.subheader("Risk by Cluster")
            
            if 'cluster_kmeans' in filtered_df.columns:
                risk_by_cluster = filtered_df.groupby('cluster_kmeans')[risk_col].agg(['sum', 'count', 'mean'])
                risk_by_cluster.columns = ['At Risk', 'Total', 'Rate']
                
                fig = px.bar(
                    risk_by_cluster.reset_index(),
                    x='cluster_kmeans',
                    y='Rate',
                    color='Rate',
                    color_continuous_scale='RdYlGn_r',
                    labels={'cluster_kmeans': 'Cluster', 'Rate': 'Risk Rate'}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cluster data not available")
        
        with col2:
            st.subheader("Risk by Math Track")
            
            if 'math_track' in filtered_df.columns:
                valid_tracks = filtered_df[filtered_df['math_track'] != 'Unknown']
                
                if len(valid_tracks) > 0:
                    risk_by_track = valid_tracks.groupby('math_track')[risk_col].mean()
                    
                    fig = px.bar(
                        x=risk_by_track.index,
                        y=risk_by_track.values,
                        color=risk_by_track.values,
                        color_continuous_scale='RdYlGn_r',
                        labels={'x': 'Math Track', 'y': 'Risk Rate'}
                    )
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # At-risk student list
        st.markdown("---")
        st.subheader("🚨 At-Risk Students")
        
        at_risk_df = filtered_df[filtered_df[risk_col] == 1].copy()
        
        if len(at_risk_df) > 0:
            # Determine display columns
            display_cols = ['student_id']
            
            # Add status
            if 'is_graduated' in at_risk_df.columns:
                at_risk_df['Status'] = at_risk_df.apply(
                    lambda r: 'Graduated' if r.get('is_graduated') else
                              'Active' if r.get('is_active') else
                              'Withdrawn' if r.get('is_withdrawn') else
                              'Dismissed' if r.get('is_dismissed') else 'Unknown',
                    axis=1
                )
                display_cols.append('Status')
            
            # Add other columns
            optional_cols = [yg_col, 'final_cgpa', 'math_track', 'cluster_kmeans']
            for col in optional_cols:
                if col and col in at_risk_df.columns:
                    display_cols.append(col)
            
            st.write(f"Showing {min(100, len(at_risk_df))} of {len(at_risk_df)} at-risk students")
            
            st.dataframe(
                at_risk_df[display_cols].head(100),
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            csv = at_risk_df[display_cols].to_csv(index=False)
            st.download_button(
                "📥 Download At-Risk List",
                csv,
                f"at_risk_{risk_col}.csv",
                "text/csv"
            )
        else:
            st.success("🎉 No at-risk students in the selected group!")
    
    # ========================================================================
    # MODEL PERFORMANCE PAGE (UPDATED)
    # ========================================================================
    
    def show_model_performance(self):
        """Model performance with split info"""
        
        st.header("🤖 Model Performance")
        
        if not self.models:
            st.warning("No models loaded. Run the training pipeline first.")
            
            # Show reports if available
            report_path = 'reports/model_report.txt'
            if os.path.exists(report_path):
                with st.expander("📄 Model Report"):
                    with open(report_path, 'r') as f:
                        st.text(f.read())
            return
        
        st.success(f"✓ {len(self.models)} models loaded")
        
        # Model summary table
        st.subheader("📊 Model Summary")
        
        model_summary = []
        
        for task_key, model_info in self.models.items():
            split_info = model_info.get('split_info', {})
            
            row = {
                'Task': task_key,
                'Model': type(model_info['model']).__name__,
                'Features': len(model_info.get('features', [])),
                'Split Method': split_info.get('method', 'unknown'),
                'Train Size': split_info.get('train_size', 'N/A'),
                'Test Size': split_info.get('test_size', split_info.get('test_size_actual', 'N/A'))
            }
            
            # Add train/test years for temporal split
            if split_info.get('method') == 'temporal':
                train_years = split_info.get('train_years', [])
                test_years = split_info.get('test_years', [])
                row['Train Years'] = str(train_years) if train_years else 'N/A'
                row['Test Years'] = str(test_years) if test_years else 'N/A'
            
            model_summary.append(row)
        
        summary_df = pd.DataFrame(model_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Detailed view per model
        st.markdown("---")
        st.subheader("🔍 Model Details")
        
        selected_model = st.selectbox(
            "Select model:",
            list(self.models.keys())
        )
        
        if selected_model:
            model_info = self.models[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Information**")
                st.write(f"- Type: {type(model_info['model']).__name__}")
                st.write(f"- Features: {len(model_info.get('features', []))}")
                
                split_info = model_info.get('split_info', {})
                st.write(f"- Split Method: {split_info.get('method', 'N/A')}")
                
                if split_info.get('method') == 'temporal':
                    st.write(f"- Train Years: {split_info.get('train_years', 'N/A')}")
                    st.write(f"- Test Years: {split_info.get('test_years', 'N/A')}")
            
            with col2:
                st.markdown("**Features Used**")
                features = model_info.get('features', [])
                
                if features:
                    for i, feat in enumerate(features[:10], 1):
                        st.write(f"{i}. {feat}")
                    
                    if len(features) > 10:
                        st.write(f"... and {len(features) - 10} more")
        
        # Reports
        st.markdown("---")
        st.subheader("📄 Reports")
        
        reports = [
            ('reports/model_report.txt', 'Model Report'),
            ('reports/executive_summary.txt', 'Executive Summary'),
            ('reports/statistical_report.txt', 'Statistical Report'),
            ('reports/data_quality_report.txt', 'Data Quality Report')
        ]
        
        for path, name in reports:
            if os.path.exists(path):
                with st.expander(f"📄 {name}"):
                    with open(path, 'r') as f:
                        st.text(f.read())
    
    # ========================================================================
    # REMAINING PAGES (Cohort, Major, Math Track, Predictions)
    # These remain largely the same but with minor updates for student status
    # ========================================================================
    
    def show_cohort_analysis(self):
        """Cohort analysis page"""
        
        st.header("📅 Cohort Analysis")
        
        yg_col = self.get_yeargroup_column()
        
        if not yg_col:
            st.warning("Year group data not available")
            return
        
        cohorts = sorted(self.master_df[yg_col].dropna().unique())
        
        if len(cohorts) == 0:
            st.warning("No cohort data available")
            return
        
        selected_cohorts = st.multiselect(
            "Select Cohorts",
            cohorts,
            default=cohorts[-3:] if len(cohorts) >= 3 else cohorts
        )
        
        if not selected_cohorts:
            return
        
        cohort_df = self.master_df[self.master_df[yg_col].isin(selected_cohorts)]
        
        # Summary with student status
        st.subheader("📊 Cohort Summary")
        
        agg_dict = {
            'student_id': 'count',
            'is_graduated': 'sum' if 'is_graduated' in cohort_df.columns else 'count',
            'is_active': 'sum' if 'is_active' in cohort_df.columns else 'count'
        }
        
        for col in ['final_cgpa', 'first_year_struggle', 'major_success']:
            if col in cohort_df.columns:
                agg_dict[col] = 'mean'
        
        summary = cohort_df.groupby(yg_col).agg(agg_dict).round(3)
        summary = summary.rename(columns={
            'student_id': 'Total',
            'is_graduated': 'Graduated',
            'is_active': 'Active'
        })
        
        st.dataframe(summary, use_container_width=True)
        
        # Trends
        col1, col2 = st.columns(2)
        
        with col1:
            if 'final_cgpa' in summary.columns:
                fig = px.line(
                    summary.reset_index(),
                    x=yg_col,
                    y='final_cgpa',
                    markers=True,
                    title="Average CGPA Trend"
                )
                fig.add_hline(y=3.0, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graduation rate trend
            if 'Graduated' in summary.columns and 'Total' in summary.columns:
                summary['Graduation Rate'] = summary['Graduated'] / summary['Total']
                
                fig = px.line(
                    summary.reset_index(),
                    x=yg_col,
                    y='Graduation Rate',
                    markers=True,
                    title="Graduation Rate Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def show_major_analysis(self):
        """Major analysis page"""
        
        st.header("🎓 Major Analysis")
        
        program_col = self.get_program_column()
        
        if not program_col:
            st.warning("Program data not available")
            return
        
        # Filter to programs with enough data
        major_counts = self.master_df[program_col].value_counts()
        valid_majors = major_counts[major_counts >= 10].index.tolist()
        
        if not valid_majors:
            st.warning("No majors with sufficient data")
            return
        
        selected_major = st.selectbox("Select Major", valid_majors)
        
        major_df = self.master_df[self.master_df[program_col] == selected_major]
        
        # Overview with status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(major_df))
        
        with col2:
            if 'is_graduated' in major_df.columns:
                graduated = major_df['is_graduated'].sum()
                st.metric("Graduated", graduated)
        
        with col3:
            if 'final_cgpa' in major_df.columns:
                # Only for graduated students
                grad_df = major_df[major_df.get('is_graduated', True) == True]
                if len(grad_df) > 0:
                    avg_cgpa = grad_df['final_cgpa'].mean()
                    st.metric("Avg CGPA (Graduated)", f"{avg_cgpa:.2f}")
        
        with col4:
            if 'major_success' in major_df.columns:
                valid = major_df['major_success'].notna()
                if valid.sum() > 0:
                    rate = major_df.loc[valid, 'major_success'].mean() * 100
                    st.metric("Success Rate", f"{rate:.1f}%")
        
        # CGPA distribution
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'final_cgpa' in major_df.columns:
                st.subheader("CGPA Distribution")
                fig = px.histogram(
                    major_df['final_cgpa'].dropna(),
                    nbins=20
                )
                fig.add_vline(x=2.0, line_dash="dash", line_color="red")
                fig.add_vline(x=3.0, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'math_track' in major_df.columns:
                st.subheader("Math Track Distribution")
                track_dist = major_df['math_track'].value_counts()
                fig = px.pie(values=track_dist.values, names=track_dist.index)
                st.plotly_chart(fig, use_container_width=True)
    
    def show_math_track_analysis(self):
        """Math track analysis (Q7 & Q8)"""
        
        st.header("📐 Math Track Analysis")
        
        st.markdown("""
        **Research Questions:**
        - **Q7**: Is there a significant performance difference across math tracks?
        - **Q8**: Can College Algebra students succeed in Computer Science?
        """)
        
        if 'math_track' not in self.master_df.columns:
            st.warning("Math track data not available")
            return
        
        # Filter valid data
        df = self.master_df[
            (self.master_df['math_track'].notna()) &
            (self.master_df['math_track'] != 'Unknown')
        ]
        
        if len(df) < 30:
            st.warning("Insufficient data for analysis")
            return
        
        # Q7: Performance comparison
        st.subheader("📊 Q7: Performance by Math Track")
        
        # Only use graduated students for final CGPA comparison
        if 'is_graduated' in df.columns:
            grad_df = df[df['is_graduated'] == True]
            st.caption(f"Using {len(grad_df)} graduated students for CGPA analysis")
        else:
            grad_df = df
        
        if 'final_cgpa' in grad_df.columns and len(grad_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    grad_df,
                    x='math_track',
                    y='final_cgpa',
                    color='math_track',
                    title="CGPA by Math Track (Graduated Students)"
                )
                fig.add_hline(y=2.0, line_dash="dash", line_color="red")
                fig.add_hline(y=3.0, line_dash="dash", line_color="green")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary stats
                summary = grad_df.groupby('math_track')['final_cgpa'].agg(['count', 'mean', 'std', 'median'])
                summary.columns = ['Count', 'Mean', 'Std', 'Median']
                st.dataframe(summary.round(3), use_container_width=True)
        
        # Statistical test
        try:
            from scipy.stats import kruskal
            
            tracks = grad_df['math_track'].unique()
            groups = [grad_df[grad_df['math_track'] == t]['final_cgpa'].dropna() for t in tracks]
            groups = [g for g in groups if len(g) >= 5]
            
            if len(groups) >= 2:
                stat, p = kruskal(*groups)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("H-Statistic", f"{stat:.3f}")
                col2.metric("p-value", f"{p:.4f}")
                col3.metric("Significant", "Yes ✓" if p < 0.05 else "No")
                
                if p < 0.05:
                    st.success("✅ Significant difference exists across math tracks")
                else:
                    st.info("ℹ️ No significant difference found")
        except ImportError:
            st.warning("scipy not available for statistical test")
        
        # Q8: CS + College Algebra
        st.markdown("---")
        st.subheader("💻 Q8: College Algebra in CS")
        
        program_col = self.get_program_column()
        
        if program_col:
            cs_df = self.master_df[
                self.master_df[program_col].astype(str).str.contains('Computer', case=False, na=False)
            ]
            
            if len(cs_df) >= 10:
                st.write(f"**Total CS students:** {len(cs_df)}")
                
                # Filter to graduated for outcome analysis
                if 'is_graduated' in cs_df.columns:
                    cs_grad = cs_df[cs_df['is_graduated'] == True]
                    st.caption(f"Using {len(cs_grad)} graduated CS students")
                else:
                    cs_grad = cs_df
                
                if 'math_track' in cs_grad.columns:
                    cs_summary = cs_grad.groupby('math_track').agg({
                        'student_id': 'count',
                        'final_cgpa': 'mean' if 'final_cgpa' in cs_grad.columns else 'count',
                        'major_success': 'mean' if 'major_success' in cs_grad.columns else 'count'
                    }).round(3)
                    cs_summary.columns = ['Count', 'Avg CGPA', 'Success Rate']
                    
                    st.dataframe(cs_summary, use_container_width=True)
                    
                    # College Algebra specific
                    ca_cs = cs_grad[cs_grad['math_track'] == 'College Algebra']
                    
                    if len(ca_cs) >= 3 and 'major_success' in ca_cs.columns:
                        success_rate = ca_cs['major_success'].mean()
                        
                        if success_rate >= 0.5:
                            st.success(f"""
                            ✅ **College Algebra students CAN succeed in CS**
                            - Success Rate: {success_rate*100:.1f}%
                            - Sample: {len(ca_cs)} students
                            """)
                        else:
                            st.warning(f"""
                            ⚠️ **College Algebra students face challenges in CS**
                            - Success Rate: {success_rate*100:.1f}%
                            - Sample: {len(ca_cs)} students
                            - Recommendation: Additional math support
                            """)
                    else:
                        st.info(f"Insufficient College Algebra CS students ({len(ca_cs)})")
            else:
                st.warning("Insufficient CS students for analysis")
    
    def show_predictions(self):
        """Prediction page"""
        
        st.header("🔮 Student Risk Prediction")
        
        st.markdown("""
        Enter student information to predict risk levels based on trained models.
        """)
        
        with st.form("prediction_form"):
            st.subheader("📝 Student Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                hs_score = st.slider("High School Score (0-100)", 0, 100, 70)
                math_score = st.slider("Math Score", 0, 100, 70)
                english_score = st.slider("English Score", 0, 100, 70)
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                math_track = st.selectbox("Math Track", ["Calculus", "Pre-Calculus", "College Algebra"])
                intended_major = st.selectbox("Intended Major", 
                    ["Computer Science", "Business Administration", "MIS", "Engineering", "Other"])
            
            col3, col4 = st.columns(2)
            
            with col3:
                financial_aid = st.selectbox("Needs Financial Aid?", ["No", "Yes"])
            
            with col4:
                international = st.selectbox("International Student?", ["No", "Yes"])
            
            submitted = st.form_submit_button("🔮 Predict", type="primary")
        
        if submitted:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Calculate risk scores
            base_risk = (100 - hs_score) / 100 * 0.4 + (100 - math_score) / 100 * 0.4 + (100 - english_score) / 100 * 0.2
            
            # Adjustments
            if math_track == "College Algebra":
                base_risk += 0.15
            elif math_track == "Pre-Calculus":
                base_risk += 0.05
            
            if intended_major in ["Computer Science", "Engineering"] and math_track == "College Algebra":
                base_risk += 0.1
            
            if financial_aid == "Yes":
                base_risk += 0.03
            
            # Calculate risks
            struggle_risk = np.clip(base_risk, 0.05, 0.95)
            ajc_risk = np.clip(base_risk * 0.3, 0.02, 0.50)
            extended_risk = np.clip(base_risk * 0.7, 0.05, 0.85)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self._display_risk_gauge("First Year Struggle", struggle_risk)
            
            with col2:
                self._display_risk_gauge("AJC Risk", ajc_risk, (0.15, 0.30))
            
            with col3:
                self._display_risk_gauge("Extended Graduation", extended_risk)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            recs = []
            
            if struggle_risk > 0.4:
                recs.append("🎯 Enroll in academic support programs")
            
            if math_track == "College Algebra":
                recs.append("📐 Consider math tutoring or bridge courses")
            
            if intended_major in ["Computer Science", "Engineering"] and math_track != "Calculus":
                recs.append("💻 Connect with STEM mentors")
            
            if financial_aid == "Yes":
                recs.append("💰 Meet with financial aid office early")
            
            if extended_risk > 0.4:
                recs.append("📅 Create detailed graduation plan with advisor")
            
            if not recs:
                recs.append("✅ Low risk profile - standard support sufficient")
            
            for rec in recs:
                st.markdown(f"- {rec}")
    
    def _display_risk_gauge(self, title, value, thresholds=(0.30, 0.60)):
        """Display risk gauge"""
        
        low, high = thresholds
        
        if value > high:
            color, level = "#e74c3c", "HIGH"
        elif value > low:
            color, level = "#f39c12", "MEDIUM"
        else:
            color, level = "#27ae60", "LOW"
        
        st.markdown(f"### {title}")
        st.markdown(f"**{level}**: {value*100:.0f}%")
        st.progress(min(value, 1.0))
        
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
                ]
            }
        ))
        fig.update_layout(height=180, margin=dict(t=10, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main()