import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessor import DataPreprocessor

def detect_outliers(df, column):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def handle_outliers(df, method='cap'):
    """Handle outliers in numerical columns."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    
    for col in numeric_cols:
        if col not in ['Completed', 'Student_ID', 'Course_ID']:  # Don't process binary or ID columns
            outliers, lower, upper = detect_outliers(df_clean, col)
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': outlier_count,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
                
                if method == 'cap':
                    # Cap outliers
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                elif method == 'remove':
                    # Remove rows with outliers
                    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    return df_clean, outlier_info


def show(navigate_to, cleanup_data):
    """
    Displays the EDA & Preprocessing page.
    
    Args:
        navigate_to: Function to change the current page.
        cleanup_data: Function to clean up session state when navigating back.
    """
    st.markdown('<div class="section-header"> Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning(" Please upload a CSV file first in the 'Upload & Validate' page")
        if st.button(" Go to Upload"):
            navigate_to("Upload & Validate")
            st.rerun()
        return
    
    df = st.session_state.data.copy()
    
    # Check for null values
    has_nulls = df.isnull().any().any()
    if has_nulls:
        null_count = df.isnull().sum().sum()
        st.markdown(f'<div class="warning-box"> Warning: Your data contains {null_count} null values. Please clean your data for better visualizations.</div>', 
                  unsafe_allow_html=True)
    
    # Show cleaning summary if available
    if st.session_state.cleaning_actions:
        st.markdown('<div class="success-box"> Data has been cleaned and is ready for analysis!</div>', 
                  unsafe_allow_html=True)
        
        with st.expander("View Cleaning Summary"):
            for col, action in st.session_state.cleaning_actions.items():
                st.write(f"- **{col}**: {action}")
    
    # Create a clean version for visualizations (drop rows with nulls in key viz columns)
    viz_df = df.dropna(subset=['Time_Spent_Hours', 'Scores', 'Completed', 'Chapter_Order'])
    
    if len(viz_df) < len(df) and has_nulls:
        st.info(f" Visualizations show {len(viz_df)} of {len(df)} records (rows with nulls excluded)")
    
    # Statistical Summary
    st.markdown("###  Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    st.markdown("###  Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Outliers", "Patterns"])
    
    with tab1:
        st.markdown("**Distribution of Numerical Features**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time Spent Distribution
            fig = px.histogram(viz_df, x='Time_Spent_Hours', 
                             title='Time Spent Distribution',
                             color='Completed', 
                             barmode='overlay',
                             labels={'Completed': 'Completed'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Scores Distribution
            fig = px.histogram(viz_df, x='Scores', 
                             title='Scores Distribution',
                             color='Completed',
                             barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Chapter Order Distribution
            fig = px.histogram(viz_df, x='Chapter_Order', 
                             title='Chapter Order Distribution',
                             color='Completed',
                             barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Completion Rate by Chapter
            completion_by_chapter = viz_df.groupby('Chapter_Order')['Completed'].mean().reset_index()
            fig = px.bar(completion_by_chapter, x='Chapter_Order', y='Completed',
                       title='Completion Rate by Chapter',
                       labels={'Completed': 'Completion Rate'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Correlation Analysis**")
        
        # Correlation heatmap
        numeric_df = viz_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = px.imshow(corr, 
                      title='Correlation Heatmap',
                      color_continuous_scale='RdBu_r',
                      aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(viz_df, x='Time_Spent_Hours', y='Scores', 
                           color='Completed', title='Time vs Scores',
                           trendline='ols')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(viz_df, x='Chapter_Order', y='Scores',
                           color='Completed', title='Chapter vs Scores',
                           size='Time_Spent_Hours')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("**Outlier Detection & Handling**")
        
        # Detect outliers
        outliers_info = {}
        for col in ['Time_Spent_Hours', 'Scores']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
        
        # Display outlier info
        cols = st.columns(len(outliers_info))
        for i, (col, count) in enumerate(outliers_info.items()):
            with cols[i]:
                st.metric(f"Outliers in {col}", count, 
                        delta="High Risk" if count > 0 else "Clean",
                        delta_color="inverse")
        
        # Box plots
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, y='Time_Spent_Hours', title='Box Plot: Time Spent')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y='Scores', title='Box Plot: Scores')
            st.plotly_chart(fig, use_container_width=True)
        
        # Outlier handling options
        st.markdown("**Handle Outliers**")
        outlier_method = st.radio("Select outlier handling method:", 
                                 ["Keep All Data", "Cap Outliers", "Remove Outliers"])
        
        if st.button("Apply Outlier Handling"):
            if outlier_method == "Keep All Data":
                st.session_state.cleaned_data = df
                st.info(" Keeping all data without outlier handling")
            elif outlier_method == "Cap Outliers":
                cleaned_df, info = handle_outliers(df, method='cap')
                st.session_state.cleaned_data = cleaned_df
                st.success(f" Outliers capped! Processed {sum([v['count'] for v in info.values()])} outliers")
            else:
                cleaned_df, info = handle_outliers(df, method='remove')
                st.session_state.cleaned_data = cleaned_df
                st.success(f" Outliers removed! Dataset reduced from {len(df)} to {len(cleaned_df)} records")
    
    with tab4:
        st.markdown("**Learning Patterns Analysis**")
        
        # Average time by completion status
        col1, col2 = st.columns(2)
        
        with col1:
            avg_time = viz_df.groupby('Completed')['Time_Spent_Hours'].mean().reset_index()
            avg_time['Completed'] = avg_time['Completed'].map({0: 'Not Completed', 1: 'Completed'})
            fig = px.bar(avg_time, x='Completed', y='Time_Spent_Hours',
                       title='Average Time by Completion Status',
                       color='Completed')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_scores = viz_df.groupby('Completed')['Scores'].mean().reset_index()
            avg_scores['Completed'] = avg_scores['Completed'].map({0: 'Not Completed', 1: 'Completed'})
            fig = px.bar(avg_scores, x='Completed', y='Scores',
                       title='Average Scores by Completion Status',
                       color='Completed')
            st.plotly_chart(fig, use_container_width=True)
    
    
    # Data Cleaning Section
    st.markdown("---")
    st.markdown("###  Data Cleaning")
    st.write("If your data has missing values or outliers, click below to clean it before preprocessing.")
    
    if st.button("Clean & Fix Data", type="primary"):
        with st.spinner("Cleaning data..."):
            preprocessor = DataPreprocessor()
            
            # 1. Handle Missing Values
            cleaned_df, actions = preprocessor.handle_missing_values(df)
            
            # 2. Handle Outliers
            cleaned_df, outlier_actions = preprocessor.handle_outliers(cleaned_df)
            
            # Store results
            st.session_state.data = cleaned_df
            st.session_state.cleaned_data = cleaned_df
            st.session_state.cleaning_actions = {**actions, **outlier_actions}
            st.session_state.preprocessed_data = None # Reset preprocessing to force re-run
            
            st.success(" Data cleaned successfully!")
            st.rerun()

    # Preprocessing Section
    st.markdown("---")
    st.markdown("###  Data Preprocessing")
    st.markdown("Prepare data for machine learning models.")
    
    if st.session_state.preprocessed_data is None:
        # Auto-preprocess if not done
        with st.spinner("Preprocessing data for models..."):
            try:
                data_to_process = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else df
                
                # Initialize preprocessor
                preprocessor = DataPreprocessor()
                
                # Prepare features
                X, y = preprocessor.prepare_features(data_to_process, target_column='Completed', fit=True)
                
                # Store in session state
                st.session_state.preprocessed_data = {
                    'X': X,
                    'y': y,
                    'preprocessor': preprocessor,
                    'original_data': data_to_process
                }
                
                st.success(" Data preprocessed successfully!")
                
                # Show preprocessing summary
                st.markdown("**Preprocessing Summary:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Features Created", X.shape[1])
                with col2:
                    st.metric("Samples", X.shape[0])
                with col3:
                    st.metric("Target Variable", "Completed")
                
                # Show feature names
                st.markdown("**Engineered Features:**")
                st.write(", ".join(preprocessor.feature_names))
                
            except Exception as e:
                st.error(f" Preprocessing error: {str(e)}")
    
    # Navigation Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Back Button with Cleanup
        def go_back_to_upload():
            cleanup_data("EDA & Preprocessing", "Upload & Validate")
            navigate_to("Upload & Validate")
            
        st.button(" Back to Upload", 
                 on_click=go_back_to_upload,
                 use_container_width=True)
        
    with col3:
         # Next Button
        if st.session_state.preprocessed_data is not None:
            st.button(" Go to Predictions", 
                     type="primary", 
                     use_container_width=True,
                     on_click=navigate_to,
                     args=("Predictions & Insights",))
        else:
            st.info("Wait for preprocessing...")
