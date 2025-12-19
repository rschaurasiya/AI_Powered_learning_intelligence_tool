import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.predictior import Predictor
from src.insights import InsightsGenerator

def show(navigate_to, cleanup_data):
    """
    Displays the Predictions & Insights page.
    
    Args:
        navigate_to: Function to change the current page.
        cleanup_data: Function to clean up session state when navigating back.
    """
    st.markdown('<div class="section-header"> Predictions & Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.preprocessed_data is None:
        st.warning(" Please preprocess your data first in the 'EDA & Preprocessing' page")
        if st.button("⬅ Go to EDA"):
            navigate_to("EDA & Preprocessing")
            st.rerun()
        return
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'model.pkl'
    
    if not model_path.exists():
        st.error(" Model file not found! Please train a model first.")
        st.code("python main.py --mode train --data data/sample_data.csv", language="bash")
        return
    
    try:
        # Load model and preprocessor
        model_data = joblib.load(model_path)
        model = model_data['model']
        
        # Get preprocessed data
        X = st.session_state.preprocessed_data['X']
        original_data = st.session_state.preprocessed_data['original_data']
        
        # Make predictions
        with st.spinner("Generating predictions..."):
            predictor = Predictor(model_path=str(model_path))
            predictor.model = model
            
            predictions = predictor.predict_with_confidence(X)
            
            # Combine with original data
            results = pd.concat([original_data.reset_index(drop=True), predictions], axis=1)
            
            st.success(" Predictions generated successfully!")
        
        # Display metrics
        st.markdown("###  Prediction Summary")
        
        # Show Model Info
        model_name = model_data.get('model_name', model.__class__.__name__)
        model_acc = model_data.get('metrics', {}).get('val_accuracy', 0.0)
        
        st.info(f" Using Best Selected Model: **{model_name}** (Validation Accuracy: {model_acc:.2%})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(predictions))
        with col2:
            st.metric("Predicted Completions", predictions['prediction'].sum())
        with col3:
            st.metric("High Output Risk", (predictions['risk_level'] == 'High').sum())
        with col4:
            st.metric("Confidence (Avg)", f"{predictions['confidence'].mean():.2f}")
        
        # Show predictions table
        st.markdown("###  Student Predictions")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            risk_filter = st.multiselect("Filter by Risk Level:", 
                                       options=['High', 'Medium', 'Low'],
                                       default=['High', 'Medium', 'Low'])
        
        filtered_results = results[results['risk_level'].isin(risk_filter)]
        
        st.dataframe(filtered_results.style.background_gradient(subset=['completion_probability'], cmap='RdYlGn'),
                    use_container_width=True)
        
        # Download button
        csv = filtered_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
        
        # Student-level Insights
        st.markdown("###  Student-Level Insights")
        
        insights_gen = InsightsGenerator()
        student_risks = insights_gen.calculate_student_risk_scores(original_data)
        
        # Display student risks
        st.dataframe(student_risks.style.background_gradient(subset=['Completion_Rate'], cmap='RdYlGn'),
                    use_container_width=True)
        
        # Chapter Difficulty Analysis
        st.markdown("###  Chapter Difficulty Analysis")
        
        chapter_difficulty = insights_gen.calculate_chapter_difficulty(original_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Chapter difficulty bar chart
            fig = px.bar(chapter_difficulty, x='Chapter_Order', y='Difficulty_Score',
                       title='Chapter Difficulty Scores',
                       color='Difficulty_Score',
                       color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Completion rate by chapter
            fig = px.line(chapter_difficulty, x='Chapter_Order', y='Completion_Rate',
                        title='Completion Rate by Chapter',
                        markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Chapter difficulty table
        st.dataframe(chapter_difficulty.style.background_gradient(subset=['Difficulty_Score'], cmap='YlOrRd'),
                    use_container_width=True)
        
        # Strategic Insights Summary
        st.markdown("---")
        st.markdown("###  Executive Summary & Strategic Insights")
        
        # Calculate high-level metrics
        total_students = len(original_data['Student_ID'].unique())
        predicted_pass = predictions['prediction'].sum()
        pass_rate = (predicted_pass / len(predictions)) * 100
        
        # Identify hardest chapter
        if not chapter_difficulty.empty:
            hardest_chapter_row = chapter_difficulty.loc[chapter_difficulty['Completion_Rate'].idxmin()]
            hardest_chap = hardest_chapter_row['Chapter_Order']
            hardest_chap_rate = hardest_chapter_row['Completion_Rate'] * 100
        else:
            hardest_chap = "N/A"
            hardest_chap_rate = 0
            
        # Time comparison
        avg_time_pass = original_data[original_data['Completed'] == 1]['Time_Spent_Hours'].mean()
        avg_time_fail = original_data[original_data['Completed'] == 0]['Time_Spent_Hours'].mean()
        
        # Handle NaN values if data is sparse
        if pd.isna(avg_time_pass): avg_time_pass = 0
        if pd.isna(avg_time_fail): avg_time_fail = 0
        
        high_risk_count = (predictions['risk_level'] == 'High').sum()
        
        # Display insights in a styled container
        st.info(f"""
         Key Takeaways for Decision Makers:
        
        1. Cohort Health Forecast: The model projects a **{pass_rate:.1f}% completion rate**. 
           *(Based on {len(predictions)} student records processed)*
           
        2.  Critical Bottleneck: **Chapter {int(hardest_chap) if hardest_chap != "N/A" else "N/A"}** is identified as the biggest hurdle, with a low completion rate of **{hardest_chap_rate:.1f}%**.
           > *Actionable:* specific review of this chapter's content or additional support workshops are recommended.

        3.  The Engagement Gap: Successful students invest approx. **{avg_time_pass:.1f} hours**, compared to just **{avg_time_fail:.1f} hours** for those who drop out.
           > *Actionable:* Early warning systems should flag students with low engagement hours in early chapters.

        4.  Immediate Intervention: **{high_risk_count} students** are flagged as 'High Risk'.
           > *Actionable:* Export the list above and schedule 1-on-1 check-ins with these specific learners immediately.
        """)

    except Exception as e:
        st.error(f" Error during prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        
    # Navigation Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Back to EDA
        def go_back_to_eda():
            cleanup_data("Predictions & Insights", "EDA & Preprocessing")
            navigate_to("EDA & Preprocessing")
            
        st.button(" Back to EDA", 
                 on_click=go_back_to_eda,
                 use_container_width=True)
