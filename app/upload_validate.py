import streamlit as st
import pandas as pd
import json
import io
import sys
from pathlib import Path

# Add parent directory to path to import src modules if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader

def show(navigate_to):
    """
    Displays the Upload & Validate page with support for multiple data sources.
    
    Args:
        navigate_to: Function to change the current page in session state.
    """
    st.markdown('<div class="section-header">Upload Your Data</div>', unsafe_allow_html=True)
    
    # Show required column suggestions prominently
    st.info("""
    ###  Required Columns
    Your data needs to contain the following columns (extra columns will be automatically ignored):
    
    | Column Name | Description | Example |
    |---|---|---|
    | `Student_ID` | Unique identifier | S001 |
    | `Course_ID` | Course identifier | C001 |
    | `Chapter_Order` | Chapter sequence number | 1 |
    | `Time_Spent_Hours` | Time spent on chapter | 2.5 |
    | `Scores` | Assessment score | 85 |
    | `Completed` | Completion status (0/1) | 1 |
    """)
    
    # Data Source Selection
    st.markdown("### Select Data Source")
    source_type = st.radio(
        "Choose how you want to provide data:",
        ["Upload File (CSV/JSON)", "API Payload (JSON)", "App-based Input (Manual)"],
        horizontal=True
    )
    
    df = None
    
    # ==========================================
    # 1. FILE UPLOAD (CSV/JSON)
    # ==========================================
    if source_type == "Upload File (CSV/JSON)":
        uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=['csv', 'json'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                if df is not None:
                    st.success(f"File uploaded successfully! Loaded {len(df)} records.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # ==========================================
    # 2. API PAYLOAD (JSON)
    # ==========================================
    elif source_type == "API Payload (JSON)":
        st.write("Paste your JSON data below (e.g., list of records or dictionary):")
        json_input = st.text_area("JSON Payload", height=200, placeholder='[{"Student_ID": "S001", "Scores": 85, ...}]')
        
        if st.button("Process JSON Payload"):
            if json_input.strip():
                try:
                    # Try parsing as standard JSON
                    data = json.loads(json_input)
                    df = pd.DataFrame(data)
                    st.success(f"JSON parsed successfully! Loaded {len(df)} records.")
                except json.JSONDecodeError as json_err:
                    # Fallback: Try parsing as Python literal (handles single quotes, trailing commas, etc.)
                    try:
                        import ast
                        data = ast.literal_eval(json_input)
                        df = pd.DataFrame(data)
                        st.success(f"JSON (relaxed) parsed successfully! Loaded {len(df)} records.")
                    except Exception:
                        # If both fail, show the original JSON error which is usually more descriptive
                        st.error(f"Invalid JSON format: {str(json_err)}")
                        st.warning("Tip: Check for trailing commas, missing braces, or ensure keys are quoted.")
                except Exception as e:
                    st.error(f"Error converting input to DataFrame: {str(e)}")
            else:
                st.warning("Please paste some JSON data first.")

    # ==========================================
    # 3. APP-BASED INPUT (MANUAL)
    # ==========================================
    elif source_type == "App-based Input (Manual)":
        st.write("Enter data manually into the table below:")
        
        # Template for manual entry
        required_cols = ['Student_ID', 'Course_ID', 'Chapter_Order', 'Time_Spent_Hours', 'Scores', 'Completed']
        
        # Initialize an empty dataframe with columns if not already there or if switching modes
        if 'manual_df' not in st.session_state:
             st.session_state.manual_df = pd.DataFrame(columns=required_cols)
        
        # Allow user to edit
        edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("Use Manual Data"):
            if not edited_df.empty:
                df = edited_df
                st.session_state.manual_df = edited_df # Save state
                st.success(f"Manual data submitted! Loaded {len(df)} records.")
            else:
                st.warning("The table is empty. Please add some rows.")

    
    # ==========================================
    # UNIFIED VALIDATION & PROCESSING
    # ==========================================
    if df is not None:
        # Load data into session state
        st.session_state.data = df
        
        # Check validation
        loader = DataLoader()
        
        try:
            # Validate and filter columns
            df = loader.validate_columns(df)
            st.session_state.data = df  # Update with filtered data
            st.session_state.validated = True
            
            st.markdown('<div class="success-box"> All required columns are present and valid!</div>', 
                      unsafe_allow_html=True)
            
            # Show data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check for missing values
            missing_info = df.isnull().sum()
            total_missing = missing_info.sum()
            
            if total_missing > 0:
                st.warning(f" Found {total_missing} missing values in your data.")
                st.write("Missing values by column:")
                st.write(missing_info[missing_info > 0])
                
                st.markdown("###  Data Info")
                st.write(f"Note: Found {missing_info.sum()} missing values. You can clean these in the EDA page.")
                
        except ValueError as e:
            st.error(f"Validation failed: {str(e)}")
            st.session_state.validated = False

    # Show validation error if no data valid yet but source selected
    elif st.session_state.data is None and source_type == "App-based Input (Manual)":
         # Don't show "Upload file" msg if in manual mode
         pass
    elif st.session_state.data is None and source_type == "API Payload (JSON)":
         pass
    elif st.session_state.data is None:
         st.info(" Please upload a file to get started")

        
    # Add Next Page Button if validated
    if st.session_state.validated:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Use callback for reliable navigation
            st.button(" Go to EDA & Preprocessing", 
                      type="primary", 
                      use_container_width=True,
                      on_click=navigate_to,
                      args=("EDA & Preprocessing",))
