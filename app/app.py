"""
AI-Powered Learning Intelligence Tool - Streamlit Web Application
Upload CSV data, perform EDA, and get student completion predictions with insights.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import src modules if needed by submodules
sys.path.append(str(Path(__file__).parent.parent))

# Import page modules
# We import them inside the file to ensure sys.path is set first, though here it is top level.
# Ideally these should be relative imports if part of a package, but as independent scripts 
# running via streamlit run app/app.py, this works if they are in the same dir.
import upload_validate
import EDA_preprocessing
import prediction_insight

# Page configuration
st.set_page_config(
    page_title="AI Learning Intelligence Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #28a745;
        color: white;
        border-left: 4px solid #1e7e34;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #856404;   /* dark warning yellow */
        color: #ffffff;              /* white text */
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Helper functions for navigation and storage
def navigate_to(page):
    st.session_state.page = page


def cleanup_data(current_page, target_page):
    """Clean up data when navigating backwards."""
    if current_page == "Predictions & Insights" and target_page == "EDA & Preprocessing":
        # Clear predictions implies clearing preprocessed data so it must be re-done?
        # User said: "current page data will deleted".
        # Predictions page generates predictions on the fly, so they are deleted by leaving the page.
        # But let's also clear preprocessed_data so user can re-run preprocessing if they want.
        # This effectively resets the "Forward" state from EDA.
        st.session_state.preprocessed_data = None
    
    if current_page == "EDA & Preprocessing" and target_page == "Upload & Validate":
        # Clear everything derived from upload AND the upload itself
        st.session_state.validated = False
        st.session_state.cleaning_actions = None
        st.session_state.preprocessed_data = None
        st.session_state.cleaned_data = None
        st.session_state.data = None # Force re-upload


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header"> AI-Powered Learning Intelligence Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze student learning patterns, predict completion risk, and generate actionable insights**")
    
    # Initialize session state FIRST (before any access to session_state variables)
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'validated' not in st.session_state:
        st.session_state.validated = False
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'page' not in st.session_state:
        st.session_state.page = "Upload & Validate"
    if 'cleaning_actions' not in st.session_state:
        st.session_state.cleaning_actions = None
    
    # Sidebar
    st.sidebar.header("Navigation")
    
    # Use session state for page navigation
    page = st.sidebar.radio("Select a page:", 
                           ["Upload & Validate", "EDA & Preprocessing", "Predictions & Insights"],
                           index=["Upload & Validate", "EDA & Preprocessing", "Predictions & Insights"].index(st.session_state.page))
    
    # Sync sidebar selection with session state if changed manually
    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()

    # ========================================
    # PAGE DISPATCHER
    # ========================================
    if st.session_state.page == "Upload & Validate":
        upload_validate.show(navigate_to)
    
    elif st.session_state.page == "EDA & Preprocessing":
        EDA_preprocessing.show(navigate_to, cleanup_data)
        
    elif st.session_state.page == "Predictions & Insights":
        prediction_insight.show(navigate_to, cleanup_data)


if __name__ == "__main__":
    main()
