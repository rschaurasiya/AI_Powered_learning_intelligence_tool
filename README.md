# AI-Powered Learning Intelligence Tool

A machine learning system designed to analyze student engagement, predict completion risks, and optimize educational content. This tool empowers educators with actionable insights derived from data, allowing for proactive intervention and curriculum improvement.

## Core Capabilities

*   **Completion Prediction**: specific algorithms assess student engagement patterns to forecast course completion likelihood.
*   **Risk Stratification**: automatically categorizes students into High, Medium, or Low risk groups for targeted support.
*   **Chapter Difficulty Analysis**: identifies specific course sections causing student drop-offs using a weighted difficulty index.
*   **Strategic Insights**: generates executive summaries that highlight key factors affecting student success.

## Technical Architecture

The system is built on a robust Python stack:
*   **Frontend**: Streamlit for an interactive, web-based dashboard.
*   **Machine Learning**: Scikit-learn and XGBoost for ensemble modeling and classification.
*   **Data Processing**: Pandas and NumPy for efficient data manipulation and cleaning.
*   **Visualization**: Plotly for dynamic, specific charts and graphs.

## Quick Start

### Prerequisites
*   Python 3.8+
*   pip

### Installation

1.  Clone the repository and navigate to the project directory.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To launch the web interface:
```bash
streamlit run app/app.py
```
The dashboard will open in your default browser at `http://localhost:8501`.

## Data Format

The system expects input data (CSV or JSON) with the following columns:

*   **Student_ID**: Unique identifier.
*   **Course_ID**: Course identifier.
*   **Chapter_Order**: Sequence number of the chapter.
*   **Time_Spent_Hours**: Duration spent on the material.
*   **Scores**: Assessment result (0-100).
*   **Completed**: Status indicator (1 for complete, 0 for incomplete).

## AI Methodology

### Model Selection
The system employs an automated ensemble approach, training multiple classifiers (Random Forest, XGBoost, Logistic Regression) and selecting the highest-performing model based on validation accuracy. This ensures reliability across varying datasets.

### Risk & Difficulty Logic
*   **Risk Score**: Calculated based on the ratio of completion rate to average time spent. Students with low completion and low engagement time are flagged as High Risk.
*   **Difficulty Score**: A composite metric weighing completion rate (60%) and normalized time spent (40%) to pinpoint challenging content.
