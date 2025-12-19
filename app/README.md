# Running the Streamlit Application

## Quick Start

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the Model (if not already done):**
```bash
python main.py --mode train --data data/sample_data.csv
```

3. **Launch Streamlit App:**
```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the Web Interface

### Page 1: Upload & Validate
1. Click "Browse files" and upload your CSV file
2. View data preview and validation status
3. Check data summary metrics and missing values

### Page 2: EDA & Preprocessing
1. Explore data distributions and correlations
2. Detect and handle outliers
3. View learning patterns
4. Click "Preprocess Data" button

### Page 3: Predictions & Insights
1. View prediction summary and risk distribution
2. Explore interactive visualizations
3. Filter results by risk level
4. Download predictions as CSV
5. Review student-level insights and chapter difficulty

## Features

✅ Interactive CSV file upload
✅ Automatic data validation
✅ Comprehensive EDA with Plotly charts
✅ Outlier detection with IQR method
✅ Data preprocessing integration
✅ ML model predictions
✅ Risk assessment (HIGH/MEDIUM/LOW)
✅ Chapter difficulty analysis
✅ Downloadable results
