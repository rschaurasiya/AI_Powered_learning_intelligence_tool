# AI-Powered Learning Intelligence Tool

A comprehensive machine learning system for analyzing student learning patterns, predicting completion/dropout risk, and generating actionable insights from educational data.

## Features

- **Data Loading & Validation**: Accepts CSV/JSON data with automatic column validation
- **Smart Preprocessing**: Handles missing values, feature encoding, and scaling
- **ML Model Training**: RandomForest classifier for completion prediction
- **Risk Assessment**: Calculates student risk scores (HIGH/MEDIUM/LOW) based on engagement metrics
- **Chapter Difficulty Analysis**: Identifies challenging content using completion rates and time spent
- **Comprehensive Insights**: Generates detailed analytics reports

## Required Data Format

Your CSV/JSON file must contain these columns:

| Column | Description | Type |
|--------|-------------|------|
| `Student_ID` | Unique student identifier | String/Integer |
| `Course_ID` | Course identifier | String/Integer |
| `Chapter_Order` | Chapter sequence number | Integer |
| `Time_Spent_Hours` | Time spent on chapter (hours) | Float |
| `Scores` | Achievement score | Float |
| `Completed` | Completion status (0/1 or True/False) | Boolean/Integer |

## Installation

```bash
# Clone or download the project
cd prasunet_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train a Model

```bash
python main.py --mode train --data data/sample_data.csv --model models/model.pkl
```

### 2. Make Predictions

```bash
python main.py --mode predict --data data/new_students.csv --model models/model.pkl --output predictions.csv
```

### 3. Generate Insights

```bash
python main.py --mode analyze --data data/sample_data.csv --output insights_report.json
```

### 4. Run Complete Pipeline

```bash
python main.py --mode all --data data/sample_data.csv --model models/model.pkl
```

## Project Structure

```
prasunet_project/
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   └── sample_data.csv    # Sample dataset
├── models/
│   └── model.pkl          # Trained model (generated)
├── src/
│   ├── data_loader.py     # Data loading & validation
│   ├── preprocessor.py    # Data preprocessing
│   ├── model_trainer.py   # Model training
│   ├── predictior.py      # Predictions
│   └── insights.py        # Analytics & insights
└── tests/
    └── test+predictor.py  # Unit tests
```

## Risk Score Calculation

Student risk levels are determined using:

- **HIGH RISK**: Completion rate < 30% AND average time < 2 hours
- **MEDIUM RISK**: Completion rate < 60%
- **LOW RISK**: All other cases

## Chapter Difficulty Formula

Difficulty score combines completion rates and normalized time:

```
Difficulty = (1 - completion_rate) × 0.6 + (normalized_time) × 0.4
```

Where:
- Low completion rates indicate harder chapters (60% weight)
- High time spent indicates harder chapters (40% weight)
- Scores range from 0 (easiest) to 1 (hardest)

## Output Files

- **`predictions.csv`**: Student predictions with completion probabilities and risk levels
- **`insights_report.json`**: Comprehensive JSON report with all analytics
- **`student_risks.csv`**: Individual student risk assessments
- **`chapter_difficulty.csv`**: Chapter-level difficulty analysis

## Example Code Usage

### Load and Validate Data

```python
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data('data/sample_data.csv')
summary = loader.get_data_summary(df)
print(f"Loaded {summary['total_students']} students")
```

### Calculate Risk Scores

```python
from src.insights import InsightsGenerator

insights = InsightsGenerator()
student_risks = insights.calculate_student_risk_scores(df)
print(student_risks)
```

### Calculate Chapter Difficulty

```python
chapter_difficulty = insights.calculate_chapter_difficulty(df)
print(chapter_difficulty.head())
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test+predictor.py -v
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- pytest >= 7.4.0 (for testing)

## License

This project is for educational purposes.

## Support

For issues or questions, please refer to the code documentation or contact the development team.
