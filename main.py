
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.predictior import Predictor
from src.insights import InsightsGenerator


def train_pipeline(data_path, model_path='models/model.pkl'):
    """
    Complete training pipeline: load data, preprocess, train model.
    
    Args:
        data_path (str): Path to training data CSV/JSON
        model_path (str): Path to save the trained model
    """
    print("\n" + "="*60)
    print("TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    loader = DataLoader()
    df = loader.load_data(data_path)
    print(f"[OK] Loaded {len(df)} records")
    
    # Show data summary
    summary = loader.get_data_summary(df)
    print(f"\nData Summary:")
    print(f"  - Total records: {summary['total_records']}")
    print(f"  - Unique students: {summary['unique_students']}")
    print(f"  - Unique courses: {summary['unique_courses']}")
    print(f"  - Overall completion rate: {summary['completion_rate']:.2%}")
    
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features(df, target_column='Completed', fit=True)
    print(f"[OK] Preprocessed {len(X)} samples with {X.shape[1]} features")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    trainer = ModelTrainer()
    metrics = trainer.train_model(X, y)
    print(f"[OK] Model trained successfully")
    
    # Show feature importance
    importance = trainer.get_feature_importance(top_n=5)
    if importance is not None:
        print("\nTop 5 Important Features:")
        for idx, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 4: Save model and preprocessor
    print(f"\nStep 4: Saving model to {model_path}...")
    model_data = {
        'model': trainer.model,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names,
        'metrics': metrics
    }
    
    # Ensure models directory exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump(model_data, model_path)
    print(f"[OK] Model and preprocessor saved successfully")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60 + "\n")
    
    return trainer, preprocessor


def predict_pipeline(data_path, model_path='models/model.pkl', output_path='predictions.csv'):
    """
    Prediction pipeline: load model, make predictions on new data.
    
    Args:
        data_path (str): Path to data for prediction
        model_path (str): Path to trained model
        output_path (str): Path to save predictions
    """
    print("\n" + "="*60)
    print("PREDICTION PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    loader = DataLoader()
    df = loader.load_data(data_path)
    print(f"[OK] Loaded {len(df)} records")
    
    # Step 2: Load model and preprocessor
    print(f"\nStep 2: Loading model from {model_path}...")
    import joblib
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    print("[OK] Model and preprocessor loaded")
    
    # Step 3: Preprocess data
    print("\nStep 3: Preprocessing data...")
    # Don't use target column for prediction
    df_temp = df.copy()
    if 'Completed' not in df_temp.columns:
        df_temp['Completed'] = 0  # Dummy target
    
    X, _ = preprocessor.prepare_features(df_temp, target_column='Completed', fit=False)
    print(f"[OK] Preprocessed {len(X)} samples")
    
    # Step 4: Make predictions
    print("\nStep 4: Making predictions...")
    predictor = Predictor(model_path=model_path)
    predictor.model = model
    
    predictions = predictor.predict_with_confidence(X)
    print("[OK] Predictions completed")
    
    # Step 5: Save results
    print(f"\nStep 5: Saving predictions to {output_path}...")
    results = pd.concat([df.reset_index(drop=True), predictions], axis=1)
    results.to_csv(output_path, index=False)
    print(f"[OK] Predictions saved successfully")
    
    # Show summary
    print("\nPrediction Summary:")
    print(f"  - Total predictions: {len(predictions)}")
    print(f"  - Predicted completions: {predictions['prediction'].sum()}")
    print(f"  - Predicted dropouts: {len(predictions) - predictions['prediction'].sum()}")
    print(f"  - High risk students: {(predictions['risk_level'] == 'High').sum()}")
    print(f"  - Medium risk students: {(predictions['risk_level'] == 'Medium').sum()}")
    print(f"  - Low risk students: {(predictions['risk_level'] == 'Low').sum()}")
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETED")
    print("="*60 + "\n")
    
    return results


def analyze_pipeline(data_path, output_path='insights_report.json'):
    """
    Analysis pipeline: generate insights and analytics reports.
    
    Args:
        data_path (str): Path to data for analysis
        output_path (str): Path to save insights report
    """
    print("\n" + "="*60)
    print("ANALYSIS PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    loader = DataLoader()
    df = loader.load_data(data_path)
    print(f"[OK] Loaded {len(df)} records")
    
    # Step 2: Generate insights
    print("\nStep 2: Generating insights...")
    insights_gen = InsightsGenerator()
    
    # Calculate student risk scores
    print("\n  Calculating student risk scores...")
    student_risks = insights_gen.calculate_student_risk_scores(df)
    print(f"  [OK] Analyzed {len(student_risks)} students")
    
    # Calculate chapter difficulty
    print("  Calculating chapter difficulty...")
    chapter_difficulty = insights_gen.calculate_chapter_difficulty(df)
    print(f"  [OK] Analyzed {len(chapter_difficulty)} chapters")
    
    # Generate summary report
    print("  Generating summary report...")
    report = insights_gen.generate_summary_report(df)
    print("  [OK] Report generated")
    
    # Step 3: Display insights
    print("\n" + "-"*60)
    print("INSIGHTS SUMMARY")
    print("-"*60)
    
    print("\nOverall Statistics:")
    for key, value in report['overall'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nStudent Risk Distribution:")
    for key, value in report['risk_distribution'].items():
        print(f"  {key}: {value}")
    
    print("\nChapter Difficulty Insights:")
    for key, value in report['chapter_insights'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTop 5 Most Difficult Chapters:")
    for idx, row in chapter_difficulty.head(5).iterrows():
        print(f"  Chapter {int(row['Chapter_Order'])}: "
              f"Difficulty={row['Difficulty_Score']:.3f}, "
              f"Completion={row['Completion_Rate']:.2%}")
    
    print("\nTop 5 Easiest Chapters:")
    for idx, row in chapter_difficulty.tail(5).iterrows():
        print(f"  Chapter {int(row['Chapter_Order'])}: "
              f"Difficulty={row['Difficulty_Score']:.3f}, "
              f"Completion={row['Completion_Rate']:.2%}")
    
    # Step 4: Save report
    print(f"\nStep 3: Saving report to {output_path}...")
    
    # Convert DataFrames to JSON-serializable format
    report_json = {
        'overall': {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v 
                    for k, v in report['overall'].items()},
        'risk_distribution': {k: int(v) for k, v in report['risk_distribution'].items()},
        'chapter_insights': {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v 
                            for k, v in report['chapter_insights'].items()},
        'student_risks': student_risks.to_dict(orient='records'),
        'chapter_difficulty': chapter_difficulty.to_dict(orient='records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    
    print(f"[OK] Report saved successfully")
    
    # Also save detailed CSVs
    student_risks.to_csv('student_risks.csv', index=False)
    chapter_difficulty.to_csv('chapter_difficulty.csv', index=False)
    print("[OK] Detailed reports saved to student_risks.csv and chapter_difficulty.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60 + "\n")
    
    return report


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='AI-Powered Learning Intelligence Tool'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict', 'analyze', 'all'],
        help='Operation mode: train, predict, analyze, or all'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/model.pkl',
        help='Path to model file (default: models/model.pkl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output file (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train_pipeline(args.data, args.model)
        
        elif args.mode == 'predict':
            output = args.output or 'predictions.csv'
            predict_pipeline(args.data, args.model, output)
        
        elif args.mode == 'analyze':
            output = args.output or 'insights_report.json'
            analyze_pipeline(args.data, output)
        
        elif args.mode == 'all':
            # Run complete pipeline
            print("\nRunning complete pipeline...\n")
            train_pipeline(args.data, args.model)
            predict_pipeline(args.data, args.model)
            analyze_pipeline(args.data)
        
        print("\n[OK] All operations completed successfully!\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
