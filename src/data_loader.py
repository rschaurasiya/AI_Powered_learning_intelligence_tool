"""
Data Loader Module
Handles reading CSV/JSON files and validates required columns for learning analytics.
"""

import pandas as pd
import json
from pathlib import Path


class DataLoader:
    """Load and validate student learning data from CSV or JSON files."""
    
    REQUIRED_COLUMNS = [
        'Student_ID',
        'Course_ID',
        'Chapter_Order',
        'Time_Spent_Hours',
        'Scores',
        'Completed'
    ]
    
    def __init__(self):
        """Initialize DataLoader."""
        pass
    
    def load_data(self, filepath):
        """
        Load data from CSV or JSON file.
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine file type and load accordingly
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix.lower() == '.json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Only CSV and JSON are supported.")
        
        # Validate columns
        df = self.validate_columns(df)
        
        return df
    
    def validate_columns(self, df):
        """
        Validate and filter DataFrame columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with only required columns
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Required columns are: {self.REQUIRED_COLUMNS}"
            )
            
        # Extract only required columns (ignore extra columns)
        df_filtered = df[self.REQUIRED_COLUMNS].copy()
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df_filtered['Time_Spent_Hours']):
            try:
                df_filtered['Time_Spent_Hours'] = pd.to_numeric(df_filtered['Time_Spent_Hours'])
            except:
                raise ValueError("Time_Spent_Hours must be numeric")
        
        if not pd.api.types.is_numeric_dtype(df_filtered['Scores']):
            try:
                df_filtered['Scores'] = pd.to_numeric(df_filtered['Scores'])
            except:
                raise ValueError("Scores must be numeric")
        
        # Validate Completed column (should be 0 or 1, or boolean)
        return df_filtered
    
    def get_data_summary(self, df):
        """
        Get a summary of the loaded data.
        
        Args:
            df (pd.DataFrame): DataFrame to summarize
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_records': len(df),
            'unique_students': df['Student_ID'].nunique(),
            'unique_courses': df['Course_ID'].nunique(),
            'unique_chapters': df['Chapter_Order'].nunique(),
            'completion_rate': df['Completed'].mean(),
            'avg_time_spent': df['Time_Spent_Hours'].mean(),
            'avg_score': df['Scores'].mean(),
            'missing_values': df.isnull().sum().to_dict()
        }
        return summary


# Convenience functions for direct use
def load_data(filepath):
    """
    Load data from CSV or JSON file.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded and validated data
    """
    loader = DataLoader()
    return loader.load_data(filepath)


def validate_columns(df):
    """
    Validate DataFrame columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if valid
    """
    loader = DataLoader()
    return loader.validate_columns(df)
