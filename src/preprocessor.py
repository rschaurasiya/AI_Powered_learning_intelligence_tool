"""
Data Preprocessor Module
Handles data cleaning, feature engineering, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Preprocess student learning data for machine learning."""
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def handle_missing_values(self, df, threshold=0.05):
        """
        Handle missing values using a smart strategy:
        - If missing % < threshold: Remove rows
        - If missing % >= threshold: Impute (mean for numeric, mode for categorical)
        
        Args:
            df (pd.DataFrame): Input DataFrame
            threshold (float): Threshold for removing vs imputing (default 5%)
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
            dict: Summary of actions taken per column
        """
        df_copy = df.copy()
        actions = {}
        initial_len = len(df_copy)
        
        # Calculate missing percentage per column
        missing_stats = df_copy.isnull().mean()
        columns_with_missing = missing_stats[missing_stats > 0].index.tolist()
        
        # Track rows to drop (indices)
        rows_to_drop = set()
        
        for col in columns_with_missing:
            missing_pct = missing_stats[col]
            
            if missing_pct < threshold:
                # Strategy: Remove rows
                null_indices = df_copy[df_copy[col].isnull()].index.tolist()
                rows_to_drop.update(null_indices)
                actions[col] = f"Removed rows ({missing_pct:.1%})"
            else:
                # Strategy: Impute
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    imputer = SimpleImputer(strategy='mean')
                    df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                    actions[col] = f"Imputed with mean ({missing_pct:.1%})"
                    
                    # Special handling for binary/integer columns like Completed
                    if col == 'Completed':
                        df_copy[col] = df_copy[col].round().astype(int)
                        
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                    actions[col] = f"Imputed with mode ({missing_pct:.1%})"
        
        # Drop accumulated rows
        if rows_to_drop:
            df_copy = df_copy.drop(index=list(rows_to_drop)).reset_index(drop=True)
            actions['global'] = f"Dropped {len(rows_to_drop)} rows total"
            
        return df_copy, actions
    
    def handle_outliers(self, df):
        """
        Handle outliers using IQR method - cap outliers to valid range.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with outliers capped
            dict: Summary of actions taken per column
        """
        df_copy = df.copy()
        actions = {}
        
        # Get numeric columns, excluding binary and ID columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        cols_to_process = [col for col in numeric_cols 
                          if col not in ['Completed', 'Student_ID', 'Course_ID']]
        
        for col in cols_to_process:
            try:
                # Calculate IQR
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers before capping
                outliers_count = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    # Cap outliers
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                    actions[col] = f"Capped {outliers_count} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]"
                    
            except Exception as e:
                actions[col] = f"Error processing outliers: {str(e)}"
                
        if not actions:
            actions['result'] = "No outliers detected"
            
        return df_copy, actions
    
    def encode_features(self, df, categorical_columns=None):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            categorical_columns (list): List of categorical column names to encode
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        df_copy = df.copy()
        
        if categorical_columns is None:
            # Auto-detect categorical columns (object type or low cardinality)
            categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
            # Also include ID columns that might be numeric but categorical
            for col in ['Student_ID', 'Course_ID']:
                if col in df_copy.columns and col not in categorical_columns:
                    categorical_columns.append(col)
        
        # Encode each categorical column
        for col in categorical_columns:
            if col in df_copy.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    # Use existing encoder for consistency
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        
        return df_copy
    
    def scale_data(self, df, columns_to_scale=None, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns_to_scale (list): List of columns to scale. If None, scale all numeric columns.
            fit (bool): Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df_copy = df.copy()
        
        if columns_to_scale is None:
            # Scale all numeric columns except binary/categorical
            columns_to_scale = []
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                # Don't scale binary columns or IDs
                if col not in ['Completed'] and not col.endswith('_ID'):
                    columns_to_scale.append(col)
        
        if len(columns_to_scale) > 0:
            if fit:
                df_copy[columns_to_scale] = self.scaler.fit_transform(df_copy[columns_to_scale])
            else:
                df_copy[columns_to_scale] = self.scaler.transform(df_copy[columns_to_scale])
        
        return df_copy
    
    def create_features(self, df):
        """
        Create additional features for better predictions.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        df_copy = df.copy()
        
        # Feature 1: Time per chapter (efficiency metric)
        df_copy['Time_Per_Chapter'] = df_copy['Time_Spent_Hours'] / (df_copy['Chapter_Order'] + 1)
        
        # Feature 2: Score rate (performance metric)
        df_copy['Score_Rate'] = df_copy['Scores'] / (df_copy['Time_Spent_Hours'] + 0.1)  # Avoid division by zero
        
        # Feature 3: Progress indicator (how far in the course)
        if 'Chapter_Order' in df_copy.columns:
            max_chapter = df_copy['Chapter_Order'].max()
            df_copy['Progress_Ratio'] = df_copy['Chapter_Order'] / (max_chapter + 1)
        
        # Feature 4: Engagement score (combination of time and completion)
        df_copy['Engagement_Score'] = df_copy['Time_Spent_Hours'] * df_copy['Completed']
        
        return df_copy
    
    def prepare_features(self, df, target_column='Completed', fit=True):
        """
        Complete preprocessing pipeline: handle missing values, outliers, create features, encode, and scale.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column
            fit (bool): Whether to fit transformers (True for training, False for prediction)
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is target vector
        """
        try:
            # Step 1: Handle missing values
            df_processed, missing_actions = self.handle_missing_values(df)
            
            # Step 2: Handle outliers
            df_processed, outlier_actions = self.handle_outliers(df_processed)
            
            # Step 3: Create additional features
            df_processed = self.create_features(df_processed)
            
            # Step 4: Separate features and target
            if target_column in df_processed.columns:
                y = df_processed[target_column].values
                X = df_processed.drop(columns=[target_column])
            else:
                y = None
                X = df_processed
            
            # Step 5: Encode categorical features
            categorical_cols = ['Student_ID', 'Course_ID']
            X = self.encode_features(X, categorical_columns=categorical_cols)
            
            # Step 6: Scale numerical features
            X = self.scale_data(X, fit=fit)
            
            # Store feature names for later use
            if fit:
                self.feature_names = X.columns.tolist()
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"Error in preprocessing pipeline: {str(e)}")
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor (for prediction).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        X, _ = self.prepare_features(df, fit=False)
        return X


# Convenience functions
def preprocess_data(df, target_column='Completed'):
    """
    Preprocess data for training.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Target column name
        
    Returns:
        tuple: (X, y, preprocessor)
    """
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features(df, target_column=target_column, fit=True)
    return X, y, preprocessor
