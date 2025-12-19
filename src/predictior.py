"""
Predictor Module
Handles loading trained models and making predictions for student completion/dropout.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class Predictor:
    """Load trained models and make predictions on student data."""
    
    def __init__(self, model_path='models/model.pkl'):
        """
        Initialize Predictor.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        
    def load_model(self, filepath=None):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the model file. If None, uses self.model_path
            
        Returns:
            bool: True if successful
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if filepath is None:
            filepath = self.model_path
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Extract model and metadata
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names')
        else:
            # Legacy format - just the model
            self.model = model_data
        
        print(f"Model loaded from {filepath}")
        return True
    
    def predict_completion(self, student_data):
        """
        Predict completion probability for students.
        
        Args:
            student_data (pd.DataFrame or np.ndarray): Student feature data
            
        Returns:
            np.ndarray: Completion probabilities (0-1)
            
        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array if DataFrame
        if isinstance(student_data, pd.DataFrame):
            X = student_data.values
        else:
            X = student_data
        
        # Get probability predictions
        if hasattr(self.model, 'predict_proba'):
            # Probability of completion (class 1)
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            # Fallback to binary predictions
            probabilities = self.model.predict(X).astype(float)
        
        return probabilities
    
    def predict_dropout_risk(self, student_data, threshold=0.5):
        """
        Predict binary dropout risk for students.
        
        Args:
            student_data (pd.DataFrame or np.ndarray): Student feature data
            threshold (float): Probability threshold for dropout prediction
            
        Returns:
            np.ndarray: Binary predictions (0 = will complete, 1 = dropout risk)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get completion probabilities
        completion_probs = self.predict_completion(student_data)
        
        # Convert to dropout risk (inverse of completion)
        dropout_risk = (completion_probs < threshold).astype(int)
        
        return dropout_risk
    
    def predict_with_confidence(self, student_data):
        """
        Predict completion with confidence scores.
        
        Args:
            student_data (pd.DataFrame or np.ndarray): Student feature data
            
        Returns:
            pd.DataFrame: Predictions with confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get predictions and probabilities
        completion_probs = self.predict_completion(student_data)
        predictions = (completion_probs >= 0.5).astype(int)
        
        # Calculate confidence (distance from 0.5 threshold)
        confidence = np.abs(completion_probs - 0.5) * 2  # Scale to 0-1
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'completion_probability': completion_probs,
            'confidence': confidence,
            'risk_level': ['Low' if p >= 0.7 else 'Medium' if p >= 0.4 else 'High' 
                          for p in completion_probs]
        })
        
        return results
    
    def predict_batch(self, student_data, include_confidence=True):
        """
        Make batch predictions on multiple students.
        
        Args:
            student_data (pd.DataFrame): Student feature data
            include_confidence (bool): Whether to include confidence scores
            
        Returns:
            pd.DataFrame: Predictions for all students
        """
        if include_confidence:
            return self.predict_with_confidence(student_data)
        else:
            completion_probs = self.predict_completion(student_data)
            return pd.DataFrame({
                'prediction': (completion_probs >= 0.5).astype(int),
                'completion_probability': completion_probs
            })


# Convenience functions
def load_model(filepath='models/model.pkl'):
    """
    Load a trained model and return a Predictor instance.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        Predictor: Initialized predictor with loaded model
    """
    predictor = Predictor(model_path=filepath)
    predictor.load_model()
    return predictor


def predict_completion(model_path, student_data):
    """
    Quick prediction function.
    
    Args:
        model_path (str): Path to the model file
        student_data: Student feature data
        
    Returns:
        np.ndarray: Completion probabilities
    """
    predictor = load_model(model_path)
    return predictor.predict_completion(student_data)


def predict_dropout_risk(model_path, student_data, threshold=0.5):
    """
    Quick dropout risk prediction.
    
    Args:
        model_path (str): Path to the model file
        student_data: Student feature data
        threshold (float): Risk threshold
        
    Returns:
        np.ndarray: Binary dropout risk predictions
    """
    predictor = load_model(model_path)
    return predictor.predict_dropout_risk(student_data, threshold)
