import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class ModelTrainer:
    """Train and evaluate machine learning models for student completion prediction."""
    
    def __init__(self, model=None):
        """
        Initialize ModelTrainer.
        
        Args:
            model: Sklearn model instance. If None, uses automatic selection.
        """
        self.model = model
        self.feature_importance = None
        self.feature_names = None
        self.metrics = {}
        self.best_model_name = "Custom Model" if model else None

    def compare_and_select_best_model(self, X, y):
        """
        Train multiple models and select the best one based on accuracy.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            object: Best trained model
        """
        print("Starting multi-model comparison...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        results = {}
        
        for name, model in models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                mean_score = scores.mean()
                results[name] = mean_score
                print(f"{name}: {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name
            except Exception as e:
                print(f"Failed to train {name}: {str(e)}")
        
        print(f"\nBest Model: {best_name} with Accuracy: {best_score:.4f}")
        
        self.model = best_model
        self.best_model_name = best_name
        
        # Retrain best model on full dataset
        self.model.fit(X, y)
        
        return self.model

    def train_model(self, X, y, validation_split=0.2, auto_select=True):
        """
        Train the machine learning model.
        
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            validation_split (float): Fraction of data to use for validation
            auto_select (bool): If True, compare multiple models and select best.
            
        Returns:
            dict: Training metrics
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        if self.model is None or auto_select:
            self.compare_and_select_best_model(X_train, y_train)
        else:
            print(f"Training {self.model.__class__.__name__}...")
            self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Store metrics
        self.metrics = {
            'model_name': self.best_model_name if self.best_model_name else self.model.__class__.__name__,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'classification_report': classification_report(y_val, val_pred),
            'confusion_matrix': confusion_matrix(y_val, val_pred).tolist()
        }
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        print(f"Training completed!")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return self.metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation on the model.
        
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        print(f"Performing {cv}-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
        
        print(f"Cross-validation Mean Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, top_n=10):
        """
        Get the top N most important features.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model.")
            return None
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath='models/model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path where the model will be saved
            
        Returns:
            str: Path where model was saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath='models/model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            bool: True if successful
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.best_model_name = model_data.get('model_name', self.model.__class__.__name__)
        self.feature_names = model_data.get('feature_names')
        self.feature_importance = model_data.get('feature_importance')
        self.metrics = model_data.get('metrics', {})
        
        print(f"Model loaded from {filepath}")
        return True


# Convenience functions
def train_model(X, y, model=None, save_path='models/model.pkl'):
    """
    Train a model and save it.
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Model instance (optional)
        save_path: Path to save the model
        
    Returns:
        ModelTrainer: Trained model trainer instance
    """
    trainer = ModelTrainer(model=model)
    trainer.train_model(X, y, auto_select=(model is None))
    trainer.save_model(save_path)
    return trainer


def save_model(model, filepath='models/model.pkl'):
    """
    Save a trained model.
    
    Args:
        model: Trained sklearn model
        filepath: Save path
        
    Returns:
        str: Path where model was saved
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model}, filepath)
    return str(filepath)
