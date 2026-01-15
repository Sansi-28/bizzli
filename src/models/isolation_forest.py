"""
Isolation Forest Model for Anomaly Detection
Unsupervised learning approach to identify anomalies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os


class IsolationForestDetector:
    """Isolation Forest based anomaly detector"""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        """
        Args:
            contamination: Expected proportion of anomalies in dataset
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.contamination = contamination
        self.feature_columns = None
        
    def fit(self, X: pd.DataFrame, feature_columns: list = None):
        """Train the model on normal data"""
        print("Training Isolation Forest...")
        
        if feature_columns is not None:
            self.feature_columns = feature_columns
            X_train = X[feature_columns]
        else:
            X_train = X
            
        self.model.fit(X_train)
        print("  ✅ Training complete!")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies
        Returns: -1 for anomalies, 1 for normal
        """
        if self.feature_columns is not None:
            X_pred = X[self.feature_columns]
        else:
            X_pred = X
            
        predictions = self.model.predict(X_pred)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower scores = more anomalous)
        Returns: Anomaly scores between -1 and 1
        """
        if self.feature_columns is not None:
            X_pred = X[self.feature_columns]
        else:
            X_pred = X
            
        scores = self.model.score_samples(X_pred)
        return scores
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """Evaluate model performance"""
        print("\nEvaluating Isolation Forest...")
        
        # Predict
        y_pred = self.predict(X)
        y_pred_binary = (y_pred == -1).astype(int)  # Convert to binary
        
        # Convert true labels to binary (assuming 'normal' = 0, others = 1)
        y_true_binary = (y_true != 'normal').astype(int)
        
        # Anomaly scores
        scores = -self.predict_proba(X)  # Negate so higher = more anomalous
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_true_binary, y_pred_binary, 
                                   target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        print(cm)
        
        try:
            auc_score = roc_auc_score(y_true_binary, scores)
            print(f"\nROC-AUC Score: {auc_score:.4f}")
        except:
            auc_score = None
            print("\nROC-AUC Score: Could not compute")
        
        results = {
            'predictions': y_pred_binary,
            'scores': scores,
            'confusion_matrix': cm,
            'roc_auc': auc_score
        }
        
        return results
    
    def save(self, filepath: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        detector = cls(contamination=data['contamination'])
        detector.model = data['model']
        detector.feature_columns = data['feature_columns']
        return detector


if __name__ == '__main__':
    print("Isolation Forest Anomaly Detector")
    print("This module is meant to be imported and used in the training pipeline")
