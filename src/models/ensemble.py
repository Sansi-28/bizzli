"""
Ensemble Anomaly Detection Model
Combines multiple models for robust detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os


class EnsembleDetector:
    """Ensemble of multiple anomaly detection models"""
    
    def __init__(self, models: dict = None, weights: dict = None):
        """
        Args:
            models: Dictionary of {'model_name': model_object}
            weights: Dictionary of {'model_name': weight}
        """
        self.models = models or {}
        self.weights = weights or {}
        self.threshold = 0.5  # Default threshold for ensemble score
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble anomaly scores
        Returns: Weighted average of model scores
        """
        self.normalize_weights()
        
        ensemble_scores = np.zeros(len(X))
        
        for name, model in self.models.items():
            print(f"  Getting predictions from {name}...")
            
            try:
                # Get scores from each model
                scores = model.predict_proba(X)
                
                # Normalize scores to [0, 1] range
                scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                
                # Add weighted score
                weight = self.weights.get(name, 1.0)
                ensemble_scores += weight * scores_normalized
                
            except Exception as e:
                print(f"    Warning: Could not get scores from {name}: {e}")
                continue
        
        return ensemble_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies using ensemble
        Returns: 1 for anomalies, 0 for normal
        """
        scores = self.predict_proba(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions
    
    def get_risk_level(self, score: float) -> str:
        """Convert anomaly score to risk level"""
        if score >= 0.9:
            return 'Critical'
        elif score >= 0.7:
            return 'High'
        elif score >= 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def predict_with_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with detailed information
        Returns: DataFrame with scores, predictions, and risk levels
        """
        print("\nGenerating ensemble predictions...")
        
        # Get ensemble scores
        scores = self.predict_proba(X)
        predictions = (scores > self.threshold).astype(int)
        risk_levels = [self.get_risk_level(s) for s in scores]
        
        # Get individual model scores
        individual_scores = {}
        for name, model in self.models.items():
            try:
                model_scores = model.predict_proba(X)
                # Normalize
                model_scores = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min() + 1e-10)
                individual_scores[f'{name}_score'] = model_scores
            except:
                individual_scores[f'{name}_score'] = np.zeros(len(X))
        
        # Create results DataFrame
        results = pd.DataFrame({
            'ensemble_score': scores,
            'is_anomaly': predictions,
            'risk_level': risk_levels,
            **individual_scores
        })
        
        return results
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """Evaluate ensemble performance"""
        print("\n" + "="*60)
        print("Evaluating Ensemble Model")
        print("="*60)
        
        # Predict
        y_pred = self.predict(X)
        scores = self.predict_proba(X)
        
        # Convert true labels to binary
        y_true_binary = (y_true != 'normal').astype(int)
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_true_binary, y_pred, 
                                   target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true_binary, y_pred)
        print(cm)
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        try:
            auc_score = roc_auc_score(y_true_binary, scores)
            print(f"\nROC-AUC Score: {auc_score:.4f}")
        except:
            auc_score = None
            print("\nROC-AUC Score: Could not compute")
        
        # Calculate metrics
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nSummary Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        results = {
            'predictions': y_pred,
            'scores': scores,
            'confusion_matrix': cm,
            'roc_auc': auc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return results
    
    def save(self, filepath: str):
        """Save ensemble configuration"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'threshold': self.threshold
        }, filepath)
        
        print(f"Ensemble model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load ensemble from disk"""
        data = joblib.load(filepath)
        
        ensemble = cls(
            models=data['models'],
            weights=data['weights']
        )
        ensemble.threshold = data['threshold']
        
        return ensemble


if __name__ == '__main__':
    print("Ensemble Anomaly Detector")
    print("This module is meant to be imported and used in the training pipeline")
