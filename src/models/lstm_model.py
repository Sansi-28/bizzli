"""
LSTM Model for Time-Series Anomaly Detection
Deep learning approach for temporal patterns
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os


class LSTMDetector:
    """LSTM-based anomaly detector for time-series data"""
    
    def __init__(self, sequence_length=24, lstm_units=64, learning_rate=0.001):
        """
        Args:
            sequence_length: Length of input sequences (e.g., 24 for 24 hours)
            lstm_units: Number of LSTM units
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None
        self.threshold = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def create_sequences(self, data: np.ndarray, sequence_length: int):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, n_features: int):
        """Build LSTM architecture"""
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, activation='relu', 
                       return_sequences=True,
                       input_shape=(self.sequence_length, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units // 2, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_features, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def fit(self, X: pd.DataFrame, feature_columns: list = None,
            epochs=30, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        print("Training LSTM...")
        
        if feature_columns is not None:
            self.feature_columns = feature_columns
            X_train = X[feature_columns].values
        else:
            X_train = X.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, self.sequence_length)
        
        print(f"  Sequences shape: {X_seq.shape}")
        
        # Build model
        if self.model is None:
            self.build_model(n_features=X_seq.shape[2])
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=False,  # Don't shuffle time series
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Calculate threshold from prediction error on training data
        predictions = self.model.predict(X_seq, verbose=0)
        mse = np.mean(np.square(y_seq - predictions), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        print(f"  ✅ Training complete!")
        print(f"  Prediction error threshold: {self.threshold:.6f}")
        
        return history
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction error as anomaly score
        Returns: Prediction errors (higher = more anomalous)
        """
        if self.feature_columns is not None:
            X_pred = X[self.feature_columns].values
        else:
            X_pred = X.values
        
        X_scaled = self.scaler.transform(X_pred)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, self.sequence_length)
        
        if len(X_seq) == 0:
            return np.array([])
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        mse = np.mean(np.square(y_seq - predictions), axis=1)
        
        # Pad the beginning (sequences not available for first sequence_length points)
        mse_padded = np.concatenate([np.zeros(self.sequence_length), mse])
        
        return mse_padded
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies based on prediction error
        Returns: 1 for anomalies, 0 for normal
        """
        mse = self.predict_proba(X)
        predictions = (mse > self.threshold).astype(int)
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """Evaluate model performance"""
        print("\nEvaluating LSTM...")
        
        # Predict
        y_pred = self.predict(X)
        scores = self.predict_proba(X)
        
        # Align lengths (due to padding)
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        scores = scores[:min_len]
        
        # Convert true labels to binary
        y_true_binary = (y_true != 'normal').astype(int)
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_true_binary, y_pred, 
                                   target_names=['Normal', 'Anomaly']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true_binary, y_pred)
        print(cm)
        
        try:
            auc_score = roc_auc_score(y_true_binary, scores)
            print(f"\nROC-AUC Score: {auc_score:.4f}")
        except:
            auc_score = None
            print("\nROC-AUC Score: Could not compute")
        
        results = {
            'predictions': y_pred,
            'scores': scores,
            'confusion_matrix': cm,
            'roc_auc': auc_score
        }
        
        return results
    
    def save(self, filepath: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        self.model.save(filepath.replace('.pkl', '_keras.h5'))
        
        # Save other attributes
        joblib.dump({
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'learning_rate': self.learning_rate,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        
        detector = cls(
            sequence_length=data['sequence_length'],
            lstm_units=data['lstm_units'],
            learning_rate=data['learning_rate']
        )
        
        # Load Keras model
        detector.model = keras.models.load_model(filepath.replace('.pkl', '_keras.h5'))
        detector.threshold = data['threshold']
        detector.feature_columns = data['feature_columns']
        detector.scaler = data['scaler']
        
        return detector


if __name__ == '__main__':
    print("LSTM Anomaly Detector")
    print("This module is meant to be imported and used in the training pipeline")
