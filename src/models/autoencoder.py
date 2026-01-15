"""
Autoencoder Model for Anomaly Detection
Deep learning approach using reconstruction error
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


class AutoencoderDetector:
    """Autoencoder-based anomaly detector"""
    
    def __init__(self, encoding_dim=32, learning_rate=0.001):
        """
        Args:
            encoding_dim: Dimension of encoded representation
            learning_rate: Learning rate for optimizer
        """
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.threshold = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim: int):
        """Build autoencoder architecture"""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = autoencoder
        return autoencoder
    
    def fit(self, X: pd.DataFrame, feature_columns: list = None,
            epochs=50, batch_size=32, validation_split=0.2):
        """Train the autoencoder on normal data"""
        print("Training Autoencoder...")
        
        if feature_columns is not None:
            self.feature_columns = feature_columns
            X_train = X[feature_columns]
        else:
            X_train = X
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Build model
        if self.model is None:
            self.build_model(input_dim=X_scaled.shape[1])
        
        # Train
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Calculate threshold from reconstruction error on training data
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        print(f"  ✅ Training complete!")
        print(f"  Reconstruction threshold: {self.threshold:.6f}")
        
        return history
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get reconstruction error as anomaly score
        Returns: Reconstruction errors (higher = more anomalous)
        """
        if self.feature_columns is not None:
            X_pred = X[self.feature_columns]
        else:
            X_pred = X
        
        X_scaled = self.scaler.transform(X_pred)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        return mse
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error
        Returns: 1 for anomalies, 0 for normal
        """
        mse = self.predict_proba(X)
        predictions = (mse > self.threshold).astype(int)
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> dict:
        """Evaluate model performance"""
        print("\nEvaluating Autoencoder...")
        
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
            'encoding_dim': self.encoding_dim,
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
            encoding_dim=data['encoding_dim'],
            learning_rate=data['learning_rate']
        )
        
        # Load Keras model
        detector.model = keras.models.load_model(filepath.replace('.pkl', '_keras.h5'))
        detector.threshold = data['threshold']
        detector.feature_columns = data['feature_columns']
        detector.scaler = data['scaler']
        
        return detector


if __name__ == '__main__':
    print("Autoencoder Anomaly Detector")
    print("This module is meant to be imported and used in the training pipeline")
