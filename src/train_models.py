"""
Main Training Pipeline
Orchestrates data processing, feature engineering, and model training
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector
from models.lstm_model import LSTMDetector
from models.ensemble import EnsembleDetector


def main():
    print("="*60)
    print("ANOMALY DETECTION TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Paths
    consumption_path = 'data/synthetic/consumption_timeseries.csv'
    metadata_path = 'data/synthetic/consumers_metadata.csv'
    models_dir = 'data/models'
    
    # Check if data exists
    if not os.path.exists(consumption_path):
        print(f"❌ Data not found at {consumption_path}")
        print("Please run: python scripts/generate_sample_data.py")
        return
    
    # ==================== STEP 1: Data Preprocessing ====================
    print("\n" + "="*60)
    print("STEP 1: Data Preprocessing")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_pipeline(
        consumption_path=consumption_path,
        metadata_path=metadata_path,
        remove_outliers=False  # Keep outliers as they might be anomalies
    )
    
    # ==================== STEP 2: Feature Engineering ====================
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_features(
        df,
        include_lags=True,
        include_rolling=True
    )
    
    # Get feature names
    feature_columns = engineer.get_feature_names()
    print(f"\nTotal features: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns[:10]}... (showing first 10)")
    
    # ==================== STEP 3: Train-Test Split ====================
    print("\n" + "="*60)
    print("STEP 3: Train-Test Split")
    print("="*60)
    
    # Time-based split
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df):,} records")
    print(f"Test set: {len(test_df):,} records")
    
    # Separate features and labels
    X_train = train_df[feature_columns]
    y_train = train_df['anomaly_label']
    
    X_test = test_df[feature_columns]
    y_test = test_df['anomaly_label']
    
    print(f"\nAnomaly distribution in train set:")
    print(y_train.value_counts())
    print(f"\nAnomaly distribution in test set:")
    print(y_test.value_counts())
    
    # ==================== STEP 4: Model Training ====================
    print("\n" + "="*60)
    print("STEP 4: Model Training")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 4.1 Isolation Forest
    print("\n--- Training Isolation Forest ---")
    iso_forest = IsolationForestDetector(contamination=0.05, n_estimators=100)
    iso_forest.fit(X_train, feature_columns=feature_columns)
    iso_forest.save(f'{models_dir}/isolation_forest.pkl')
    
    # 4.2 Autoencoder (on a sample for speed)
    print("\n--- Training Autoencoder ---")
    # Use sample for faster training
    sample_size = min(50000, len(X_train))
    train_sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train.iloc[train_sample_idx]
    
    autoencoder = AutoencoderDetector(encoding_dim=32, learning_rate=0.001)
    autoencoder.fit(
        X_train_sample, 
        feature_columns=feature_columns,
        epochs=20,  # Reduced for hackathon speed
        batch_size=128,
        validation_split=0.2
    )
    autoencoder.save(f'{models_dir}/autoencoder.pkl')
    
    # 4.3 LSTM (on even smaller sample for speed)
    print("\n--- Training LSTM ---")
    # Use smaller sample for LSTM (more computationally expensive)
    lstm_sample_size = min(10000, len(X_train))
    lstm_sample_idx = np.random.choice(len(X_train), lstm_sample_size, replace=False)
    X_train_lstm = X_train.iloc[lstm_sample_idx]
    
    # Use fewer features for LSTM
    important_features = [f for f in feature_columns if any(x in f for x in 
                         ['consumption', 'hour', 'day', 'mean', 'std'])][:20]
    
    lstm = LSTMDetector(sequence_length=24, lstm_units=64, learning_rate=0.001)
    lstm.fit(
        X_train_lstm,
        feature_columns=important_features,
        epochs=10,  # Reduced for hackathon speed
        batch_size=64,
        validation_split=0.2
    )
    lstm.save(f'{models_dir}/lstm.pkl')
    
    # ==================== STEP 5: Ensemble Model ====================
    print("\n" + "="*60)
    print("STEP 5: Building Ensemble Model")
    print("="*60)
    
    ensemble = EnsembleDetector()
    ensemble.add_model('isolation_forest', iso_forest, weight=0.3)
    ensemble.add_model('autoencoder', autoencoder, weight=0.35)
    ensemble.add_model('lstm', lstm, weight=0.35)
    
    ensemble.save(f'{models_dir}/ensemble.pkl')
    
    # ==================== STEP 6: Evaluation ====================
    print("\n" + "="*60)
    print("STEP 6: Model Evaluation")
    print("="*60)
    
    # Evaluate on test set (use sample for speed)
    test_sample_size = min(10000, len(X_test))
    test_sample_idx = np.random.choice(len(X_test), test_sample_size, replace=False)
    X_test_sample = X_test.iloc[test_sample_idx]
    y_test_sample = y_test.iloc[test_sample_idx]
    
    print("\n--- Individual Model Evaluations ---")
    
    # Isolation Forest
    print("\n1. ISOLATION FOREST")
    iso_results = iso_forest.evaluate(X_test_sample, y_test_sample.values)
    
    # Autoencoder
    print("\n2. AUTOENCODER")
    ae_results = autoencoder.evaluate(X_test_sample, y_test_sample.values)
    
    # LSTM
    print("\n3. LSTM")
    # Use same important features
    lstm_results = lstm.evaluate(
        X_test_sample[important_features] if len(X_test_sample) > lstm.sequence_length else X_test_sample,
        y_test_sample.values
    )
    
    # Ensemble
    print("\n4. ENSEMBLE")
    ensemble_results = ensemble.evaluate(X_test_sample, y_test_sample.values)
    
    # ==================== STEP 7: Save Results ====================
    print("\n" + "="*60)
    print("STEP 7: Saving Results")
    print("="*60)
    
    # Save feature columns
    pd.DataFrame({'feature': feature_columns}).to_csv(
        f'{models_dir}/feature_columns.csv', index=False
    )
    
    # Save evaluation results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'n_features': len(feature_columns),
        'models': {
            'isolation_forest': {
                'roc_auc': float(iso_results.get('roc_auc', 0)) if iso_results.get('roc_auc') else 0
            },
            'autoencoder': {
                'roc_auc': float(ae_results.get('roc_auc', 0)) if ae_results.get('roc_auc') else 0
            },
            'lstm': {
                'roc_auc': float(lstm_results.get('roc_auc', 0)) if lstm_results.get('roc_auc') else 0
            },
            'ensemble': {
                'roc_auc': float(ensemble_results.get('roc_auc', 0)) if ensemble_results.get('roc_auc') else 0,
                'precision': float(ensemble_results.get('precision', 0)),
                'recall': float(ensemble_results.get('recall', 0)),
                'f1_score': float(ensemble_results.get('f1_score', 0))
            }
        }
    }
    
    import json
    with open(f'{models_dir}/training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results saved to {models_dir}/training_results.json")
    
    # ==================== COMPLETION ====================
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels saved in: {models_dir}/")
    print("\nNext steps:")
    print("1. Run dashboard: streamlit run dashboard/app.py")
    print("2. Or start API: uvicorn api.main:app --reload")
    print("="*60)


if __name__ == '__main__':
    main()
