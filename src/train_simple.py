"""
Simplified Training Pipeline for Hackathon
Uses only scikit-learn models (no TensorFlow required)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# Set random seed
np.random.seed(42)


def create_features(df):
    """Create features from consumption data"""
    print("Creating features...")
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Sort by consumer and time
    df = df.sort_values(['consumer_id', 'timestamp'])
    
    # Lag features
    for lag in [1, 24]:
        df[f'consumption_lag_{lag}'] = df.groupby('consumer_id')['consumption_kwh'].shift(lag)
    
    # Rolling features
    df['consumption_mean_24h'] = df.groupby('consumer_id')['consumption_kwh'].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )
    df['consumption_std_24h'] = df.groupby('consumer_id')['consumption_kwh'].transform(
        lambda x: x.rolling(24, min_periods=1).std()
    )
    df['consumption_max_24h'] = df.groupby('consumer_id')['consumption_kwh'].transform(
        lambda x: x.rolling(24, min_periods=1).max()
    )
    df['consumption_min_24h'] = df.groupby('consumer_id')['consumption_kwh'].transform(
        lambda x: x.rolling(24, min_periods=1).min()
    )
    
    # Ratio features
    df['consumption_ratio'] = df['consumption_kwh'] / (df['consumption_mean_24h'] + 0.001)
    df['consumption_range'] = df['consumption_max_24h'] - df['consumption_min_24h']
    
    # Z-score
    df['z_score'] = (df['consumption_kwh'] - df['consumption_mean_24h']) / (df['consumption_std_24h'] + 0.001)
    
    # Zero consumption
    df['is_zero'] = (df['consumption_kwh'] < 0.1).astype(int)
    df['zero_count_24h'] = df.groupby('consumer_id')['is_zero'].transform(
        lambda x: x.rolling(24, min_periods=1).sum()
    )
    
    # Fill NaN
    df = df.fillna(0)
    
    print(f"  ✅ Created {len([c for c in df.columns if c not in ['timestamp', 'consumer_id', 'anomaly_label']])} features")
    
    return df


def train_isolation_forest(X_train, contamination=0.05):
    """Train Isolation Forest model"""
    print("\n--- Training Isolation Forest ---")
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train)
    print("  ✅ Isolation Forest trained!")
    
    return model


def train_statistical_detector(X_train):
    """Train a statistical anomaly detector using Z-scores"""
    print("\n--- Training Statistical Detector ---")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Calculate thresholds
    X_scaled = scaler.transform(X_train)
    thresholds = {
        'mean': np.mean(X_scaled, axis=0),
        'std': np.std(X_scaled, axis=0),
        'q99': np.percentile(np.abs(X_scaled), 99, axis=0)
    }
    
    print("  ✅ Statistical detector trained!")
    
    return scaler, thresholds


def predict_ensemble(iso_forest, scaler, thresholds, X, weights=[0.6, 0.4]):
    """Combine predictions from multiple models"""
    
    # Isolation Forest scores
    iso_scores = -iso_forest.score_samples(X)  # Higher = more anomalous
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 0.001)
    
    # Statistical scores
    X_scaled = scaler.transform(X)
    stat_scores = np.mean(np.abs(X_scaled), axis=1)
    stat_scores_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min() + 0.001)
    
    # Ensemble
    ensemble_scores = weights[0] * iso_scores_norm + weights[1] * stat_scores_norm
    
    return ensemble_scores


def get_risk_level(score):
    """Convert score to risk level"""
    if score >= 0.8:
        return 'Critical'
    elif score >= 0.6:
        return 'High'
    elif score >= 0.4:
        return 'Medium'
    else:
        return 'Low'


def main():
    print("="*60)
    print("ANOMALY DETECTION TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Paths
    consumption_path = 'data/synthetic/consumption_timeseries.csv'
    models_dir = 'data/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # ==================== STEP 1: Load Data ====================
    print("\n" + "="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    
    if not os.path.exists(consumption_path):
        print(f"❌ Data not found at {consumption_path}")
        print("Please run: python scripts/generate_quick_data.py")
        return
    
    df = pd.read_csv(consumption_path)
    print(f"Loaded {len(df):,} records")
    print(f"Consumers: {df['consumer_id'].nunique():,}")
    print(f"Anomaly distribution:\n{df['anomaly_label'].value_counts()}")
    
    # ==================== STEP 2: Feature Engineering ====================
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    
    df = create_features(df)
    
    # Feature columns
    feature_cols = [
        'consumption_kwh', 'hour', 'day_of_week', 'is_weekend', 'is_night',
        'hour_sin', 'hour_cos', 'consumption_lag_1', 'consumption_lag_24',
        'consumption_mean_24h', 'consumption_std_24h', 'consumption_max_24h',
        'consumption_min_24h', 'consumption_ratio', 'consumption_range',
        'z_score', 'is_zero', 'zero_count_24h'
    ]
    
    print(f"Using {len(feature_cols)} features")
    
    # ==================== STEP 3: Train-Test Split ====================
    print("\n" + "="*60)
    print("STEP 3: Train-Test Split")
    print("="*60)
    
    # Time-based split
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Sample for faster training
    sample_size = min(100000, len(train_df))
    train_sample = train_df.sample(n=sample_size, random_state=42)
    
    X_train = train_sample[feature_cols]
    y_train = train_sample['anomaly_label']
    
    test_sample_size = min(50000, len(test_df))
    test_sample = test_df.sample(n=test_sample_size, random_state=42)
    
    X_test = test_sample[feature_cols]
    y_test = test_sample['anomaly_label']
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # ==================== STEP 4: Train Models ====================
    print("\n" + "="*60)
    print("STEP 4: Model Training")
    print("="*60)
    
    # Train Isolation Forest
    iso_forest = train_isolation_forest(X_train, contamination=0.05)
    
    # Train Statistical Detector
    scaler, thresholds = train_statistical_detector(X_train)
    
    # ==================== STEP 5: Evaluation ====================
    print("\n" + "="*60)
    print("STEP 5: Model Evaluation")
    print("="*60)
    
    # Get ensemble predictions
    print("\nGenerating predictions...")
    scores = predict_ensemble(iso_forest, scaler, thresholds, X_test)
    
    # Convert to binary predictions (threshold = 0.5)
    threshold = 0.5
    y_pred = (scores > threshold).astype(int)
    
    # True labels (normal=0, anomaly=1)
    y_true = (y_test != 'normal').astype(int)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"  TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")
    
    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, scores)
        print(f"\n  ROC-AUC Score: {auc:.4f}")
    except:
        auc = 0
    
    # Calculate metrics
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- Summary Metrics ---")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # ==================== STEP 6: Save Models ====================
    print("\n" + "="*60)
    print("STEP 6: Saving Models")
    print("="*60)
    
    # Save models
    joblib.dump(iso_forest, f'{models_dir}/isolation_forest.pkl')
    joblib.dump(scaler, f'{models_dir}/scaler.pkl')
    joblib.dump(thresholds, f'{models_dir}/thresholds.pkl')
    joblib.dump(feature_cols, f'{models_dir}/feature_columns.pkl')
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_cols),
        'metrics': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4)
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(f'{models_dir}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Models saved to {models_dir}/")
    
    # ==================== COMPLETE ====================
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel Performance:")
    print(f"  • Precision: {precision:.1%}")
    print(f"  • Recall: {recall:.1%}")
    print(f"  • F1-Score: {f1:.1%}")
    print(f"  • ROC-AUC: {auc:.4f}")
    print(f"\nNext step: streamlit run dashboard/app.py")
    print("="*60)


if __name__ == '__main__':
    main()
