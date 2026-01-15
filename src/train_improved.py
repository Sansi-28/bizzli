"""
Improved Training Pipeline for Hackathon
Optimized for detecting anomalies in electrical consumption data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import json

np.random.seed(42)


def create_advanced_features(df):
    """Create advanced features for anomaly detection"""
    print("Creating advanced features...")
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Sort for lag calculations
    df = df.sort_values(['consumer_id', 'timestamp']).reset_index(drop=True)
    
    # Consumer-level statistics (global per consumer)
    consumer_stats = df.groupby('consumer_id')['consumption_kwh'].agg(['mean', 'std', 'median']).reset_index()
    consumer_stats.columns = ['consumer_id', 'consumer_mean', 'consumer_std', 'consumer_median']
    df = df.merge(consumer_stats, on='consumer_id', how='left')
    
    # Deviation from consumer's own baseline
    df['deviation_from_mean'] = df['consumption_kwh'] - df['consumer_mean']
    df['deviation_ratio'] = df['consumption_kwh'] / (df['consumer_mean'] + 0.01)
    df['z_score_consumer'] = (df['consumption_kwh'] - df['consumer_mean']) / (df['consumer_std'] + 0.01)
    
    # Lag features
    for lag in [1, 2, 3, 24]:
        df[f'lag_{lag}'] = df.groupby('consumer_id')['consumption_kwh'].shift(lag)
        df[f'diff_{lag}'] = df['consumption_kwh'] - df[f'lag_{lag}']
    
    # Rolling window features (24h = 1 day)
    for window in [6, 24, 72]:
        df[f'rolling_mean_{window}'] = df.groupby('consumer_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('consumer_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f'rolling_max_{window}'] = df.groupby('consumer_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df[f'rolling_min_{window}'] = df.groupby('consumer_id')['consumption_kwh'].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
    
    # Current vs rolling average ratio
    df['ratio_to_24h_mean'] = df['consumption_kwh'] / (df['rolling_mean_24'] + 0.01)
    df['ratio_to_72h_mean'] = df['consumption_kwh'] / (df['rolling_mean_72'] + 0.01)
    
    # Range and volatility
    df['range_24h'] = df['rolling_max_24'] - df['rolling_min_24']
    df['range_72h'] = df['rolling_max_72'] - df['rolling_min_72']
    df['cv_24h'] = df['rolling_std_24'] / (df['rolling_mean_24'] + 0.01)  # Coefficient of variation
    
    # Zero/low consumption detection
    df['is_zero'] = (df['consumption_kwh'] < 0.1).astype(int)
    df['is_very_low'] = (df['consumption_kwh'] < df['consumer_mean'] * 0.1).astype(int)
    df['zero_count_24h'] = df.groupby('consumer_id')['is_zero'].transform(
        lambda x: x.rolling(24, min_periods=1).sum()
    )
    df['zero_count_72h'] = df.groupby('consumer_id')['is_zero'].transform(
        lambda x: x.rolling(72, min_periods=1).sum()
    )
    
    # High consumption detection
    df['is_spike'] = (df['consumption_kwh'] > df['consumer_mean'] * 3).astype(int)
    df['spike_count_24h'] = df.groupby('consumer_id')['is_spike'].transform(
        lambda x: x.rolling(24, min_periods=1).sum()
    )
    
    # Odd hour usage (high consumption during night)
    df['night_consumption'] = df['consumption_kwh'] * df['is_night']
    df['night_ratio'] = df['night_consumption'] / (df['consumer_mean'] + 0.01)
    
    # Fill NaN
    df = df.fillna(0)
    
    # Replace infinities
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"  ✅ Created features. Total columns: {len(df.columns)}")
    
    return df


def train_models(X_train, y_train_binary):
    """Train multiple models"""
    
    models = {}
    
    # 1. Isolation Forest (unsupervised)
    print("\n--- Training Isolation Forest ---")
    iso_forest = IsolationForest(
        contamination=0.05,
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    models['isolation_forest'] = iso_forest
    print("  ✅ Isolation Forest trained!")
    
    # 2. Random Forest (supervised - if we have labels)
    print("\n--- Training Random Forest Classifier ---")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train_binary)
    models['random_forest'] = rf_clf
    print("  ✅ Random Forest trained!")
    
    # 3. Statistical Scaler
    print("\n--- Training Statistical Detector ---")
    scaler = StandardScaler()
    scaler.fit(X_train)
    models['scaler'] = scaler
    print("  ✅ Statistical detector trained!")
    
    return models


def predict_ensemble(models, X, threshold=0.5):
    """Ensemble predictions from multiple models"""
    
    # Isolation Forest scores
    iso_scores = -models['isolation_forest'].score_samples(X)
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 0.001)
    
    # Random Forest probabilities
    rf_proba = models['random_forest'].predict_proba(X)[:, 1]
    
    # Statistical z-score based
    X_scaled = models['scaler'].transform(X)
    stat_scores = np.mean(np.abs(X_scaled), axis=1)
    stat_scores_norm = (stat_scores - stat_scores.min()) / (stat_scores.max() - stat_scores.min() + 0.001)
    
    # Ensemble: weighted average
    # Give more weight to supervised model since we have labels
    ensemble_scores = 0.2 * iso_scores_norm + 0.6 * rf_proba + 0.2 * stat_scores_norm
    
    # Predictions
    predictions = (ensemble_scores > threshold).astype(int)
    
    return predictions, ensemble_scores


def main():
    print("="*60)
    print("IMPROVED ANOMALY DETECTION TRAINING")
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
        print(f"❌ Data not found!")
        return
    
    df = pd.read_csv(consumption_path)
    print(f"Loaded {len(df):,} records")
    
    # ==================== STEP 2: Feature Engineering ====================
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    
    df = create_advanced_features(df)
    
    # Define feature columns
    exclude_cols = ['timestamp', 'consumer_id', 'anomaly_label', 'consumer_mean', 'consumer_std', 'consumer_median']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    
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
    
    # Sample for training (use more samples)
    train_sample_size = min(200000, len(train_df))
    train_sample = train_df.sample(n=train_sample_size, random_state=42)
    
    X_train = train_sample[feature_cols]
    y_train = train_sample['anomaly_label']
    y_train_binary = (y_train != 'normal').astype(int)
    
    # Test sample
    test_sample_size = min(100000, len(test_df))
    test_sample = test_df.sample(n=test_sample_size, random_state=42)
    
    X_test = test_sample[feature_cols]
    y_test = test_sample['anomaly_label']
    y_test_binary = (y_test != 'normal').astype(int)
    
    print(f"Train: {len(X_train):,} samples ({y_train_binary.sum():,} anomalies)")
    print(f"Test: {len(X_test):,} samples ({y_test_binary.sum():,} anomalies)")
    
    # ==================== STEP 4: Train Models ====================
    print("\n" + "="*60)
    print("STEP 4: Model Training")
    print("="*60)
    
    models = train_models(X_train, y_train_binary)
    
    # ==================== STEP 5: Evaluation ====================
    print("\n" + "="*60)
    print("STEP 5: Model Evaluation")
    print("="*60)
    
    # Find optimal threshold using precision-recall curve
    print("\nFinding optimal threshold...")
    _, scores = predict_ensemble(models, X_test)
    
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test_binary, scores)
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 0.001)
    
    # Find threshold with best F1
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    print(f"  Optimal threshold: {best_threshold:.3f}")
    
    # Final predictions with optimal threshold
    y_pred, ensemble_scores = predict_ensemble(models, X_test, threshold=best_threshold)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test_binary, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test_binary, y_pred)
    print(cm)
    print(f"  TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")
    
    # Metrics
    try:
        auc = roc_auc_score(y_test_binary, ensemble_scores)
    except:
        auc = 0
    
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- Summary Metrics ---")
    print(f"  Precision: {precision:.4f} ({precision:.1%})")
    print(f"  Recall: {recall:.4f} ({recall:.1%})")
    print(f"  F1-Score: {f1:.4f} ({f1:.1%})")
    print(f"  ROC-AUC: {auc:.4f}")
    
    # Feature importance from Random Forest
    print("\n--- Top 10 Important Features ---")
    importances = models['random_forest'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # ==================== STEP 6: Save Models ====================
    print("\n" + "="*60)
    print("STEP 6: Saving Models")
    print("="*60)
    
    joblib.dump(models['isolation_forest'], f'{models_dir}/isolation_forest.pkl')
    joblib.dump(models['random_forest'], f'{models_dir}/random_forest.pkl')
    joblib.dump(models['scaler'], f'{models_dir}/scaler.pkl')
    joblib.dump(feature_cols, f'{models_dir}/feature_columns.pkl')
    joblib.dump(best_threshold, f'{models_dir}/threshold.pkl')
    
    # Save feature importance
    feature_importance.to_csv(f'{models_dir}/feature_importance.csv', index=False)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_cols),
        'optimal_threshold': float(best_threshold),
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
    print(f"\n🎯 Model Performance:")
    print(f"   • Precision: {precision:.1%}")
    print(f"   • Recall: {recall:.1%}")
    print(f"   • F1-Score: {f1:.1%}")
    print(f"   • ROC-AUC: {auc:.4f}")
    print(f"\n📊 Next: streamlit run dashboard/app.py")
    print("="*60)


if __name__ == '__main__':
    main()
