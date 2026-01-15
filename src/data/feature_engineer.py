"""
Feature Engineering Module
Creates derived features from raw consumption data for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


class FeatureEngineer:
    """Extract and engineer features from electrical consumption data"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 24, 168]) -> pd.DataFrame:
        """Create lag features for consumption"""
        df = df.copy()
        df = df.sort_values(['consumer_id', 'timestamp'])
        
        for lag in lags:
            df[f'consumption_lag_{lag}'] = df.groupby('consumer_id')['consumption_kwh'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [24, 168, 720]) -> pd.DataFrame:
        """Create rolling window statistics"""
        df = df.copy()
        df = df.sort_values(['consumer_id', 'timestamp'])
        
        for window in windows:
            window_label = f'{window}h' if window < 168 else f'{window//24}d'
            
            # Rolling statistics
            df[f'consumption_mean_{window_label}'] = (
                df.groupby('consumer_id')['consumption_kwh']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            df[f'consumption_std_{window_label}'] = (
                df.groupby('consumer_id')['consumption_kwh']
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            
            df[f'consumption_min_{window_label}'] = (
                df.groupby('consumer_id')['consumption_kwh']
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )
            
            df[f'consumption_max_{window_label}'] = (
                df.groupby('consumer_id')['consumption_kwh']
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
            
            # Ratio to rolling mean
            df[f'consumption_ratio_{window_label}'] = (
                df['consumption_kwh'] / (df[f'consumption_mean_{window_label}'] + 1e-6)
            )
        
        return df
    
    def create_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rate of change features"""
        df = df.copy()
        df = df.sort_values(['consumer_id', 'timestamp'])
        
        # First derivative (rate of change)
        df['consumption_diff_1h'] = df.groupby('consumer_id')['consumption_kwh'].diff(1)
        df['consumption_diff_24h'] = df.groupby('consumer_id')['consumption_kwh'].diff(24)
        df['consumption_diff_168h'] = df.groupby('consumer_id')['consumption_kwh'].diff(168)
        
        # Percentage change
        df['consumption_pct_change_1h'] = df.groupby('consumer_id')['consumption_kwh'].pct_change(1)
        df['consumption_pct_change_24h'] = df.groupby('consumer_id')['consumption_kwh'].pct_change(24)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features per consumer"""
        df = df.copy()
        
        # Group by consumer and calculate statistics
        consumer_stats = df.groupby('consumer_id')['consumption_kwh'].agg([
            'mean', 'std', 'min', 'max', 'median',
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('skew', lambda x: x.skew()),
            ('kurtosis', lambda x: x.kurtosis())
        ]).reset_index()
        
        consumer_stats.columns = ['consumer_id'] + [
            f'consumer_{col}' for col in consumer_stats.columns[1:]
        ]
        
        df = df.merge(consumer_stats, on='consumer_id', how='left')
        
        # Deviation from consumer's own statistics
        df['deviation_from_mean'] = df['consumption_kwh'] - df['consumer_mean']
        df['z_score'] = (df['consumption_kwh'] - df['consumer_mean']) / (df['consumer_std'] + 1e-6)
        df['iqr'] = df['consumer_q75'] - df['consumer_q25']
        df['is_outlier_iqr'] = (
            (df['consumption_kwh'] < df['consumer_q25'] - 1.5 * df['iqr']) |
            (df['consumption_kwh'] > df['consumer_q75'] + 1.5 * df['iqr'])
        ).astype(int)
        
        return df
    
    def create_zero_consumption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features related to zero or near-zero consumption"""
        df = df.copy()
        df = df.sort_values(['consumer_id', 'timestamp'])
        
        # Zero consumption indicator
        df['is_zero_consumption'] = (df['consumption_kwh'] < 0.1).astype(int)
        
        # Consecutive zero consumption count
        df['zero_consumption_streak'] = (
            df.groupby('consumer_id')['is_zero_consumption']
            .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
        )
        
        # Zero consumption in last 24h/7d
        df['zero_count_24h'] = (
            df.groupby('consumer_id')['is_zero_consumption']
            .transform(lambda x: x.rolling(24, min_periods=1).sum())
        )
        
        df['zero_count_168h'] = (
            df.groupby('consumer_id')['is_zero_consumption']
            .transform(lambda x: x.rolling(168, min_periods=1).sum())
        )
        
        return df
    
    def create_peak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features related to peak consumption"""
        df = df.copy()
        
        # Peak to average ratio
        df['peak_to_avg_ratio'] = df['consumption_kwh'] / (df['consumption_mean_24h'] + 1e-6)
        
        # Is peak hour
        df['is_peak_consumption'] = (
            df['consumption_kwh'] > df['consumption_mean_24h'] + df['consumption_std_24h']
        ).astype(int)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, 
                         include_lags: bool = True,
                         include_rolling: bool = True) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        
        print("Starting feature engineering...")
        original_shape = df.shape
        
        # Time features
        print("  - Creating time features...")
        df = self.create_time_features(df)
        
        # Lag features
        if include_lags:
            print("  - Creating lag features...")
            df = self.create_lag_features(df)
        
        # Rolling features
        if include_rolling:
            print("  - Creating rolling window features...")
            df = self.create_rolling_features(df)
        
        # Derivative features
        print("  - Creating derivative features...")
        df = self.create_derivative_features(df)
        
        # Statistical features
        print("  - Creating statistical features...")
        df = self.create_statistical_features(df)
        
        # Zero consumption features
        print("  - Creating zero consumption features...")
        df = self.create_zero_consumption_features(df)
        
        # Peak features
        print("  - Creating peak features...")
        df = self.create_peak_features(df)
        
        # Handle missing values from lag/rolling operations
        print("  - Handling missing values...")
        df = df.fillna(method='bfill').fillna(0)
        
        print(f"\n✅ Feature engineering complete!")
        print(f"   Original shape: {original_shape}")
        print(f"   Final shape: {df.shape}")
        print(f"   New features added: {df.shape[1] - original_shape[1]}")
        
        # Store feature names (excluding metadata columns)
        exclude_cols = ['timestamp', 'consumer_id', 'consumption_kwh', 'anomaly_label']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names"""
        return self.feature_names


if __name__ == '__main__':
    # Example usage
    print("Feature Engineering Module")
    print("This module is meant to be imported and used in the training pipeline")
