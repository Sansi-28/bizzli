"""
Data Preprocessing Module
Cleans and prepares raw consumption data for modeling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os


class DataPreprocessor:
    """Preprocess electrical consumption data"""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, consumption_path: str, metadata_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load consumption and metadata files"""
        print(f"Loading data...")
        consumption = pd.read_csv(consumption_path)
        metadata = pd.read_csv(metadata_path)
        
        print(f"  - Consumption records: {len(consumption):,}")
        print(f"  - Consumers: {len(metadata):,}")
        
        return consumption, metadata
    
    def merge_with_metadata(self, consumption: pd.DataFrame, 
                           metadata: pd.DataFrame) -> pd.DataFrame:
        """Merge consumption data with consumer metadata"""
        df = consumption.merge(metadata, on='consumer_id', how='left')
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        missing_before = df.isnull().sum().sum()
        
        # Fill missing consumption with 0 (likely meter issues)
        df['consumption_kwh'] = df['consumption_kwh'].fillna(0)
        
        # Forward fill other missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        missing_after = df.isnull().sum().sum()
        
        print(f"  - Missing values before: {missing_before}")
        print(f"  - Missing values after: {missing_after}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       column: str = 'consumption_kwh',
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """Remove extreme outliers from consumption data"""
        print(f"Removing outliers using {method} method...")
        
        original_len = len(df)
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < threshold]
        
        removed = original_len - len(df)
        print(f"  - Removed {removed:,} outliers ({removed/original_len*100:.2f}%)")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        # Consumer type one-hot encoding
        if 'consumer_type' in df.columns:
            consumer_type_dummies = pd.get_dummies(df['consumer_type'], prefix='type')
            df = pd.concat([df, consumer_type_dummies], axis=1)
        
        # Connection type one-hot encoding
        if 'connection_type' in df.columns:
            connection_type_dummies = pd.get_dummies(df['connection_type'], prefix='connection')
            df = pd.concat([df, connection_type_dummies], axis=1)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                          feature_columns: list,
                          scaler_type: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features"""
        print(f"Normalizing features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        
        # Only normalize numeric columns
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        self.feature_columns = numeric_features
        
        print(f"  - Normalized {len(numeric_features)} features")
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        print(f"Splitting data (test_size={test_size})...")
        
        if time_based:
            # Time-based split (last test_size% of data)
            df = df.sort_values('timestamp')
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            # Random split by consumer
            consumers = df['consumer_id'].unique()
            test_consumers = np.random.choice(
                consumers, 
                size=int(len(consumers) * test_size),
                replace=False
            )
            train_df = df[~df['consumer_id'].isin(test_consumers)]
            test_df = df[df['consumer_id'].isin(test_consumers)]
        
        print(f"  - Train set: {len(train_df):,} records")
        print(f"  - Test set: {len(test_df):,} records")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           output_dir: str = 'data/processed'):
        """Save processed data and scaler"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving processed data to {output_dir}...")
        
        train_df.to_csv(f'{output_dir}/train_data.csv', index=False)
        test_df.to_csv(f'{output_dir}/test_data.csv', index=False)
        
        if self.scaler is not None:
            joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
            joblib.dump(self.feature_columns, f'{output_dir}/feature_columns.pkl')
        
        print("  ✅ Processed data saved!")
    
    def preprocess_pipeline(self, 
                           consumption_path: str,
                           metadata_path: str,
                           output_dir: str = 'data/processed',
                           remove_outliers: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete preprocessing pipeline"""
        
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # Load data
        consumption, metadata = self.load_data(consumption_path, metadata_path)
        
        # Merge with metadata
        print("\nMerging with metadata...")
        df = self.merge_with_metadata(consumption, metadata)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers (optional)
        if remove_outliers:
            df = self.remove_outliers(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Final dataset shape: {df.shape}")
        
        return df


if __name__ == '__main__':
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Paths
    consumption_path = 'data/synthetic/consumption_timeseries.csv'
    metadata_path = 'data/synthetic/consumers_metadata.csv'
    
    # Run preprocessing
    df = preprocessor.preprocess_pipeline(
        consumption_path=consumption_path,
        metadata_path=metadata_path,
        output_dir='data/processed'
    )
    
    print("\n" + "="*60)
    print("Preprocessing complete! Data ready for feature engineering.")
    print("="*60)
