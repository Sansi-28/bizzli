"""
Fast Data Generator for Hackathon Demo
Generates smaller but realistic dataset quickly
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

np.random.seed(42)
random.seed(42)

def generate_quick_dataset(n_consumers=1000, n_days=90, anomaly_rate=0.05):
    """
    Generate a smaller dataset quickly for demo purposes
    
    Args:
        n_consumers: Number of consumers (default 1000)
        n_days: Days of data (default 90 = 3 months)
        anomaly_rate: Percentage of anomalous consumers
    """
    print("="*60)
    print("FAST DATA GENERATOR FOR HACKATHON")
    print("="*60)
    print(f"Consumers: {n_consumers}")
    print(f"Days: {n_days}")
    print(f"Anomaly rate: {anomaly_rate*100}%")
    print("="*60)
    
    output_dir = 'data/synthetic'
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Generate Consumer Metadata ====================
    print("\n1. Generating consumer metadata...")
    
    consumer_types = ['residential', 'commercial', 'industrial']
    connection_types = ['single_phase', 'three_phase']
    
    consumers = []
    for i in range(n_consumers):
        c_type = random.choices(consumer_types, weights=[0.7, 0.2, 0.1])[0]
        
        if c_type == 'residential':
            base = np.random.uniform(5, 15)
        elif c_type == 'commercial':
            base = np.random.uniform(20, 100)
        else:
            base = np.random.uniform(100, 500)
        
        consumers.append({
            'consumer_id': f'C{str(i+1).zfill(6)}',
            'consumer_type': c_type,
            'connection_type': random.choice(connection_types),
            'latitude': np.random.uniform(28.4, 28.8),  # Delhi region
            'longitude': np.random.uniform(77.0, 77.4),
            'base_consumption': base,
            'is_anomalous': i < int(n_consumers * anomaly_rate)
        })
    
    consumers_df = pd.DataFrame(consumers)
    consumers_df.to_csv(f'{output_dir}/consumers_metadata.csv', index=False)
    print(f"   ✅ Saved {len(consumers_df)} consumers")
    
    # ==================== Generate Consumption Data ====================
    print("\n2. Generating consumption time series...")
    
    start_date = datetime(2025, 10, 15)
    hours = n_days * 24
    timestamps = [start_date + timedelta(hours=h) for h in range(hours)]
    
    # Hourly patterns
    residential_pattern = np.array([
        0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 0.7, 0.5, 0.5,
        0.6, 0.6, 0.5, 0.5, 0.6, 0.7, 0.9, 1.0, 0.9, 0.8, 0.6, 0.4
    ])
    commercial_pattern = np.array([
        0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2
    ])
    industrial_pattern = np.array([
        0.8, 0.8, 0.8, 0.8, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0,
        0.95, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8
    ])
    
    patterns = {
        'residential': residential_pattern,
        'commercial': commercial_pattern,
        'industrial': industrial_pattern
    }
    
    anomaly_types = ['sudden_spike', 'zero_consumption', 'odd_hour_usage', 
                     'gradual_theft', 'erratic_pattern']
    
    all_data = []
    
    for idx, consumer in consumers_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx+1}/{n_consumers}")
        
        base = consumer['base_consumption']
        pattern = patterns[consumer['consumer_type']]
        
        # Generate base consumption
        consumption = np.zeros(hours)
        for h in range(hours):
            hour_of_day = h % 24
            day = h // 24
            
            # Base + pattern + noise
            value = base * pattern[hour_of_day]
            value *= np.random.normal(1.0, 0.15)  # Add noise
            
            # Weekend reduction
            if (start_date + timedelta(days=day)).weekday() >= 5:
                value *= 0.8
            
            consumption[h] = max(0, value)
        
        # Determine anomaly type
        if consumer['is_anomalous']:
            anom_type = random.choice(anomaly_types)
            
            if anom_type == 'sudden_spike':
                spike_start = hours // 2
                consumption[spike_start:] *= np.random.uniform(2.5, 4.0)
                
            elif anom_type == 'zero_consumption':
                zero_start = hours // 2
                zero_len = min(24*7, hours - zero_start)
                consumption[zero_start:zero_start+zero_len] *= 0.05
                
            elif anom_type == 'odd_hour_usage':
                for i in range(0, hours, 24):
                    if i+5 < hours:
                        consumption[i+2:i+5] *= 4.0
                        consumption[i+10:i+16] *= 0.4
                        
            elif anom_type == 'gradual_theft':
                decline = np.linspace(1.0, 0.3, hours // 2)
                consumption[hours//2:] *= decline[:len(consumption[hours//2:])]
                
            elif anom_type == 'erratic_pattern':
                noise = np.random.uniform(0.3, 3.0, hours)
                consumption *= noise
        else:
            anom_type = 'normal'
        
        # Create records
        for h in range(hours):
            all_data.append({
                'timestamp': timestamps[h],
                'consumer_id': consumer['consumer_id'],
                'consumption_kwh': round(consumption[h], 4),
                'anomaly_label': anom_type
            })
    
    print("\n3. Creating DataFrame...")
    consumption_df = pd.DataFrame(all_data)
    
    print("\n4. Saving data...")
    consumption_df.to_csv(f'{output_dir}/consumption_timeseries.csv', index=False)
    print(f"   ✅ Saved {len(consumption_df):,} consumption records")
    
    # Summary
    summary = {
        'total_consumers': n_consumers,
        'n_days': n_days,
        'total_readings': len(consumption_df),
        'anomalous_consumers': int(n_consumers * anomaly_rate),
        'date_range': {
            'start': str(start_date.date()),
            'end': str((start_date + timedelta(days=n_days)).date())
        },
        'anomaly_distribution': consumption_df['anomaly_label'].value_counts().to_dict()
    }
    
    with open(f'{output_dir}/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"Consumers: {n_consumers:,}")
    print(f"Records: {len(consumption_df):,}")
    print(f"Anomalous consumers: {int(n_consumers * anomaly_rate)}")
    print(f"\nAnomalies by type:")
    for atype, count in summary['anomaly_distribution'].items():
        pct = count / len(consumption_df) * 100
        print(f"  - {atype}: {count:,} ({pct:.1f}%)")
    print("="*60)
    
    return consumers_df, consumption_df


if __name__ == '__main__':
    # Generate quick dataset for demo
    generate_quick_dataset(
        n_consumers=1000,   # 1K consumers
        n_days=90,          # 3 months
        anomaly_rate=0.05   # 5% anomalies
    )
