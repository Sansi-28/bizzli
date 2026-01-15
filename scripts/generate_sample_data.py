"""
Data Generator for Synthetic Electrical Consumption Data
Generates realistic consumption patterns with anomalies for testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import random
import json
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class ElectricalDataGenerator:
    """Generate synthetic electrical consumption data with anomalies"""
    
    def __init__(self, n_consumers=10000, n_days=365, anomaly_rate=0.05):
        """
        Args:
            n_consumers: Number of consumers to generate
            n_days: Number of days of historical data
            anomaly_rate: Percentage of consumers with anomalous behavior
        """
        self.n_consumers = n_consumers
        self.n_days = n_days
        self.anomaly_rate = anomaly_rate
        self.start_date = datetime(2025, 1, 15) - timedelta(days=n_days)
        
    def generate_consumer_metadata(self):
        """Generate consumer profile metadata"""
        consumer_types = ['residential', 'commercial', 'industrial']
        connection_types = ['single_phase', 'three_phase']
        
        metadata = []
        for i in range(self.n_consumers):
            consumer_type = random.choice(consumer_types)
            
            # Base consumption varies by type
            if consumer_type == 'residential':
                base_consumption = np.random.uniform(5, 15)  # kWh per day
            elif consumer_type == 'commercial':
                base_consumption = np.random.uniform(20, 100)
            else:  # industrial
                base_consumption = np.random.uniform(100, 500)
            
            metadata.append({
                'consumer_id': f'C{str(i+1).zfill(6)}',
                'name': fake.company() if consumer_type != 'residential' else fake.name(),
                'consumer_type': consumer_type,
                'connection_type': random.choice(connection_types),
                'latitude': float(fake.latitude()),
                'longitude': float(fake.longitude()),
                'address': fake.address().replace('\n', ', '),
                'meter_id': f'M{str(i+1).zfill(6)}',
                'installation_date': fake.date_between(start_date='-5y', end_date='-1y'),
                'base_consumption': base_consumption,
                'is_anomalous': i < int(self.n_consumers * self.anomaly_rate)
            })
        
        return pd.DataFrame(metadata)
    
    def generate_hourly_pattern(self, consumer_type):
        """Generate typical hourly consumption pattern"""
        if consumer_type == 'residential':
            # Peak in morning (7-9) and evening (18-22)
            pattern = [
                0.3, 0.2, 0.2, 0.2, 0.2, 0.3,  # 0-5 (night)
                0.5, 0.8, 1.0, 0.7, 0.5, 0.5,  # 6-11 (morning peak)
                0.6, 0.6, 0.5, 0.5, 0.6, 0.7,  # 12-17 (afternoon)
                0.9, 1.0, 0.9, 0.8, 0.6, 0.4   # 18-23 (evening peak)
            ]
        elif consumer_type == 'commercial':
            # Peak during business hours (9-18)
            pattern = [
                0.2, 0.2, 0.2, 0.2, 0.2, 0.3,  # 0-5
                0.4, 0.6, 0.8, 1.0, 1.0, 1.0,  # 6-11
                1.0, 1.0, 1.0, 0.9, 0.8, 0.7,  # 12-17
                0.5, 0.4, 0.3, 0.3, 0.2, 0.2   # 18-23
            ]
        else:  # industrial
            # Relatively constant with slight variations
            pattern = [
                0.8, 0.8, 0.8, 0.8, 0.8, 0.85,  # 0-5
                0.9, 0.95, 1.0, 1.0, 1.0, 1.0,  # 6-11
                0.95, 0.95, 0.95, 0.9, 0.9, 0.85,  # 12-17
                0.85, 0.85, 0.8, 0.8, 0.8, 0.8  # 18-23
            ]
        
        return np.array(pattern)
    
    def add_seasonal_variation(self, base_consumption, day_of_year):
        """Add seasonal variation (higher in summer/winter)"""
        # Peak in summer (day 150-250) and winter (day 0-50, 300-365)
        season_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        return base_consumption * season_factor
    
    def add_weekly_variation(self, consumption, day_of_week):
        """Add weekly variation (lower on weekends)"""
        if day_of_week >= 5:  # Weekend
            return consumption * 0.8
        return consumption
    
    def inject_anomaly(self, data, consumer_info, anomaly_type):
        """Inject specific anomaly pattern"""
        
        if anomaly_type == 'sudden_spike':
            # Sudden dramatic increase (possible illegal connection)
            spike_start = np.random.randint(len(data) // 2, len(data))
            data[spike_start:] *= np.random.uniform(2.5, 5.0)
            
        elif anomaly_type == 'zero_consumption':
            # Periods of zero consumption (meter bypass)
            zero_start = np.random.randint(len(data) // 2, len(data) - 24*7)
            zero_duration = np.random.randint(24*3, 24*14)  # 3-14 days
            data[zero_start:zero_start+zero_duration] *= 0.1
            
        elif anomaly_type == 'odd_hour_usage':
            # High consumption at odd hours (2-5 AM)
            for i in range(0, len(data), 24):
                data[i+2:i+5] *= np.random.uniform(3.0, 5.0)
                data[i+9:i+18] *= 0.5  # Low during normal hours
                
        elif anomaly_type == 'gradual_theft':
            # Gradual decrease in reported consumption
            decline_start = len(data) // 3
            decline_factor = np.linspace(1.0, 0.3, len(data) - decline_start)
            data[decline_start:] *= decline_factor
            
        elif anomaly_type == 'erratic_pattern':
            # Very irregular consumption pattern
            noise = np.random.uniform(0.3, 3.0, len(data))
            data *= noise
            
        return data
    
    def generate_consumption_timeseries(self, consumer_info):
        """Generate hourly consumption time series for a consumer"""
        
        hours = self.n_days * 24
        timestamps = [self.start_date + timedelta(hours=h) for h in range(hours)]
        
        # Get base pattern
        hourly_pattern = self.generate_hourly_pattern(consumer_info['consumer_type'])
        base_consumption = consumer_info['base_consumption']
        
        # Generate consumption
        consumption = []
        for h in range(hours):
            hour_of_day = h % 24
            day = h // 24
            day_of_week = (self.start_date + timedelta(days=day)).weekday()
            day_of_year = (self.start_date + timedelta(days=day)).timetuple().tm_yday
            
            # Base consumption with patterns
            value = base_consumption * hourly_pattern[hour_of_day]
            
            # Add variations
            value = self.add_seasonal_variation(value, day_of_year)
            value = self.add_weekly_variation(value, day_of_week)
            
            # Add random noise
            value *= np.random.normal(1.0, 0.1)
            
            consumption.append(max(0, value))
        
        consumption = np.array(consumption)
        
        # Inject anomaly if marked
        if consumer_info['is_anomalous']:
            anomaly_types = ['sudden_spike', 'zero_consumption', 'odd_hour_usage', 
                           'gradual_theft', 'erratic_pattern']
            anomaly_type = random.choice(anomaly_types)
            consumption = self.inject_anomaly(consumption, consumer_info, anomaly_type)
            anomaly_label = anomaly_type
        else:
            anomaly_label = 'normal'
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'consumer_id': consumer_info['consumer_id'],
            'consumption_kwh': consumption,
            'anomaly_label': anomaly_label
        })
        
        return df
    
    def generate_dataset(self, output_dir='data/synthetic'):
        """Generate complete dataset"""
        
        print(f"Generating data for {self.n_consumers} consumers...")
        print(f"Anomaly rate: {self.anomaly_rate*100}%")
        
        # Generate consumer metadata
        print("\n1. Generating consumer metadata...")
        consumers_df = self.generate_consumer_metadata()
        
        # Save metadata
        os.makedirs(output_dir, exist_ok=True)
        consumers_df.to_csv(f'{output_dir}/consumers_metadata.csv', index=False)
        print(f"   Saved: {output_dir}/consumers_metadata.csv")
        
        # Generate consumption data
        print("\n2. Generating consumption time series...")
        all_consumption = []
        
        for idx, consumer in consumers_df.iterrows():
            if (idx + 1) % 500 == 0:
                print(f"   Progress: {idx+1}/{self.n_consumers} consumers")
            
            consumption_df = self.generate_consumption_timeseries(consumer)
            all_consumption.append(consumption_df)
        
        # Combine all consumption data
        print("\n3. Combining data...")
        consumption_full = pd.concat(all_consumption, ignore_index=True)
        
        # Save consumption data
        consumption_full.to_csv(f'{output_dir}/consumption_timeseries.csv', index=False)
        print(f"   Saved: {output_dir}/consumption_timeseries.csv")
        
        # Generate summary statistics
        print("\n4. Generating summary statistics...")
        summary = {
            'total_consumers': self.n_consumers,
            'n_days': self.n_days,
            'total_readings': len(consumption_full),
            'anomalous_consumers': int(self.n_consumers * self.anomaly_rate),
            'normal_consumers': int(self.n_consumers * (1 - self.anomaly_rate)),
            'date_range': {
                'start': str(self.start_date.date()),
                'end': str((self.start_date + timedelta(days=self.n_days)).date())
            },
            'anomaly_types': consumption_full['anomaly_label'].value_counts().to_dict()
        }
        
        with open(f'{output_dir}/dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved: {output_dir}/dataset_summary.json")
        
        print("\n✅ Dataset generation complete!")
        print(f"\nSummary:")
        print(f"  - Total consumers: {summary['total_consumers']:,}")
        print(f"  - Anomalous: {summary['anomalous_consumers']:,}")
        print(f"  - Normal: {summary['normal_consumers']:,}")
        print(f"  - Total readings: {summary['total_readings']:,}")
        print(f"  - Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"\nAnomalies by type:")
        for anom_type, count in summary['anomaly_types'].items():
            print(f"  - {anom_type}: {count:,}")
        
        return consumers_df, consumption_full


if __name__ == '__main__':
    # Generate dataset
    generator = ElectricalDataGenerator(
        n_consumers=10000,
        n_days=365,
        anomaly_rate=0.05
    )
    
    consumers, consumption = generator.generate_dataset()
    
    print("\n" + "="*60)
    print("Dataset ready for training!")
    print("="*60)
