"""
Manipur India Specific Data Generator
Generates realistic electrical consumption data for Manipur region
With authentic Manipuri names, locations, and consumption patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

np.random.seed(42)
random.seed(42)

# ============================================================
# MANIPUR SPECIFIC DATA
# ============================================================

# Authentic Manipuri Names (Common names from Meitei, Naga, Kuki communities)
MANIPURI_FIRST_NAMES_MALE = [
    "Thoiba", "Ibomcha", "Tomba", "Biren", "Okram", "Rajkumar", "Thangjam",
    "Yumnam", "Moirangthem", "Laishram", "Ningombam", "Sagolsem", "Thokchom",
    "Konsam", "Khumanthem", "Lourembam", "Maibam", "Naorem", "Oinam", "Pukhrambam",
    "Salam", "Wangkhem", "Yumlembam", "Heisnam", "Irengbam", "Jilangamba",
    "Khundrakpam", "Leishangthem", "Mangang", "Ngangom", "Kshetrimayum",
    "Sorokhaibam", "Thiyam", "Usham", "Wahengbam", "Yendrembam", "Ahongshangbam",
    "Bijoy", "Chinglen", "Deben", "Ebomcha", "Gokul", "Hemanta", "Ibotombi",
    "Joykumar", "Khogen", "Leikai", "Manihar", "Nabachandra", "Oken"
]

MANIPURI_FIRST_NAMES_FEMALE = [
    "Thoibi", "Sanatombi", "Sanahanbi", "Memcha", "Chanu", "Devi", "Tombi",
    "Ibemhal", "Nongthombam", "Ongbi", "Saikhom", "Thangjam", "Yumnam",
    "Lanchenbi", "Mangolnganbi", "Naoroibam", "Oinam", "Phurailatpam",
    "Radhamani", "Sanasam", "Takhellambam", "Umakanta", "Victoria", "Waikhom",
    "Yaiphabi", "Ahongsangbam", "Bidyarani", "Chandrakala", "Dinamani",
    "Elangbam", "Geetarani", "Hemabati", "Ibeyaima", "Jamuna", "Kalpana",
    "Laishram", "Maipaksana", "Naoba", "Ojita", "Premila"
]

MANIPURI_SURNAMES = [
    "Singh", "Devi", "Chanu", "Meitei", "Meetei", "Haokip", "Vaiphei",
    "Gangte", "Zou", "Simte", "Paite", "Thadou", "Hmar", "Kom", "Aimol",
    "Chothe", "Koireng", "Lamkang", "Moyon", "Monsang", "Anal", "Angami",
    "Mao", "Maram", "Poumai", "Rongmei", "Tangkhul", "Zeliang", "Liangmai",
    "Thangal", "Kabui", "Inpui", "Kharam", "Chiru", "Tarao", "Khoibu"
]

# Manipur Districts and Locations
MANIPUR_LOCATIONS = {
    "Imphal East": {
        "areas": ["Porompat", "Khongman", "Heingang", "Wangkhei", "Sagolband", 
                  "Langol", "Mantripukhri", "Koirengei", "Nagamapal", "Thangmeiband"],
        "lat_range": (24.78, 24.85),
        "lon_range": (93.93, 94.00)
    },
    "Imphal West": {
        "areas": ["Lamphel", "Singjamei", "Keishampat", "Uripok", "Kwakeithel",
                  "Chingmeirong", "Kakching", "Kongba", "Naoremthong", "Irilbung"],
        "lat_range": (24.75, 24.82),
        "lon_range": (93.88, 93.95)
    },
    "Thoubal": {
        "areas": ["Thoubal Bazar", "Kakching", "Wangjing", "Wabagai", "Yairipok",
                  "Lilong", "Khongjom", "Heirok", "Mayang Imphal", "Leisho"],
        "lat_range": (24.60, 24.70),
        "lon_range": (93.95, 94.10)
    },
    "Bishnupur": {
        "areas": ["Bishnupur Bazar", "Nambol", "Moirang", "Kumbi", "Ningthoukhong",
                  "Kwasiphai", "Laphupat Tera", "Toubul", "Thanga", "Karang"],
        "lat_range": (24.55, 24.65),
        "lon_range": (93.75, 93.90)
    },
    "Churachandpur": {
        "areas": ["Churachandpur Town", "Tuibong", "Rengkai", "Saikot", "Henglep",
                  "Singngat", "Tipaimukh", "Thanlon", "Sangaikot", "Mualkoi"],
        "lat_range": (24.30, 24.45),
        "lon_range": (93.65, 93.80)
    },
    "Ukhrul": {
        "areas": ["Ukhrul Town", "Phungyar", "Kamjong", "Kasom", "Lunghar",
                  "Chingai", "Litan", "Shangshak", "Pushing", "Jessami"],
        "lat_range": (25.00, 25.20),
        "lon_range": (94.30, 94.50)
    },
    "Senapati": {
        "areas": ["Senapati Town", "Kangpokpi", "Saikul", "Maram", "Purul",
                  "Mao", "Tadubi", "Liyai", "Willong", "Karong"],
        "lat_range": (25.20, 25.40),
        "lon_range": (93.95, 94.15)
    },
    "Chandel": {
        "areas": ["Chandel Town", "Tengnoupal", "Machi", "Moreh", "Chakpikarong",
                  "Khudengthabi", "Liwa Sarei", "New Samtal", "Molnoi", "Khengjoy"],
        "lat_range": (24.30, 24.50),
        "lon_range": (94.00, 94.30)
    }
}

# Manipur-specific business names
MANIPUR_BUSINESS_NAMES = [
    "Ima Keithel", "Khwairamband Bazaar", "Ema Market", "Paona Bazaar",
    "Thangal Bazaar", "BT Road Market", "Lamlong Market", "Singjamei Market",
    "Hotel Nirmala", "Hotel Classic Grande", "Imphal Hotel", "The Classic Hotel",
    "Luxmi Kitchen", "Sangai Continental", "Rice Bowl Restaurant", "Gomti Restaurant",
    "RIMS Hospital", "Shija Hospital", "Raj Medicity", "Babina Diagnostics",
    "City Pharmacy", "Apollo Pharmacy Imphal", "Medplus Manipur",
    "DM College", "Manipur University", "NIT Manipur", "CMC Imphal",
    "Sunrise School", "St. Joseph School", "Nirmalabas High School",
    "SBI Imphal Branch", "HDFC Bank Imphal", "Manipur Rural Bank",
    "Manipur Handloom", "Moirang Pheijom", "Longpi Pottery", "Kauna Craft",
    "Shamu Automobile", "Tata Motors Imphal", "Hero MotoCorp Manipur",
    "Reliance Fresh Imphal", "Big Bazaar Imphal", "Vishal Mega Mart"
]

# Industrial establishments
MANIPUR_INDUSTRIES = [
    "MSPDCL Substation", "Loktak Power Station", "NEEPCO Office",
    "Manipur Spinning Mills", "Imphal Rice Mill", "MANIREDA Office",
    "Manipur Food Industries", "Cement Corporation Manipur", "Steel Plant Imphal",
    "Handloom Cluster Moirang", "Bamboo Processing Unit", "Ginger Processing Plant"
]


def generate_manipuri_name(is_business=False, is_industrial=False):
    """Generate authentic Manipuri name"""
    if is_industrial:
        return random.choice(MANIPUR_INDUSTRIES)
    elif is_business:
        return random.choice(MANIPUR_BUSINESS_NAMES)
    else:
        gender = random.choice(['male', 'female'])
        if gender == 'male':
            first = random.choice(MANIPURI_FIRST_NAMES_MALE)
        else:
            first = random.choice(MANIPURI_FIRST_NAMES_FEMALE)
        surname = random.choice(MANIPURI_SURNAMES)
        return f"{first} {surname}"


def get_manipur_location():
    """Get random location in Manipur with authentic coordinates"""
    district = random.choice(list(MANIPUR_LOCATIONS.keys()))
    loc_data = MANIPUR_LOCATIONS[district]
    area = random.choice(loc_data["areas"])
    
    lat = np.random.uniform(*loc_data["lat_range"])
    lon = np.random.uniform(*loc_data["lon_range"])
    
    return {
        "district": district,
        "area": area,
        "address": f"{area}, {district}, Manipur",
        "latitude": round(lat, 6),
        "longitude": round(lon, 6)
    }


def generate_manipur_dataset(n_consumers=2000, n_days=90, anomaly_rate=0.05):
    """
    Generate Manipur-specific electrical consumption dataset
    
    Args:
        n_consumers: Number of consumers (default 2000)
        n_days: Days of data (default 90 = 3 months)
        anomaly_rate: Percentage of anomalous consumers (default 5%)
    """
    print("="*60)
    print("🏔️  MANIPUR ELECTRICITY DATA GENERATOR")
    print("="*60)
    print(f"Consumers: {n_consumers}")
    print(f"Days: {n_days}")
    print(f"Anomaly rate: {anomaly_rate*100}%")
    print(f"Region: Manipur, India")
    print("="*60)
    
    output_dir = 'data/synthetic'
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Generate Consumer Metadata ====================
    print("\n1. Generating Manipur consumer profiles...")
    
    consumer_types = ['residential', 'commercial', 'industrial']
    connection_types = ['single_phase', 'three_phase']
    
    # Manipur-specific tariff categories (as per MSPDCL)
    tariff_categories = {
        'residential': ['Domestic LT-I', 'Domestic LT-II', 'BPL Category'],
        'commercial': ['Commercial LT', 'Commercial HT', 'Shops & Establishments'],
        'industrial': ['Industrial LT', 'Industrial HT', 'Small Scale Industry']
    }
    
    consumers = []
    for i in range(n_consumers):
        # Manipur has more residential consumers
        c_type = random.choices(consumer_types, weights=[0.75, 0.18, 0.07])[0]
        
        # Get location
        location = get_manipur_location()
        
        # Generate name based on type
        if c_type == 'residential':
            name = generate_manipuri_name(is_business=False)
            base = np.random.uniform(3, 12)  # Lower consumption in Manipur
            conn_type = random.choices(connection_types, weights=[0.85, 0.15])[0]
        elif c_type == 'commercial':
            name = generate_manipuri_name(is_business=True)
            base = np.random.uniform(15, 80)
            conn_type = random.choices(connection_types, weights=[0.5, 0.5])[0]
        else:  # industrial
            name = generate_manipuri_name(is_industrial=True)
            base = np.random.uniform(80, 400)
            conn_type = 'three_phase'
        
        # Consumer ID format: MN-DISTRICT_CODE-NUMBER (Manipur style)
        district_code = location['district'][:3].upper()
        consumer_no = f"MN-{district_code}-{str(i+1).zfill(6)}"
        
        consumers.append({
            'consumer_id': consumer_no,
            'consumer_name': name,
            'consumer_type': c_type,
            'connection_type': conn_type,
            'tariff_category': random.choice(tariff_categories[c_type]),
            'district': location['district'],
            'area': location['area'],
            'address': location['address'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'sanctioned_load_kw': round(base * 0.5, 2),
            'meter_number': f"MNP{random.randint(100000, 999999)}",
            'installation_date': (datetime(2025, 1, 1) - timedelta(days=random.randint(365, 2000))).strftime('%Y-%m-%d'),
            'base_consumption': round(base, 2),
            'is_anomalous': i < int(n_consumers * anomaly_rate)
        })
    
    consumers_df = pd.DataFrame(consumers)
    consumers_df.to_csv(f'{output_dir}/consumers_metadata.csv', index=False)
    print(f"   ✅ Saved {len(consumers_df)} Manipur consumers")
    
    # Show district distribution
    print("\n   District-wise consumers:")
    for district, count in consumers_df['district'].value_counts().items():
        print(f"      {district}: {count}")
    
    # ==================== Generate Consumption Data ====================
    print("\n2. Generating consumption time series...")
    
    # Start date: October 2025 (includes winter - higher consumption in Manipur)
    start_date = datetime(2025, 10, 15)
    hours = n_days * 24
    timestamps = [start_date + timedelta(hours=h) for h in range(hours)]
    
    # Manipur-specific hourly patterns (accounting for load shedding, climate)
    # Manipur has cooler climate, different consumption patterns
    residential_pattern = np.array([
        0.25, 0.20, 0.18, 0.18, 0.20, 0.35,  # 0-5: Night (lower, some heating)
        0.55, 0.75, 0.85, 0.60, 0.45, 0.50,  # 6-11: Morning peak
        0.55, 0.50, 0.45, 0.50, 0.60, 0.80,  # 12-17: Afternoon
        1.00, 0.95, 0.85, 0.70, 0.50, 0.35   # 18-23: Evening peak (lighting + heating)
    ])
    
    commercial_pattern = np.array([
        0.15, 0.15, 0.15, 0.15, 0.15, 0.20,  # 0-5: Closed
        0.35, 0.55, 0.80, 1.00, 1.00, 0.95,  # 6-11: Opening hours
        0.90, 0.85, 0.85, 0.90, 0.95, 0.85,  # 12-17: Business hours
        0.70, 0.55, 0.40, 0.30, 0.20, 0.15   # 18-23: Closing
    ])
    
    industrial_pattern = np.array([
        0.60, 0.60, 0.60, 0.60, 0.65, 0.75,  # 0-5: Night shift (reduced)
        0.85, 0.95, 1.00, 1.00, 1.00, 0.95,  # 6-11: Day shift
        0.90, 0.95, 0.95, 0.95, 0.90, 0.85,  # 12-17: Afternoon
        0.75, 0.70, 0.65, 0.65, 0.60, 0.60   # 18-23: Evening shift
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
        if (idx + 1) % 200 == 0:
            print(f"   Progress: {idx+1}/{n_consumers}")
        
        base = consumer['base_consumption']
        pattern = patterns[consumer['consumer_type']]
        
        # Generate consumption
        consumption = np.zeros(hours)
        for h in range(hours):
            hour_of_day = h % 24
            day = h // 24
            current_date = start_date + timedelta(days=day)
            
            # Base + pattern
            value = base * pattern[hour_of_day]
            
            # Manipur winter factor (Nov-Feb higher consumption)
            month = current_date.month
            if month in [11, 12, 1, 2]:
                value *= 1.25  # 25% higher in winter
            
            # Weekend effect (less for commercial)
            if current_date.weekday() >= 5:
                if consumer['consumer_type'] == 'residential':
                    value *= 1.1  # Slightly higher at home
                elif consumer['consumer_type'] == 'commercial':
                    value *= 0.6  # Many shops closed
            
            # Manipur has frequent power cuts - simulate occasional dips
            if random.random() < 0.02:  # 2% chance of load shedding
                value *= 0.3
            
            # Add noise
            value *= np.random.normal(1.0, 0.12)
            consumption[h] = max(0, value)
        
        # Inject anomalies
        if consumer['is_anomalous']:
            anom_type = random.choice(anomaly_types)
            
            if anom_type == 'sudden_spike':
                # Illegal connection added midway
                spike_start = hours // 2
                consumption[spike_start:] *= np.random.uniform(2.5, 4.5)
                
            elif anom_type == 'zero_consumption':
                # Meter bypass - near zero for 1-2 weeks
                zero_start = hours // 2
                zero_len = min(24*random.randint(5, 14), hours - zero_start)
                consumption[zero_start:zero_start+zero_len] *= 0.02
                
            elif anom_type == 'odd_hour_usage':
                # High consumption during odd hours (2-5 AM)
                for i in range(0, hours, 24):
                    if i+5 < hours:
                        consumption[i+2:i+5] *= 5.0
                        consumption[i+10:i+16] *= 0.3
                        
            elif anom_type == 'gradual_theft':
                # Slow meter tampering over time
                decline = np.linspace(1.0, 0.25, hours // 2)
                consumption[hours//2:] *= decline[:len(consumption[hours//2:])]
                
            elif anom_type == 'erratic_pattern':
                # Very irregular (unstable meter or tampering)
                noise = np.random.uniform(0.2, 3.5, hours)
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
        'region': 'Manipur, India',
        'utility': 'MSPDCL (Manipur State Power Distribution Company Limited)',
        'total_consumers': n_consumers,
        'n_days': n_days,
        'total_readings': len(consumption_df),
        'anomalous_consumers': int(n_consumers * anomaly_rate),
        'date_range': {
            'start': str(start_date.date()),
            'end': str((start_date + timedelta(days=n_days)).date())
        },
        'district_distribution': consumers_df['district'].value_counts().to_dict(),
        'consumer_type_distribution': consumers_df['consumer_type'].value_counts().to_dict(),
        'anomaly_distribution': consumption_df['anomaly_label'].value_counts().to_dict()
    }
    
    with open(f'{output_dir}/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ MANIPUR DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"Region: Manipur, India")
    print(f"Consumers: {n_consumers:,}")
    print(f"Records: {len(consumption_df):,}")
    print(f"Anomalous: {int(n_consumers * anomaly_rate)} consumers")
    print(f"\nDistrict Distribution:")
    for dist, count in summary['district_distribution'].items():
        print(f"  - {dist}: {count}")
    print(f"\nAnomaly Types:")
    for atype, count in summary['anomaly_distribution'].items():
        pct = count / len(consumption_df) * 100
        print(f"  - {atype}: {count:,} ({pct:.1f}%)")
    print("="*60)
    
    return consumers_df, consumption_df


if __name__ == '__main__':
    # Generate Manipur dataset with 1000 consumers
    generate_manipur_dataset(
        n_consumers=1000,   # 1000 consumers
        n_days=90,          # 3 months
        anomaly_rate=0.05   # 5% anomalies
    )
