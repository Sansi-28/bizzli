"""
Flask Backend for Manipur PowerGuard Dashboard
Replaces Streamlit with REST API endpoints
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ==============================================================================
# Data Loading
# ==============================================================================

def get_data_path(filename):
    """Get absolute path to data files"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, filename)

def load_consumption_data():
    """Load consumption timeseries data"""
    path = get_data_path('data/synthetic/consumption_timeseries.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

def load_metadata():
    """Load consumer metadata"""
    path = get_data_path('data/synthetic/consumers_metadata.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Rename columns for compatibility
        rename_map = {'consumer_name': 'name', 'latitude': 'lat', 'longitude': 'lon'}
        df = df.rename(columns=rename_map)
        return df
    return None

def load_model_results():
    """Load training results if available"""
    path = get_data_path('data/models/training_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_feature_importance():
    """Load feature importance data"""
    path = get_data_path('data/models/feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ==============================================================================
# API Routes
# ==============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'manipur-powerguard-api'})


@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get list of all districts"""
    df_meta = load_metadata()
    if df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    districts = sorted(df_meta['district'].unique().tolist())
    return jsonify({'districts': districts})


@app.route('/api/kpis', methods=['GET'])
def get_kpis():
    """Get key performance indicators for dashboard"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_meta = df_meta
        filtered_cons = df_cons
    
    total_consumers = len(filtered_meta)
    total_anomalies = filtered_cons[filtered_cons['anomaly_label'] != 'normal']['consumer_id'].nunique()
    anomaly_rate = (total_anomalies / total_consumers) * 100 if total_consumers > 0 else 0
    
    # Calculate approximate loss (₹7/unit)
    anomalous_consumption = filtered_cons[filtered_cons['anomaly_label'] != 'normal']['consumption_kwh'].sum()
    est_loss = float(anomalous_consumption * 7)
    
    return jsonify({
        'total_consumers': int(total_consumers),
        'flagged_consumers': int(total_anomalies),
        'anomaly_rate': round(anomaly_rate, 1),
        'estimated_loss': round(est_loss, 0),
        'grid_efficiency': 94.2
    })


@app.route('/api/consumption/daily', methods=['GET'])
def get_daily_consumption():
    """Get daily aggregated consumption data"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_cons = df_cons
    
    # Daily aggregation
    daily_cons = filtered_cons.set_index('timestamp').resample('D')['consumption_kwh'].sum().reset_index()
    daily_cons['timestamp'] = daily_cons['timestamp'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'data': daily_cons.to_dict(orient='records')
    })


@app.route('/api/anomalies/distribution', methods=['GET'])
def get_anomaly_distribution():
    """Get anomaly type distribution"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_cons = df_cons
    
    anomaly_counts = filtered_cons[filtered_cons['anomaly_label'] != 'normal']['anomaly_label'].value_counts()
    
    return jsonify({
        'data': [{'type': k, 'count': int(v)} for k, v in anomaly_counts.items()]
    })


@app.route('/api/anomalies/recent', methods=['GET'])
def get_recent_anomalies():
    """Get recent anomalies with consumer details"""
    district = request.args.get('district', None)
    limit = request.args.get('limit', 10, type=int)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_meta = df_meta
        filtered_cons = df_cons
    
    # Get recent anomalies
    anomalies = filtered_cons[filtered_cons['anomaly_label'] != 'normal'].copy()
    anomalies = anomalies.sort_values('timestamp', ascending=False).head(limit)
    
    # Merge with metadata
    enriched = anomalies.merge(filtered_meta, on='consumer_id', how='left')
    enriched['timestamp'] = enriched['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    columns = ['timestamp', 'consumer_id', 'name', 'district', 'anomaly_label', 'consumption_kwh']
    result = enriched[columns].to_dict(orient='records')
    
    return jsonify({'data': result})


@app.route('/api/map/consumers', methods=['GET'])
def get_map_consumers():
    """Get consumer data for map visualization"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_meta = df_meta
        filtered_cons = df_cons
    
    # Define high-risk anomaly types
    high_risk_types = {'meter_tampering', 'theft_suspected', 'bypass_detected', 'sudden_spike'}
    low_risk_types = {'unusual_pattern', 'irregular_consumption', 'odd_hour_usage', 'gradual_theft', 'erratic_pattern'}
    
    # Classify each consumer based on their most severe anomaly
    def classify_risk(labels):
        anomaly_labels = [l for l in labels if l != 'normal']
        if not anomaly_labels:
            return 'normal'
        if any(l in high_risk_types for l in anomaly_labels):
            return 'high_risk'
        return 'low_risk'
    
    risk_agg = filtered_cons.groupby('consumer_id')['anomaly_label'].apply(classify_risk).reset_index()
    risk_agg.columns = ['consumer_id', 'risk_class']
    
    map_data = filtered_meta.merge(risk_agg, on='consumer_id', how='left')
    map_data['risk_class'] = map_data['risk_class'].fillna('normal')
    map_data['has_anomaly'] = map_data['risk_class'] != 'normal'
    map_data['status'] = map_data['risk_class'].apply(lambda x: 'High Risk' if x == 'high_risk' else ('Low Risk' if x == 'low_risk' else 'Normal'))
    
    columns = ['consumer_id', 'name', 'district', 'consumer_type', 'lat', 'lon', 'status', 'risk_class', 'has_anomaly']
    result = map_data[columns].to_dict(orient='records')
    
    # Add risk distribution summary
    risk_counts = map_data['risk_class'].value_counts().to_dict()
    
    return jsonify({
        'data': result,
        'risk_distribution': {
            'normal': risk_counts.get('normal', 0),
            'low_risk': risk_counts.get('low_risk', 0),
            'high_risk': risk_counts.get('high_risk', 0)
        }
    })


@app.route('/api/districts/risk', methods=['GET'])
def get_district_risk():
    """Get risk percentage by district"""
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Get anomaly status per consumer
    anomalies_agg = df_cons.groupby('consumer_id').agg(
        has_anomaly=('anomaly_label', lambda x: (x != 'normal').any())
    ).reset_index()
    
    merged = df_meta.merge(anomalies_agg, on='consumer_id', how='left')
    merged['has_anomaly'] = merged['has_anomaly'].fillna(False)
    
    district_risk = merged.groupby('district')['has_anomaly'].mean().sort_values(ascending=False) * 100
    
    result = [{'district': k, 'risk_percentage': round(v, 1)} for k, v in district_risk.items()]
    
    return jsonify({'data': result})


@app.route('/api/consumers/search', methods=['GET'])
def search_consumers():
    """Search consumers by name or ID"""
    query = request.args.get('q', '')
    
    df_meta = load_metadata()
    
    if df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    if not query:
        return jsonify({'data': []})
    
    matches = df_meta[
        df_meta['consumer_id'].astype(str).str.contains(query, case=False) |
        df_meta['name'].str.contains(query, case=False)
    ]
    
    columns = ['consumer_id', 'name', 'district', 'consumer_type', 'lat', 'lon']
    result = matches[columns].head(20).to_dict(orient='records')
    
    return jsonify({'data': result})


@app.route('/api/consumers/<consumer_id>', methods=['GET'])
def get_consumer_details(consumer_id):
    """Get detailed info for a specific consumer"""
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Get consumer metadata
    consumer_meta = df_meta[df_meta['consumer_id'] == consumer_id]
    
    if consumer_meta.empty:
        return jsonify({'error': 'Consumer not found'}), 404
    
    consumer_info = consumer_meta.iloc[0].to_dict()
    
    # Get consumption data
    cons_data = df_cons[df_cons['consumer_id'] == consumer_id].sort_values('timestamp')
    
    # Define risk weights for different anomaly types
    high_risk_types = {'meter_tampering', 'theft_suspected', 'bypass_detected', 'sudden_spike'}
    medium_risk_types = {'zero_consumption', 'odd_hour_usage'}
    low_risk_types = {'unusual_pattern', 'irregular_consumption', 'gradual_theft', 'erratic_pattern'}
    
    # Calculate risk score based on anomaly type severity
    total_readings = len(cons_data)
    anomaly_labels = cons_data['anomaly_label'].unique().tolist()
    
    # Remove 'normal' from list if present
    anomaly_types = [label for label in anomaly_labels if label != 'normal']
    
    # Count anomalies by severity
    high_risk_count = sum(1 for label in anomaly_types if label in high_risk_types)
    medium_risk_count = sum(1 for label in anomaly_types if label in medium_risk_types)
    low_risk_count = sum(1 for label in anomaly_types if label in low_risk_types)
    
    # Calculate risk score based on severity levels
    # High risk types: 70-100 score
    # Medium risk types: 40-69 score  
    # Low risk types: 15-39 score
    # Normal: 0-14 score
    if high_risk_count > 0:
        # High risk: base 70, add up to 30 more based on anomaly percentage
        anom_percentage = (cons_data['anomaly_label'] != 'normal').sum() / total_readings if total_readings > 0 else 0
        risk_score = 70 + int(anom_percentage * 30)
        risk_class = 'high_risk'
    elif medium_risk_count > 0:
        # Medium risk: 40-69
        anom_percentage = (cons_data['anomaly_label'] != 'normal').sum() / total_readings if total_readings > 0 else 0
        risk_score = 40 + int(anom_percentage * 29)
        risk_class = 'high_risk'  # Still high risk for UI
    elif low_risk_count > 0:
        # Low risk: 15-39
        anom_percentage = (cons_data['anomaly_label'] != 'normal').sum() / total_readings if total_readings > 0 else 0
        risk_score = 15 + int(anom_percentage * 24)
        risk_class = 'low_risk'
    else:
        # Normal consumer
        risk_score = 0
        risk_class = 'normal'
    
    # Total anomaly count for display
    anom_count = (cons_data['anomaly_label'] != 'normal').sum()
    high_risk_anom_count = sum((cons_data['anomaly_label'] == t).sum() for t in high_risk_types) + \
                           sum((cons_data['anomaly_label'] == t).sum() for t in medium_risk_types)
    low_risk_anom_count = sum((cons_data['anomaly_label'] == t).sum() for t in low_risk_types)
    
    # Prepare consumption timeline with proper baseline calculation
    # Baseline should represent expected consumption for this consumer type
    consumer_type = consumer_info.get('consumer_type', 'residential')
    
    # Get average consumption pattern for this consumer type from all similar consumers
    similar_consumers = df_meta[df_meta['consumer_type'] == consumer_type]['consumer_id'].tolist()
    similar_data = df_cons[df_cons['consumer_id'].isin(similar_consumers)]
    
    # Calculate hourly average pattern by day of week for similar consumers (excluding anomalies)
    similar_normal = similar_data[similar_data['anomaly_label'] == 'normal'].copy()
    
    cons_data = cons_data.copy()
    cons_data['datetime'] = pd.to_datetime(cons_data['timestamp'])
    cons_data['hour'] = cons_data['datetime'].dt.hour
    cons_data['dayofweek'] = cons_data['datetime'].dt.dayofweek
    cons_data['date'] = cons_data['datetime'].dt.date
    
    if len(similar_normal) > 0:
        similar_normal['datetime'] = pd.to_datetime(similar_normal['timestamp'])
        similar_normal['hour'] = similar_normal['datetime'].dt.hour
        similar_normal['dayofweek'] = similar_normal['datetime'].dt.dayofweek
        
        # Calculate baseline by hour and day of week for more variation
        hourly_dow_baseline = similar_normal.groupby(['dayofweek', 'hour'])['consumption_kwh'].mean()
        
        # Map baseline to consumer data
        cons_data['baseline'] = cons_data.apply(
            lambda row: hourly_dow_baseline.get((row['dayofweek'], row['hour']), 2.0), 
            axis=1
        )
    else:
        cons_data['baseline'] = 2.0  # Default baseline
    
    # Scale baseline to match consumer's average (some consumers use more/less than average)
    normal_data = cons_data[cons_data['anomaly_label'] == 'normal']
    if len(normal_data) > 0:
        consumer_avg = normal_data['consumption_kwh'].mean()
        baseline_avg = cons_data['baseline'].mean()
        if baseline_avg > 0 and not pd.isna(consumer_avg):
            scale_factor = consumer_avg / baseline_avg
            cons_data['baseline'] = cons_data['baseline'] * scale_factor
    
    # Add slight daily variation to baseline (weather, seasonal effects simulation)
    daily_variation = cons_data.groupby('date')['consumption_kwh'].transform('mean')
    daily_baseline_mean = cons_data.groupby('date')['baseline'].transform('mean')
    
    # Adjust baseline slightly towards actual daily pattern (but not too much)
    adjustment_factor = 0.3  # 30% adjustment towards actual pattern
    cons_data['baseline'] = cons_data['baseline'] * (1 - adjustment_factor) + \
                            (cons_data['baseline'] * (daily_variation / daily_baseline_mean.replace(0, 1))) * adjustment_factor
    
    cons_data['timestamp'] = cons_data['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    
    timeline = cons_data[['timestamp', 'consumption_kwh', 'baseline', 'anomaly_label']].to_dict(orient='records')
    
    return jsonify({
        'consumer': consumer_info,
        'risk_score': risk_score,
        'risk_class': risk_class,
        'high_risk_count': int(high_risk_anom_count),
        'low_risk_count': int(low_risk_anom_count),
        'timeline': timeline,
        'anomaly_count': int(anom_count),
        'total_readings': int(total_readings)
    })


@app.route('/api/model/results', methods=['GET'])
def get_model_results():
    """Get model training results and metrics"""
    results = load_model_results()
    
    if results is None:
        return jsonify({'error': 'Model results not found'}), 404
    
    return jsonify(results)


@app.route('/api/model/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance data"""
    fi_df = load_feature_importance()
    
    if fi_df is None:
        return jsonify({'error': 'Feature importance data not found'}), 404
    
    result = fi_df.head(10).to_dict(orient='records')
    
    return jsonify({'data': result})


# ==============================================================================
# NEW FEATURE ENDPOINTS
# ==============================================================================

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get alerts with severity levels and status"""
    district = request.args.get('district', None)
    severity = request.args.get('severity', None)
    status = request.args.get('status', 'all')  # all, acknowledged, pending
    limit = request.args.get('limit', 50, type=int)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_meta = df_meta
        filtered_cons = df_cons
    
    # Get anomalies
    anomalies = filtered_cons[filtered_cons['anomaly_label'] != 'normal'].copy()
    anomalies = anomalies.merge(filtered_meta, on='consumer_id', how='left')
    
    # Assign severity based on anomaly type
    severity_map = {
        'sudden_spike': 'critical',
        'unusual_pattern': 'high', 
        'meter_tampering': 'critical',
        'theft_suspected': 'critical',
        'irregular_consumption': 'medium',
        'bypass_detected': 'critical'
    }
    anomalies['severity'] = anomalies['anomaly_label'].map(severity_map).fillna('low')
    
    # Calculate loss estimate per anomaly
    anomalies['estimated_loss'] = anomalies['consumption_kwh'] * 7 * 0.3  # 30% assumed theft
    
    # Filter by severity if specified
    if severity and severity != 'all':
        anomalies = anomalies[anomalies['severity'] == severity]
    
    anomalies = anomalies.sort_values('timestamp', ascending=False).head(limit)
    anomalies['timestamp'] = anomalies['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    anomalies['acknowledged'] = False  # Default status
    
    columns = ['timestamp', 'consumer_id', 'name', 'district', 'anomaly_label', 
               'consumption_kwh', 'severity', 'estimated_loss', 'acknowledged']
    result = anomalies[columns].to_dict(orient='records')
    
    return jsonify({
        'data': result,
        'total': len(result),
        'severity_counts': {
            'critical': len([r for r in result if r['severity'] == 'critical']),
            'high': len([r for r in result if r['severity'] == 'high']),
            'medium': len([r for r in result if r['severity'] == 'medium']),
            'low': len([r for r in result if r['severity'] == 'low'])
        }
    })


@app.route('/api/revenue/impact', methods=['GET'])
def get_revenue_impact():
    """Get detailed revenue impact analysis"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_cons = df_cons
    
    # Calculate revenue metrics
    anomalous = filtered_cons[filtered_cons['anomaly_label'] != 'normal']
    normal = filtered_cons[filtered_cons['anomaly_label'] == 'normal']
    
    total_consumption = filtered_cons['consumption_kwh'].sum()
    anomalous_consumption = anomalous['consumption_kwh'].sum()
    
    # Revenue calculations (₹7/unit)
    rate_per_unit = 7
    total_expected_revenue = total_consumption * rate_per_unit
    revenue_at_risk = anomalous_consumption * rate_per_unit * 0.3  # 30% assumed loss
    
    # Monthly breakdown
    anomalous_monthly = anomalous.set_index('timestamp').resample('M')['consumption_kwh'].sum()
    monthly_loss = (anomalous_monthly * rate_per_unit * 0.3).reset_index()
    monthly_loss['timestamp'] = monthly_loss['timestamp'].dt.strftime('%Y-%m')
    monthly_loss.columns = ['month', 'estimated_loss']
    
    # By anomaly type
    by_type = anomalous.groupby('anomaly_label')['consumption_kwh'].sum() * rate_per_unit * 0.3
    by_type_data = [{'type': k, 'loss': round(v, 2)} for k, v in by_type.items()]
    
    # By district
    if district == 'All Districts' or not district:
        anomalous_with_district = anomalous.merge(df_meta[['consumer_id', 'district']], on='consumer_id')
        by_district = anomalous_with_district.groupby('district')['consumption_kwh'].sum() * rate_per_unit * 0.3
        by_district_data = [{'district': k, 'loss': round(v, 2)} for k, v in by_district.items()]
    else:
        by_district_data = []
    
    return jsonify({
        'summary': {
            'total_consumption_kwh': round(total_consumption, 2),
            'anomalous_consumption_kwh': round(anomalous_consumption, 2),
            'total_expected_revenue': round(total_expected_revenue, 2),
            'revenue_at_risk': round(revenue_at_risk, 2),
            'recovery_potential': round(revenue_at_risk * 0.8, 2),  # 80% recoverable
            'rate_per_unit': rate_per_unit
        },
        'monthly_trend': monthly_loss.to_dict(orient='records'),
        'by_type': by_type_data,
        'by_district': by_district_data
    })


@app.route('/api/anomalies/classification', methods=['GET'])
def get_anomaly_classification():
    """Get detailed anomaly classification breakdown"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_cons = df_cons
    
    anomalies = filtered_cons[filtered_cons['anomaly_label'] != 'normal']
    
    # Classification definitions
    classifications = {
        'sudden_spike': {
            'category': 'Usage Anomaly',
            'description': 'Sudden unusual increase in consumption',
            'action': 'Schedule meter verification',
            'priority': 2
        },
        'unusual_pattern': {
            'category': 'Pattern Anomaly', 
            'description': 'Consumption pattern deviates from normal behavior',
            'action': 'Monitor for 7 days',
            'priority': 3
        },
        'meter_tampering': {
            'category': 'Equipment Issue',
            'description': 'Possible meter manipulation detected',
            'action': 'Immediate physical inspection required',
            'priority': 1
        },
        'theft_suspected': {
            'category': 'Theft',
            'description': 'High probability of electricity theft',
            'action': 'Dispatch investigation team',
            'priority': 1
        },
        'irregular_consumption': {
            'category': 'Usage Anomaly',
            'description': 'Irregular consumption patterns detected',
            'action': 'Review consumption history',
            'priority': 3
        },
        'bypass_detected': {
            'category': 'Theft',
            'description': 'Meter bypass suspected',
            'action': 'Emergency inspection required',
            'priority': 1
        }
    }
    
    # Build classification summary
    type_counts = anomalies['anomaly_label'].value_counts()
    classification_data = []
    
    for anom_type, count in type_counts.items():
        info = classifications.get(anom_type, {
            'category': 'Unknown',
            'description': 'Unclassified anomaly',
            'action': 'Manual review required',
            'priority': 4
        })
        affected_consumers = anomalies[anomalies['anomaly_label'] == anom_type]['consumer_id'].nunique()
        
        classification_data.append({
            'type': anom_type,
            'count': int(count),
            'affected_consumers': affected_consumers,
            **info
        })
    
    # Sort by priority
    classification_data.sort(key=lambda x: x['priority'])
    
    return jsonify({
        'data': classification_data,
        'total_anomalies': len(anomalies),
        'total_affected_consumers': anomalies['consumer_id'].nunique()
    })


@app.route('/api/consumers/compare', methods=['GET'])
def compare_consumers():
    """Compare multiple consumers' consumption patterns"""
    consumer_ids = request.args.get('ids', '').split(',')
    consumer_ids = [c.strip() for c in consumer_ids if c.strip()]
    
    if not consumer_ids:
        return jsonify({'error': 'No consumer IDs provided'}), 400
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    results = []
    for cid in consumer_ids[:5]:  # Max 5 consumers
        consumer_meta = df_meta[df_meta['consumer_id'] == cid]
        if consumer_meta.empty:
            continue
            
        cons_data = df_cons[df_cons['consumer_id'] == cid]
        
        # Daily aggregation
        daily = cons_data.set_index('timestamp').resample('D')['consumption_kwh'].sum().reset_index()
        daily['timestamp'] = daily['timestamp'].dt.strftime('%Y-%m-%d')
        
        results.append({
            'consumer_id': cid,
            'name': consumer_meta.iloc[0]['name'],
            'district': consumer_meta.iloc[0]['district'],
            'avg_consumption': round(cons_data['consumption_kwh'].mean(), 2),
            'max_consumption': round(cons_data['consumption_kwh'].max(), 2),
            'anomaly_count': int((cons_data['anomaly_label'] != 'normal').sum()),
            'timeline': daily.to_dict(orient='records')
        })
    
    return jsonify({'data': results})


@app.route('/api/export/anomalies', methods=['GET'])
def export_anomalies():
    """Export anomaly data as CSV-ready JSON"""
    district = request.args.get('district', None)
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Apply district filter
    if district and district != 'All Districts':
        filtered_meta = df_meta[df_meta['district'] == district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_meta = df_meta
        filtered_cons = df_cons
    
    anomalies = filtered_cons[filtered_cons['anomaly_label'] != 'normal'].copy()
    anomalies = anomalies.merge(filtered_meta, on='consumer_id', how='left')
    anomalies['timestamp'] = anomalies['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    anomalies['estimated_loss'] = anomalies['consumption_kwh'] * 7 * 0.3
    
    columns = ['timestamp', 'consumer_id', 'name', 'district', 'consumer_type',
               'anomaly_label', 'consumption_kwh', 'estimated_loss', 'lat', 'lon']
    
    result = anomalies[columns].to_dict(orient='records')
    
    return jsonify({
        'data': result,
        'total_records': len(result),
        'export_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/stats/summary', methods=['GET'])
def get_summary_stats():
    """Get quick summary statistics for dashboard widgets"""
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    total_consumers = len(df_meta)
    total_readings = len(df_cons)
    
    anomalies = df_cons[df_cons['anomaly_label'] != 'normal']
    total_anomalies = len(anomalies)
    affected_consumers = anomalies['consumer_id'].nunique()
    
    # Time range
    date_range = {
        'start': df_cons['timestamp'].min().strftime('%Y-%m-%d'),
        'end': df_cons['timestamp'].max().strftime('%Y-%m-%d')
    }
    
    # Consumption stats
    total_consumption = df_cons['consumption_kwh'].sum()
    avg_daily_consumption = df_cons.set_index('timestamp').resample('D')['consumption_kwh'].sum().mean()
    
    # District breakdown
    district_counts = df_meta['district'].value_counts().to_dict()
    
    # Consumer type breakdown
    type_counts = df_meta['consumer_type'].value_counts().to_dict()
    
    return jsonify({
        'consumers': {
            'total': int(total_consumers),
            'affected': int(affected_consumers),
            'healthy': int(total_consumers - affected_consumers)
        },
        'readings': {
            'total': int(total_readings),
            'anomalies': int(total_anomalies),
            'normal': int(total_readings - total_anomalies)
        },
        'consumption': {
            'total_kwh': round(total_consumption, 2),
            'avg_daily_kwh': round(avg_daily_consumption, 2)
        },
        'date_range': date_range,
        'districts': district_counts,
        'consumer_types': type_counts
    })


# ==============================================================================
# CHATBOT ENDPOINT
# ==============================================================================

@app.route('/api/chat', methods=['POST'])
def chat_query():
    """Process natural language queries about the data"""
    data = request.get_json()
    query = data.get('query', '').lower().strip()
    
    if not query:
        return jsonify({'response': "Please ask me a question about the grid data.", 'type': 'text'})
    
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'response': "Sorry, I couldn't access the data.", 'type': 'error'})
    
    # Get aggregated data for queries
    anomalies_agg = df_cons.groupby('consumer_id').agg(
        has_anomaly=('anomaly_label', lambda x: (x != 'normal').any()),
        anomaly_count=('anomaly_label', lambda x: (x != 'normal').sum()),
        total_consumption=('consumption_kwh', 'sum'),
        avg_consumption=('consumption_kwh', 'mean')
    ).reset_index()
    
    merged = df_meta.merge(anomalies_agg, on='consumer_id', how='left')
    merged['has_anomaly'] = merged['has_anomaly'].fillna(False)
    
    # District risk calculation
    district_risk = merged.groupby('district').agg(
        total_consumers=('consumer_id', 'count'),
        flagged_consumers=('has_anomaly', 'sum'),
        total_anomalies=('anomaly_count', 'sum'),
        total_consumption=('total_consumption', 'sum')
    ).reset_index()
    district_risk['risk_percentage'] = (district_risk['flagged_consumers'] / district_risk['total_consumers']) * 100
    district_risk = district_risk.sort_values('risk_percentage', ascending=False)
    
    response = None
    response_type = 'text'
    data_payload = None
    
    # Query matching patterns
    if any(word in query for word in ['highest risk', 'most risk', 'riskiest', 'most dangerous', 'worst district']):
        top_district = district_risk.iloc[0]
        response = f"**{top_district['district']}** has the highest risk at **{top_district['risk_percentage']:.1f}%** with {int(top_district['flagged_consumers'])} flagged consumers out of {int(top_district['total_consumers'])} total."
        data_payload = {'district': top_district['district'], 'risk': round(top_district['risk_percentage'], 1)}
        
    elif any(word in query for word in ['lowest risk', 'safest', 'least risk', 'best district']):
        safe_district = district_risk.iloc[-1]
        response = f"**{safe_district['district']}** is the safest with only **{safe_district['risk_percentage']:.1f}%** risk rate."
        data_payload = {'district': safe_district['district'], 'risk': round(safe_district['risk_percentage'], 1)}
        
    elif any(word in query for word in ['total consumers', 'how many consumers', 'number of consumers']):
        total = len(df_meta)
        response = f"There are **{total:,}** consumers being monitored across all districts."
        data_payload = {'total_consumers': total}
        
    elif any(word in query for word in ['total anomalies', 'how many anomalies', 'anomaly count']):
        total_anom = int((df_cons['anomaly_label'] != 'normal').sum())
        affected = int(merged['has_anomaly'].sum())
        response = f"There are **{total_anom:,}** anomaly readings detected, affecting **{affected}** consumers."
        data_payload = {'total_anomalies': total_anom, 'affected_consumers': affected}
        
    elif any(word in query for word in ['revenue', 'loss', 'money', 'financial']):
        anomalous_kwh = df_cons[df_cons['anomaly_label'] != 'normal']['consumption_kwh'].sum()
        est_loss = anomalous_kwh * 7 * 0.3  # ₹7/unit, 30% theft assumption
        response = f"Estimated revenue at risk: **₹{est_loss:,.0f}** based on anomalous consumption of {anomalous_kwh:,.0f} kWh."
        data_payload = {'estimated_loss': round(est_loss, 0), 'anomalous_kwh': round(anomalous_kwh, 0)}
        
    elif any(word in query for word in ['all districts', 'list districts', 'district list', 'which districts']):
        districts = district_risk['district'].tolist()
        response = f"Monitoring **{len(districts)}** districts: {', '.join(districts)}"
        data_payload = {'districts': districts}
        
    elif any(word in query for word in ['district risk', 'risk by district', 'district ranking']):
        top_5 = district_risk.head(5)
        lines = [f"**Top 5 Risk Districts:**"]
        for i, row in top_5.iterrows():
            lines.append(f"• {row['district']}: {row['risk_percentage']:.1f}%")
        response = "\n".join(lines)
        response_type = 'list'
        data_payload = top_5[['district', 'risk_percentage']].to_dict(orient='records')
        
    elif any(word in query for word in ['anomaly type', 'types of anomaly', 'anomaly distribution', 'what anomalies']):
        type_counts = df_cons[df_cons['anomaly_label'] != 'normal']['anomaly_label'].value_counts()
        lines = ["**Anomaly Types Detected:**"]
        for anom_type, count in type_counts.items():
            lines.append(f"• {anom_type}: {count:,} incidents")
        response = "\n".join(lines)
        response_type = 'list'
        data_payload = type_counts.to_dict()
        
    elif any(word in query for word in ['top consumer', 'highest consumption', 'most consumption']):
        top_consumer = merged.nlargest(1, 'total_consumption').iloc[0]
        response = f"**{top_consumer['name']}** ({top_consumer['consumer_id']}) has the highest consumption at **{top_consumer['total_consumption']:,.0f} kWh**."
        data_payload = {'consumer': top_consumer['name'], 'consumption': round(top_consumer['total_consumption'], 0)}
        
    elif any(word in query for word in ['model', 'accuracy', 'performance', 'precision', 'recall']):
        results = load_model_results()
        if results and 'metrics' in results:
            metrics = results['metrics']
            response = f"**Model Performance:**\n• Precision: {metrics.get('precision', 0):.1%}\n• Recall: {metrics.get('recall', 0):.1%}\n• F1 Score: {metrics.get('f1', 0):.1%}\n• ROC-AUC: {metrics.get('roc_auc', 0):.3f}"
            response_type = 'list'
            data_payload = metrics
        else:
            response = "Model metrics are not available."
            
    elif any(word in query for word in ['help', 'what can you', 'commands', 'how to use']):
        response = """**I can answer questions like:**
• "Which district has the highest risk?"
• "How many consumers are being monitored?"
• "What's the total revenue at risk?"
• "Show me district risk ranking"
• "What types of anomalies are detected?"
• "What is the model performance?"
• "Which consumer has highest consumption?"
• "How many anomalies are there?"
• "Which is the safest district?"
"""
        response_type = 'help'
        
    else:
        response = "I'm not sure how to answer that. Try asking about:\n• District risk levels\n• Consumer counts\n• Anomaly statistics\n• Revenue loss estimates\n\nType **help** for more options."
        response_type = 'fallback'
    
    return jsonify({
        'response': response,
        'type': response_type,
        'data': data_payload
    })


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
