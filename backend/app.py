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
    
    # Check if consumer has anomalies
    anomalies_agg = filtered_cons.groupby('consumer_id')['anomaly_label'].apply(
        lambda x: (x != 'normal').sum() > 0
    ).reset_index()
    anomalies_agg.columns = ['consumer_id', 'has_anomaly']
    
    map_data = filtered_meta.merge(anomalies_agg, on='consumer_id', how='left')
    map_data['has_anomaly'] = map_data['has_anomaly'].fillna(False)
    map_data['status'] = map_data['has_anomaly'].apply(lambda x: 'CRITICAL' if x else 'Normal')
    
    columns = ['consumer_id', 'name', 'district', 'consumer_type', 'lat', 'lon', 'status', 'has_anomaly']
    result = map_data[columns].to_dict(orient='records')
    
    return jsonify({'data': result})


@app.route('/api/districts/risk', methods=['GET'])
def get_district_risk():
    """Get risk percentage by district"""
    df_cons = load_consumption_data()
    df_meta = load_metadata()
    
    if df_cons is None or df_meta is None:
        return jsonify({'error': 'Data not found'}), 404
    
    # Get anomaly status per consumer
    anomalies_agg = df_cons.groupby('consumer_id')['anomaly_label'].apply(
        lambda x: (x != 'normal').sum() > 0
    ).reset_index()
    anomalies_agg.columns = ['consumer_id', 'is_anomalous']
    
    merged = df_meta.merge(anomalies_agg, on='consumer_id', how='left')
    merged['is_anomalous'] = merged['is_anomalous'].fillna(False)
    
    district_risk = merged.groupby('district')['is_anomalous'].mean().sort_values(ascending=False) * 100
    
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
    
    # Calculate risk score
    anom_count = (cons_data['anomaly_label'] != 'normal').sum()
    total_readings = len(cons_data)
    risk_score = min(100, int((anom_count / total_readings) * 500)) if total_readings > 0 else 0
    
    # Prepare consumption timeline
    cons_data['baseline'] = cons_data['consumption_kwh'].rolling(24, center=True, min_periods=1).mean()
    cons_data['timestamp'] = cons_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    timeline = cons_data[['timestamp', 'consumption_kwh', 'baseline', 'anomaly_label']].to_dict(orient='records')
    
    return jsonify({
        'consumer': consumer_info,
        'risk_score': risk_score,
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
# Main
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
