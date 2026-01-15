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
# Main
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
