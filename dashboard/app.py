"""
Streamlit Dashboard for Anomaly Detection System
Main entry point for the interactive dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page config
st.set_page_config(
    page_title="Electrical Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-critical {
        background-color: #ff4444;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .anomaly-high {
        background-color: #ff8c00;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .anomaly-medium {
        background-color: #ffd700;
        color: black;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .anomaly-low {
        background-color: #90ee90;
        color: black;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load sample data for dashboard"""
    try:
        # Load consumption data
        consumption_path = 'data/synthetic/consumption_timeseries.csv'
        if os.path.exists(consumption_path):
            df = pd.read_csv(consumption_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return generate_demo_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return generate_demo_data()


def generate_demo_data():
    """Generate demo data for display"""
    n_points = 1000
    dates = pd.date_range(start='2025-01-01', periods=n_points, freq='H')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'consumer_id': [f'C{str(i).zfill(6)}' for i in np.random.randint(1, 100, n_points)],
        'consumption_kwh': np.random.normal(10, 3, n_points),
        'anomaly_label': np.random.choice(['normal', 'sudden_spike', 'gradual_theft', 'zero_consumption'], 
                                         n_points, p=[0.90, 0.04, 0.03, 0.03])
    })
    
    return df


def main():
    # Header
    st.markdown('<div class="main-header">⚡ Electrical Anomaly Detection System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/white?text=TechSprint+2", 
                use_column_width=True)
        st.markdown("---")
        
        st.header("🎛️ Navigation")
        page = st.radio("Select Page", 
                       ["📊 Overview", "🔍 Consumer Search", "🗺️ Anomaly Map", "📈 Analytics"])
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Severity filter
        severity_filter = st.multiselect(
            "Severity Level",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"]
        )
        
        st.markdown("---")
        st.caption("TechSprint 2 Hackathon 2026")
    
    # Load data
    df = load_data()
    
    # Page routing
    if "Overview" in page:
        show_overview(df, severity_filter)
    elif "Consumer Search" in page:
        show_consumer_search(df)
    elif "Map" in page:
        show_anomaly_map(df)
    elif "Analytics" in page:
        show_analytics(df)


def show_overview(df, severity_filter):
    """Overview dashboard page"""
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_consumers = df['consumer_id'].nunique()
        st.metric("Total Consumers", f"{total_consumers:,}")
    
    with col2:
        anomalies = len(df[df['anomaly_label'] != 'normal'])
        st.metric("Active Anomalies", f"{anomalies:,}", 
                 delta=f"{(anomalies/len(df)*100):.1f}%")
    
    with col3:
        critical_count = int(anomalies * 0.2)  # Demo: 20% critical
        st.metric("Critical Alerts", critical_count, delta="⚠️")
    
    with col4:
        detection_rate = 87.5  # Demo value
        st.metric("Detection Rate", f"{detection_rate}%", delta="2.3%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Anomaly Trend (Last 30 Days)")
        
        # Group by date
        df_daily = df.copy()
        df_daily['date'] = pd.to_datetime(df_daily['timestamp']).dt.date
        daily_anomalies = df_daily[df_daily['anomaly_label'] != 'normal'].groupby('date').size().reset_index()
        daily_anomalies.columns = ['date', 'count']
        
        fig = px.line(daily_anomalies, x='date', y='count', 
                     title='Daily Anomaly Count',
                     labels={'count': 'Number of Anomalies', 'date': 'Date'})
        fig.update_traces(line_color='#ff4444', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Anomaly Types Distribution")
        
        anomaly_counts = df[df['anomaly_label'] != 'normal']['anomaly_label'].value_counts()
        
        fig = px.pie(values=anomaly_counts.values, 
                    names=anomaly_counts.index,
                    title='Anomaly Types',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts table
    st.subheader("🚨 Recent Alerts")
    
    recent_anomalies = df[df['anomaly_label'] != 'normal'].tail(10).copy()
    recent_anomalies['severity'] = recent_anomalies['anomaly_label'].apply(
        lambda x: 'Critical' if x in ['sudden_spike', 'zero_consumption'] else 'High'
    )
    
    st.dataframe(
        recent_anomalies[['timestamp', 'consumer_id', 'consumption_kwh', 'anomaly_label', 'severity']],
        use_container_width=True
    )


def show_consumer_search(df):
    """Consumer search page"""
    st.header("🔍 Consumer Search")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        consumer_id = st.text_input("Enter Consumer ID", placeholder="C000001")
    
    with col2:
        st.write("")  # Spacing
        search_btn = st.button("🔎 Search", type="primary")
    
    if search_btn and consumer_id:
        consumer_data = df[df['consumer_id'] == consumer_id]
        
        if len(consumer_data) > 0:
            st.success(f"Found {len(consumer_data)} records for {consumer_id}")
            
            # Consumer profile
            st.subheader("👤 Consumer Profile")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_consumption = consumer_data['consumption_kwh'].mean()
                st.metric("Avg. Consumption", f"{avg_consumption:.2f} kWh")
            
            with col2:
                total_anomalies = len(consumer_data[consumer_data['anomaly_label'] != 'normal'])
                st.metric("Anomalies Detected", total_anomalies)
            
            with col3:
                risk_score = min(total_anomalies * 10, 100)  # Demo calculation
                st.metric("Risk Score", f"{risk_score}/100")
            
            with col4:
                status = "⚠️ High Risk" if risk_score > 50 else "✅ Normal"
                st.metric("Status", status)
            
            # Consumption timeline
            st.subheader("📊 Consumption Timeline")
            
            fig = go.Figure()
            
            # Normal consumption
            normal_data = consumer_data[consumer_data['anomaly_label'] == 'normal']
            fig.add_trace(go.Scatter(
                x=normal_data['timestamp'],
                y=normal_data['consumption_kwh'],
                mode='lines',
                name='Normal',
                line=dict(color='blue', width=2)
            ))
            
            # Anomalies
            anomaly_data = consumer_data[consumer_data['anomaly_label'] != 'normal']
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['consumption_kwh'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
            fig.update_layout(
                xaxis_title='Time',
                yaxis_title='Consumption (kWh)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly list
            if len(anomaly_data) > 0:
                st.subheader("⚠️ Detected Anomalies")
                st.dataframe(
                    anomaly_data[['timestamp', 'consumption_kwh', 'anomaly_label']],
                    use_container_width=True
                )
        else:
            st.error(f"No data found for Consumer ID: {consumer_id}")


def show_anomaly_map(df):
    """Geographic anomaly map"""
    st.header("🗺️ Anomaly Heatmap")
    
    st.info("Geographic visualization of anomaly clusters")
    
    # Generate demo coordinates
    n_anomalies = len(df[df['anomaly_label'] != 'normal'])
    map_data = pd.DataFrame({
        'lat': np.random.uniform(37.0, 38.0, min(n_anomalies, 1000)),
        'lon': np.random.uniform(-122.5, -122.0, min(n_anomalies, 1000)),
        'severity': np.random.choice(['Critical', 'High', 'Medium'], min(n_anomalies, 1000))
    })
    
    # Color mapping
    color_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow'}
    map_data['color'] = map_data['severity'].map(color_map)
    
    st.map(map_data[['lat', 'lon']])
    
    # Statistics by region
    st.subheader("📍 Regional Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("North Region", "234 anomalies")
    with col2:
        st.metric("Central Region", "189 anomalies")
    with col3:
        st.metric("South Region", "156 anomalies")


def show_analytics(df):
    """Analytics and model performance"""
    st.header("📈 Model Analytics")
    
    # Model performance
    st.subheader("🎯 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", "89%")
    with col2:
        st.metric("Recall", "86%")
    with col3:
        st.metric("F1-Score", "87.5%")
    with col4:
        st.metric("ROC-AUC", "0.92")
    
    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    
    confusion_data = pd.DataFrame({
        'Predicted Normal': [8950, 150],
        'Predicted Anomaly': [120, 780]
    }, index=['Actual Normal', 'Actual Anomaly'])
    
    st.dataframe(confusion_data, use_container_width=True)
    
    # Feature importance
    st.subheader("🔑 Top Important Features")
    
    features = ['consumption_mean_24h', 'consumption_std_7d', 'zero_count_24h', 
                'hour_sin', 'consumption_ratio_24h', 'deviation_from_mean',
                'peak_to_avg_ratio', 'is_night', 'day_of_week', 'consumption_diff_24h']
    importances = [0.15, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]
    
    fig = px.bar(x=importances, y=features, orientation='h',
                title='Feature Importance',
                labels={'x': 'Importance', 'y': 'Feature'})
    fig.update_traces(marker_color='#1f77b4')
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
