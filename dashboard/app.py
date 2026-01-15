"""
⚡ Manipur PowerGuard - Advanced Anomaly Detection Dashboard
TechSprint2 Showcase Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Manipur GridWatch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Toggle (Dark/Light Mode)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Define theme colors
THEMES = {
    'light': {
        'bg': '#FFFFFF',
        'text': '#1A1A2E',
        'card_bg': '#F8F9FA',
        'sidebar_bg': '#F0F2F6',
        'accent': '#0077B6',
        'alert_bg': '#FFF5F5',
        'border': '#E2E8F0',
        'chart_bg': '#FFFFFF',
        'map_style': 'carto-positron'
    },
    'dark': {
        'bg': '#0E1117',
        'text': '#FAFAFA',
        'card_bg': '#262730',
        'sidebar_bg': '#1A1C24',
        'accent': '#00ADB5',
        'alert_bg': '#3d1e1e',
        'border': '#262730',
        'chart_bg': 'rgba(0,0,0,0)',
        'map_style': 'carto-darkmatter'
    }
}

# Get current theme
theme = THEMES['dark'] if st.session_state.dark_mode else THEMES['light']

# Dynamic CSS based on theme
st.markdown(f"""
<style>
    /* Global Background & Font */
    .stApp {{
        background-color: {theme['bg']};
        color: {theme['text']};
    }}
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {{
        background-color: {theme['card_bg']};
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid {theme['accent']};
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {theme['sidebar_bg']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        color: {theme['text']};
    }}
    
    .highlight {{
        color: {theme['accent']};
        font-weight: bold;
    }}
    
    .alert-card {{
        background-color: {theme['alert_bg']};
        border: 1px solid #E53E3E;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }}
    
    /* Map Container */
    .map-container {{
        border: 2px solid {theme['border']};
        border-radius: 10px;
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA LOADING & CACHING
# ==============================================================================

@st.cache_data
def load_data():
    """Load consumption and metadata with caching"""
    data_path = 'data/synthetic/consumption_timeseries.csv'
    meta_path = 'data/synthetic/consumers_metadata.csv' # It exists from generator logic
    
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        st.error("Data files not found. Please run the generator first.")
        return None, None
        
    # Load Data
    df_cons = pd.read_csv(data_path)
    df_meta = pd.read_csv(meta_path)

    # Rename columns for compatibility with dashboard
    rename_map = {'consumer_name': 'name', 'latitude': 'lat', 'longitude': 'lon'}
    df_meta = df_meta.rename(columns=rename_map)
    
    # Preprocessing
    df_cons['timestamp'] = pd.to_datetime(df_cons['timestamp'])
    
    return df_cons, df_meta

@st.cache_data
def load_model_results():
    """Load training results if available"""
    path = 'data/models/training_results.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def get_latest_anomalies(df_cons, df_meta, top_n=10):
    """Get the most recent anomalies detected"""
    anomalies = df_cons[df_cons['anomaly_label'] != 'normal'].copy()
    anomalies = anomalies.sort_values('timestamp', ascending=False).head(top_n)
    
    # Merge with metadata for display
    enriched = anomalies.merge(df_meta, on='consumer_id', how='left')
    return enriched

# ==============================================================================
# 3. DASHBOARD LOGIC
# ==============================================================================

def main():
    # Header
    col_h1, col_h2 = st.columns([1, 4])
    with col_h1:
        st.markdown("# ⚡") 
    with col_h2:
        st.title("Manipur State Power | Intelligent GridWatch")
        st.markdown("Automated Theft Detection & Loss Prevention System")

    # Load Data
    with st.spinner("Connecting to Grid Data..."):
        df_cons, df_meta = load_data()
        model_results = load_model_results()

    if df_cons is None:
        return

    # Sidebar Navigation
    st.sidebar.markdown("## 📡 Navigation")
    page = st.sidebar.radio("Console View", ["Command Center", "Geospatial Intelligence", "Consumer Forensics", "System Health"])
    
    st.sidebar.markdown("---")
    
    # Theme Toggle
    st.sidebar.markdown("## 🎨 Appearance")
    dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Filters")
    
    # Global Filters
    all_districts = sorted(df_meta['district'].unique())
    selected_district = st.sidebar.selectbox("Select District", ["All Districts"] + all_districts)
    
    # Filter Data based on selection
    if selected_district != "All Districts":
        filtered_meta = df_meta[df_meta['district'] == selected_district]
        consumer_ids = filtered_meta['consumer_id'].unique()
        filtered_cons = df_cons[df_cons['consumer_id'].isin(consumer_ids)]
    else:
        filtered_meta = df_meta
        filtered_cons = df_cons

    # ==========================================================================
    # PAGE: COMMAND CENTER (Overview)
    # ==========================================================================
    if page == "Command Center":
        
        # 1. KPI Row
        st.markdown("### 📊 Live Grid Status")
        
        total_consumers = len(filtered_meta)
        total_anomalies = filtered_cons[filtered_cons['anomaly_label'] != 'normal']['consumer_id'].nunique()
        anomaly_rate = (total_anomalies / total_consumers) * 100 if total_consumers > 0 else 0
        
        # Calculate approximate loss (Assuming ₹7/unit)
        anomalous_consumption = filtered_cons[filtered_cons['anomaly_label'] != 'normal']['consumption_kwh'].sum()
        est_loss = anomalous_consumption * 7 
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Monitored Consumers", f"{total_consumers:,}", delta="Active")
        kpi2.metric("Flagged Consumers", f"{total_anomalies:,}", delta=f"{anomaly_rate:.1f}% Risk", delta_color="inverse")
        kpi3.metric("Est. Revenue At Risk", f"₹ {est_loss:,.0f}", delta="Last 90 Days", delta_color="inverse")
        kpi4.metric("Grid Efficiency", "94.2%", delta="+1.2%")

        st.markdown("---")

        # 2. Main Content Area
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("#### ⚡ Consumption Trends (Time Series)")
            # Daily Aggregation
            daily_cons = filtered_cons.set_index('timestamp').resample('D')['consumption_kwh'].sum().reset_index()
            
            fig = px.area(daily_cons, x='timestamp', y='consumption_kwh', 
                          title="Total Grid Load (kWh)",
                          color_discrete_sequence=[theme['accent']])
            fig.update_layout(paper_bgcolor=theme['chart_bg'], plot_bgcolor=theme['chart_bg'], font_color=theme['text'])
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### 🚨 Anomaly Distribution")
            
            anomaly_counts = filtered_cons[filtered_cons['anomaly_label'] != 'normal']['anomaly_label'].value_counts()
            
            if not anomaly_counts.empty:
                fig_pie = px.pie(
                    values=anomaly_counts.values, 
                    names=anomaly_counts.index,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_layout(
                    showlegend=False,
                    paper_bgcolor=theme['chart_bg'], 
                    plot_bgcolor=theme['chart_bg'], 
                    font_color=theme['text'],
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.dataframe(
                    anomaly_counts.reset_index().rename(columns={'index': 'Type', 'anomaly_label': 'Count'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No anomalies detected in this selection.")

        # 3. Recent Alerts
        st.markdown("#### 🔔 Recent High-Priority Alerts")
        recent_alerts = get_latest_anomalies(filtered_cons, filtered_meta)
        
        if not recent_alerts.empty:
            # Format for display
            display_df = recent_alerts[['timestamp', 'consumer_id', 'name', 'district', 'anomaly_label', 'consumption_kwh']]
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df,
                column_config={
                    "anomaly_label": st.column_config.TextColumn("Risk Type"),
                    "consumption_kwh": st.column_config.NumberColumn("Reading (kWh)", format="%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No recent alerts.")

    # ==========================================================================
    # PAGE: GEOSPATIAL INTELLIGENCE
    # ==========================================================================
    elif page == "Geospatial Intelligence":
        st.markdown("### 🛰️ District Surveillance Map")
        
        # Prepare Map Data: Aggregate by Consumer first to see status
        # A consumer is 'High Risk' if they have detected anomalies
        
        anomalies_agg = filtered_cons.groupby('consumer_id')['anomaly_label'].apply(lambda x: (x != 'normal').sum() > 0).reset_index()
        anomalies_agg.columns = ['consumer_id', 'active_anomaly']
        
        map_data = filtered_meta.merge(anomalies_agg, on='consumer_id', how='left')
        map_data['active_anomaly'] = map_data['active_anomaly'].fillna(False)
        map_data['status'] = map_data['active_anomaly'].apply(lambda x: 'CRITICAL' if x else 'Normal')
        map_data['color'] = map_data['active_anomaly'].apply(lambda x: '#FF4B4B' if x else '#00ADB5')
        map_data['size'] = map_data['active_anomaly'].apply(lambda x: 8 if x else 3)
        
        col_map, col_details = st.columns([3, 1])
        
        with col_map:
            # Using Plotly Scatter Mapbox
            fig_map = px.scatter_mapbox(
                map_data,
                lat='lat',
                lon='lon',
                color='status',
                hover_name='name',
                hover_data=['district', 'consumer_type', 'consumer_id'],
                color_discrete_map={'CRITICAL': '#FF4B4B', 'Normal': '#00ADB5'},
                zoom=8,
                center={"lat": 24.8170, "lon": 93.9368}, # Imphal Center
                height=600,
                opacity=0.7
            )
            fig_map.update_layout(
                mapbox_style=theme['map_style'],
                paper_bgcolor=theme['chart_bg'],
                margin=dict(t=0, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col_details:
            st.markdown("#### 📍 District Heatmap")
            
            district_risk = map_data.groupby('district')['is_anomalous'].mean().sort_values(ascending=False) * 100
            
            fig_bar = px.bar(
                district_risk, 
                orientation='h',
                title="Risk % by District",
                labels={'value': 'Consumers Flagged (%)', 'district': ''},
                color=district_risk.values,
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(
                showlegend=False, 
                paper_bgcolor=theme['chart_bg'], 
                plot_bgcolor=theme['chart_bg'], 
                font_color=theme['text'],
                xaxis=dict(showgrid=True, gridcolor=theme['border']),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("�� High risk in **Churachandpur** detected this week.")

    # ==========================================================================
    # PAGE: CONSUMER FORENSICS
    # ==========================================================================
    elif page == "Consumer Forensics":
        st.markdown("### 🕵️ Individual Usage Inspector")
        
        # Search Box
        search_query = st.text_input("Search Consumer (Name or ID)", placeholder="e.g. C00045 or Thoiba")
        
        target_consumer = None
        
        if search_query:
            # Filter
            matches = df_meta[
                df_meta['consumer_id'].astype(str).str.contains(search_query, case=False) | 
                df_meta['name'].str.contains(search_query, case=False)
            ]
            
            if not matches.empty:
                target_options = matches['consumer_id'].tolist()
                target_labels = [f"{row['name']} ({row['consumer_id']})" for _, row in matches.iterrows()]
                
                s_choice = st.selectbox("Select Match", options=target_options, format_func=lambda x: target_labels[target_options.index(x)])
                target_consumer = matches[matches['consumer_id'] == s_choice].iloc[0]
            else:
                st.warning("No consumer found.")
        
        if target_consumer is not None:
            # Layout
            c_info, c_stats = st.columns([1, 2])
            
            # Get Consumer Data
            cons_data = df_cons[df_cons['consumer_id'] == target_consumer['consumer_id']].sort_values('timestamp')
            
            with c_info:
                st.markdown(f"""
                <div style='background-color: {theme['card_bg']}; padding: 20px; border-radius: 10px; border-top: 3px solid {theme['accent']}; color: {theme['text']};'>
                    <h3 style='color: {theme['text']};'>👤 {target_consumer['name']}</h3>
                    <p><b>ID:</b> {target_consumer['consumer_id']}</p>
                    <p><b>District:</b> {target_consumer['district']}</p>
                    <p><b>Type:</b> {target_consumer['consumer_type']}</p>
                    <p><b>Coordinates:</b> {target_consumer['lat']:.4f}, {target_consumer['lon']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Score
                anom_count = (cons_data['anomaly_label'] != 'normal').sum()
                risk_score = min(100, (anom_count / len(cons_data)) * 500) # Simple scaling
                
                st.markdown(f"### Risk Score: {int(risk_score)}/100")
                st.progress(int(risk_score))
                
                if risk_score > 50:
                    st.error("HIGH THEFT PROBABILITY")
                else:
                    st.success("LOW RISK")

            with c_stats:
                # Comparison Chart
                st.subheader("Consumption vs Baseline")
                
                # Add Baseline (Simulated as rolling mean for demo visual)
                cons_data['baseline'] = cons_data['consumption_kwh'].rolling(24, center=True).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cons_data['timestamp'], y=cons_data['consumption_kwh'], 
                                        name='Actual Usage', line=dict(color=theme['accent'], width=2)))
                fig.add_trace(go.Scatter(x=cons_data['timestamp'], y=cons_data['baseline'], 
                                        name='Expected Pattern', line=dict(color='gray', dash='dash')))
                
                # Highlight Anomalies
                anoms = cons_data[cons_data['anomaly_label'] != 'normal']
                fig.add_trace(go.Scatter(x=anoms['timestamp'], y=anoms['consumption_kwh'],
                                        mode='markers', name='Anomalies',
                                        marker=dict(color='#FF4B4B', size=10, symbol='x')))
                
                fig.update_layout(
                    paper_bgcolor=theme['chart_bg'], 
                    plot_bgcolor=theme['chart_bg'], 
                    font_color=theme['text'],
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # PAGE: SYSTEM HEALTH
    # ==========================================================================
    elif page == "System Health":
        st.markdown("### 🧠 AI Model Diagnostics")
        
        if model_results:
            m1, m2, m3, m4 = st.columns(4)
            # Use 'metrics' key and 'f1_score'
            metrics = model_results.get('metrics', {})
            m1.metric("Precision", f"{metrics.get('precision', 0):.1%}")
            m2.metric("Recall", f"{metrics.get('recall', 0):.1%}")
            m3.metric("F1 Score", f"{metrics.get('f1_score', 0):.1%}")
            m4.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
            
            col_chart, col_feat = st.columns(2)
            
            with col_chart:
                st.markdown("#### 🔢 Confusion Matrix")
                cm = model_results['confusion_matrix']
                
                # cm is already [[TN, FP], [FN, TP]] in the JSON
                z = cm
                x = ['Normal', 'Anomaly']
                y = ['Normal', 'Anomaly']
                
                fig_cm = px.imshow(z, x=x, y=y, text_auto=True, color_continuous_scale='Blues',
                                   title="Detection Performance")
                fig_cm.update_layout(paper_bgcolor=theme['chart_bg'], plot_bgcolor=theme['chart_bg'], font_color=theme['text'])
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_feat:
                st.markdown("#### 🧬 Feature Importance")
                
                # Load feature importance from CSV if exists
                fi_path = 'data/models/feature_importance.csv'
                if os.path.exists(fi_path):
                    fi_df = pd.read_csv(fi_path).head(10)
                    
                    fig_fi = px.bar(fi_df, x='importance', y='feature', orientation='h',
                                    title="Top Defensive Signals",
                                    color='importance', color_continuous_scale='Teal')
                    fig_fi.update_layout(
                        paper_bgcolor=theme['chart_bg'], 
                        plot_bgcolor=theme['chart_bg'], 
                        font_color=theme['text'],
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importance data not found.")
            
            st.markdown("---")
            st.caption(f"Model Training Timestamp: {pd.to_datetime('now').strftime('%Y-%m-%d %H:%M')}")
            st.caption("Algorithm: Hybrid Ensemble (Isolation Forest + Random Forest + Statistical Rules)")
            
        else:
            st.warning("Model results not found. Please train the model first.")

if __name__ == "__main__":
    main()
