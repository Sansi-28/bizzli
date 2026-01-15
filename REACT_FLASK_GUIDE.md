# Manipur PowerGuard - React + Flask Architecture

This document describes the new React + Flask architecture that replaces the original Streamlit dashboard.

## Architecture Overview

```
TechSprint2/
├── backend/                 # Flask API Backend
│   ├── app.py              # Main Flask application
│   ├── config.py           # Configuration settings
│   └── __init__.py
├── frontend/               # React Frontend
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   │   ├── Sidebar.js
│   │   │   ├── KPICard.js
│   │   │   ├── DataTable.js
│   │   │   ├── ChartContainer.js
│   │   │   └── Loading.js
│   │   ├── pages/          # Page components
│   │   │   ├── CommandCenter.js
│   │   │   ├── GeospatialIntelligence.js
│   │   │   ├── ConsumerForensics.js
│   │   │   └── SystemHealth.js
│   │   ├── context/        # React Context (Theme)
│   │   ├── services/       # API service layer
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── data/                   # Data files
├── src/                    # ML/Data processing code
└── dashboard/              # (Legacy) Streamlit dashboard
```

## Quick Start

### 1. Install Backend Dependencies

```bash
cd /path/to/TechSprint2
pip install -r requirements.txt
```

### 2. Start Flask Backend

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Start React Frontend

```bash
npm start
```

The dashboard will be available at `http://localhost:3000`

## API Endpoints

### Health & Status
- `GET /api/health` - Health check

### Districts
- `GET /api/districts` - List all districts

### KPIs & Metrics
- `GET /api/kpis?district=<name>` - Get KPI metrics

### Consumption Data
- `GET /api/consumption/daily?district=<name>` - Daily consumption trends

### Anomalies
- `GET /api/anomalies/distribution?district=<name>` - Anomaly type distribution
- `GET /api/anomalies/recent?district=<name>&limit=10` - Recent anomalies

### Map Data
- `GET /api/map/consumers?district=<name>` - Consumer locations with status
- `GET /api/districts/risk` - Risk percentage by district

### Consumer Search
- `GET /api/consumers/search?q=<query>` - Search consumers
- `GET /api/consumers/<consumer_id>` - Consumer details with timeline

### Model Results
- `GET /api/model/results` - Model training metrics
- `GET /api/model/feature-importance` - Feature importance data

## Features

### Command Center
- Live grid status KPIs
- Consumption trends (Area chart)
- Anomaly distribution (Pie chart)
- Recent high-priority alerts table

### Geospatial Intelligence
- Interactive map with consumer markers
- Critical vs Normal status visualization
- District risk heatmap (Bar chart)

### Consumer Forensics
- Consumer search by name/ID
- Detailed consumer profile
- Risk score calculation
- Consumption vs baseline timeline
- Anomaly point highlighting

### System Health
- Model performance metrics (Precision, Recall, F1, AUC)
- Confusion matrix visualization
- Feature importance chart

## Technology Stack

### Backend
- **Flask** - Python web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations

### Frontend
- **React 18** - UI framework
- **React Router** - Client-side routing
- **Recharts** - Chart library
- **React-Leaflet** - Map visualization
- **Axios** - HTTP client
- **React Icons** - Icon library

## Development

### Environment Variables

Create `.env` files for configuration:

**Frontend (.env)**
```
REACT_APP_API_URL=http://localhost:5000/api
```

### Running in Development

1. Start backend: `python backend/app.py`
2. Start frontend: `cd frontend && npm start`

### Building for Production

```bash
cd frontend
npm run build
```

The build output will be in `frontend/build/`

## Migration Notes

The following Streamlit features have been migrated:

| Streamlit Feature | React + Flask Equivalent |
|-------------------|--------------------------|
| `st.set_page_config` | React Router + Layout |
| `st.session_state` | React Context (Theme) |
| `st.sidebar` | Sidebar component |
| `st.metric` | KPICard component |
| `st.plotly_chart` | Recharts components |
| `st.dataframe` | DataTable component |
| `st.cache_data` | Browser caching + API |
| `st.selectbox` | Native select element |
| `st.text_input` | Native input element |

## Troubleshooting

### CORS Issues
Ensure Flask-CORS is properly configured in `backend/app.py`

### API Connection Failed
1. Check if Flask backend is running on port 5000
2. Verify the proxy setting in `frontend/package.json`

### Map Not Loading
Ensure Leaflet CSS is included in `public/index.html`
