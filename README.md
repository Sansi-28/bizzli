# Manipur PowerGuard - Electricity Theft Detection System

An AI-powered electricity theft detection and loss prevention system for Manipur State Power Distribution Company Limited (MSPDCL).

## Features

### Core Functionality
- **Anomaly Detection**: ML-based detection of electricity theft patterns
- **3-Class Risk Classification**: Normal, Low Risk, High Risk categorization
- **Real-time Monitoring**: Live dashboard for grid monitoring

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Command Center** | Main dashboard with KPIs, anomaly distribution, revenue impact analysis |
| **Geospatial Intelligence** | Interactive map with consumer locations, satellite/terrain views, district risk heatmap |
| **Consumer Forensics** | Individual consumer analysis with consumption vs baseline charts, zoom functionality |
| **System Health** | Model performance metrics and feature importance visualization |

### Key Components
- **Alert Panel**: Real-time anomaly alerts with severity levels
- **Revenue Impact**: Loss estimation by district and anomaly type
- **Anomaly Classification**: Categorized breakdown of detected anomalies
- **Consumer Comparison**: Multi-consumer analysis tool
- **AI Chatbot**: Natural language query interface for data insights

### UI Features
- **Ink Wash Theme**: Elegant dark/light mode support
- **Responsive Design**: Works on desktop and tablet
- **Interactive Charts**: Zoomable, filterable visualizations
- **Map Views**: Street, Satellite, and Terrain options

## Project Structure

```
TechSprint2/
├── backend/
│   └── app.py              # Flask API server
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/     # Reusable UI components
│       ├── pages/          # Main dashboard pages
│       ├── context/        # React context (theme)
│       └── services/       # API service layer
├── data/
│   └── synthetic/          # Generated dataset
├── scripts/
│   └── generate_manipur_data.py  # Data generation script
├── src/
│   ├── models/             # ML model implementations
│   └── data/               # Data preprocessing
├── config/
│   └── config.yaml         # Configuration settings
└── requirements.txt        # Python dependencies
```

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TechSprint2
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Generate sample data** (if not present)
   ```bash
   python scripts/generate_manipur_data.py
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```
   Backend runs on: http://localhost:5000

2. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   npm start
   ```
   Frontend runs on: http://localhost:3000

3. **Open browser** and navigate to http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Dashboard KPI statistics |
| `/api/districts` | GET | List of all districts |
| `/api/anomalies/distribution` | GET | Anomaly type distribution |
| `/api/anomalies/recent` | GET | Recent anomaly alerts |
| `/api/map/consumers` | GET | Consumer locations for map |
| `/api/consumers/search` | GET | Search consumers by name/ID |
| `/api/consumers/<id>` | GET | Consumer details with timeline |
| `/api/revenue/impact` | GET | Revenue loss analysis |
| `/api/chatbot` | POST | AI chatbot queries |

## Data Schema

### Consumer Metadata
| Field | Type | Description |
|-------|------|-------------|
| consumer_id | string | Unique identifier (e.g., MN-IMP-000001) |
| name | string | Consumer name |
| district | string | District location |
| consumer_type | string | residential/commercial/industrial/agricultural |
| lat, lon | float | GPS coordinates |

### Consumption Timeseries
| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime | Hourly reading timestamp |
| consumer_id | string | Consumer reference |
| consumption_kwh | float | Energy consumption in kWh |
| anomaly_label | string | normal/sudden_spike/meter_tampering/etc. |

## Risk Classification

| Risk Level | Anomaly Types | Score Range |
|------------|---------------|-------------|
| **High Risk** | meter_tampering, theft_suspected, bypass_detected, sudden_spike | 70-100 |
| **Medium Risk** | zero_consumption, odd_hour_usage | 40-69 |
| **Low Risk** | gradual_theft, erratic_pattern, irregular_consumption | 15-39 |
| **Normal** | No anomalies | 0 |

## Technology Stack

### Backend
- **Flask 3.0+**: REST API server
- **Pandas**: Data processing
- **NumPy**: Numerical computations

### Frontend
- **React 18**: UI framework
- **Recharts**: Data visualization
- **React-Leaflet**: Interactive maps
- **React Router**: Navigation

### ML Models (in src/)
- Isolation Forest
- LSTM Autoencoder
- Ensemble Model

## Configuration

Edit `config/config.yaml` for:
- Model parameters
- Detection thresholds
- Alert settings

## License

MIT License

## Contributors

Developed for Manipur State Power Distribution Company Limited (MSPDCL)
