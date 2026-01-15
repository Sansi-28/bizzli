# Manipur PowerGuard - Feature Documentation

## 🎯 System Overview

Manipur PowerGuard is an intelligent grid monitoring and electricity theft detection system designed for the Manipur State Power Distribution Company. The system uses machine learning to identify anomalous consumption patterns and provides comprehensive tools for investigation and revenue protection.

---

## 📋 Feature List & Objectives

### 1. **Dashboard Command Center**
**Location:** `/` (Home Page)

| Feature | Objective |
|---------|-----------|
| **Quick Stats Summary** | Provide at-a-glance overview of total consumers, flagged consumers, readings, and data date range |
| **District Filter** | Allow filtering of all dashboard data by specific districts for regional analysis |
| **KPI Cards** | Display key metrics: Monitored Consumers, Flagged Consumers, Revenue at Risk, Grid Efficiency |
| **Consumption Trend Chart** | Visualize daily consumption patterns over time to identify systemic issues |
| **Anomaly Distribution Pie Chart** | Show breakdown of anomaly types to understand threat landscape |

---

### 2. **Alert Management System**
**Location:** Command Center Page

| Feature | Objective |
|---------|-----------|
| **Severity Classification** | Categorize alerts as Critical, High, Medium, Low for prioritization |
| **Severity Filter** | Filter alerts by severity level to focus on high-priority items |
| **Alert Acknowledgment** | Track which alerts have been reviewed/acknowledged by operators |
| **Estimated Loss Display** | Show potential revenue loss per alert for prioritization |
| **Real-time Count Badges** | Display counts per severity level for quick assessment |

**Alert Severity Definitions:**
- **Critical:** Meter tampering, theft suspected, bypass detected
- **High:** Unusual patterns indicating potential theft
- **Medium:** Irregular consumption requiring monitoring
- **Low:** Minor deviations, informational alerts

---

### 3. **Revenue Impact Analysis**
**Location:** Command Center Page

| Feature | Objective |
|---------|-----------|
| **Revenue at Risk Summary** | Calculate total estimated revenue loss from anomalies |
| **Recovery Potential** | Show 80% of at-risk revenue as recoverable amount |
| **Anomalous Consumption Tracking** | Track kWh consumed under anomalous conditions |
| **Loss by Anomaly Type** | Pie chart showing which anomaly types cause most loss |
| **Loss by District** | Bar chart comparing revenue impact across districts |
| **Monthly Loss Trend** | Time-series showing loss evolution for trend analysis |

**Calculation Method:**
- Rate: ₹7 per kWh
- Assumed loss factor: 30% of anomalous consumption
- Recovery potential: 80% of estimated loss

---

### 4. **Anomaly Classification System**
**Location:** Command Center Page

| Feature | Objective |
|---------|-----------|
| **Category Grouping** | Organize anomalies into categories: Theft, Equipment Issue, Usage Anomaly |
| **Priority Ranking** | Sort by urgency (Urgent → Low) for action prioritization |
| **Affected Consumer Count** | Show how many consumers are impacted per anomaly type |
| **Recommended Actions** | Provide suggested next steps for each anomaly type |
| **Expandable Details** | Click to reveal full description and action items |

**Anomaly Categories:**
| Type | Category | Priority | Recommended Action |
|------|----------|----------|-------------------|
| meter_tampering | Equipment Issue | Urgent | Immediate physical inspection |
| theft_suspected | Theft | Urgent | Dispatch investigation team |
| bypass_detected | Theft | Urgent | Emergency inspection required |
| sudden_spike | Usage Anomaly | High | Schedule meter verification |
| unusual_pattern | Pattern Anomaly | Medium | Monitor for 7 days |
| irregular_consumption | Usage Anomaly | Medium | Review consumption history |

---

### 5. **Geospatial Intelligence**
**Location:** `/geospatial`

| Feature | Objective |
|---------|-----------|
| **Interactive Map** | Visualize consumer locations across Manipur state |
| **Risk Markers** | Color-coded markers showing normal (green) vs anomalous (red) consumers |
| **Consumer Popup** | Click markers to view consumer details and status |
| **District Filtering** | Filter map view by specific districts |
| **Risk Heatmap Data** | District-wise risk percentage breakdown |

---

### 6. **Consumer Forensics**
**Location:** `/forensics`

| Feature | Objective |
|---------|-----------|
| **Consumer Search** | Search by name or consumer ID for quick lookup |
| **Risk Score Display** | 0-100 risk score with visual indicator |
| **Consumption vs Baseline Chart** | Compare actual usage against expected patterns |
| **Anomaly Point Markers** | Highlight specific timestamps with anomalies on chart |
| **Consumer Profile Card** | Display full consumer metadata (ID, district, type, coordinates) |
| **Reading Statistics** | Show total readings and anomaly count |

---

### 7. **Consumer Comparison Tool**
**Location:** `/forensics`

| Feature | Objective |
|---------|-----------|
| **Multi-Consumer Selection** | Select up to 5 consumers for comparison |
| **Consumption Timeline Overlay** | Compare daily consumption patterns on single chart |
| **Statistics Table** | Side-by-side comparison of avg consumption, max consumption, anomaly counts |
| **Visual Differentiation** | Color-coded lines for easy identification |

**Use Cases:**
- Compare suspicious consumer against neighbors
- Identify consumption pattern differences
- Detect coordinated theft patterns

---

### 8. **Data Export System**
**Location:** Command Center (header), Alert Panel

| Feature | Objective |
|---------|-----------|
| **CSV Export** | Download anomaly data as CSV for external analysis |
| **Timestamped Files** | Auto-generated filenames with export date |
| **District Filtering** | Export filtered to specific district if selected |
| **Complete Data** | Includes timestamp, consumer info, location, estimated loss |

**Export Fields:**
- Timestamp, Consumer ID, Name, District, Consumer Type
- Anomaly Label, Consumption (kWh), Estimated Loss
- Latitude, Longitude

---

### 9. **System Health & Model Performance**
**Location:** `/health`

| Feature | Objective |
|---------|-----------|
| **Model Metrics Display** | Show Precision, Recall, F1-Score, ROC-AUC |
| **Feature Importance Chart** | Visualize which features contribute most to detection |
| **Training Results** | Display model training configuration and outcomes |
| **Performance Tracking** | Monitor detection system effectiveness over time |

**Current Model Performance:**
- Recall: 55.9%
- Precision: 87.5%
- F1-Score: 0.682
- ROC-AUC: 0.926

---

### 10. **Theme & Accessibility**
**Location:** Sidebar

| Feature | Objective |
|---------|-----------|
| **Dark/Light Mode Toggle** | Switch between dark and light themes |
| **Professional Color Palette** | Clean, readable colors for extended use |
| **Monospace Data Values** | Clear typography for numerical data |
| **Responsive Design** | Works on desktop and tablet devices |

---

## 🔧 Technical Features

### Backend API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | System health check |
| `GET /api/districts` | List all districts |
| `GET /api/kpis` | Key performance indicators |
| `GET /api/consumption/daily` | Daily aggregated consumption |
| `GET /api/anomalies/distribution` | Anomaly type distribution |
| `GET /api/anomalies/recent` | Recent anomaly list |
| `GET /api/anomalies/classification` | Detailed classification breakdown |
| `GET /api/alerts` | Alert management with severity |
| `GET /api/revenue/impact` | Revenue impact analysis |
| `GET /api/map/consumers` | Consumer map data |
| `GET /api/districts/risk` | District risk percentages |
| `GET /api/consumers/search` | Consumer search |
| `GET /api/consumers/:id` | Consumer details |
| `GET /api/consumers/compare` | Multi-consumer comparison |
| `GET /api/export/anomalies` | Export anomaly data |
| `GET /api/stats/summary` | Quick summary statistics |
| `GET /api/model/results` | ML model results |
| `GET /api/model/feature-importance` | Feature importance data |

---

## 📊 Dashboard Metrics Explained

### Revenue at Risk Calculation
```
Revenue at Risk = Anomalous Consumption (kWh) × Rate (₹7) × Loss Factor (30%)
Recovery Potential = Revenue at Risk × 80%
```

### Risk Score Calculation (Per Consumer)
```
Risk Score = min(100, (Anomaly Count / Total Readings) × 500)
```

### Grid Efficiency
```
Grid Efficiency = (Normal Readings / Total Readings) × 100
```

---

## 🚀 Future Enhancement Possibilities

1. **Real-time Data Integration** - Connect to live smart meter feeds
2. **SMS/Email Alerts** - Push notifications for critical alerts
3. **Investigation Workflow** - Track case status from detection to resolution
4. **Predictive Analytics** - Forecast potential theft patterns
5. **Mobile App** - Field inspector mobile application
6. **Report Scheduling** - Automated daily/weekly report generation
7. **User Role Management** - Admin, Analyst, Inspector access levels
8. **Audit Trail** - Track all user actions and acknowledgments

---

## 📁 File Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── AlertPanel.js        # Alert management
│   │   ├── RevenueImpact.js     # Financial analysis
│   │   ├── AnomalyClassification.js  # Category breakdown
│   │   ├── ConsumerComparison.js     # Multi-consumer analysis
│   │   ├── StatsSummary.js      # Quick stats widget
│   │   ├── ExportButton.js      # CSV export
│   │   ├── KPICard.js           # Metric cards
│   │   ├── ChartContainer.js    # Chart wrapper
│   │   ├── DataTable.js         # Data grid
│   │   └── Loading.js           # Loading spinner
│   ├── pages/
│   │   ├── CommandCenter.js     # Main dashboard
│   │   ├── GeospatialIntelligence.js  # Map view
│   │   ├── ConsumerForensics.js # Investigation
│   │   └── SystemHealth.js      # Model performance
│   ├── services/
│   │   └── api.js               # API client
│   └── context/
│       └── ThemeContext.js      # Theme management

backend/
├── app.py                       # Flask API server
├── config.py                    # Configuration
└── requirements.txt             # Python dependencies
```
