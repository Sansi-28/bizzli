# ⚡ AI-Based Anomaly Detection for Electrical Consumption

An intelligent system to detect non-technical losses (NTL), electricity theft, and meter tampering using machine learning and deep learning techniques.

## 🎯 Overview

This project implements an AI-driven anomaly detection system to identify unusual electrical consumption patterns that may indicate:
- **Illegal connections**
- **Meter tampering**
- **Energy theft**
- **Billing fraud**
- **Non-technical losses**

## 🚀 Features

- **Multi-Model Approach**: Combines Isolation Forest, Autoencoders, and LSTM networks
- **Real-time Detection**: Process consumption data in real-time
- **Interactive Dashboard**: Streamlit-based visualization and monitoring
- **Risk Scoring**: Prioritize cases by severity (Low/Medium/High/Critical)
- **Explainable AI**: Understand why a case was flagged
- **Geographic Visualization**: Identify hotspots and clusters
- **REST API**: Easy integration with existing systems

## 📊 Architecture

```
Data Sources → Preprocessing → Feature Engineering → ML Models → Ensemble → Alerts → Dashboard
```

### Models Implemented:
1. **Isolation Forest**: Unsupervised anomaly detection
2. **Autoencoder**: Deep learning reconstruction error
3. **LSTM/GRU**: Time-series pattern detection
4. **Ensemble**: Weighted combination of all models

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd TechSprint2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
TechSprint2/
├── data/
│   ├── raw/                    # Raw consumption data
│   ├── processed/              # Cleaned and processed data
│   ├── synthetic/              # Generated sample data
│   └── models/                 # Saved model files
├── src/
│   ├── data/
│   │   ├── data_generator.py   # Generate synthetic data
│   │   ├── preprocessor.py     # Data cleaning
│   │   └── feature_engineer.py # Feature creation
│   ├── models/
│   │   ├── isolation_forest.py # Isolation Forest model
│   │   ├── autoencoder.py      # Autoencoder model
│   │   ├── lstm_model.py       # LSTM model
│   │   └── ensemble.py         # Ensemble model
│   ├── utils/
│   │   ├── config.py           # Configuration
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── logger.py           # Logging utilities
│   └── train_models.py         # Training pipeline
├── dashboard/
│   ├── app.py                  # Main dashboard
│   ├── pages/
│   │   ├── overview.py         # Overview page
│   │   ├── consumer_search.py  # Consumer search
│   │   ├── anomaly_map.py      # Geographic view
│   │   └── analytics.py        # Model analytics
│   └── components/             # Reusable components
├── api/
│   ├── main.py                 # FastAPI application
│   ├── routes/                 # API endpoints
│   └── schemas/                # Data schemas
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
├── tests/
│   ├── test_models.py
│   └── test_preprocessing.py
├── config/
│   ├── config.yaml            # Main configuration
│   └── model_params.yaml      # Model parameters
├── scripts/
│   └── generate_sample_data.py # Data generation script
├── requirements.txt
├── README.md
└── ROADMAP.md                 # 48-hour hackathon plan
```

## 🎬 Quick Start

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates synthetic electrical consumption data with:
- 10,000 consumers
- 1 year of hourly readings
- 5% anomalous patterns (theft, tampering, etc.)

### 2. Train Models

```bash
python src/train_models.py
```

Trains all models:
- Isolation Forest
- Autoencoder
- LSTM
- Ensemble model

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Access the dashboard at: `http://localhost:8501`

### 4. Start API Server (Optional)

```bash
uvicorn api.main:app --reload --port 8000
```

API documentation at: `http://localhost:8000/docs`

## 📊 Usage Examples

### Detect Anomalies for a Consumer

```python
from src.models.ensemble import EnsembleDetector

# Load trained model
detector = EnsembleDetector.load('data/models/ensemble_model.pkl')

# Predict
consumer_data = pd.read_csv('data/processed/consumer_123.csv')
result = detector.predict(consumer_data)

print(f"Anomaly Score: {result['score']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Anomaly Type: {result['anomaly_type']}")
```

### Using the API

```bash
# Predict single consumer
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"consumer_id": "123", "consumption_data": [...]}'

# Get consumer risk profile
curl -X GET "http://localhost:8000/api/v1/consumer/123"
```

## 🔍 Anomaly Types Detected

| Type | Description | Indicators |
|------|-------------|-----------|
| **Sudden Spike** | Dramatic consumption increase | consumption > 3σ above mean |
| **Zero Consumption** | Meter bypass or failure | consecutive zero readings |
| **Odd Hour Usage** | Unusual timing patterns | high consumption at 2-5 AM |
| **Gradual Theft** | Slow increasing deviation | consumption trend divergence |
| **Neighborhood Deviation** | Differs from similar consumers | ratio vs neighbors > 2.0 |

## 📈 Model Performance

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Isolation Forest | 0.82 | 0.55 | 0.66 | ~0.85 |
| Statistical Detector | 0.78 | 0.52 | 0.62 | ~0.80 |
| **Ensemble** | **0.875** | **0.559** | **0.682** | **0.926** |

*Tested on synthetic dataset with 5% anomaly rate. Lower recall is due to class imbalance and conservative thresholds prioritizing precision.*

**Note**: The model prioritizes high precision (fewer false alarms) over recall. To catch more anomalies, the detection threshold can be lowered at the cost of more false positives.

## 🎨 Dashboard Features

### 1. Overview Page
- Total consumers monitored
- Active alerts by severity
- Detection trends
- Geographic hotspots map

### 2. Consumer Search
- Search by ID or location
- View consumption history
- Anomaly timeline
- Risk score breakdown

### 3. Anomaly Map
- Interactive geographic view
- Filter by severity and date
- Cluster detection
- Drill-down capability

### 4. Analytics
- Model performance metrics
- Feature importance
- Confusion matrix
- ROC/PR curves

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  
models:
  isolation_forest:
    contamination: 0.05
    n_estimators: 100
  
  autoencoder:
    encoding_dim: 32
    epochs: 50
    batch_size: 32
  
  lstm:
    units: 64
    sequence_length: 24
    epochs: 30

ensemble:
  weights:
    isolation_forest: 0.3
    autoencoder: 0.35
    lstm: 0.35

alerts:
  thresholds:
    critical: 0.9
    high: 0.7
    medium: 0.5
    low: 0.3
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_models.py::test_isolation_forest
```

## 📚 Documentation

- [ROADMAP.md](ROADMAP.md) - 48-hour hackathon development plan
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server is running)
- Notebooks in `notebooks/` folder for detailed analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is developed for the TechSprint 2 Hackathon.

## 🙏 Acknowledgments

- Inspired by real-world electricity theft detection systems
- Uses state-of-the-art ML techniques for anomaly detection
- Built for rapid prototyping and demonstration

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Built for TechSprint 2 Hackathon - January 2026** 🚀
