# 🚀 Quick Start Guide

Get up and running with the Anomaly Detection System in minutes!

## ⚡ Fast Track (5 Minutes)

### 1. Setup Environment

```bash
# Navigate to project
cd /home/sansi/Desktop/TechSprint2

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
# Generate 10,000 consumers with 1 year of data
python scripts/generate_sample_data.py
```

**Expected output:** ~87 million consumption records generated in ~2-3 minutes

### 3. Train Models

```bash
# Train all models (Isolation Forest, Autoencoder, LSTM, Ensemble)
python src/train_models.py
```

**Expected duration:** 
- Isolation Forest: ~10 seconds
- Autoencoder: ~2-3 minutes
- LSTM: ~3-5 minutes
- Total: ~10-15 minutes

### 4. Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py
```

**Dashboard URL:** http://localhost:8501

---

## 📋 Detailed Step-by-Step

### Prerequisites

- Python 3.9+
- pip
- 4GB RAM minimum
- 2GB free disk space

### Installation

```bash
# Clone or navigate to project
cd TechSprint2

# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Or on Windows
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Data Generation

```bash
python scripts/generate_sample_data.py
```

**What it creates:**
- `data/synthetic/consumers_metadata.csv` - 10,000 consumer profiles
- `data/synthetic/consumption_timeseries.csv` - 87.6M hourly readings
- `data/synthetic/dataset_summary.json` - Dataset statistics

**Customization:**
Edit `scripts/generate_sample_data.py` to change:
- `n_consumers`: Number of consumers (default: 10,000)
- `n_days`: Days of historical data (default: 365)
- `anomaly_rate`: Percentage of anomalous consumers (default: 5%)

### Model Training

```bash
python src/train_models.py
```

**Training Pipeline:**
1. ✅ Data Preprocessing
2. ✅ Feature Engineering (50+ features)
3. ✅ Train-Test Split (80/20)
4. ✅ Isolation Forest Training
5. ✅ Autoencoder Training
6. ✅ LSTM Training
7. ✅ Ensemble Model Creation
8. ✅ Evaluation & Metrics

**Output Files:**
- `data/models/isolation_forest.pkl`
- `data/models/autoencoder.pkl` + `autoencoder_keras.h5`
- `data/models/lstm.pkl` + `lstm_keras.h5`
- `data/models/ensemble.pkl`
- `data/models/training_results.json`

### Dashboard Launch

```bash
streamlit run dashboard/app.py
```

**Dashboard Features:**
- 📊 **Overview**: System metrics, trends, alerts
- 🔍 **Consumer Search**: Detailed consumer analysis
- 🗺️ **Anomaly Map**: Geographic visualization
- 📈 **Analytics**: Model performance metrics

---

## 🎯 Quick Demo Scenarios

### Scenario 1: Detect Sudden Spike
1. Launch dashboard
2. Navigate to "Consumer Search"
3. Search for consumers with anomaly type "sudden_spike"
4. View consumption timeline

### Scenario 2: View Anomaly Map
1. Go to "Anomaly Map" page
2. Filter by severity: Critical, High
3. Identify geographic clusters

### Scenario 3: Check Model Performance
1. Go to "Analytics" page
2. View confusion matrix
3. Check feature importance
4. Analyze ROC-AUC scores

---

## 🛠️ Troubleshooting

### Issue: Import Errors

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: CUDA/TensorFlow Errors

For CPU-only (no GPU):
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```

### Issue: Memory Errors During Training

Reduce sample sizes in `src/train_models.py`:
```python
# Line ~90
sample_size = min(10000, len(X_train))  # Reduce to 10K

# Line ~110
lstm_sample_size = min(5000, len(X_train))  # Reduce to 5K
```

### Issue: Dashboard Not Loading Data

```bash
# Check if data exists
ls -lh data/synthetic/

# If empty, regenerate
python scripts/generate_sample_data.py
```

---

## 📊 Understanding the Output

### Training Results

Check `data/models/training_results.json`:

```json
{
  "models": {
    "isolation_forest": {"roc_auc": 0.82},
    "autoencoder": {"roc_auc": 0.85},
    "lstm": {"roc_auc": 0.87},
    "ensemble": {
      "roc_auc": 0.89,
      "precision": 0.89,
      "recall": 0.86,
      "f1_score": 0.875
    }
  }
}
```

**What it means:**
- **ROC-AUC > 0.85**: Excellent detection
- **Precision 89%**: Of flagged cases, 89% are actual anomalies
- **Recall 86%**: Catches 86% of all anomalies
- **F1-Score 87.5%**: Balanced performance

### Anomaly Types Detected

| Type | Description | Frequency |
|------|-------------|-----------|
| `normal` | No anomaly | ~95% |
| `sudden_spike` | Dramatic consumption increase | ~1% |
| `zero_consumption` | Meter bypass or tampering | ~1% |
| `odd_hour_usage` | Unusual timing patterns | ~1% |
| `gradual_theft` | Slow increasing deviation | ~1% |
| `erratic_pattern` | Very irregular consumption | ~1% |

---

## 🎨 Customization

### Change Anomaly Rate

Edit `scripts/generate_sample_data.py`:

```python
generator = ElectricalDataGenerator(
    n_consumers=10000,
    n_days=365,
    anomaly_rate=0.10  # Change to 10%
)
```

### Adjust Model Parameters

Edit `config/config.yaml`:

```yaml
models:
  isolation_forest:
    contamination: 0.10  # Increase expected anomaly rate
    n_estimators: 200    # More trees for better accuracy
```

### Modify Dashboard

Edit `dashboard/app.py` to:
- Add new pages
- Change color schemes
- Add custom metrics
- Integrate with live data sources

---

## 🔄 Retraining Models

```bash
# Delete old models
rm -rf data/models/*.pkl data/models/*.h5

# Retrain
python src/train_models.py
```

---

## 📚 Next Steps

1. **Integrate Real Data**: Replace synthetic data with actual smart meter readings
2. **Deploy API**: Create REST API for real-time predictions
3. **Add Notifications**: Email/SMS alerts for critical anomalies
4. **Scale Up**: Optimize for millions of consumers
5. **Explainability**: Add SHAP values for model interpretability

---

## 🆘 Need Help?

**Common Commands:**

```bash
# Check Python version
python --version

# List installed packages
pip list

# Check file sizes
du -sh data/synthetic/*

# View logs
tail -f logs/anomaly_detection.log
```

**Support:**
- Check [README.md](README.md) for detailed documentation
- See [ROADMAP.md](ROADMAP.md) for full development plan
- Review code comments in source files

---

## 🎉 You're Ready!

Your AI-based anomaly detection system is now ready to:
- ✅ Process large-scale consumption data
- ✅ Detect theft and tampering patterns
- ✅ Visualize anomalies in real-time
- ✅ Prioritize investigation cases
- ✅ Provide explainable insights

**Happy detecting! ⚡🔍**
