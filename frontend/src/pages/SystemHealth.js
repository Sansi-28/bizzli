import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import KPICard from '../components/KPICard';
import ChartContainer from '../components/ChartContainer';
import Loading from '../components/Loading';
import { useTheme } from '../context/ThemeContext';
import { getModelResults, getFeatureImportance } from '../services/api';
import './SystemHealth.css';

const SystemHealth = () => {
  const { darkMode } = useTheme();
  const [loading, setLoading] = useState(true);
  const [modelResults, setModelResults] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const [modelRes, fiRes] = await Promise.all([
        getModelResults(),
        getFeatureImportance()
      ]);

      setModelResults(modelRes.data);
      setFeatureImportance(fiRes.data.data || []);
    } catch (err) {
      console.error('Error loading model data:', err);
      setError('Model results not found. Please train the model first.');
    } finally {
      setLoading(false);
    }
  };

  const getConfusionMatrixData = () => {
    if (!modelResults?.confusion_matrix) return [];
    const cm = modelResults.confusion_matrix;
    return [
      { name: 'True Negative', value: cm[0][0], category: 'Normal → Normal' },
      { name: 'False Positive', value: cm[0][1], category: 'Normal → Anomaly' },
      { name: 'False Negative', value: cm[1][0], category: 'Anomaly → Normal' },
      { name: 'True Positive', value: cm[1][1], category: 'Anomaly → Anomaly' },
    ];
  };

  if (loading) {
    return <Loading message="Loading model diagnostics..." />;
  }

  if (error) {
    return (
      <div className="system-health">
        <div className="page-header">
          <h1>🧠 AI Model Diagnostics</h1>
          <p>Model performance metrics and feature analysis</p>
        </div>
        <div className="warning-message">
          <p>⚠️ {error}</p>
        </div>
      </div>
    );
  }

  const metrics = modelResults?.metrics || {};
  const cmData = getConfusionMatrixData();

  return (
    <div className="system-health">
      <div className="page-header">
        <h1>🧠 AI Model Diagnostics</h1>
        <p>Model performance metrics and feature analysis</p>
      </div>

      <div className="kpi-grid">
        <KPICard
          label="Precision"
          value={`${((metrics.precision || 0) * 100).toFixed(1)}%`}
          deltaType="neutral"
        />
        <KPICard
          label="Recall"
          value={`${((metrics.recall || 0) * 100).toFixed(1)}%`}
          deltaType="neutral"
        />
        <KPICard
          label="F1 Score"
          value={`${((metrics.f1_score || 0) * 100).toFixed(1)}%`}
          deltaType="neutral"
        />
        <KPICard
          label="ROC AUC"
          value={(metrics.roc_auc || 0).toFixed(4)}
          deltaType="neutral"
        />
      </div>

      <div className="grid-equal">
        <ChartContainer title="🔢 Confusion Matrix">
          <div className="confusion-matrix">
            <div className="matrix-grid">
              <div className="matrix-header"></div>
              <div className="matrix-header">Predicted Normal</div>
              <div className="matrix-header">Predicted Anomaly</div>
              
              <div className="matrix-label">Actual Normal</div>
              <div className="matrix-cell tn">
                <span className="cell-value">{cmData[0]?.value || 0}</span>
                <span className="cell-label">TN</span>
              </div>
              <div className="matrix-cell fp">
                <span className="cell-value">{cmData[1]?.value || 0}</span>
                <span className="cell-label">FP</span>
              </div>
              
              <div className="matrix-label">Actual Anomaly</div>
              <div className="matrix-cell fn">
                <span className="cell-value">{cmData[2]?.value || 0}</span>
                <span className="cell-label">FN</span>
              </div>
              <div className="matrix-cell tp">
                <span className="cell-value">{cmData[3]?.value || 0}</span>
                <span className="cell-label">TP</span>
              </div>
            </div>
          </div>
        </ChartContainer>

        <ChartContainer title="🧬 Feature Importance">
          {featureImportance.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={featureImportance}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
                <XAxis
                  type="number"
                  stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                  tick={{ fontSize: 12 }}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                  tick={{ fontSize: 11 }}
                  width={95}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: darkMode ? '#1A1C24' : '#FFFFFF',
                    border: `1px solid ${darkMode ? '#2D3748' : '#E2E8F0'}`,
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="importance" name="Importance">
                  {featureImportance.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={`hsl(${175 + index * 10}, 70%, ${50 - index * 3}%)`}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state-small">
              <p>Feature importance data not available</p>
            </div>
          )}
        </ChartContainer>
      </div>

      <div className="model-info">
        <p><strong>Model Training Timestamp:</strong> {new Date().toLocaleString()}</p>
        <p><strong>Algorithm:</strong> Hybrid Ensemble (Isolation Forest + Random Forest + Statistical Rules)</p>
      </div>
    </div>
  );
};

export default SystemHealth;
