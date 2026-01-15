import React, { useState, useEffect } from 'react';
import { FiAlertTriangle, FiAlertCircle, FiInfo, FiCheck, FiFilter, FiDownload } from 'react-icons/fi';
import { getAlerts } from '../services/api';
import './AlertPanel.css';

const AlertPanel = ({ district, onExport }) => {
  const [alerts, setAlerts] = useState([]);
  const [severityCounts, setSeverityCounts] = useState({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [acknowledgedIds, setAcknowledgedIds] = useState(new Set());

  useEffect(() => {
    loadAlerts();
  }, [district, filter]);

  const loadAlerts = async () => {
    setLoading(true);
    try {
      const response = await getAlerts(district, filter, 50);
      setAlerts(response.data.data || []);
      setSeverityCounts(response.data.severity_counts || {});
    } catch (error) {
      console.error('Error loading alerts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledge = (index) => {
    setAcknowledgedIds(prev => new Set([...prev, index]));
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return <FiAlertTriangle className="icon critical" />;
      case 'high': return <FiAlertCircle className="icon high" />;
      case 'medium': return <FiInfo className="icon medium" />;
      default: return <FiInfo className="icon low" />;
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <div className="alert-panel">
      <div className="alert-header">
        <h3>Alert Management</h3>
        <div className="alert-actions">
          <div className="filter-dropdown">
            <FiFilter />
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="all">All Severities</option>
              <option value="critical">Critical ({severityCounts.critical || 0})</option>
              <option value="high">High ({severityCounts.high || 0})</option>
              <option value="medium">Medium ({severityCounts.medium || 0})</option>
              <option value="low">Low ({severityCounts.low || 0})</option>
            </select>
          </div>
          {onExport && (
            <button className="export-btn" onClick={onExport}>
              <FiDownload /> Export
            </button>
          )}
        </div>
      </div>

      <div className="severity-summary">
        <div className="severity-badge critical">
          <span className="count">{severityCounts.critical || 0}</span>
          <span className="label">Critical</span>
        </div>
        <div className="severity-badge high">
          <span className="count">{severityCounts.high || 0}</span>
          <span className="label">High</span>
        </div>
        <div className="severity-badge medium">
          <span className="count">{severityCounts.medium || 0}</span>
          <span className="label">Medium</span>
        </div>
        <div className="severity-badge low">
          <span className="count">{severityCounts.low || 0}</span>
          <span className="label">Low</span>
        </div>
      </div>

      <div className="alert-list">
        {loading ? (
          <div className="loading-state">Loading alerts...</div>
        ) : alerts.length === 0 ? (
          <div className="empty-state">No alerts found</div>
        ) : (
          alerts.map((alert, index) => (
            <div 
              key={index} 
              className={`alert-item ${alert.severity} ${acknowledgedIds.has(index) ? 'acknowledged' : ''}`}
            >
              <div className="alert-icon">
                {getSeverityIcon(alert.severity)}
              </div>
              <div className="alert-content">
                <div className="alert-title">
                  <span className="consumer-name">{alert.name}</span>
                  <span className="consumer-id">({alert.consumer_id})</span>
                </div>
                <div className="alert-details">
                  <span className="anomaly-type">{alert.anomaly_label}</span>
                  <span className="district">{alert.district}</span>
                  <span className="loss">{formatCurrency(alert.estimated_loss)}</span>
                </div>
                <div className="alert-time">{alert.timestamp}</div>
              </div>
              <div className="alert-action">
                {acknowledgedIds.has(index) ? (
                  <span className="ack-badge"><FiCheck /> Acknowledged</span>
                ) : (
                  <button 
                    className="ack-btn"
                    onClick={() => handleAcknowledge(index)}
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertPanel;
