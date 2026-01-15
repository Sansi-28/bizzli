import React, { useState, useEffect } from 'react';
import { FiAlertTriangle, FiTool, FiZap, FiActivity, FiChevronRight } from 'react-icons/fi';
import { getAnomalyClassification } from '../services/api';
import './AnomalyClassification.css';

const AnomalyClassification = ({ district, onSelectType }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedType, setExpandedType] = useState(null);

  useEffect(() => {
    loadData();
  }, [district]);

  const loadData = async () => {
    setLoading(true);
    try {
      const response = await getAnomalyClassification(district);
      setData(response.data);
    } catch (error) {
      console.error('Error loading classification:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'Theft': return <FiAlertTriangle className="cat-icon theft" />;
      case 'Equipment Issue': return <FiTool className="cat-icon equipment" />;
      case 'Usage Anomaly': return <FiZap className="cat-icon usage" />;
      default: return <FiActivity className="cat-icon" />;
    }
  };

  const getPriorityLabel = (priority) => {
    switch (priority) {
      case 1: return { label: 'Urgent', class: 'urgent' };
      case 2: return { label: 'High', class: 'high' };
      case 3: return { label: 'Medium', class: 'medium' };
      default: return { label: 'Low', class: 'low' };
    }
  };

  if (loading) {
    return <div className="classification-loading">Analyzing anomalies...</div>;
  }

  if (!data) {
    return <div className="classification-error">Unable to load classification data</div>;
  }

  return (
    <div className="anomaly-classification">
      <div className="classification-header">
        <h3>Anomaly Classification</h3>
        <div className="summary-stats">
          <span className="stat">
            <strong>{data.total_anomalies}</strong> total anomalies
          </span>
          <span className="stat">
            <strong>{data.total_affected_consumers}</strong> consumers affected
          </span>
        </div>
      </div>

      <div className="classification-list">
        {data.data.map((item, index) => (
          <div 
            key={item.type} 
            className={`classification-item ${expandedType === index ? 'expanded' : ''}`}
          >
            <div 
              className="item-header"
              onClick={() => setExpandedType(expandedType === index ? null : index)}
            >
              <div className="item-icon">
                {getCategoryIcon(item.category)}
              </div>
              <div className="item-main">
                <div className="item-title">
                  <span className="type-name">{item.type.replace(/_/g, ' ')}</span>
                  <span className={`priority-badge ${getPriorityLabel(item.priority).class}`}>
                    {getPriorityLabel(item.priority).label}
                  </span>
                </div>
                <div className="item-category">{item.category}</div>
              </div>
              <div className="item-counts">
                <span className="count-value">{item.count}</span>
                <span className="count-label">incidents</span>
              </div>
              <div className="item-expand">
                <FiChevronRight className={expandedType === index ? 'rotated' : ''} />
              </div>
            </div>

            {expandedType === index && (
              <div className="item-details">
                <div className="detail-row">
                  <span className="detail-label">Description</span>
                  <span className="detail-value">{item.description}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Affected Consumers</span>
                  <span className="detail-value">{item.affected_consumers}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Recommended Action</span>
                  <span className="detail-value action">{item.action}</span>
                </div>
                {onSelectType && (
                  <button 
                    className="view-cases-btn"
                    onClick={() => onSelectType(item.type)}
                  >
                    View All Cases <FiChevronRight />
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AnomalyClassification;
