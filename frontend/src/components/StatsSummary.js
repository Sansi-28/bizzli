import React, { useState, useEffect } from 'react';
import { FiUsers, FiAlertCircle, FiActivity, FiCalendar, FiZap } from 'react-icons/fi';
import { getSummaryStats } from '../services/api';
import './StatsSummary.css';

const StatsSummary = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await getSummaryStats();
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !stats) {
    return null;
  }

  const healthPercentage = Math.round(
    (stats.consumers.healthy / stats.consumers.total) * 100
  );

  return (
    <div className="stats-summary">
      <div className="stats-row">
        <div className="stat-item">
          <FiUsers className="stat-icon" />
          <div className="stat-content">
            <span className="stat-value">{stats.consumers.total.toLocaleString()}</span>
            <span className="stat-label">Total Consumers</span>
          </div>
        </div>

        <div className="stat-divider"></div>

        <div className="stat-item">
          <FiAlertCircle className="stat-icon warning" />
          <div className="stat-content">
            <span className="stat-value">{stats.consumers.affected.toLocaleString()}</span>
            <span className="stat-label">Flagged</span>
          </div>
        </div>

        <div className="stat-divider"></div>

        <div className="stat-item">
          <FiActivity className="stat-icon" />
          <div className="stat-content">
            <span className="stat-value">{stats.readings.total.toLocaleString()}</span>
            <span className="stat-label">Readings</span>
          </div>
        </div>

        <div className="stat-divider"></div>

        <div className="stat-item">
          <FiZap className="stat-icon success" />
          <div className="stat-content">
            <span className="stat-value">{healthPercentage}%</span>
            <span className="stat-label">Healthy</span>
          </div>
        </div>

        <div className="stat-divider"></div>

        <div className="stat-item">
          <FiCalendar className="stat-icon" />
          <div className="stat-content">
            <span className="stat-value date">{stats.date_range.start}</span>
            <span className="stat-label">to {stats.date_range.end}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsSummary;
