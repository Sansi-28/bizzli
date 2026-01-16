import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { FiDollarSign, FiTrendingDown, FiTarget, FiPieChart } from 'react-icons/fi';
import { getRevenueImpact } from '../services/api';
import { useTheme } from '../context/ThemeContext';
import './RevenueImpact.css';

const COLORS = ['#ff2d55', '#f9a825', '#9d4edd', '#00f5d4', '#3182CE', '#805AD5'];

const RevenueImpact = ({ district }) => {
  const { darkMode } = useTheme();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('summary');

  useEffect(() => {
    loadData();
  }, [district]);

  const loadData = async () => {
    setLoading(true);
    try {
      const response = await getRevenueImpact(district);
      setData(response.data);
    } catch (error) {
      console.error('Error loading revenue impact:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    if (value >= 10000000) {
      return `₹${(value / 10000000).toFixed(2)} Cr`;
    } else if (value >= 100000) {
      return `₹${(value / 100000).toFixed(2)} L`;
    } else if (value >= 1000) {
      return `₹${(value / 1000).toFixed(1)} K`;
    }
    return `₹${value.toFixed(0)}`;
  };

  if (loading) {
    return <div className="revenue-loading">Loading revenue analysis...</div>;
  }

  if (!data) {
    return <div className="revenue-error">Unable to load revenue data</div>;
  }

  return (
    <div className="revenue-impact">
      <div className="revenue-header">
        <h3><FiDollarSign /> Revenue Impact Analysis</h3>
        <div className="view-toggle">
          <button 
            className={activeView === 'summary' ? 'active' : ''} 
            onClick={() => setActiveView('summary')}
          >
            Summary
          </button>
          <button 
            className={activeView === 'breakdown' ? 'active' : ''} 
            onClick={() => setActiveView('breakdown')}
          >
            Breakdown
          </button>
          <button 
            className={activeView === 'trend' ? 'active' : ''} 
            onClick={() => setActiveView('trend')}
          >
            Trend
          </button>
        </div>
      </div>

      {activeView === 'summary' && (
        <div className="summary-view">
          <div className="metric-cards">
            <div className="metric-card">
              <div className="metric-icon loss">
                <FiTrendingDown />
              </div>
              <div className="metric-content">
                <span className="metric-value">{formatCurrency(data.summary.revenue_at_risk)}</span>
                <span className="metric-label">Revenue at Risk</span>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-icon recovery">
                <FiTarget />
              </div>
              <div className="metric-content">
                <span className="metric-value">{formatCurrency(data.summary.recovery_potential)}</span>
                <span className="metric-label">Recovery Potential (80%)</span>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-icon consumption">
                <FiPieChart />
              </div>
              <div className="metric-content">
                <span className="metric-value">{data.summary.anomalous_consumption_kwh.toLocaleString()} kWh</span>
                <span className="metric-label">Anomalous Consumption</span>
              </div>
            </div>
          </div>

          <div className="rate-info">
            <span>Tariff Rate: ₹{data.summary.rate_per_unit}/kWh</span>
            <span>Total Expected Revenue: {formatCurrency(data.summary.total_expected_revenue)}</span>
          </div>
        </div>
      )}

      {activeView === 'breakdown' && (
        <div className="breakdown-view">
          <div className="breakdown-section">
            <h4>Loss by Anomaly Type</h4>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={data.by_type}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={70}
                  paddingAngle={3}
                  dataKey="loss"
                  nameKey="type"
                >
                  {data.by_type.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
            <div className="legend-list">
              {data.by_type.map((item, index) => (
                <div key={item.type} className="legend-item">
                  <span className="legend-color" style={{ background: COLORS[index % COLORS.length] }}></span>
                  <span className="legend-label">{item.type}</span>
                  <span className="legend-value">{formatCurrency(item.loss)}</span>
                </div>
              ))}
            </div>
          </div>

          {data.by_district && data.by_district.length > 0 && (
            <div className="breakdown-section">
              <h4>Loss by District</h4>
              <ResponsiveContainer width="100%" height={Math.max(250, data.by_district.length * 35)}>
                <BarChart data={data.by_district} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
                  <XAxis type="number" tickFormatter={(v) => formatCurrency(v)} />
                  <YAxis 
                    dataKey="district" 
                    type="category" 
                    width={120} 
                    tick={{ fontSize: 11 }} 
                    interval={0}
                  />
                  <Tooltip formatter={(value) => formatCurrency(value)} />
                  <Bar dataKey="loss" fill="#ff2d55" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {activeView === 'trend' && (
        <div className="trend-view">
          <h4>Monthly Loss Trend</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={data.monthly_trend}>
              <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tickFormatter={(v) => formatCurrency(v)} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value) => formatCurrency(value)} />
              <Bar dataKey="estimated_loss" fill="#f9a825" radius={[4, 4, 0, 0]} name="Estimated Loss" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default RevenueImpact;
