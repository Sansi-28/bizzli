import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceDot
} from 'recharts';
import { FiSearch, FiUser } from 'react-icons/fi';
import ChartContainer from '../components/ChartContainer';
import Loading from '../components/Loading';
import { useTheme } from '../context/ThemeContext';
import { searchConsumers, getConsumerDetails } from '../services/api';
import './ConsumerForensics.css';

const ConsumerForensics = () => {
  const { darkMode } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedConsumer, setSelectedConsumer] = useState(null);
  const [consumerDetails, setConsumerDetails] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (searchQuery.length >= 2) {
      const timer = setTimeout(() => {
        handleSearch();
      }, 300);
      return () => clearTimeout(timer);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);

  const handleSearch = async () => {
    try {
      const response = await searchConsumers(searchQuery);
      setSearchResults(response.data.data || []);
    } catch (error) {
      console.error('Error searching consumers:', error);
    }
  };

  const handleSelectConsumer = async (consumer) => {
    setSelectedConsumer(consumer);
    setSearchResults([]);
    setLoading(true);
    
    try {
      const response = await getConsumerDetails(consumer.consumer_id);
      setConsumerDetails(response.data);
    } catch (error) {
      console.error('Error loading consumer details:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (score) => {
    if (score > 50) return { label: 'HIGH THEFT PROBABILITY', color: 'danger' };
    if (score > 25) return { label: 'MEDIUM RISK', color: 'warning' };
    return { label: 'LOW RISK', color: 'success' };
  };

  // Prepare chart data with anomaly markers
  const getChartData = () => {
    if (!consumerDetails?.timeline) return [];
    return consumerDetails.timeline.map((item) => ({
      ...item,
      isAnomaly: item.anomaly_label !== 'normal'
    }));
  };

  const chartData = getChartData();
  const anomalyPoints = chartData.filter(d => d.isAnomaly);

  return (
    <div className="forensics-page">
      <div className="page-header">
        <h1>🕵️ Individual Usage Inspector</h1>
        <p>Deep dive into consumer behavior and anomaly patterns</p>
      </div>

      <div className="search-section">
        <div className="search-input-wrapper">
          <FiSearch className="search-icon" />
          <input
            type="text"
            placeholder="Search Consumer (Name or ID) e.g. C00045 or Thoiba"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        
        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((consumer) => (
              <div
                key={consumer.consumer_id}
                className="search-result-item"
                onClick={() => handleSelectConsumer(consumer)}
              >
                <FiUser />
                <span className="name">{consumer.name}</span>
                <span className="id">({consumer.consumer_id})</span>
                <span className="district">{consumer.district}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {loading && <Loading message="Loading consumer data..." />}

      {selectedConsumer && consumerDetails && !loading && (
        <div className="consumer-analysis">
          <div className="grid-consumer">
            <div className="consumer-card">
              <h2>
                <FiUser /> {consumerDetails.consumer.name}
              </h2>
              <div className="info-row">
                <span className="label">ID</span>
                <span className="value">{consumerDetails.consumer.consumer_id}</span>
              </div>
              <div className="info-row">
                <span className="label">District</span>
                <span className="value">{consumerDetails.consumer.district}</span>
              </div>
              <div className="info-row">
                <span className="label">Type</span>
                <span className="value">{consumerDetails.consumer.consumer_type}</span>
              </div>
              <div className="info-row">
                <span className="label">Coordinates</span>
                <span className="value">
                  {consumerDetails.consumer.lat?.toFixed(4)}, {consumerDetails.consumer.lon?.toFixed(4)}
                </span>
              </div>

              <div className="risk-score">
                <h3>Risk Score: {consumerDetails.risk_score}/100</h3>
                <div className="risk-bar">
                  <div
                    className={`fill ${consumerDetails.risk_score > 50 ? 'high' : 'low'}`}
                    style={{ width: `${consumerDetails.risk_score}%` }}
                  ></div>
                </div>
                <div className={`risk-label ${getRiskLevel(consumerDetails.risk_score).color}`}>
                  {getRiskLevel(consumerDetails.risk_score).label}
                </div>
              </div>

              <div className="stats-grid">
                <div className="stat">
                  <span className="stat-value">{consumerDetails.total_readings}</span>
                  <span className="stat-label">Total Readings</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{consumerDetails.anomaly_count}</span>
                  <span className="stat-label">Anomalies</span>
                </div>
              </div>
            </div>

            <ChartContainer title="Consumption vs Baseline">
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
                  <XAxis
                    dataKey="timestamp"
                    stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => value?.split(' ')[0] || ''}
                  />
                  <YAxis
                    stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: darkMode ? '#1A1C24' : '#FFFFFF',
                      border: `1px solid ${darkMode ? '#2D3748' : '#E2E8F0'}`,
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="consumption_kwh"
                    stroke={darkMode ? '#00ADB5' : '#0077B6'}
                    strokeWidth={2}
                    dot={false}
                    name="Actual Usage"
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline"
                    stroke="#A0AEC0"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Expected Pattern"
                  />
                  {anomalyPoints.map((point, index) => (
                    <ReferenceDot
                      key={index}
                      x={point.timestamp}
                      y={point.consumption_kwh}
                      r={6}
                      fill="#FF4B4B"
                      stroke="#FF4B4B"
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div className="chart-legend-custom">
                <span className="legend-marker anomaly">✕</span>
                <span>Anomaly Points</span>
              </div>
            </ChartContainer>
          </div>
        </div>
      )}

      {!selectedConsumer && !loading && (
        <div className="empty-state">
          <FiSearch size={48} />
          <h3>Search for a Consumer</h3>
          <p>Enter a consumer name or ID to view detailed usage patterns and risk analysis</p>
        </div>
      )}
    </div>
  );
};

export default ConsumerForensics;
