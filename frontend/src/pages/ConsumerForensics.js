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
  ReferenceDot,
  Brush
} from 'recharts';
import { FiSearch, FiUser, FiUsers } from 'react-icons/fi';
import ChartContainer from '../components/ChartContainer';
import Loading from '../components/Loading';
import ConsumerComparison from '../components/ConsumerComparison';
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
  const [timePeriod, setTimePeriod] = useState('daily');

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

  const getRiskLevel = (riskClass, score) => {
    if (riskClass === 'high_risk') return { label: 'HIGH RISK', color: 'danger' };
    if (riskClass === 'low_risk') return { label: 'LOW RISK', color: 'warning' };
    return { label: 'NORMAL', color: 'success' };
  };

  const getRiskBarClass = (riskClass) => {
    if (riskClass === 'high_risk') return 'high';
    if (riskClass === 'low_risk') return 'medium';
    return 'low';
  };

  // Prepare chart data with aggregation based on time period
  const getChartData = () => {
    if (!consumerDetails?.timeline) return [];
    
    const timeline = consumerDetails.timeline.map((item) => ({
      ...item,
      isAnomaly: item.anomaly_label !== 'normal'
    }));
    
    if (timePeriod === 'hourly') {
      // Sample hourly data for cleaner visualization
      if (timeline.length > 200) {
        const samplingRate = Math.ceil(timeline.length / 150);
        return timeline.filter((item, index) => 
          item.isAnomaly || index % samplingRate === 0
        );
      }
      return timeline;
    }
    
    // Aggregate data by day or month
    const aggregated = {};
    
    timeline.forEach((item) => {
      if (!item.timestamp) return;
      
      let key;
      if (timePeriod === 'daily') {
        key = item.timestamp.split(' ')[0]; // YYYY-MM-DD
      } else {
        // Monthly: YYYY-MM
        key = item.timestamp.slice(0, 7);
      }
      
      if (!aggregated[key]) {
        aggregated[key] = {
          timestamp: key,
          consumption_kwh: 0,
          baseline: 0,
          count: 0,
          hasAnomaly: false
        };
      }
      
      aggregated[key].consumption_kwh += item.consumption_kwh || 0;
      aggregated[key].baseline += item.baseline || 0;
      aggregated[key].count += 1;
      if (item.isAnomaly) aggregated[key].hasAnomaly = true;
    });
    
    return Object.values(aggregated)
      .map((item) => ({
        timestamp: item.timestamp,
        consumption_kwh: Math.round(item.consumption_kwh * 100) / 100,
        baseline: Math.round(item.baseline * 100) / 100,
        isAnomaly: item.hasAnomaly
      }))
      .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
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
                    className={`fill ${getRiskBarClass(consumerDetails.risk_class)}`}
                    style={{ width: `${Math.max(consumerDetails.risk_score, 5)}%` }}
                  ></div>
                </div>
                <div className={`risk-label ${getRiskLevel(consumerDetails.risk_class, consumerDetails.risk_score).color}`}>
                  {getRiskLevel(consumerDetails.risk_class, consumerDetails.risk_score).label}
                </div>
              </div>

              <div className="stats-grid">
                <div className="stat">
                  <span className="stat-value">{consumerDetails.total_readings}</span>
                  <span className="stat-label">Total Readings</span>
                </div>
                <div className="stat">
                  <span className="stat-value high-risk">{consumerDetails.high_risk_count || 0}</span>
                  <span className="stat-label">High Risk</span>
                </div>
                <div className="stat">
                  <span className="stat-value low-risk">{consumerDetails.low_risk_count || 0}</span>
                  <span className="stat-label">Low Risk</span>
                </div>
              </div>
            </div>

            <ChartContainer title="Consumption vs Baseline">
              <div className="chart-controls">
                <label>View by:</label>
                <select 
                  value={timePeriod} 
                  onChange={(e) => setTimePeriod(e.target.value)}
                  className="time-period-select"
                >
                  <option value="hourly">Hourly</option>
                  <option value="daily">Daily</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} opacity={0.5} />
                  <XAxis
                    dataKey="timestamp"
                    stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => {
                      if (!value) return '';
                      if (timePeriod === 'monthly') {
                        // Show YYYY-MM as "Jan 25", "Feb 25" etc.
                        const [year, month] = value.split('-');
                        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                        return `${months[parseInt(month) - 1]} '${year?.slice(2)}`;
                      }
                      if (timePeriod === 'daily') {
                        // Show MM-DD format
                        return value.slice(5); // MM-DD
                      }
                      // Hourly: show date only
                      const parts = value.split(' ');
                      return parts[0]?.slice(5) || '';
                    }}
                    interval={timePeriod === 'monthly' ? 0 : 'preserveStartEnd'}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                    tick={{ fontSize: 11 }}
                    tickFormatter={(value) => `${value.toFixed(1)}`}
                    width={50}
                    label={{ 
                      value: 'kWh', 
                      angle: -90, 
                      position: 'insideLeft',
                      style: { fontSize: 11, fill: darkMode ? '#A0AEC0' : '#4A5568' }
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: darkMode ? '#1A1C24' : '#FFFFFF',
                      border: `1px solid ${darkMode ? '#2D3748' : '#E2E8F0'}`,
                      borderRadius: '8px',
                      padding: '12px'
                    }}
                    labelFormatter={(label) => `Time: ${label}`}
                    formatter={(value, name) => [`${value?.toFixed(2)} kWh`, name]}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Line
                    type="monotone"
                    dataKey="consumption_kwh"
                    stroke={darkMode ? '#00f5d4' : '#0077B6'}
                    strokeWidth={2}
                    dot={false}
                    name="Actual Usage"
                    activeDot={{ r: 4, fill: darkMode ? '#00f5d4' : '#0077B6' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline"
                    stroke={darkMode ? '#A0AEC0' : '#718096'}
                    strokeWidth={1.5}
                    strokeDasharray="8 4"
                    dot={false}
                    name="Expected Pattern"
                  />
                  {anomalyPoints.map((point, index) => (
                    <ReferenceDot
                      key={index}
                      x={point.timestamp}
                      y={point.consumption_kwh}
                      r={5}
                      fill="#FF4B4B"
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  ))}
                  <Brush
                    dataKey="timestamp"
                    height={40}
                    stroke={darkMode ? '#3d5a80' : '#0077B6'}
                    fill={darkMode ? '#1a1c24' : '#f5f5f5'}
                    tickFormatter={(value) => {
                      if (!value) return '';
                      if (timePeriod === 'monthly') return value.slice(5);
                      return value.slice(5, 10);
                    }}
                  />
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

      <h3 className="section-title"><FiUsers /> Consumer Comparison Tool</h3>
      <ConsumerComparison />
    </div>
  );
};

export default ConsumerForensics;
