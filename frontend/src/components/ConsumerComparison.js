import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { FiSearch, FiX, FiPlus, FiUsers } from 'react-icons/fi';
import { searchConsumers, compareConsumers } from '../services/api';
import { useTheme } from '../context/ThemeContext';
import './ConsumerComparison.css';

const COMPARISON_COLORS = ['#00f5d4', '#ff2d55', '#f9a825', '#9d4edd', '#3182CE'];

const ConsumerComparison = () => {
  const { darkMode } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedConsumers, setSelectedConsumers] = useState([]);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (searchQuery.length < 2) return;
    try {
      const response = await searchConsumers(searchQuery);
      setSearchResults(response.data.data || []);
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  const addConsumer = (consumer) => {
    if (selectedConsumers.length >= 5) return;
    if (selectedConsumers.find(c => c.consumer_id === consumer.consumer_id)) return;
    
    setSelectedConsumers([...selectedConsumers, consumer]);
    setSearchResults([]);
    setSearchQuery('');
  };

  const removeConsumer = (consumerId) => {
    setSelectedConsumers(selectedConsumers.filter(c => c.consumer_id !== consumerId));
  };

  const runComparison = async () => {
    if (selectedConsumers.length < 2) return;
    
    setLoading(true);
    try {
      const ids = selectedConsumers.map(c => c.consumer_id);
      const response = await compareConsumers(ids);
      setComparisonData(response.data.data);
    } catch (error) {
      console.error('Comparison error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Merge timelines for chart
  const getChartData = () => {
    if (!comparisonData) return [];
    
    const dateMap = {};
    comparisonData.forEach(consumer => {
      consumer.timeline.forEach(point => {
        if (!dateMap[point.timestamp]) {
          dateMap[point.timestamp] = { date: point.timestamp };
        }
        dateMap[point.timestamp][consumer.consumer_id] = point.consumption_kwh;
      });
    });
    
    return Object.values(dateMap).sort((a, b) => 
      new Date(a.date) - new Date(b.date)
    );
  };

  return (
    <div className="consumer-comparison">
      <div className="comparison-header">
        <h3><FiUsers /> Consumer Comparison</h3>
        <span className="hint">Compare up to 5 consumers</span>
      </div>

      <div className="comparison-body">
        <div className="selection-section">
          <div className="search-box">
            <FiSearch />
            <input
              type="text"
              placeholder="Search consumer by name or ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button onClick={handleSearch}>Search</button>
          </div>

          {searchResults.length > 0 && (
            <div className="search-dropdown">
              {searchResults.map(consumer => (
                <div 
                  key={consumer.consumer_id}
                  className="search-option"
                  onClick={() => addConsumer(consumer)}
                >
                  <span className="name">{consumer.name}</span>
                  <span className="id">{consumer.consumer_id}</span>
                  <span className="district">{consumer.district}</span>
                  <FiPlus className="add-icon" />
                </div>
              ))}
            </div>
          )}

          <div className="selected-consumers">
            {selectedConsumers.map((consumer, index) => (
              <div 
                key={consumer.consumer_id} 
                className="selected-tag"
                style={{ borderColor: COMPARISON_COLORS[index] }}
              >
                <span 
                  className="color-dot" 
                  style={{ background: COMPARISON_COLORS[index] }}
                ></span>
                <span className="name">{consumer.name}</span>
                <button onClick={() => removeConsumer(consumer.consumer_id)}>
                  <FiX />
                </button>
              </div>
            ))}
          </div>

          {selectedConsumers.length >= 2 && (
            <button 
              className="compare-btn"
              onClick={runComparison}
              disabled={loading}
            >
              {loading ? 'Comparing...' : 'Compare Consumption'}
            </button>
          )}
        </div>

        {comparisonData && (
          <div className="comparison-results">
            <div className="stats-comparison">
              <table>
                <thead>
                  <tr>
                    <th>Consumer</th>
                    <th>Avg. Consumption</th>
                    <th>Max Consumption</th>
                    <th>Anomalies</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.map((consumer, index) => (
                    <tr key={consumer.consumer_id}>
                      <td>
                        <span 
                          className="color-dot" 
                          style={{ background: COMPARISON_COLORS[index] }}
                        ></span>
                        {consumer.name}
                      </td>
                      <td>{consumer.avg_consumption.toFixed(2)} kWh</td>
                      <td>{consumer.max_consumption.toFixed(2)} kWh</td>
                      <td className={consumer.anomaly_count > 0 ? 'has-anomaly' : ''}>
                        {consumer.anomaly_count}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="comparison-chart">
              <h4>Daily Consumption Comparison</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={getChartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 10 }} 
                    tickFormatter={(v) => v?.split('-').slice(1).join('/')}
                  />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  {comparisonData.map((consumer, index) => (
                    <Line
                      key={consumer.consumer_id}
                      type="monotone"
                      dataKey={consumer.consumer_id}
                      stroke={COMPARISON_COLORS[index]}
                      strokeWidth={2}
                      dot={false}
                      name={consumer.name}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConsumerComparison;
