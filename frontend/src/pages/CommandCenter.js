import React, { useState, useEffect } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import KPICard from '../components/KPICard';
import ChartContainer from '../components/ChartContainer';
import DataTable from '../components/DataTable';
import Loading from '../components/Loading';
import StatsSummary from '../components/StatsSummary';
import AlertPanel from '../components/AlertPanel';
import RevenueImpact from '../components/RevenueImpact';
import AnomalyClassification from '../components/AnomalyClassification';
import ExportButton from '../components/ExportButton';
import { useTheme } from '../context/ThemeContext';
import {
  getDistricts,
  getKPIs,
  getDailyConsumption,
  getAnomalyDistribution,
  getRecentAnomalies,
  exportAnomalies
} from '../services/api';
import './CommandCenter.css';

const COLORS = ['#E53E3E', '#DD6B20', '#D69E2E', '#38A169', '#3182CE', '#805AD5'];

const CommandCenter = () => {
  const { darkMode } = useTheme();
  const [loading, setLoading] = useState(true);
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('All Districts');
  const [kpis, setKpis] = useState(null);
  const [dailyConsumption, setDailyConsumption] = useState([]);
  const [anomalyDistribution, setAnomalyDistribution] = useState([]);
  const [recentAnomalies, setRecentAnomalies] = useState([]);

  useEffect(() => {
    loadDistricts();
  }, []);

  useEffect(() => {
    loadData();
  }, [selectedDistrict]);

  const loadDistricts = async () => {
    try {
      const response = await getDistricts();
      setDistricts(response.data.districts || []);
    } catch (error) {
      console.error('Error loading districts:', error);
    }
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const [kpiRes, consumptionRes, anomalyRes, alertsRes] = await Promise.all([
        getKPIs(selectedDistrict),
        getDailyConsumption(selectedDistrict),
        getAnomalyDistribution(selectedDistrict),
        getRecentAnomalies(selectedDistrict, 10)
      ]);

      setKpis(kpiRes.data);
      setDailyConsumption(consumptionRes.data.data || []);
      setAnomalyDistribution(anomalyRes.data.data || []);
      setRecentAnomalies(alertsRes.data.data || []);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const alertColumns = [
    { key: 'timestamp', label: 'Timestamp' },
    { key: 'consumer_id', label: 'Consumer ID' },
    { key: 'name', label: 'Name' },
    { key: 'district', label: 'District' },
    {
      key: 'anomaly_label',
      label: 'Risk Type',
      render: (value) => (
        <span className="status-badge critical">{value}</span>
      )
    },
    {
      key: 'consumption_kwh',
      label: 'Reading (kWh)',
      render: (value) => value?.toFixed(2)
    }
  ];

  if (loading && !kpis) {
    return <Loading message="Connecting to Grid Data..." />;
  }

  return (
    <div className="command-center">
      <div className="page-header">
        <div className="header-content">
          <h1>⚡ Manipur State Power | Intelligent GridWatch</h1>
          <p>Automated Theft Detection & Loss Prevention System</p>
        </div>
        <div className="header-actions">
          <ExportButton district={selectedDistrict} label="Export Report" />
        </div>
      </div>

      <StatsSummary />

      <div className="filter-bar">
        <label>Select District:</label>
        <select
          value={selectedDistrict}
          onChange={(e) => setSelectedDistrict(e.target.value)}
        >
          <option value="All Districts">All Districts</option>
          {districts.map((district) => (
            <option key={district} value={district}>
              {district}
            </option>
          ))}
        </select>
      </div>

      <h3 className="section-title">📊 Key Performance Indicators</h3>

      <div className="kpi-grid">
        <KPICard
          label="Monitored Consumers"
          value={kpis?.total_consumers?.toLocaleString() || '0'}
          delta="Active"
          deltaType="positive"
        />
        <KPICard
          label="Flagged Consumers"
          value={kpis?.flagged_consumers?.toLocaleString() || '0'}
          delta={`${kpis?.anomaly_rate || 0}% Risk`}
          deltaType="negative"
        />
        <KPICard
          label="Est. Revenue At Risk"
          value={`₹ ${kpis?.estimated_loss?.toLocaleString() || 0}`}
          delta="Last 90 Days"
          deltaType="negative"
        />
        <KPICard
          label="Grid Efficiency"
          value={`${kpis?.grid_efficiency || 0}%`}
          delta="+1.2%"
          deltaType="positive"
        />
      </div>

      <div className="grid-2">
        <ChartContainer title="⚡ Consumption Trends (Time Series)" icon="">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={dailyConsumption}>
              <defs>
                <linearGradient id="colorConsumption" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={darkMode ? '#00ADB5' : '#0077B6'} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={darkMode ? '#00ADB5' : '#0077B6'} stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
              <XAxis 
                dataKey="timestamp" 
                stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                tick={{ fontSize: 12 }}
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
              <Area
                type="monotone"
                dataKey="consumption_kwh"
                stroke={darkMode ? '#00ADB5' : '#0077B6'}
                fillOpacity={1}
                fill="url(#colorConsumption)"
                name="Total Load (kWh)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="🚨 Anomaly Distribution" icon="">
          {anomalyDistribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={anomalyDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="count"
                  nameKey="type"
                  label={({ type, count }) => `${type}: ${count}`}
                >
                  {anomalyDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state">
              <p>No anomalies detected in this selection.</p>
            </div>
          )}
        </ChartContainer>
      </div>

      <h3 className="section-title">💰 Revenue Impact Analysis</h3>
      <RevenueImpact district={selectedDistrict} />

      <h3 className="section-title">🔔 Alert Management</h3>
      <AlertPanel district={selectedDistrict} />

      <h3 className="section-title">🏷️ Anomaly Classification</h3>
      <AnomalyClassification district={selectedDistrict} />
    </div>
  );
};

export default CommandCenter;
