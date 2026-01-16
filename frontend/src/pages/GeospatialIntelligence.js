import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
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
import ChartContainer from '../components/ChartContainer';
import Loading from '../components/Loading';
import { useTheme } from '../context/ThemeContext';
import { getDistricts, getMapConsumers, getDistrictRisk } from '../services/api';
import 'leaflet/dist/leaflet.css';
import './GeospatialIntelligence.css';

const GeospatialIntelligence = () => {
  const { darkMode } = useTheme();
  const [loading, setLoading] = useState(true);
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('All Districts');
  const [mapData, setMapData] = useState([]);
  const [districtRisk, setDistrictRisk] = useState([]);
  const [mapView, setMapView] = useState('street');

  useEffect(() => {
    loadDistricts();
    loadDistrictRisk();
  }, []);

  useEffect(() => {
    loadMapData();
  }, [selectedDistrict]);

  const loadDistricts = async () => {
    try {
      const response = await getDistricts();
      setDistricts(response.data.districts || []);
    } catch (error) {
      console.error('Error loading districts:', error);
    }
  };

  const loadMapData = async () => {
    setLoading(true);
    try {
      const response = await getMapConsumers(selectedDistrict);
      setMapData(response.data.data || []);
    } catch (error) {
      console.error('Error loading map data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadDistrictRisk = async () => {
    try {
      const response = await getDistrictRisk();
      setDistrictRisk(response.data.data || []);
    } catch (error) {
      console.error('Error loading district risk:', error);
    }
  };

  const getTileUrl = () => {
    if (mapView === 'satellite') {
      return 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
    }
    if (mapView === 'terrain') {
      return 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png';
    }
    // Street view
    return darkMode
      ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
  };

  const getAttribution = () => {
    if (mapView === 'satellite') {
      return '&copy; <a href="https://www.esri.com/">Esri</a>';
    }
    if (mapView === 'terrain') {
      return '&copy; <a href="https://opentopomap.org">OpenTopoMap</a>';
    }
    return '&copy; <a href="https://carto.com/">CARTO</a>';
  };

  const tileUrl = getTileUrl();

  // Manipur center coordinates
  const mapCenter = [24.8170, 93.9368];

  if (loading && mapData.length === 0) {
    return <Loading message="Loading geospatial data..." />;
  }

  return (
    <div className="geospatial-page">
      <div className="page-header">
        <h1>🛰️ District Surveillance Map</h1>
        <p>Real-time consumer monitoring with anomaly detection overlay</p>
      </div>

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

        <label>Map View:</label>
        <select
          value={mapView}
          onChange={(e) => setMapView(e.target.value)}
          className="map-view-select"
        >
          <option value="street">Street Map</option>
          <option value="satellite">Satellite</option>
          <option value="terrain">Terrain</option>
        </select>
      </div>

      <div className="grid-3-1">
        <ChartContainer title="" className="map-chart-container">
          <div className="map-container">
            <MapContainer
              center={mapCenter}
              zoom={8}
              style={{ height: '500px', width: '100%' }}
            >
              <TileLayer
                attribution={getAttribution()}
                url={tileUrl}
              />
              {mapData.map((consumer) => {
                const getColor = (riskClass) => {
                  switch(riskClass) {
                    case 'high_risk': return '#FF4B4B';
                    case 'low_risk': return '#f9a825';
                    default: return '#00ADB5';
                  }
                };
                const color = getColor(consumer.risk_class);
                return (
                <CircleMarker
                  key={consumer.consumer_id}
                  center={[consumer.lat, consumer.lon]}
                  radius={consumer.risk_class === 'high_risk' ? 8 : (consumer.risk_class === 'low_risk' ? 6 : 4)}
                  fillColor={color}
                  color={color}
                  weight={1}
                  opacity={0.8}
                  fillOpacity={0.6}
                >
                  <Popup>
                    <div className="map-popup">
                      <h4>{consumer.name}</h4>
                      <p><strong>ID:</strong> {consumer.consumer_id}</p>
                      <p><strong>District:</strong> {consumer.district}</p>
                      <p><strong>Type:</strong> {consumer.consumer_type}</p>
                      <p>
                        <strong>Status:</strong>{' '}
                        <span className={consumer.risk_class}>
                          {consumer.status}
                        </span>
                      </p>
                    </div>
                  </Popup>
                </CircleMarker>
              )})}
            </MapContainer>
          </div>
          <div className="map-legend">
            <div className="legend-item">
              <span className="legend-dot high-risk"></span>
              High Risk
            </div>
            <div className="legend-item">
              <span className="legend-dot low-risk"></span>
              Low Risk
            </div>
            <div className="legend-item">
              <span className="legend-dot normal"></span>
              Normal
            </div>
          </div>
        </ChartContainer>

        <ChartContainer title="📍 District Risk Heatmap">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={districtRisk}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#2D3748' : '#E2E8F0'} />
              <XAxis
                type="number"
                stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                type="category"
                dataKey="district"
                stroke={darkMode ? '#A0AEC0' : '#4A5568'}
                tick={{ fontSize: 11 }}
                width={75}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: darkMode ? '#1A1C24' : '#FFFFFF',
                  border: `1px solid ${darkMode ? '#2D3748' : '#E2E8F0'}`,
                  borderRadius: '8px'
                }}
                formatter={(value) => [`${value.toFixed(1)}%`, 'Risk']}
              />
              <Bar dataKey="risk_percentage" name="Risk %" radius={[0, 4, 4, 0]}>
                {districtRisk.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      entry.risk_percentage > 10
                        ? '#ff2d55'
                        : entry.risk_percentage > 5
                        ? '#f9a825'
                        : '#00f5d4'
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="risk-info">
            <p>🔴 High risk areas require immediate attention</p>
          </div>
        </ChartContainer>
      </div>
    </div>
  );
};

export default GeospatialIntelligence;
