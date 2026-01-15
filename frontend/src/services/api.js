import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const healthCheck = () => api.get('/health');

// Districts
export const getDistricts = () => api.get('/districts');

// KPIs
export const getKPIs = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/kpis', { params });
};

// Consumption data
export const getDailyConsumption = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/consumption/daily', { params });
};

// Anomalies
export const getAnomalyDistribution = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/anomalies/distribution', { params });
};

export const getRecentAnomalies = (district = null, limit = 10) => {
  const params = { limit };
  if (district && district !== 'All Districts') {
    params.district = district;
  }
  return api.get('/anomalies/recent', { params });
};

// Map data
export const getMapConsumers = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/map/consumers', { params });
};

export const getDistrictRisk = () => api.get('/districts/risk');

// Consumer search and details
export const searchConsumers = (query) => {
  return api.get('/consumers/search', { params: { q: query } });
};

export const getConsumerDetails = (consumerId) => {
  return api.get(`/consumers/${consumerId}`);
};

// Model results
export const getModelResults = () => api.get('/model/results');

export const getFeatureImportance = () => api.get('/model/feature-importance');

// ============================================
// NEW FEATURE ENDPOINTS
// ============================================

// Alert Management
export const getAlerts = (district = null, severity = null, limit = 50) => {
  const params = { limit };
  if (district && district !== 'All Districts') params.district = district;
  if (severity && severity !== 'all') params.severity = severity;
  return api.get('/alerts', { params });
};

// Revenue Impact Analysis
export const getRevenueImpact = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/revenue/impact', { params });
};

// Anomaly Classification
export const getAnomalyClassification = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/anomalies/classification', { params });
};

// Consumer Comparison
export const compareConsumers = (consumerIds) => {
  return api.get('/consumers/compare', { params: { ids: consumerIds.join(',') } });
};

// Export Data
export const exportAnomalies = (district = null) => {
  const params = district && district !== 'All Districts' ? { district } : {};
  return api.get('/export/anomalies', { params });
};

// Summary Statistics
export const getSummaryStats = () => api.get('/stats/summary');

export default api;
