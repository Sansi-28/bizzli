import React from 'react';
import './KPICard.css';

const KPICard = ({ label, value, delta, deltaType = 'neutral', prefix = '', suffix = '' }) => {
  return (
    <div className="kpi-card">
      <div className="label">{label}</div>
      <div className="value">{prefix}{value}{suffix}</div>
      {delta && (
        <span className={`delta ${deltaType}`}>
          {delta}
        </span>
      )}
    </div>
  );
};

export default KPICard;
