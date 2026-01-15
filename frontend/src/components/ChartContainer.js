import React from 'react';
import './ChartContainer.css';

const ChartContainer = ({ title, icon, children, className = '' }) => {
  return (
    <div className={`chart-container ${className}`}>
      {title && (
        <h3>
          {icon && <span className="icon">{icon}</span>}
          {title}
        </h3>
      )}
      <div className="chart-content">
        {children}
      </div>
    </div>
  );
};

export default ChartContainer;
