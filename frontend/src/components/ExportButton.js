import React, { useState } from 'react';
import { FiDownload, FiFileText, FiCheck } from 'react-icons/fi';
import { exportAnomalies } from '../services/api';
import './ExportButton.css';

const ExportButton = ({ district, label = 'Export Data' }) => {
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState(false);

  const handleExport = async () => {
    setExporting(true);
    try {
      const response = await exportAnomalies(district);
      const data = response.data;
      
      // Convert to CSV
      if (data.data && data.data.length > 0) {
        const headers = Object.keys(data.data[0]);
        const csvContent = [
          headers.join(','),
          ...data.data.map(row => 
            headers.map(h => {
              const val = row[h];
              // Escape quotes and wrap in quotes if contains comma
              if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
                return `"${val.replace(/"/g, '""')}"`;
              }
              return val;
            }).join(',')
          )
        ].join('\n');
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `anomaly_report_${new Date().toISOString().split('T')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        setExported(true);
        setTimeout(() => setExported(false), 3000);
      }
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  return (
    <button 
      className={`export-button ${exported ? 'exported' : ''}`}
      onClick={handleExport}
      disabled={exporting}
    >
      {exporting ? (
        <>
          <span className="spinner-small"></span>
          Exporting...
        </>
      ) : exported ? (
        <>
          <FiCheck />
          Exported!
        </>
      ) : (
        <>
          <FiDownload />
          {label}
        </>
      )}
    </button>
  );
};

export default ExportButton;
