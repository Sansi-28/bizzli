import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { 
  FiHome, 
  FiMap, 
  FiSearch, 
  FiActivity,
  FiMoon,
  FiSun
} from 'react-icons/fi';
import './Sidebar.css';

const Sidebar = () => {
  const { darkMode, toggleTheme } = useTheme();
  const location = useLocation();

  const navItems = [
    { path: '/', icon: <FiHome />, label: 'Command Center' },
    { path: '/geospatial', icon: <FiMap />, label: 'Geospatial Intelligence' },
    { path: '/forensics', icon: <FiSearch />, label: 'Consumer Forensics' },
    { path: '/health', icon: <FiActivity />, label: 'System Health' },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="icon">⚡</span>
        <div>
          <h1>Manipur GridWatch</h1>
          <p className="subtitle">PowerGuard System</p>
        </div>
      </div>

      <nav className="nav-section">
        <h3>📡 Navigation</h3>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <span className="icon">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="nav-section">
        <h3>🎨 Appearance</h3>
        <div className="theme-toggle" onClick={toggleTheme}>
          <span className="icon">{darkMode ? <FiMoon /> : <FiSun />}</span>
          <span>Dark Mode</span>
          <div className={`switch ${darkMode ? 'active' : ''}`}></div>
        </div>
      </div>

      <div className="sidebar-footer">
        <p>Manipur State Power</p>
        <p className="version">v1.0.0</p>
      </div>
    </aside>
  );
};

export default Sidebar;
