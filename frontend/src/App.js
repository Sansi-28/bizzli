import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatBot from './components/ChatBot';
import CommandCenter from './pages/CommandCenter';
import GeospatialIntelligence from './pages/GeospatialIntelligence';
import ConsumerForensics from './pages/ConsumerForensics';
import SystemHealth from './pages/SystemHealth';
import { ThemeProvider } from './context/ThemeContext';
import './App.css';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="app-container">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<CommandCenter />} />
              <Route path="/geospatial" element={<GeospatialIntelligence />} />
              <Route path="/forensics" element={<ConsumerForensics />} />
              <Route path="/health" element={<SystemHealth />} />
            </Routes>
          </main>
          <ChatBot />
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
