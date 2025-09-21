import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Import components we'll create
import BusMap from './components/BusMap';
import PredictionChart from './components/PredictionChart';
import OptimizationPanel from './components/OptimizationPanel';
import RealTimeMetrics from './components/RealTimeMetrics';

function App() {
  const [buses, setBuses] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [optimizations, setOptimizations] = useState({});
  const [selectedRoute, setSelectedRoute] = useState('RT001');
  const [analytics, setAnalytics] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [apiStatus, setApiStatus] = useState('connecting');

  const routes = [
    { id: 'RT001', name: 'Route 1 - Central to Airport', color: '#007bff' },
    { id: 'RT002', name: 'Route 2 - Mall to University', color: '#28a745' },
    { id: 'RT003', name: 'Route 3 - Station to Hospital', color: '#dc3545' }
  ];

  // Fetch data from backend
  const fetchData = async () => {
    try {
      // Check API status
      const statusResponse = await axios.get('http://localhost:5000/api/status');
      setApiStatus('connected');

      // Get real-time bus data
      const busResponse = await axios.get('http://localhost:5000/api/buses');
      setBuses(busResponse.data.buses);

      // Get predictions for selected route
      const predResponse = await axios.get(`http://localhost:5000/api/predictions/${selectedRoute}`);
      setPredictions(prev => ({
        ...prev,
        [selectedRoute]: predResponse.data.predictions
      }));

      // Get optimization data
      const optResponse = await axios.get(`http://localhost:5000/api/optimize/${selectedRoute}`);
      setOptimizations(prev => ({
        ...prev,
        [selectedRoute]: optResponse.data
      }));

      // Get analytics
      const analyticsResponse = await axios.get('http://localhost:5000/api/analytics');
      setAnalytics(analyticsResponse.data);

      setLastUpdate(new Date());

    } catch (error) {
      console.error('Error fetching data:', error);
      setApiStatus('error');
    }
  };

  // Auto-refresh data every 10 seconds
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [selectedRoute]);

  const getStatusBadge = () => {
    const statusConfig = {
      'connected': { color: 'success', text: '🟢 Live', pulse: true },
      'connecting': { color: 'warning', text: '🟡 Connecting...', pulse: true },
      'error': { color: 'danger', text: '🔴 Error', pulse: false }
    };
    
    const config = statusConfig[apiStatus];
    return (
      <span className={`badge bg-${config.color} ${config.pulse ? 'pulse' : ''}`}>
        {config.text}
      </span>
    );
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="bg-primary text-white shadow-sm">
        <div className="container-fluid py-3">
          <div className="row align-items-center">
            <div className="col-md-6">
              <h1 className="h3 mb-0">
                🚌 Smart Bus Management System
              </h1>
              <small className="text-light">
                Real-time ML-powered optimization & prediction
              </small>
            </div>
            <div className="col-md-6 text-end">
              {getStatusBadge()}
              <small className="text-light ms-3">
                Last Update: {lastUpdate.toLocaleTimeString()}
              </small>
            </div>
          </div>
        </div>
      </header>

      <div className="container-fluid mt-3">
        {/* Route Selection */}
        <div className="row mb-4">
          <div className="col-12">
            <div className="card">
              <div className="card-body py-2">
                <div className="btn-group" role="group">
                  {routes.map(route => (
                    <button
                      key={route.id}
                      type="button"
                      className={`btn ${selectedRoute === route.id ? 'btn-primary' : 'btn-outline-primary'}`}
                      onClick={() => setSelectedRoute(route.id)}
                    >
                      <span 
                        className="badge me-2" 
                        style={{backgroundColor: route.color}}
                      >
                        ●
                      </span>
                      {route.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Dashboard */}
        <div className="row">
          {/* Real-time Metrics */}
          <div className="col-md-3 mb-4">
            <RealTimeMetrics 
              buses={buses} 
              selectedRoute={selectedRoute}
              optimization={optimizations[selectedRoute]}
            />
          </div>

          {/* Bus Map */}
          <div className="col-md-9 mb-4">
            <div className="card h-100">
              <div className="card-header">
                <h5 className="mb-0">🗺️ Live Bus Tracking</h5>
              </div>
              <div className="card-body">
                <BusMap 
                  buses={buses.filter(bus => bus.route_id === selectedRoute)} 
                  route={routes.find(r => r.id === selectedRoute)}
                />
              </div>
            </div>
          </div>

          {/* Predictions Chart */}
          <div className="col-md-6 mb-4">
            <div className="card h-100">
              <div className="card-header">
                <h5 className="mb-0">📊 ML Demand Predictions</h5>
              </div>
              <div className="card-body">
                <PredictionChart 
                  predictions={predictions[selectedRoute] || []}
                />
              </div>
            </div>
          </div>

          {/* Optimization Panel */}
          <div className="col-md-6 mb-4">
            <OptimizationPanel 
              optimization={optimizations[selectedRoute]}
              routeName={routes.find(r => r.id === selectedRoute)?.name}
            />
          </div>

          {/* Analytics */}
          {analytics && (
            <div className="col-12 mb-4">
              <div className="card">
                <div className="card-header">
                  <h5 className="mb-0">📈 System Analytics</h5>
                </div>
                <div className="card-body">
                  <div className="row">
                    <div className="col-md-6">
                      <h6>Hourly Ridership Patterns</h6>
                      <div className="small">
                        {analytics.hourly_patterns.slice(6, 22).map(pattern => (
                          <div key={pattern.hour} className="d-flex justify-content-between">
                            <span>{pattern.hour}:00</span>
                            <span className="fw-bold">{pattern.avg_passengers} passengers</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="col-md-6">
                      <h6>Route Performance</h6>
                      <div className="small">
                        {analytics.route_performance.map(route => (
                          <div key={route.route_id} className="d-flex justify-content-between">
                            <span>{route.route_id}</span>
                            <span className="fw-bold">{route.avg_passengers} avg passengers</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
