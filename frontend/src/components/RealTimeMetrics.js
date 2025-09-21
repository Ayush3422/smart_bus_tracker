import React from 'react';

function RealTimeMetrics({ buses, selectedRoute, optimization }) {
  // Filter buses for selected route
  const routeBuses = buses.filter(bus => bus.route_id === selectedRoute);
  
  // Calculate metrics
  const totalPassengers = routeBuses.reduce((sum, bus) => sum + bus.passengers, 0);
  const avgSpeed = routeBuses.length > 0 ? 
    Math.round(routeBuses.reduce((sum, bus) => sum + bus.speed_kmh, 0) / routeBuses.length) : 0;
  const busCapacity = routeBuses.length * 120; // Assuming 120 passenger capacity per bus
  const occupancyRate = busCapacity > 0 ? Math.round((totalPassengers / busCapacity) * 100) : 0;

  // Get optimization status
  const currentDemand = optimization?.current_demand || 0;
  const optimizedFrequency = optimization?.schedule_optimization?.optimized_frequency || 15;

  const getOccupancyColor = () => {
    if (occupancyRate > 80) return 'danger';
    if (occupancyRate > 60) return 'warning';
    return 'success';
  };

  const getSpeedColor = () => {
    if (avgSpeed < 15) return 'danger';
    if (avgSpeed < 25) return 'warning';
    return 'success';
  };

  return (
    <div>
      {/* Route Overview Card */}
      <div className="card mb-3">
        <div className="card-header">
          <h6 className="mb-0">📊 Route Overview</h6>
        </div>
        <div className="card-body">
          <div className="row text-center">
            <div className="col-6">
              <div className="h4 mb-0 text-primary">{routeBuses.length}</div>
              <small className="text-muted">Active Buses</small>
            </div>
            <div className="col-6">
              <div className="h4 mb-0 text-success">{totalPassengers}</div>
              <small className="text-muted">Total Passengers</small>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Metrics */}
      <div className="card mb-3">
        <div className="card-header">
          <h6 className="mb-0">🚌 Live Metrics</h6>
        </div>
        <div className="card-body">
          {/* Occupancy Rate */}
          <div className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <small className="text-muted">Occupancy Rate</small>
              <span className={`badge bg-${getOccupancyColor()}`}>{occupancyRate}%</span>
            </div>
            <div className="progress" style={{height: '8px'}}>
              <div 
                className={`progress-bar bg-${getOccupancyColor()}`}
                style={{width: `${Math.min(100, occupancyRate)}%`}}
              ></div>
            </div>
          </div>

          {/* Average Speed */}
          <div className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <small className="text-muted">Average Speed</small>
              <span className={`badge bg-${getSpeedColor()}`}>{avgSpeed} km/h</span>
            </div>
            <div className="progress" style={{height: '8px'}}>
              <div 
                className={`progress-bar bg-${getSpeedColor()}`}
                style={{width: `${Math.min(100, (avgSpeed / 50) * 100)}%`}}
              ></div>
            </div>
          </div>

          {/* Current Demand */}
          <div className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <small className="text-muted">ML Predicted Demand</small>
              <span className="badge bg-info">{currentDemand}</span>
            </div>
            <div className="progress" style={{height: '8px'}}>
              <div 
                className="progress-bar bg-info"
                style={{width: `${Math.min(100, (currentDemand / 120) * 100)}%`}}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Optimization Status */}
      <div className="card mb-3">
        <div className="card-header">
          <h6 className="mb-0">⚡ Optimization Status</h6>
        </div>
        <div className="card-body">
          <div className="text-center">
            <div className="h5 mb-1 text-warning">{optimizedFrequency} min</div>
            <small className="text-muted">Optimized Frequency</small>
          </div>
          
          {optimization?.bunching_prevention?.bunching_detected && (
            <div className="alert alert-warning mt-3 py-2">
              <small>⚠️ Bunching detected</small>
            </div>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="card">
        <div className="card-header">
          <h6 className="mb-0">🔔 Alerts & Status</h6>
        </div>
        <div className="card-body">
          {/* High Occupancy Alert */}
          {occupancyRate > 80 && (
            <div className="alert alert-danger py-2 mb-2">
              <small>🔴 High occupancy detected</small>
            </div>
          )}
          
          {/* Low Speed Alert */}
          {avgSpeed < 15 && routeBuses.length > 0 && (
            <div className="alert alert-warning py-2 mb-2">
              <small>🟡 Traffic congestion detected</small>
            </div>
          )}
          
          {/* Normal Operation */}
          {occupancyRate <= 80 && avgSpeed >= 15 && (
            <div className="alert alert-success py-2 mb-2">
              <small>✅ Normal operation</small>
            </div>
          )}
          
          {/* No buses alert */}
          {routeBuses.length === 0 && (
            <div className="alert alert-info py-2">
              <small>ℹ️ No buses active on route</small>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RealTimeMetrics;