import React from 'react';

function BusMap({ buses, route }) {
  return (
    <div className="text-center">
      <div className="mb-3">
        <span 
          className="badge fs-6 px-3 py-2" 
          style={{backgroundColor: route?.color}}
        >
          {route?.name}
        </span>
      </div>
      
      <div className="row g-3">
        {buses.map(bus => (
          <div key={bus.bus_id} className="col-md-6">
            <div className="card border-primary">
              <div className="card-body p-3">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <h6 className="card-title mb-0">{bus.bus_id}</h6>
                  <span className="badge bg-success">
                    {bus.passengers} 👥
                  </span>
                </div>
                
                <div className="small text-muted">
                  <div>📍 {bus.latitude.toFixed(4)}, {bus.longitude.toFixed(4)}</div>
                  <div>⚡ {bus.speed_kmh} km/h</div>
                  <div>🕐 {new Date(bus.last_update).toLocaleTimeString()}</div>
                </div>
                
                {/* Passenger Load Indicator */}
                <div className="mt-2">
                  <div className="progress" style={{height: '8px'}}>
                    <div 
                      className={`progress-bar ${
                        bus.passengers > 80 ? 'bg-danger' : 
                        bus.passengers > 50 ? 'bg-warning' : 'bg-success'
                      }`}
                      style={{width: `${Math.min(100, (bus.passengers / 120) * 100)}%`}}
                    ></div>
                  </div>
                  <small className="text-muted">
                    {bus.passengers > 80 ? 'High Load' : 
                     bus.passengers > 50 ? 'Medium Load' : 'Normal Load'}
                  </small>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {buses.length === 0 && (
        <div className="alert alert-info">
          <i className="bi bi-info-circle me-2"></i>
          No buses currently active on this route
        </div>
      )}
      
      {/* Route Summary */}
      {buses.length > 0 && (
        <div className="mt-4 p-3 bg-light rounded">
          <div className="row text-center">
            <div className="col-4">
              <div className="h5 mb-0 text-primary">{buses.length}</div>
              <small className="text-muted">Active Buses</small>
            </div>
            <div className="col-4">
              <div className="h5 mb-0 text-success">
                {Math.round(buses.reduce((sum, bus) => sum + bus.passengers, 0) / buses.length)}
              </div>
              <small className="text-muted">Avg Passengers</small>
            </div>
            <div className="col-4">
              <div className="h5 mb-0 text-info">
                {Math.round(buses.reduce((sum, bus) => sum + bus.speed_kmh, 0) / buses.length)}
              </div>
              <small className="text-muted">Avg Speed (km/h)</small>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default BusMap;