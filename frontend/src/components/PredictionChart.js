import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function PredictionChart({ predictions }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="text-center text-muted">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p className="mt-2">Loading ML predictions...</p>
      </div>
    );
  }

  const next12Hours = predictions.slice(0, 12);

  const data = {
    labels: next12Hours.map(p => `${p.hour}:00`),
    datasets: [
      {
        label: 'Predicted Passengers',
        data: next12Hours.map(p => p.predicted_passengers),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.4,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointBackgroundColor: next12Hours.map(p => p.is_peak ? '#ff6384' : '#75c2c2')
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Next 12 Hours ML Predictions (Ultra-Advanced Model)'
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context) {
            const prediction = next12Hours[context.dataIndex];
            return [
              `Day: ${prediction.day_type}`,
              `Peak Hour: ${prediction.is_peak ? 'Yes' : 'No'}`
            ];
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Passengers'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time (Hours)'
        }
      }
    }
  };

  // Peak hour detection
  const currentHour = new Date().getHours();
  const isPeakHour = (currentHour >= 7 && currentHour <= 9) || (currentHour >= 17 && currentHour <= 19);
  
  // Calculate statistics
  const avgPrediction = Math.round(next12Hours.reduce((sum, p) => sum + p.predicted_passengers, 0) / next12Hours.length);
  const maxPrediction = Math.max(...next12Hours.map(p => p.predicted_passengers));
  const nextHourPrediction = next12Hours[0]?.predicted_passengers || 0;

  return (
    <div>
      <Line data={data} options={options} />
      
      <div className="mt-3">
        {isPeakHour && (
          <div className="alert alert-warning py-2">
            ⚡ Peak hour detected - higher passenger demand expected
          </div>
        )}
        
        <div className="row text-center">
          <div className="col-4">
            <div className="border-end">
              <div className="h4 mb-0 text-primary">
                {nextHourPrediction}
              </div>
              <small className="text-muted">Next Hour</small>
            </div>
          </div>
          <div className="col-4">
            <div className="border-end">
              <div className="h4 mb-0 text-success">
                {avgPrediction}
              </div>
              <small className="text-muted">12hr Average</small>
            </div>
          </div>
          <div className="col-4">
            <div className="h4 mb-0 text-info">
              {maxPrediction}
            </div>
            <small className="text-muted">Peak Today</small>
          </div>
        </div>
        
        {/* Peak Hours Indicator */}
        <div className="mt-3 p-2 bg-light rounded">
          <div className="row">
            <div className="col-6">
              <small className="text-muted">🔥 Peak Hours:</small>
              <div className="small">
                {next12Hours.filter(p => p.is_peak).map(p => (
                  <span key={p.hour} className="badge bg-danger me-1">
                    {p.hour}:00
                  </span>
                ))}
              </div>
            </div>
            <div className="col-6">
              <small className="text-muted">🌙 Off-Peak:</small>
              <div className="small">
                {next12Hours.filter(p => !p.is_peak).slice(0, 4).map(p => (
                  <span key={p.hour} className="badge bg-secondary me-1">
                    {p.hour}:00
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PredictionChart;