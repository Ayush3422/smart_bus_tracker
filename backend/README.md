# Smart Bus Backend - Quick Start Guide

## 🚀 Features
- **Ultra-Advanced ML Integration**: Uses your 23-algorithm model with 144 features
- **Real-Time Bus Simulation**: Live GPS tracking and passenger counts
- **ML-Powered Predictions**: 24-hour ridership forecasting
- **Smart Schedule Optimization**: Dynamic frequency adjustment based on demand
- **Bunching Prevention**: Detects and alerts for bus bunching issues
- **Analytics Dashboard**: Comprehensive system performance metrics

## 📋 API Endpoints

### 1. System Status
```
GET /api/status
```
Returns server status, model information, and health check.

### 2. Real-Time Bus Data
```
GET /api/buses
```
Live bus locations, passenger counts, and speeds for all routes.

### 3. ML Predictions
```
GET /api/predictions/<route_id>
```
24-hour ridership predictions using ultra-advanced ML model.
- Example: `/api/predictions/RT001`

### 4. Schedule Optimization
```
GET /api/optimize/<route_id>
```
Smart schedule optimization with bunching prevention.
- Example: `/api/optimize/RT001`

### 5. Analytics Dashboard
```
GET /api/analytics
```
System-wide analytics including hourly patterns and route performance.

### 6. Model Information
```
GET /api/model/info
```
Detailed information about the loaded ML model.

## 🗄️ Database Schema

### Bus Locations (Real-time)
- timestamp, bus_id, route_id, latitude, longitude, passengers, speed_kmh

### Ridership Predictions
- route_id, timestamp, predicted_passengers, actual_passengers, created_at

### Schedule Optimizations
- route_id, original_frequency, optimized_frequency, improvement_percentage

## 🎯 Key Features

### ML Model Integration
- Automatically loads `ultra_advanced_bus_ridership_model.pkl`
- Fallback to basic model if advanced model unavailable
- Supports all 144 features and 23 algorithms
- Real-time passenger predictions

### Real-Time Simulation
- Updates every 10 seconds
- Simulates 6 buses across 3 routes
- Uses ML predictions for realistic passenger counts
- GPS coordinate simulation for Delhi area

### Smart Optimization
- Dynamic frequency adjustment (8-25 minutes)
- Demand-based scheduling
- Bunching detection and prevention
- Cost optimization for low-demand periods

## 🔧 Installation & Usage

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   python app.py
   ```

3. **Access API**:
   - Server runs on `http://localhost:5000`
   - Test with: `http://localhost:5000/api/status`

## 📊 Sample API Responses

### Status Response
```json
{
  "status": "online",
  "model_loaded": true,
  "model_type": "ultra_advanced_model",
  "features_count": 144,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Predictions Response
```json
{
  "route_id": "RT001",
  "predictions": [
    {
      "hour": 10,
      "predicted_passengers": 45,
      "is_peak": false,
      "day_type": "weekday"
    }
  ],
  "model_used": "ultra_advanced_model"
}
```

### Optimization Response
```json
{
  "route_id": "RT001",
  "current_demand": 85,
  "schedule_optimization": {
    "original_frequency": 15,
    "optimized_frequency": 8,
    "improvement_percentage": 46.7,
    "reason": "High demand detected - increased frequency"
  },
  "bunching_prevention": {
    "bunching_detected": false,
    "recommendation": "Normal operation"
  }
}
```

## 🎨 Frontend Integration

The backend is CORS-enabled and ready for frontend integration:

```javascript
// Example fetch requests
const status = await fetch('http://localhost:5000/api/status');
const buses = await fetch('http://localhost:5000/api/buses');
const predictions = await fetch('http://localhost:5000/api/predictions/RT001');
```

## 🔍 Monitoring & Logs

The server provides detailed console logging:
- ML model loading status
- Real-time data updates every 10 seconds
- Error handling and fallback mechanisms
- Database operations

## 🚌 Route Information

**Route RT001**: Central to Airport (BUS001, BUS002)
**Route RT002**: Mall to University (BUS003, BUS004)  
**Route RT003**: Station to Hospital (BUS005, BUS006)

## 🏆 Performance Highlights

- **Ultra-Advanced ML**: 23 algorithms, 144 features
- **Real-Time Updates**: 10-second intervals
- **Smart Optimization**: Dynamic frequency adjustment
- **Scalable**: SQLite database with efficient queries
- **Production-Ready**: Error handling and fallbacks