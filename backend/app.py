from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import sqlite3
import joblib
import random
import os
import sys

# Add parent directory to path to import ML model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the ML model classes to enable loading the ultra-advanced model
try:
    from ml_model import RobustFeatureEngineer, UltraAdvancedModelOptimizer
    print("✅ Imported ML model classes successfully")
except ImportError as e:
    print(f"⚠️ Could not import ML model classes: {e}")
    print("📝 Model will use fallback prediction method")

app = Flask(__name__)
CORS(app)

# Global variables for ML model
ml_model = None
best_scaler = None
feature_columns = []
model_name = "ultra_advanced_model"
model_loaded = False

def load_ml_model():
    """Load the ultra-advanced ML model"""
    global ml_model, best_scaler, feature_columns, model_name, model_loaded
    
    print("🤖 Loading Ultra-Advanced ML model...")
    
    try:
        # Try to load the ultra-advanced model first
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ultra_advanced_bus_ridership_model.pkl')
        print(f"🔍 Looking for model at: {model_path}")
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            ml_model = model_data.get('best_model')
            best_scaler = model_data.get('best_scaler')
            feature_columns = model_data.get('feature_columns', [])
            model_name = model_data.get('best_model_name', 'ultra_advanced_model')
            
            if ml_model is not None:
                model_loaded = True
                print(f"✅ Loaded {model_name} model with {len(feature_columns)} features")
                return True
        
        # Fallback to basic model
        fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bus_ridership_model.pkl')
        print(f"🔍 Looking for fallback model at: {fallback_path}")
        if os.path.exists(fallback_path):
            print("⚠️ Ultra-advanced model not found, loading fallback model...")
            model_data = joblib.load(fallback_path)
            ml_model = model_data.get('model', model_data.get('best_model'))
            best_scaler = model_data.get('scaler', model_data.get('best_scaler'))
            feature_columns = model_data.get('feature_columns', ['hour', 'day_of_week', 'is_weekend'])
            model_name = model_data.get('model_name', 'fallback_model')
            
            if ml_model is not None:
                model_loaded = True
                print(f"✅ Loaded fallback {model_name} model")
                return True
        
        print("❌ No ML model found! Creating mock predictor...")
        model_loaded = False
        return False
        
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        print("🔧 Using mock predictor for demonstration...")
        model_loaded = False
        return False

# Database setup
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('bus_system.db')
    
    # Real-time bus data
    conn.execute('''
        CREATE TABLE IF NOT EXISTS bus_locations (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            bus_id TEXT,
            route_id TEXT,
            latitude REAL,
            longitude REAL,
            passengers INTEGER,
            speed_kmh REAL
        )
    ''')
    
    # Predictions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ridership_predictions (
            id INTEGER PRIMARY KEY,
            route_id TEXT,
            timestamp TEXT,
            predicted_passengers INTEGER,
            actual_passengers INTEGER,
            created_at TEXT
        )
    ''')
    
    # Schedule optimization
    conn.execute('''
        CREATE TABLE IF NOT EXISTS schedule_optimizations (
            id INTEGER PRIMARY KEY,
            route_id TEXT,
            original_frequency INTEGER,
            optimized_frequency INTEGER,
            improvement_percentage REAL,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# ML Prediction Function
def predict_ridership(hour, day_of_week, is_weekend, weather_factor=1.0, route_id='RT001'):
    """Use trained ultra-advanced ML model to predict ridership"""
    global ml_model, best_scaler, feature_columns, model_loaded
    
    if not model_loaded or ml_model is None:
        # Mock prediction for demonstration
        base_ridership = 50
        hour_factor = 1.5 if hour in [7, 8, 9, 17, 18, 19] else 0.7 if hour in [22, 23, 0, 1, 2] else 1.0
        weekend_factor = 0.6 if is_weekend else 1.0
        weather_factor = max(0.5, min(1.5, weather_factor))
        
        prediction = base_ridership * hour_factor * weekend_factor * weather_factor
        return max(1, int(prediction + random.uniform(-10, 10)))
    
    try:
        # Prepare features for the ultra-advanced model
        current_time = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Create feature dictionary with all possible features
        features = {}
        
        # Basic temporal features
        features['hour'] = hour
        features['day_of_week'] = day_of_week
        features['month'] = current_time.month
        features['day'] = current_time.day
        features['week_of_year'] = current_time.isocalendar()[1]
        features['day_of_year'] = current_time.timetuple().tm_yday
        features['quarter'] = (current_time.month - 1) // 3 + 1
        features['is_weekend'] = is_weekend
        features['weather_factor'] = weather_factor
        
        # Cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * current_time.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * current_time.month / 12)
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Peak hour features
        features['is_morning_peak'] = hour in [7, 8, 9]
        features['is_evening_peak'] = hour in [17, 18, 19]
        features['is_lunch_hour'] = hour in [12, 13]
        features['is_late_night'] = hour in [22, 23, 0, 1, 2]
        
        # Holiday features (simplified)
        features['is_holiday'] = False
        features['days_to_holiday'] = 10
        features['is_day_before_holiday'] = False
        features['is_day_after_holiday'] = False
        
        # Interaction features
        features['hour_dow_interaction'] = hour * day_of_week
        features['peak_weekend_interaction'] = (hour in [7, 8, 9, 17, 18, 19]) * is_weekend
        features['weather_hour_interaction'] = weather_factor * hour
        features['weather_peak_interaction'] = weather_factor * (hour in [7, 8, 9, 17, 18, 19])
        
        # Route features (defaults for prediction)
        route_features = {
            'route_id_encoded': 0,
            'route_avg_passengers': 50.0,
            'route_std_passengers': 15.0,
            'route_max_passengers': 120.0,
            'route_min_passengers': 5.0,
            'route_hour_avg_passengers': 50.0
        }
        features.update(route_features)
        
        # Lag and rolling features (use defaults for prediction)
        for col in feature_columns:
            if 'lag' in col and col not in features:
                features[col] = 45.0
            elif 'rolling' in col and col not in features:
                if 'mean' in col:
                    features[col] = 50.0
                elif 'std' in col:
                    features[col] = 12.0
                else:
                    features[col] = 25.0
            elif col not in features:
                features[col] = 0.0
        
        # Create DataFrame with only the features that exist in the model
        available_features = {col: features.get(col, 0.0) for col in feature_columns}
        test_df = pd.DataFrame([available_features])
        
        # Make prediction with proper scaling
        requires_scaling = any(model_type in model_name.lower() 
                             for model_type in ['neural_network', 'svr', 'ridge', 'lasso', 
                                              'elastic_net', 'bayesian', 'huber', 'theil_sen', 
                                              'ransac', 'gaussian_process'])
        
        if requires_scaling and best_scaler is not None:
            test_scaled = best_scaler.transform(test_df)
            prediction = ml_model.predict(test_scaled)[0]
        else:
            prediction = ml_model.predict(test_df)[0]
        
        return max(1, int(prediction))
        
    except Exception as e:
        print(f"⚠️ Prediction error: {e}, using fallback")
        # Fallback prediction
        base_ridership = 50
        hour_factor = 1.5 if hour in [7, 8, 9, 17, 18, 19] else 0.7 if hour in [22, 23, 0, 1, 2] else 1.0
        weekend_factor = 0.6 if is_weekend else 1.0
        
        prediction = base_ridership * hour_factor * weekend_factor * weather_factor
        return max(1, int(prediction + random.uniform(-5, 5)))

# Real-time data simulator
class BusSimulator:
    def __init__(self):
        self.routes = [
            {'id': 'RT001', 'name': 'Route 1 - Central to Airport', 'buses': ['BUS001', 'BUS002']},
            {'id': 'RT002', 'name': 'Route 2 - Mall to University', 'buses': ['BUS003', 'BUS004']},  
            {'id': 'RT003', 'name': 'Route 3 - Station to Hospital', 'buses': ['BUS005', 'BUS006']}
        ]
        
        # Starting positions (Delhi coordinates)
        self.bus_positions = {
            'BUS001': {'lat': 28.6139, 'lng': 77.2090, 'route': 'RT001'},
            'BUS002': {'lat': 28.6200, 'lng': 77.2100, 'route': 'RT001'},
            'BUS003': {'lat': 28.5355, 'lng': 77.3910, 'route': 'RT002'},
            'BUS004': {'lat': 28.5400, 'lng': 77.3950, 'route': 'RT002'},
            'BUS005': {'lat': 28.6692, 'lng': 77.4538, 'route': 'RT003'},
            'BUS006': {'lat': 28.6700, 'lng': 77.4600, 'route': 'RT003'}
        }
        self.running = True
        
    def simulate_real_time_data(self):
        """Generate real-time bus data with ML predictions"""
        while self.running:
            try:
                conn = sqlite3.connect('bus_system.db')
                current_time = datetime.now()
                
                for bus_id, position in self.bus_positions.items():
                    # Simulate GPS movement
                    position['lat'] += random.uniform(-0.001, 0.001)
                    position['lng'] += random.uniform(-0.001, 0.001)
                    
                    # Use ML model for realistic passenger prediction
                    predicted_passengers = predict_ridership(
                        hour=current_time.hour,
                        day_of_week=current_time.weekday(),
                        is_weekend=current_time.weekday() >= 5,
                        weather_factor=random.uniform(0.8, 1.2),
                        route_id=position['route']
                    )
                    
                    # Add some randomness to make it more realistic
                    actual_passengers = max(1, predicted_passengers + random.randint(-10, 15))
                    
                    # Insert real-time data
                    conn.execute('''
                        INSERT INTO bus_locations 
                        (timestamp, bus_id, route_id, latitude, longitude, passengers, speed_kmh)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        current_time.isoformat(),
                        bus_id,
                        position['route'],
                        position['lat'],
                        position['lng'],
                        actual_passengers,
                        random.randint(15, 45)  # Speed
                    ))
                
                conn.commit()
                conn.close()
                
                print(f"📡 Updated real-time data at {current_time.strftime('%H:%M:%S')}")
                
            except Exception as e:
                print(f"⚠️ Simulation error: {e}")
            
            time.sleep(10)  # Update every 10 seconds
    
    def start_simulation(self):
        thread = threading.Thread(target=self.simulate_real_time_data)
        thread.daemon = True
        thread.start()
        return thread

# API Endpoints
@app.route('/api/status')
def api_status():
    """Get API status and model information"""
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'model_type': model_name,
        'features_count': len(feature_columns),
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/api/buses')
def get_buses():
    """Get current bus positions and passenger counts"""
    try:
        conn = sqlite3.connect('bus_system.db')
        
        query = '''
            SELECT bus_id, route_id, latitude, longitude, passengers, speed_kmh, timestamp
            FROM bus_locations 
            WHERE timestamp >= datetime('now', '-5 minutes')
            ORDER BY timestamp DESC
        '''
        
        result = conn.execute(query).fetchall()
        conn.close()
        
        buses = []
        for row in result:
            buses.append({
                'bus_id': row[0],
                'route_id': row[1],
                'latitude': round(row[2], 6),
                'longitude': round(row[3], 6),
                'passengers': row[4],
                'speed_kmh': row[5],
                'last_update': row[6]
            })
        
        return jsonify({
            'buses': buses,
            'count': len(buses),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<route_id>')
def get_predictions(route_id):
    """Get ML-powered ridership predictions for next 24 hours"""
    try:
        current_time = datetime.now()
        predictions = []
        
        # Generate predictions for next 24 hours
        for i in range(24):
            future_time = current_time + timedelta(hours=i)
            
            predicted_passengers = predict_ridership(
                hour=future_time.hour,
                day_of_week=future_time.weekday(),
                is_weekend=future_time.weekday() >= 5,
                weather_factor=1.0,  # Normal weather
                route_id=route_id
            )
            
            predictions.append({
                'hour': future_time.hour,
                'timestamp': future_time.isoformat(),
                'predicted_passengers': predicted_passengers,
                'day_type': 'weekend' if future_time.weekday() >= 5 else 'weekday',
                'is_peak': future_time.hour in [7, 8, 9, 17, 18, 19]
            })
        
        return jsonify({
            'route_id': route_id,
            'predictions': predictions,
            'model_used': model_name,
            'generated_at': current_time.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize/<route_id>')
def optimize_schedule(route_id):
    """Smart schedule optimization with ML insights"""
    try:
        current_time = datetime.now()
        
        # Get current demand prediction
        current_demand = predict_ridership(
            hour=current_time.hour,
            day_of_week=current_time.weekday(),
            is_weekend=current_time.weekday() >= 5,
            route_id=route_id
        )
        
        # Get next hour demand
        next_hour_demand = predict_ridership(
            hour=(current_time.hour + 1) % 24,
            day_of_week=current_time.weekday(),
            is_weekend=current_time.weekday() >= 5,
            route_id=route_id
        )
        
        # Original schedule (baseline)
        original_frequency = 15  # Every 15 minutes
        
        # Smart optimization logic based on ultra-advanced predictions
        if current_demand > 80:
            optimized_frequency = 8  # Every 8 minutes
            optimization_reason = "High demand detected - increased frequency"
        elif current_demand > 60:
            optimized_frequency = 12  # Every 12 minutes  
            optimization_reason = "Moderate demand - slight frequency increase"
        elif current_demand < 20:
            optimized_frequency = 25  # Every 25 minutes
            optimization_reason = "Low demand - reduced frequency to save costs"
        else:
            optimized_frequency = 15  # Keep original
            optimization_reason = "Optimal frequency maintained"
        
        improvement = abs(original_frequency - optimized_frequency)
        improvement_pct = (improvement / original_frequency) * 100
        
        # Bus bunching detection (simplified)
        conn = sqlite3.connect('bus_system.db')
        bunching_query = '''
            SELECT COUNT(*) as bus_count
            FROM bus_locations 
            WHERE route_id = ? AND timestamp >= datetime('now', '-10 minutes')
            GROUP BY ROUND(latitude, 3), ROUND(longitude, 3)
            HAVING bus_count > 1
        '''
        
        bunching_result = conn.execute(bunching_query, (route_id,)).fetchall()
        conn.close()
        
        bunching_detected = len(bunching_result) > 0
        bunching_alerts = [f"Multiple buses detected at same location"] if bunching_detected else []
        
        return jsonify({
            'route_id': route_id,
            'current_demand': current_demand,
            'next_hour_demand': next_hour_demand,
            'schedule_optimization': {
                'original_frequency': original_frequency,
                'optimized_frequency': optimized_frequency,
                'improvement_minutes': improvement,
                'improvement_percentage': round(improvement_pct, 1),
                'reason': optimization_reason
            },
            'bunching_prevention': {
                'bunching_detected': bunching_detected,
                'alerts': bunching_alerts,
                'recommendation': "Slow down leading bus" if bunching_detected else "Normal operation"
            },
            'ml_model_used': model_name,
            'timestamp': current_time.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get system analytics and performance metrics"""
    try:
        conn = sqlite3.connect('bus_system.db')
        
        # Average passengers by hour
        hourly_query = '''
            SELECT 
                strftime('%H', timestamp) as hour,
                AVG(passengers) as avg_passengers,
                COUNT(*) as data_points
            FROM bus_locations 
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        '''
        
        hourly_data = conn.execute(hourly_query).fetchall()
        
        # Route performance
        route_query = '''
            SELECT 
                route_id,
                AVG(passengers) as avg_passengers,
                MAX(passengers) as max_passengers,
                MIN(passengers) as min_passengers,
                COUNT(*) as data_points
            FROM bus_locations 
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY route_id
        '''
        
        route_data = conn.execute(route_query).fetchall()
        
        # Real-time summary
        summary_query = '''
            SELECT 
                COUNT(DISTINCT bus_id) as active_buses,
                AVG(passengers) as avg_current_passengers,
                AVG(speed_kmh) as avg_speed
            FROM bus_locations 
            WHERE timestamp >= datetime('now', '-10 minutes')
        '''
        
        summary_result = conn.execute(summary_query).fetchone()
        conn.close()
        
        analytics = {
            'hourly_patterns': [
                {
                    'hour': int(row[0]), 
                    'avg_passengers': round(row[1], 1),
                    'data_points': row[2]
                }
                for row in hourly_data
            ],
            'route_performance': [
                {
                    'route_id': row[0],
                    'avg_passengers': round(row[1], 1),
                    'max_passengers': row[2],
                    'min_passengers': row[3],
                    'data_points': row[4]
                }
                for row in route_data
            ],
            'system_summary': {
                'active_buses': summary_result[0] if summary_result[0] else 0,
                'avg_current_passengers': round(summary_result[1], 1) if summary_result[1] else 0,
                'avg_speed_kmh': round(summary_result[2], 1) if summary_result[2] else 0
            },
            'model_info': {
                'model_name': model_name,
                'model_loaded': model_loaded,
                'features_count': len(feature_columns)
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/info')
def get_model_info():
    """Get detailed information about the loaded ML model"""
    return jsonify({
        'model_name': model_name,
        'model_loaded': model_loaded,
        'features_count': len(feature_columns),
        'feature_sample': feature_columns[:10] if feature_columns else [],
        'scaler_available': best_scaler is not None,
        'model_type': type(ml_model).__name__ if ml_model else None,
        'timestamp': datetime.now().isoformat()
    })

# Initialize and run
if __name__ == '__main__':
    print("🚀 Starting Smart Bus Backend Server...")
    print("=" * 50)
    
    # Load ML model
    load_ml_model()
    
    # Initialize database
    init_database()
    
    # Start real-time simulation
    simulator = BusSimulator()
    simulator.start_simulation()
    
    print("\n✅ Backend server ready!")
    print("🌐 API endpoints available:")
    print("  - GET /api/status")
    print("  - GET /api/buses")
    print("  - GET /api/predictions/<route_id>")
    print("  - GET /api/optimize/<route_id>")
    print("  - GET /api/analytics")
    print("  - GET /api/model/info")
    print(f"🚀 Server starting on http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, port=5000, host='0.0.0.0')