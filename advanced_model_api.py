
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def predict_bus_ridership(hour, day_of_week, is_weekend, weather_factor=1.0, route_id="DTC_01"):
    """
    Advanced Bus Ridership Prediction API
    
    Predict bus ridership for Delhi routes with comprehensive feature engineering
    
    Args:
        hour (int): Hour of day (0-23)
        day_of_week (int): Day of week (0=Monday, 6=Sunday)
        is_weekend (bool): True if weekend
        weather_factor (float): Weather impact (0.5=bad, 1.0=normal, 1.5=excellent)
        route_id (str): Route identifier (e.g., "DTC_01", "DTC_34")
    
    Returns:
        dict: {
            'predicted_passengers': int,
            'confidence_level': str,
            'peak_hour': bool,
            'weather_impact': str,
            'model_version': str
        }
    """
    try:
        # Load the advanced model
        model_data = joblib.load('advanced_bus_ridership_model.pkl')
        model = model_data['best_model']
        scaler = model_data['scaler']
        model_name = model_data['model_name']
        feature_columns = model_data['feature_columns']
        
        current_time = datetime.now().replace(hour=hour, minute=0, second=0)
        
        # Prepare comprehensive features
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': current_time.month,
            'day': current_time.day,
            'week_of_year': current_time.isocalendar()[1],
            'day_of_year': current_time.timetuple().tm_yday,
            'quarter': (current_time.month - 1) // 3 + 1,
            'is_weekend': is_weekend,
            'weather_factor': weather_factor,
            
            # Cyclical features
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * current_time.month / 12),
            'month_cos': np.cos(2 * np.pi * current_time.month / 12),
            'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
            
            # Holiday features
            'is_holiday': False,
            'days_to_holiday': 10,
            'is_day_before_holiday': False,
            'is_day_after_holiday': False,
            
            # Peak hour features
            'is_morning_peak': hour in [7, 8, 9],
            'is_evening_peak': hour in [17, 18, 19],
            'is_lunch_hour': hour in [12, 13],
            'is_late_night': hour in [22, 23, 0, 1, 2],
            
            # Interaction features
            'hour_dow_interaction': hour * day_of_week,
            'peak_weekend_interaction': (hour in [7, 8, 9, 17, 18, 19]) * is_weekend,
            'weather_hour_interaction': weather_factor * hour,
            'weather_peak_interaction': weather_factor * (hour in [7, 8, 9, 17, 18, 19]),
        }
        
        # Add default values for missing features
        for col in feature_columns:
            if col not in features:
                if 'route' in col:
                    if 'encoded' in col:
                        features[col] = 0
                    elif 'avg' in col:
                        features[col] = 50.0
                    else:
                        features[col] = 15.0 if 'std' in col else 50.0
                elif 'lag' in col:
                    features[col] = 45.0
                elif 'rolling' in col:
                    features[col] = 50.0 if 'mean' in col else 12.0
                else:
                    features[col] = 0.0
        
        # Create DataFrame and predict
        test_df = pd.DataFrame([features])[feature_columns]
        
        if model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
            test_scaled = scaler.transform(test_df)
            prediction = model.predict(test_scaled)[0]
        else:
            prediction = model.predict(test_df)[0]
        
        passengers = max(1, int(prediction))
        
        # Determine confidence and characteristics
        is_peak = hour in [7, 8, 9, 17, 18, 19]
        confidence = "High" if is_peak or not is_weekend else "Medium"
        
        if weather_factor <= 0.6:
            weather_impact = "Significant Decrease"
        elif weather_factor <= 0.8:
            weather_impact = "Moderate Decrease"
        elif weather_factor >= 1.2:
            weather_impact = "Moderate Increase"
        else:
            weather_impact = "Normal"
        
        return {
            'predicted_passengers': passengers,
            'confidence_level': confidence,
            'peak_hour': is_peak,
            'weather_impact': weather_impact,
            'model_version': model_name + ' v2.0',
            'route_id': route_id,
            'timestamp': current_time.isoformat()
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'predicted_passengers': 50,  # Fallback value
            'confidence_level': 'Low',
            'peak_hour': False,
            'weather_impact': 'Unknown',
            'model_version': 'Fallback'
        }

# Example usage:
if __name__ == "__main__":
    # Test the API
    result = predict_bus_ridership(8, 1, False, 1.0, "DTC_01")
    print("API Test Result:", result)
