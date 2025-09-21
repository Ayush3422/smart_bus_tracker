#!/usr/bin/env python3
"""
Quick test of the fixed ultra-advanced model predictions
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def test_model_predictions():
    """Test if the model now gives reasonable predictions"""
    
    print("🔍 Testing Ultra-Advanced Model Predictions...")
    print("=" * 60)
    
    try:
        # Load the saved model
        model_data = joblib.load('ultra_advanced_bus_ridership_model.pkl')
        print(f"✅ Model loaded: {model_data.get('best_model_name', 'Unknown')}")
        print(f"📊 Features: {len(model_data.get('feature_columns', []))}")
        
        # Get model components
        best_model = model_data.get('best_model')
        feature_columns = model_data.get('feature_columns', [])
        best_scaler = model_data.get('best_scaler')
        best_model_name = model_data.get('best_model_name', '')
        
        if not best_model or not feature_columns:
            print("❌ Model components missing")
            return False
            
        # Test scenarios
        test_scenarios = [
            {'hour': 8, 'day_of_week': 1, 'is_weekend': False, 'description': 'Monday Morning Peak', 'expected_range': (70, 110)},
            {'hour': 18, 'day_of_week': 4, 'is_weekend': False, 'description': 'Thursday Evening Peak', 'expected_range': (70, 110)},
            {'hour': 14, 'day_of_week': 2, 'is_weekend': False, 'description': 'Tuesday Afternoon', 'expected_range': (45, 75)},
            {'hour': 10, 'day_of_week': 6, 'is_weekend': True, 'description': 'Saturday Morning', 'expected_range': (30, 60)},
            {'hour': 23, 'day_of_week': 1, 'is_weekend': False, 'description': 'Monday Late Night', 'expected_range': (15, 35)},
        ]
        
        accurate_predictions = 0
        
        print("\n🎯 Testing Scenarios:")
        print("-" * 60)
        
        for scenario in test_scenarios:
            # Create test features
            test_data = {}
            
            for feature in feature_columns:
                if feature == 'hour':
                    test_data[feature] = scenario['hour']
                elif feature == 'day_of_week':
                    test_data[feature] = scenario['day_of_week']
                elif feature == 'is_weekend':
                    test_data[feature] = scenario['is_weekend']
                elif feature == 'hour_sin':
                    test_data[feature] = np.sin(2 * np.pi * scenario['hour'] / 24)
                elif feature == 'hour_cos':
                    test_data[feature] = np.cos(2 * np.pi * scenario['hour'] / 24)
                elif feature == 'dow_sin':
                    test_data[feature] = np.sin(2 * np.pi * scenario['day_of_week'] / 7)
                elif feature == 'dow_cos':
                    test_data[feature] = np.cos(2 * np.pi * scenario['day_of_week'] / 7)
                elif feature == 'is_morning_peak':
                    test_data[feature] = scenario['hour'] in [7, 8, 9]
                elif feature == 'is_evening_peak':
                    test_data[feature] = scenario['hour'] in [17, 18, 19]
                elif feature == 'is_late_night':
                    test_data[feature] = scenario['hour'] in [22, 23, 0, 1, 2]
                elif feature == 'weather_factor':
                    test_data[feature] = 1.0
                elif 'route' in feature:
                    test_data[feature] = 50.0
                elif 'lag' in feature:
                    test_data[feature] = 45.0
                elif 'rolling' in feature:
                    test_data[feature] = 50.0 if 'mean' in feature else 15.0
                elif 'peak_weekend_interaction' in feature:
                    test_data[feature] = (scenario['hour'] in [7, 8, 9, 17, 18, 19]) * scenario['is_weekend']
                else:
                    test_data[feature] = 0.0
            
            # Create DataFrame
            test_df = pd.DataFrame([test_data])
            
            # Make prediction
            requires_scaling = any(model_type in best_model_name.lower() 
                                 for model_type in ['neural_network', 'svr', 'ridge', 'lasso', 
                                                  'elastic_net', 'bayesian', 'huber', 'theil_sen', 
                                                  'ransac', 'gaussian_process'])
            
            if requires_scaling and best_scaler is not None:
                test_scaled = best_scaler.transform(test_df)
                prediction = best_model.predict(test_scaled)[0]
            else:
                prediction = best_model.predict(test_df)[0]
            
            # Check accuracy
            expected_min, expected_max = scenario['expected_range']
            is_accurate = expected_min <= prediction <= expected_max
            
            if is_accurate:
                accurate_predictions += 1
                status = "✅ ACCURATE"
            else:
                status = "⚠️ OUTSIDE RANGE"
            
            print(f"  📊 {scenario['description']:20} {prediction:6.1f} passengers (Expected: {expected_min:3d}-{expected_max:3d}) {status}")
        
        # Summary
        accuracy_percentage = (accurate_predictions / len(test_scenarios)) * 100
        print(f"\n🎯 Model Accuracy: {accuracy_percentage:.1f}% ({accurate_predictions}/{len(test_scenarios)} scenarios)")
        
        if accuracy_percentage >= 80:
            print("🎉 EXCELLENT! Model predictions are now reasonable!")
            return True
        elif accuracy_percentage >= 60:
            print("✅ GOOD! Model predictions are much improved!")
            return True
        else:
            print("⚠️ Still needs work, but much better than millions of passengers!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model_predictions()
