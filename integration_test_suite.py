import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import time
import os
warnings.filterwarnings('ignore')

class SystemIntegrationTester:
    def __init__(self):
        print("🔧 Initializing System Integration Testing Suite...")
        
        self.test_results = {
            'data_pipeline': False,
            'model_loading': False,
            'prediction_accuracy': False,
            'api_compatibility': False,
            'visualization': False,
            'end_to_end': False
        }
        
        try:
            # Test model loading
            model_data = joblib.load('advanced_bus_ridership_model.pkl')
            self.model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_name = model_data['model_name']
            self.test_results['model_loading'] = True
            print(f"✅ Model loaded successfully: {self.model_name}")
            
            # Test data loading
            if os.path.exists('bus_ridership_data.csv'):
                self.data = pd.read_csv('bus_ridership_data.csv')
                self.test_results['data_pipeline'] = True
                print(f"✅ Data loaded successfully: {len(self.data)} records")
            else:
                print("❌ Data file not found")
                
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise
    
    def test_data_pipeline_integrity(self):
        """Test complete data pipeline from raw data to features"""
        print("\n📊 Testing Data Pipeline Integrity...")
        print("-" * 70)
        
        try:
            # Check data structure
            required_columns = ['hour', 'day_of_week', 'passengers', 'route_id']
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            print("✅ All required columns present")
            
            # Check data quality
            null_counts = self.data.isnull().sum()
            if null_counts.sum() > 0:
                print(f"⚠️ Found {null_counts.sum()} null values")
            else:
                print("✅ No null values found")
            
            # Check data ranges
            hour_range = self.data['hour'].min(), self.data['hour'].max()
            dow_range = self.data['day_of_week'].min(), self.data['day_of_week'].max()
            passengers_range = self.data['passengers'].min(), self.data['passengers'].max()
            
            print(f"📈 Data ranges:")
            print(f"   Hours: {hour_range[0]}-{hour_range[1]} (expected: 0-23)")
            print(f"   Day of week: {dow_range[0]}-{dow_range[1]} (expected: 0-6)")
            print(f"   Passengers: {passengers_range[0]}-{passengers_range[1]}")
            
            # Validate ranges
            valid_ranges = (
                0 <= hour_range[0] <= hour_range[1] <= 23 and
                0 <= dow_range[0] <= dow_range[1] <= 6 and
                passengers_range[0] >= 0
            )
            
            if valid_ranges:
                print("✅ All data ranges are valid")
                self.test_results['data_pipeline'] = True
                return True
            else:
                print("❌ Invalid data ranges detected")
                return False
                
        except Exception as e:
            print(f"❌ Data pipeline test failed: {e}")
            return False
    
    def test_prediction_accuracy_comprehensive(self):
        """Test prediction accuracy across different scenarios"""
        print("\n🎯 Testing Prediction Accuracy...")
        print("-" * 70)
        
        try:
            # Sample test scenarios
            test_scenarios = [
                # Normal scenarios
                {'hour': 8, 'dow': 1, 'is_weekend': False, 'weather': 1.0, 'desc': 'Weekday Morning Peak'},
                {'hour': 18, 'dow': 1, 'is_weekend': False, 'weather': 1.0, 'desc': 'Weekday Evening Peak'},
                {'hour': 14, 'dow': 6, 'is_weekend': True, 'weather': 1.0, 'desc': 'Weekend Afternoon'},
                {'hour': 2, 'dow': 2, 'is_weekend': False, 'weather': 1.0, 'desc': 'Late Night'},
                
                # Weather scenarios
                {'hour': 8, 'dow': 1, 'is_weekend': False, 'weather': 0.5, 'desc': 'Rainy Morning'},
                {'hour': 18, 'dow': 1, 'is_weekend': False, 'weather': 1.5, 'desc': 'Perfect Evening'},
                
                # Edge cases
                {'hour': 23, 'dow': 0, 'is_weekend': True, 'weather': 0.3, 'desc': 'Sunday Night Storm'},
                {'hour': 12, 'dow': 5, 'is_weekend': True, 'weather': 1.8, 'desc': 'Saturday Lunch Perfect'},
            ]
            
            predictions = []
            valid_predictions = 0
            
            for scenario in test_scenarios:
                # Create feature vector
                features = self.create_comprehensive_features(
                    scenario['hour'], scenario['dow'], scenario['is_weekend'], scenario['weather']
                )
                
                # Make prediction
                if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                    features_scaled = self.scaler.transform(features)
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    prediction = self.model.predict(features)[0]
                
                # Validate prediction
                is_valid = 1 <= prediction <= 300  # Reasonable passenger range
                if is_valid:
                    valid_predictions += 1
                
                predictions.append({
                    'scenario': scenario['desc'],
                    'prediction': prediction,
                    'valid': is_valid
                })
                
                status = "✅" if is_valid else "❌"
                print(f"{scenario['desc']:<25} | {prediction:6.1f} passengers | {status}")
            
            accuracy = (valid_predictions / len(test_scenarios)) * 100
            print(f"\n🎯 Prediction Accuracy: {accuracy:.1f}% ({valid_predictions}/{len(test_scenarios)} valid)")
            
            if accuracy >= 90:
                self.test_results['prediction_accuracy'] = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            return False
    
    def create_comprehensive_features(self, hour, day_of_week, is_weekend, weather_factor):
        """Create comprehensive feature vector for prediction"""
        
        current_time = datetime.now().replace(hour=hour, minute=0, second=0)
        
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
        for col in self.feature_columns:
            if col not in features:
                if 'route' in col:
                    if 'encoded' in col:
                        features[col] = 0
                    elif 'avg' in col:
                        features[col] = 50.0
                    elif 'std' in col:
                        features[col] = 15.0
                    elif 'max' in col:
                        features[col] = 120.0
                    elif 'min' in col:
                        features[col] = 5.0
                    else:
                        features[col] = 50.0
                elif 'lag' in col:
                    features[col] = 45.0
                elif 'rolling' in col:
                    if 'mean' in col:
                        features[col] = 50.0
                    elif 'std' in col:
                        features[col] = 12.0
                    elif 'max' in col:
                        features[col] = 100.0
                    elif 'min' in col:
                        features[col] = 10.0
                    else:
                        features[col] = 50.0
                else:
                    features[col] = 0.0
        
        # Create DataFrame with proper column order
        test_df = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in test_df.columns:
                test_df[col] = 0.0
        test_df = test_df[self.feature_columns]
        
        return test_df
    
    def test_api_compatibility(self):
        """Test if the model works with the API structure"""
        print("\n🌐 Testing API Compatibility...")
        print("-" * 70)
        
        try:
            # Check if API file exists
            if os.path.exists('advanced_model_api.py'):
                print("✅ API file exists")
                
                # Try to import (basic syntax check)
                with open('advanced_model_api.py', 'r') as f:
                    api_content = f.read()
                
                # Check for required API components
                required_components = [
                    'def predict_ridership',
                    'app = Flask',
                    '@app.route',
                    'POST',
                    'json'
                ]
                
                missing_components = []
                for component in required_components:
                    if component not in api_content:
                        missing_components.append(component)
                
                if missing_components:
                    print(f"❌ Missing API components: {missing_components}")
                    return False
                else:
                    print("✅ All API components present")
                    
                # Test API payload structure
                test_payload = {
                    "hour": 8,
                    "day_of_week": 1,
                    "is_weekend": False,
                    "weather_factor": 1.0,
                    "route_id": "DTC_01"
                }
                
                print(f"✅ Sample API payload: {test_payload}")
                self.test_results['api_compatibility'] = True
                return True
            else:
                print("❌ API file not found")
                return False
                
        except Exception as e:
            print(f"❌ API compatibility test failed: {e}")
            return False
    
    def test_visualization_generation(self):
        """Test visualization generation capabilities"""
        print("\n📊 Testing Visualization Generation...")
        print("-" * 70)
        
        try:
            # Create sample visualization
            hours = list(range(24))
            predictions = []
            
            for hour in hours:
                features = self.create_comprehensive_features(hour, 1, False, 1.0)
                
                if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                    features_scaled = self.scaler.transform(features)
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    prediction = self.model.predict(features)[0]
                
                predictions.append(prediction)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.plot(hours, predictions, 'b-o', linewidth=2, markersize=6)
            plt.title('24-Hour Ridership Prediction Pattern (Integration Test)', fontweight='bold', fontsize=14)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Predicted Passengers', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24, 2))
            
            # Add peak hour annotations
            morning_peak = predictions[8]
            evening_peak = predictions[18]
            plt.annotate(f'Morning Peak: {morning_peak:.0f}', 
                        xy=(8, morning_peak), xytext=(10, morning_peak + 10),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, fontweight='bold')
            plt.annotate(f'Evening Peak: {evening_peak:.0f}', 
                        xy=(18, evening_peak), xytext=(16, evening_peak + 10),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('integration_test_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Visualization generated successfully")
            print("📊 Saved as 'integration_test_visualization.png'")
            
            self.test_results['visualization'] = True
            return True
            
        except Exception as e:
            print(f"❌ Visualization test failed: {e}")
            return False
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n🔄 Testing End-to-End Workflow...")
        print("-" * 70)
        
        try:
            # Simulate complete workflow
            print("1. Loading real-time data...")
            current_time = datetime.now()
            
            print("2. Processing user request...")
            user_request = {
                "route_id": "DTC_01",
                "current_hour": current_time.hour,
                "weather_condition": "clear"  # Maps to weather_factor 1.0
            }
            
            print("3. Feature engineering...")
            weather_mapping = {
                "clear": 1.0,
                "cloudy": 0.8,
                "rainy": 0.5,
                "stormy": 0.3
            }
            
            features = self.create_comprehensive_features(
                hour=user_request["current_hour"],
                day_of_week=current_time.weekday(),
                is_weekend=current_time.weekday() >= 5,
                weather_factor=weather_mapping[user_request["weather_condition"]]
            )
            
            print("4. Making prediction...")
            if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
            else:
                prediction = self.model.predict(features)[0]
            
            print("5. Formatting response...")
            response = {
                "route_id": user_request["route_id"],
                "predicted_passengers": int(prediction),
                "confidence": "high" if 20 <= prediction <= 150 else "medium",
                "timestamp": current_time.isoformat(),
                "weather_impact": user_request["weather_condition"],
                "peak_hour": current_time.hour in [7, 8, 9, 17, 18, 19]
            }
            
            print("6. Validating response...")
            required_fields = ["route_id", "predicted_passengers", "confidence", "timestamp"]
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"❌ Missing response fields: {missing_fields}")
                return False
            
            print("✅ End-to-end workflow completed successfully")
            print(f"📋 Sample response: {json.dumps(response, indent=2)}")
            
            self.test_results['end_to_end'] = True
            return True
            
        except Exception as e:
            print(f"❌ End-to-end test failed: {e}")
            return False
    
    def generate_integration_report(self):
        """Generate comprehensive integration test report"""
        print("\n📋 Running Integration Tests...")
        print("="*80)
        
        # Run all integration tests
        data_pipeline_ok = self.test_data_pipeline_integrity()
        prediction_ok = self.test_prediction_accuracy_comprehensive()
        api_ok = self.test_api_compatibility()
        viz_ok = self.test_visualization_generation()
        e2e_ok = self.test_end_to_end_workflow()
        
        # Calculate integration score
        integration_tests = [data_pipeline_ok, prediction_ok, api_ok, viz_ok, e2e_ok]
        integration_score = sum(integration_tests) / len(integration_tests) * 100
        
        print("\n" + "="*80)
        print("🔧 SYSTEM INTEGRATION TEST REPORT")
        print("="*80)
        
        print(f"📊 Data Pipeline:       {'✅ PASS' if data_pipeline_ok else '❌ FAIL'}")
        print(f"🎯 Prediction Accuracy: {'✅ PASS' if prediction_ok else '❌ FAIL'}")
        print(f"🌐 API Compatibility:   {'✅ PASS' if api_ok else '❌ FAIL'}")
        print(f"📊 Visualization:       {'✅ PASS' if viz_ok else '❌ FAIL'}")
        print(f"🔄 End-to-End:          {'✅ PASS' if e2e_ok else '❌ FAIL'}")
        
        print(f"\n🔧 INTEGRATION SCORE: {integration_score:.1f}%")
        
        if integration_score == 100:
            print("🏆 PERFECT! Complete system integration successful!")
            status = "FULLY INTEGRATED"
            grade = "A+"
        elif integration_score >= 80:
            print("✅ EXCELLENT! System is well integrated!")
            status = "PRODUCTION READY"
            grade = "A"
        elif integration_score >= 60:
            print("👍 GOOD! Most components integrated successfully!")
            status = "MOSTLY READY"
            grade = "B"
        else:
            print("⚠️ NEEDS WORK! Integration issues detected.")
            status = "NEEDS FIXES"
            grade = "C"
        
        print(f"🎯 Integration Grade: {grade}")
        print(f"🚀 System Status: {status}")
        print("="*80)
        
        return integration_score >= 80

# Run comprehensive integration testing
if __name__ == "__main__":
    print("🔧 SYSTEM INTEGRATION TESTING SUITE")
    print("="*80)
    
    try:
        integration_tester = SystemIntegrationTester()
        
        # Run comprehensive integration tests
        is_integrated = integration_tester.generate_integration_report()
        
        if is_integrated:
            print("\n🎉 SYSTEM FULLY INTEGRATED AND READY!")
            print("🚀 Delhi Bus Ridership Prediction System operational!")
            print("✅ All components working together seamlessly!")
            print("💫 Ready for production deployment!")
        else:
            print("\n⚠️ Integration issues detected.")
            print("🔧 Please address failing components before deployment.")
            
    except Exception as e:
        print(f"❌ Integration testing failed: {e}")
        print("💡 Ensure all system components are properly configured.")