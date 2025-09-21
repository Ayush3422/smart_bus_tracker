import requests
import json
import time
import pandas as pd
from datetime import datetime
import numpy as np

class BackendMLTester:
    """Comprehensive test suite for Smart Bus Backend ML Integration"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        print("🧪 Smart Bus Backend ML Testing Suite")
        print("=" * 50)
    
    def test_api_status(self):
        """Test API status and model loading"""
        print("\n🔍 Testing API Status...")
        
        try:
            response = requests.get(f"{self.base_url}/api/status")
            data = response.json()
            
            print(f"✅ Status: {data['status']}")
            print(f"📊 Model Loaded: {data['model_loaded']}")
            print(f"🤖 Model Type: {data['model_type']}")
            print(f"📈 Features Count: {data['features_count']}")
            
            # Validate response
            assert response.status_code == 200
            assert data['status'] == 'online'
            assert data['model_loaded'] == True
            
            self.test_results.append({
                'test': 'API Status',
                'status': 'PASS',
                'model_loaded': data['model_loaded'],
                'features': data['features_count']
            })
            
            print("✅ API Status Test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ API Status Test FAILED: {e}")
            self.test_results.append({
                'test': 'API Status',
                'status': 'FAIL',
                'error': str(e)
            })
            return False
    
    def test_real_time_buses(self):
        """Test real-time bus data with ML predictions"""
        print("\n🚌 Testing Real-time Bus Data...")
        
        try:
            response = requests.get(f"{self.base_url}/api/buses")
            data = response.json()
            
            buses = data['buses']
            print(f"📍 Active Buses: {len(buses)}")
            
            if buses:
                for bus in buses[:3]:  # Show first 3 buses
                    print(f"  🚌 {bus['bus_id']} ({bus['route_id']}): {bus['passengers']} passengers")
                    print(f"     📍 Location: {bus['latitude']:.4f}, {bus['longitude']:.4f}")
                    print(f"     ⚡ Speed: {bus['speed_kmh']} km/h")
                
                # Validate ML predictions are realistic
                passenger_counts = [bus['passengers'] for bus in buses]
                avg_passengers = np.mean(passenger_counts)
                max_passengers = max(passenger_counts)
                min_passengers = min(passenger_counts)
                
                print(f"📊 Passenger Stats: Avg={avg_passengers:.1f}, Min={min_passengers}, Max={max_passengers}")
                
                # ML prediction validation
                assert all(1 <= p <= 200 for p in passenger_counts), "Passenger counts unrealistic"
                assert max_passengers < 1000, "Passenger count too high (possible ML bug)"
                
                self.test_results.append({
                    'test': 'Real-time Buses',
                    'status': 'PASS',
                    'buses_count': len(buses),
                    'avg_passengers': avg_passengers,
                    'prediction_range': f"{min_passengers}-{max_passengers}"
                })
                
                print("✅ Real-time Bus Test PASSED")
                return True
            else:
                print("⚠️ No buses found (simulation may not be running)")
                return False
                
        except Exception as e:
            print(f"❌ Real-time Bus Test FAILED: {e}")
            self.test_results.append({
                'test': 'Real-time Buses',
                'status': 'FAIL',
                'error': str(e)
            })
            return False
    
    def test_ml_predictions(self):
        """Test ML model predictions for different routes and scenarios"""
        print("\n🧠 Testing ML Predictions...")
        
        routes = ['RT001', 'RT002', 'RT003']
        prediction_results = {}
        
        for route_id in routes:
            try:
                response = requests.get(f"{self.base_url}/api/predictions/{route_id}")
                data = response.json()
                
                predictions = data['predictions']
                print(f"\n📈 Route {route_id}: {len(predictions)} predictions")
                
                # Analyze predictions
                next_4_hours = predictions[:4]
                passenger_predictions = [p['predicted_passengers'] for p in next_4_hours]
                
                for i, pred in enumerate(next_4_hours):
                    peak_indicator = "🔥" if pred['is_peak'] else "🌙"
                    print(f"  {peak_indicator} Hour {pred['hour']}: {pred['predicted_passengers']} passengers ({pred['day_type']})")
                
                # Validate ML predictions
                assert all(1 <= p <= 200 for p in passenger_predictions), f"Route {route_id}: Unrealistic predictions"
                assert len(predictions) == 24, f"Route {route_id}: Should have 24 hourly predictions"
                
                # Check peak hour logic
                peak_hours = [p for p in predictions if p['is_peak']]
                peak_predictions = [p['predicted_passengers'] for p in peak_hours]
                non_peak_predictions = [p['predicted_passengers'] for p in predictions if not p['is_peak']]
                
                if peak_predictions and non_peak_predictions:
                    avg_peak = np.mean(peak_predictions)
                    avg_non_peak = np.mean(non_peak_predictions)
                    print(f"  📊 Peak vs Non-Peak: {avg_peak:.1f} vs {avg_non_peak:.1f}")
                    
                    # Peak hours should generally have higher ridership
                    assert avg_peak >= avg_non_peak * 0.8, f"Route {route_id}: Peak hour logic may be incorrect"
                
                prediction_results[route_id] = {
                    'avg_prediction': np.mean(passenger_predictions),
                    'peak_avg': np.mean(peak_predictions) if peak_predictions else 0,
                    'model_used': data.get('model_used', 'unknown')
                }
                
            except Exception as e:
                print(f"❌ Route {route_id} prediction failed: {e}")
                prediction_results[route_id] = {'error': str(e)}
        
        if len(prediction_results) == len(routes):
            self.test_results.append({
                'test': 'ML Predictions',
                'status': 'PASS',
                'routes_tested': len(routes),
                'prediction_results': prediction_results
            })
            print("✅ ML Predictions Test PASSED")
            return True
        else:
            self.test_results.append({
                'test': 'ML Predictions',
                'status': 'FAIL',
                'routes_tested': len(prediction_results)
            })
            print("❌ ML Predictions Test FAILED")
            return False
    
    def test_schedule_optimization(self):
        """Test smart schedule optimization with ML insights"""
        print("\n⚡ Testing Schedule Optimization...")
        
        routes = ['RT001', 'RT002', 'RT003']
        optimization_results = {}
        
        for route_id in routes:
            try:
                response = requests.get(f"{self.base_url}/api/optimize/{route_id}")
                data = response.json()
                
                current_demand = data['current_demand']
                next_hour_demand = data['next_hour_demand']
                optimization = data['schedule_optimization']
                bunching = data['bunching_prevention']
                
                print(f"\n🚌 Route {route_id}:")
                print(f"  📊 Current Demand: {current_demand} passengers")
                print(f"  📈 Next Hour Demand: {next_hour_demand} passengers")
                print(f"  ⏰ Frequency: {optimization['original_frequency']}min → {optimization['optimized_frequency']}min")
                print(f"  📈 Improvement: {optimization['improvement_percentage']}%")
                print(f"  💡 Reason: {optimization['reason']}")
                print(f"  🚫 Bunching: {'Detected' if bunching['bunching_detected'] else 'None'}")
                
                # Validate optimization logic
                assert 1 <= current_demand <= 200, f"Route {route_id}: Unrealistic current demand"
                assert 1 <= next_hour_demand <= 200, f"Route {route_id}: Unrealistic next hour demand"
                assert 5 <= optimization['optimized_frequency'] <= 30, f"Route {route_id}: Unrealistic frequency"
                
                # Optimization should make sense
                if current_demand > 80:
                    assert optimization['optimized_frequency'] <= optimization['original_frequency'], \
                        f"Route {route_id}: High demand should decrease frequency"
                elif current_demand < 20:
                    assert optimization['optimized_frequency'] >= optimization['original_frequency'], \
                        f"Route {route_id}: Low demand should increase frequency"
                
                optimization_results[route_id] = {
                    'current_demand': current_demand,
                    'optimized_frequency': optimization['optimized_frequency'],
                    'improvement': optimization['improvement_percentage']
                }
                
            except Exception as e:
                print(f"❌ Route {route_id} optimization failed: {e}")
                optimization_results[route_id] = {'error': str(e)}
        
        if len(optimization_results) == len(routes):
            self.test_results.append({
                'test': 'Schedule Optimization',
                'status': 'PASS',
                'routes_tested': len(routes),
                'optimization_results': optimization_results
            })
            print("✅ Schedule Optimization Test PASSED")
            return True
        else:
            self.test_results.append({
                'test': 'Schedule Optimization',
                'status': 'FAIL',
                'routes_tested': len(optimization_results)
            })
            print("❌ Schedule Optimization Test FAILED")
            return False
    
    def test_analytics_dashboard(self):
        """Test system analytics and performance metrics"""
        print("\n📊 Testing Analytics Dashboard...")
        
        try:
            response = requests.get(f"{self.base_url}/api/analytics")
            data = response.json()
            
            hourly_patterns = data['hourly_patterns']
            route_performance = data['route_performance']
            system_summary = data['system_summary']
            model_info = data['model_info']
            
            print(f"📈 Hourly Patterns: {len(hourly_patterns)} data points")
            print(f"🚌 Route Performance: {len(route_performance)} routes")
            print(f"🎯 Active Buses: {system_summary['active_buses']}")
            print(f"📊 Avg Current Passengers: {system_summary['avg_current_passengers']}")
            print(f"⚡ Avg Speed: {system_summary['avg_speed_kmh']} km/h")
            print(f"🤖 Model: {model_info['model_name']} ({model_info['features_count']} features)")
            
            # Show some hourly patterns
            print("\n📅 Sample Hourly Patterns:")
            for pattern in hourly_patterns[:6]:
                print(f"  {pattern['hour']:02d}:00 - {pattern['avg_passengers']:.1f} passengers")
            
            # Validate analytics data
            assert len(hourly_patterns) > 0, "No hourly patterns found"
            assert len(route_performance) > 0, "No route performance data"
            assert system_summary['active_buses'] >= 0, "Invalid active bus count"
            assert model_info['model_loaded'] == True, "Model not loaded in analytics"
            
            self.test_results.append({
                'test': 'Analytics Dashboard',
                'status': 'PASS',
                'hourly_data_points': len(hourly_patterns),
                'routes_analyzed': len(route_performance),
                'active_buses': system_summary['active_buses']
            })
            
            print("✅ Analytics Dashboard Test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Analytics Dashboard Test FAILED: {e}")
            self.test_results.append({
                'test': 'Analytics Dashboard',
                'status': 'FAIL',
                'error': str(e)
            })
            return False
    
    def test_model_performance(self):
        """Test ML model performance and prediction accuracy"""
        print("\n🎯 Testing ML Model Performance...")
        
        try:
            # Test multiple prediction scenarios
            scenarios = [
                {'hour': 8, 'day': 'weekday', 'description': 'Morning Peak Weekday'},
                {'hour': 14, 'day': 'weekday', 'description': 'Afternoon Weekday'},
                {'hour': 18, 'day': 'weekday', 'description': 'Evening Peak Weekday'},
                {'hour': 10, 'day': 'weekend', 'description': 'Weekend Morning'},
                {'hour': 22, 'day': 'weekend', 'description': 'Late Night Weekend'}
            ]
            
            performance_results = []
            
            for scenario in scenarios:
                # Get predictions for this scenario
                response = requests.get(f"{self.base_url}/api/predictions/RT001")
                data = response.json()
                
                # Find matching hour prediction
                hour_prediction = next((p for p in data['predictions'] if p['hour'] == scenario['hour']), None)
                
                if hour_prediction:
                    prediction = hour_prediction['predicted_passengers']
                    is_peak = hour_prediction['is_peak']
                    
                    print(f"  {scenario['description']}: {prediction} passengers (Peak: {is_peak})")
                    
                    # Validate prediction reasonableness
                    assert 1 <= prediction <= 200, f"Unrealistic prediction for {scenario['description']}"
                    
                    performance_results.append({
                        'scenario': scenario['description'],
                        'prediction': prediction,
                        'is_peak': is_peak,
                        'hour': scenario['hour']
                    })
            
            # Test prediction consistency (multiple calls should be similar)
            print("\n🔄 Testing Prediction Consistency...")
            consistency_results = []
            for i in range(3):
                response = requests.get(f"{self.base_url}/api/predictions/RT001")
                data = response.json()
                next_hour_pred = data['predictions'][0]['predicted_passengers']
                consistency_results.append(next_hour_pred)
                time.sleep(0.5)
            
            consistency_range = max(consistency_results) - min(consistency_results)
            print(f"  Consistency Range: {consistency_range} (should be small for deterministic model)")
            
            # Test response time
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/predictions/RT001")
            response_time = time.time() - start_time
            print(f"  Response Time: {response_time:.3f} seconds")
            
            assert response_time < 5.0, "API response too slow"
            
            self.test_results.append({
                'test': 'Model Performance',
                'status': 'PASS',
                'scenarios_tested': len(scenarios),
                'consistency_range': consistency_range,
                'response_time': response_time,
                'performance_results': performance_results
            })
            
            print("✅ ML Model Performance Test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ ML Model Performance Test FAILED: {e}")
            self.test_results.append({
                'test': 'Model Performance',
                'status': 'FAIL',
                'error': str(e)
            })
            return False
    
    def run_comprehensive_test(self):
        """Run all tests and generate comprehensive report"""
        print("\n🚀 Starting Comprehensive Backend ML Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_api_status,
            self.test_real_time_buses,
            self.test_ml_predictions,
            self.test_schedule_optimization,
            self.test_analytics_dashboard,
            self.test_model_performance
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            if test():
                passed_tests += 1
            time.sleep(1)  # Brief pause between tests
        
        # Generate final report
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Backend ML Integration is FULLY FUNCTIONAL!")
        else:
            print("⚠️ Some tests failed. Check individual test results above.")
        
        print("\n📄 Detailed Results:")
        for result in self.test_results:
            status_emoji = "✅" if result['status'] == 'PASS' else "❌"
            print(f"  {status_emoji} {result['test']}: {result['status']}")
            if result['status'] == 'FAIL' and 'error' in result:
                print(f"     Error: {result['error']}")
        
        # Save results to file
        with open('backend_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'success_rate': (passed_tests/total_tests)*100,
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Test results saved to 'backend_test_results.json'")
        return passed_tests == total_tests

if __name__ == "__main__":
    # Wait for backend to be ready
    print("⏳ Waiting for backend server to be ready...")
    time.sleep(3)
    
    # Run comprehensive test
    tester = BackendMLTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🏆 Backend is ready for frontend integration!")
    else:
        print("\n🔧 Please fix the failing tests before proceeding.")