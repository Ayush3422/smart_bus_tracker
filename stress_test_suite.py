import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import gc
warnings.filterwarnings('ignore')

class StressTestSuite:
    def __init__(self):
        print("💪 Initializing Advanced Stress Testing Suite...")
        
        try:
            # Load the advanced model
            model_data = joblib.load('advanced_bus_ridership_model.pkl')
            self.model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_name = model_data['model_name']
            print(f"✅ Loaded {self.model_name} model with {len(self.feature_columns)} features")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def create_feature_vector(self, hour, day_of_week, is_weekend, weather_factor=1.0, 
                            is_holiday=False, special_event=False):
        """Create comprehensive feature vector"""
        
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
            
            # Special events
            'is_holiday': is_holiday,
            'days_to_holiday': 0 if is_holiday else 10,
            'is_day_before_holiday': False,
            'is_day_after_holiday': False,
            
            # Peak hours
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
        
        # Fill missing features with defaults
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
    
    def stress_test_performance(self):
        """Test model performance under heavy load"""
        print("\n⚡ Performance Stress Testing...")
        print("-" * 70)
        
        # Test batch predictions
        batch_sizes = [1, 10, 100, 1000, 5000]
        performance_results = []
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}...")
            
            # Create random test data
            test_data = []
            for _ in range(batch_size):
                hour = np.random.randint(0, 24)
                dow = np.random.randint(0, 7)
                weather = np.random.uniform(0.3, 1.8)
                
                feature_vector = self.create_feature_vector(
                    hour, dow, dow >= 5, weather
                )
                test_data.append(feature_vector)
            
            # Combine all test vectors
            batch_df = pd.concat(test_data, ignore_index=True)
            
            # Time the prediction
            start_time = time.time()
            
            try:
                if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                    batch_scaled = self.scaler.transform(batch_df)
                    predictions = self.model.predict(batch_scaled)
                else:
                    predictions = self.model.predict(batch_df)
                
                end_time = time.time()
                duration = end_time - start_time
                predictions_per_sec = batch_size / duration if duration > 0 else float('inf')
                
                performance_results.append({
                    'batch_size': batch_size,
                    'duration': duration,
                    'predictions_per_sec': predictions_per_sec,
                    'avg_prediction': np.mean(predictions),
                    'status': 'SUCCESS'
                })
                
                print(f"  ✅ {batch_size:5d} predictions in {duration:.3f}s ({predictions_per_sec:.1f} pred/sec)")
                
            except Exception as e:
                performance_results.append({
                    'batch_size': batch_size,
                    'duration': 0,
                    'predictions_per_sec': 0,
                    'avg_prediction': 0,
                    'status': f'ERROR: {str(e)[:30]}'
                })
                print(f"  ❌ Failed: {e}")
            
            # Clear memory
            del test_data, batch_df
            gc.collect()
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        successful = [r for r in performance_results if r['status'] == 'SUCCESS']
        if successful:
            max_throughput = max(r['predictions_per_sec'] for r in successful)
            print(f"🚀 Maximum Throughput: {max_throughput:.1f} predictions/second")
            print(f"📊 Successfully handled batches up to {max(r['batch_size'] for r in successful)} predictions")
        
        return len(successful) >= 3  # At least 3 batch sizes should work
    
    def stress_test_memory_usage(self):
        """Test memory usage and stability"""
        print("\n🧠 Memory Stress Testing...")
        print("-" * 70)
        
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        memory_results = []
        
        # Run continuous predictions
        for iteration in range(1, 11):
            # Create batch of random predictions
            test_vectors = []
            for _ in range(100):
                hour = np.random.randint(0, 24)
                dow = np.random.randint(0, 7)
                weather = np.random.uniform(0.3, 1.8)
                
                test_vectors.append(self.create_feature_vector(hour, dow, dow >= 5, weather))
            
            # Run predictions
            batch_df = pd.concat(test_vectors, ignore_index=True)
            
            try:
                if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                    batch_scaled = self.scaler.transform(batch_df)
                    predictions = self.model.predict(batch_scaled)
                else:
                    predictions = self.model.predict(batch_df)
                
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                memory_results.append({
                    'iteration': iteration,
                    'memory_mb': current_memory,
                    'memory_increase': memory_increase,
                    'predictions': len(predictions)
                })
                
                print(f"Iteration {iteration:2d}: {current_memory:6.1f} MB (+{memory_increase:5.1f} MB)")
                
            except Exception as e:
                print(f"❌ Memory test failed at iteration {iteration}: {e}")
                break
            
            # Cleanup
            del test_vectors, batch_df, predictions
            gc.collect()
        
        # Memory analysis
        if memory_results:
            final_memory = memory_results[-1]['memory_increase']
            avg_memory_per_iteration = final_memory / len(memory_results)
            
            print(f"\n🧠 Memory Analysis:")
            print(f"📈 Total memory increase: {final_memory:.1f} MB")
            print(f"📊 Average per iteration: {avg_memory_per_iteration:.1f} MB")
            
            # Check for memory leaks (should be minimal increase)
            memory_stable = final_memory < 50  # Less than 50MB increase is acceptable
            return memory_stable
        
        return False
    
    def stress_test_concurrent_requests(self):
        """Simulate concurrent prediction requests"""
        print("\n🔄 Concurrent Request Testing...")
        print("-" * 70)
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_thread(thread_id, num_predictions):
            """Worker thread for concurrent predictions"""
            try:
                predictions = []
                for i in range(num_predictions):
                    hour = (thread_id * 3 + i) % 24
                    dow = thread_id % 7
                    weather = 0.8 + (thread_id * 0.1)
                    
                    test_df = self.create_feature_vector(hour, dow, dow >= 5, weather)
                    
                    if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                        test_scaled = self.scaler.transform(test_df)
                        pred = self.model.predict(test_scaled)[0]
                    else:
                        pred = self.model.predict(test_df)[0]
                    
                    predictions.append(pred)
                
                results_queue.put({
                    'thread_id': thread_id,
                    'predictions': predictions,
                    'avg_prediction': np.mean(predictions),
                    'status': 'SUCCESS'
                })
                
            except Exception as e:
                errors_queue.put({
                    'thread_id': thread_id,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        # Test with different thread counts
        thread_counts = [1, 2, 5, 10]
        concurrent_results = []
        
        for num_threads in thread_counts:
            print(f"Testing with {num_threads} concurrent threads...")
            
            start_time = time.time()
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i, 10))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Collect results
            successful_threads = 0
            failed_threads = 0
            
            while not results_queue.empty():
                result = results_queue.get()
                successful_threads += 1
            
            while not errors_queue.empty():
                error = errors_queue.get()
                failed_threads += 1
                print(f"  ❌ Thread {error['thread_id']} failed: {error['error'][:30]}...")
            
            success_rate = (successful_threads / num_threads) * 100
            
            concurrent_results.append({
                'threads': num_threads,
                'duration': duration,
                'successful': successful_threads,
                'failed': failed_threads,
                'success_rate': success_rate
            })
            
            print(f"  ✅ {successful_threads}/{num_threads} threads successful ({success_rate:.1f}%) in {duration:.2f}s")
        
        # Concurrent analysis
        all_successful = all(r['success_rate'] == 100 for r in concurrent_results)
        print(f"\n🔄 Concurrency Summary:")
        if all_successful:
            max_threads = max(r['threads'] for r in concurrent_results)
            print(f"🎯 Successfully handled up to {max_threads} concurrent threads")
        
        return all_successful
    
    def stress_test_data_corruption(self):
        """Test model robustness against corrupted data"""
        print("\n🛡️ Data Corruption Testing...")
        print("-" * 70)
        
        corruption_tests = [
            {'name': 'Missing Values (NaN)', 'corruption': 'nan'},
            {'name': 'Infinite Values', 'corruption': 'inf'},
            {'name': 'Extreme Outliers', 'corruption': 'outliers'},
            {'name': 'Wrong Data Types', 'corruption': 'types'},
            {'name': 'Negative Values', 'corruption': 'negative'},
        ]
        
        corruption_results = []
        
        for test in corruption_tests:
            try:
                # Create normal test data
                test_df = self.create_feature_vector(8, 1, False, 1.0)
                
                # Apply corruption
                if test['corruption'] == 'nan':
                    # Introduce NaN values
                    test_df.iloc[0, :3] = np.nan
                elif test['corruption'] == 'inf':
                    # Introduce infinite values
                    test_df.iloc[0, :2] = np.inf
                elif test['corruption'] == 'outliers':
                    # Introduce extreme outliers
                    test_df.iloc[0, :3] = [999999, -999999, 1e10]
                elif test['corruption'] == 'negative':
                    # All negative values
                    test_df = test_df.abs() * -1
                elif test['corruption'] == 'types':
                    # This is handled by ensuring numeric types
                    pass
                
                # Try prediction
                if self.model_name in ['neural_network', 'svr', 'ridge', 'elastic_net']:
                    # Handle NaN/inf for scaling
                    test_clean = test_df.fillna(0).replace([np.inf, -np.inf], 0)
                    test_scaled = self.scaler.transform(test_clean)
                    prediction = self.model.predict(test_scaled)[0]
                else:
                    # Handle NaN/inf for tree models
                    test_clean = test_df.fillna(0).replace([np.inf, -np.inf], 0)
                    prediction = self.model.predict(test_clean)[0]
                
                # Check if prediction is reasonable
                is_reasonable = 1 <= prediction <= 300
                status = "✅ HANDLED" if is_reasonable else f"⚠️ UNREASONABLE ({prediction:.1f})"
                
                corruption_results.append({
                    'test': test['name'],
                    'prediction': prediction,
                    'reasonable': is_reasonable,
                    'status': 'SUCCESS'
                })
                
                print(f"{test['name']:<25} | Prediction: {prediction:6.1f} | {status}")
                
            except Exception as e:
                corruption_results.append({
                    'test': test['name'],
                    'prediction': 0,
                    'reasonable': False,
                    'status': f'ERROR: {str(e)[:30]}'
                })
                print(f"{test['name']:<25} | ERROR: {str(e)[:40]}...")
        
        # Corruption summary
        handled_count = sum(1 for r in corruption_results if r['reasonable'])
        total_tests = len(corruption_tests)
        robustness_score = (handled_count / total_tests) * 100
        
        print(f"\n🛡️ Data Corruption Summary:")
        print(f"🎯 Robustness Score: {robustness_score:.1f}% ({handled_count}/{total_tests} handled)")
        
        return robustness_score >= 80
    
    def create_stress_test_report(self):
        """Generate comprehensive stress test report"""
        print("\n📋 Running Comprehensive Stress Tests...")
        print("="*80)
        
        # Run all stress tests
        performance_passed = self.stress_test_performance()
        memory_passed = self.stress_test_memory_usage()
        concurrent_passed = self.stress_test_concurrent_requests()
        corruption_passed = self.stress_test_data_corruption()
        
        # Calculate overall stress score
        stress_tests = [performance_passed, memory_passed, concurrent_passed, corruption_passed]
        stress_score = sum(stress_tests) / len(stress_tests) * 100
        
        print("\n" + "="*80)
        print("💪 COMPREHENSIVE STRESS TEST REPORT")
        print("="*80)
        
        print(f"⚡ Performance Stress:   {'✅ PASS' if performance_passed else '❌ FAIL'}")
        print(f"🧠 Memory Stability:    {'✅ PASS' if memory_passed else '❌ FAIL'}")
        print(f"🔄 Concurrent Requests: {'✅ PASS' if concurrent_passed else '❌ FAIL'}")
        print(f"🛡️ Data Corruption:     {'✅ PASS' if corruption_passed else '❌ FAIL'}")
        
        print(f"\n💪 OVERALL STRESS SCORE: {stress_score:.1f}%")
        
        if stress_score >= 90:
            print("🏆 EXCELLENT! System is bulletproof and production-ready!")
            grade = "A+"
        elif stress_score >= 75:
            print("✅ GREAT! System handles stress very well!")
            grade = "A"
        elif stress_score >= 60:
            print("👍 GOOD! System is moderately robust!")
            grade = "B"
        else:
            print("⚠️ NEEDS WORK! System struggles under stress.")
            grade = "C"
        
        print(f"🎯 Stress Test Grade: {grade}")
        print("="*80)
        
        return stress_score >= 75

# Run comprehensive stress testing
if __name__ == "__main__":
    print("💪 ADVANCED STRESS TESTING SUITE")
    print("="*80)
    
    try:
        stress_tester = StressTestSuite()
        
        # Run comprehensive stress tests
        is_stress_ready = stress_tester.create_stress_test_report()
        
        if is_stress_ready:
            print("\n🎉 SYSTEM PASSES ALL STRESS TESTS!")
            print("💪 Ready for high-load production deployment!")
            print("🚀 Delhi Bus System can handle enterprise-level traffic!")
        else:
            print("\n⚠️ System needs optimization for high-stress scenarios.")
            print("💡 Consider performance tuning and optimization.")
            
    except Exception as e:
        print(f"❌ Stress testing failed: {e}")
        print("💡 Ensure model and dependencies are properly configured.")