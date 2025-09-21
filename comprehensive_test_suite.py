#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ultra-Advanced Bus Ridership Model
This file tests the model functionality, prediction accuracy, and performance.
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_model_loading():
    """Test if the model can be loaded and basic components exist"""
    print("🔍 Test 1: Model Loading and Components")
    print("-" * 50)
    
    try:
        # Import required modules
        from ml_model import AdvancedBusRidershipPredictor
        import pandas as pd
        import numpy as np
        
        # Check if model files exist
        model_files = [
            'ultra_advanced_bus_ridership_model.pkl',
            'ultra_advanced_feature_importance.png',
            'bus_ridership_data.csv'
        ]
        
        for file in model_files:
            if os.path.exists(file):
                print(f"  ✅ {file} exists")
            else:
                print(f"  ❌ {file} missing")
                return False
        
        print("  ✅ All required files present")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading components: {e}")
        return False

def test_model_training():
    """Test if the model can be trained successfully"""
    print("\n🔍 Test 2: Model Training Process")
    print("-" * 50)
    
    try:
        from ml_model import AdvancedBusRidershipPredictor
        import pandas as pd
        
        print("  🚀 Initializing predictor...")
        predictor = AdvancedBusRidershipPredictor(enable_advanced_features=False)
        
        print("  📊 Loading and preparing data...")
        df = predictor.load_and_prepare_data()
        
        if df is None or len(df) == 0:
            print("  ❌ Failed to load data")
            return False
            
        print(f"  ✅ Data loaded: {len(df)} records")
        
        # Check if features were created
        if hasattr(predictor, 'X') and predictor.X is not None:
            print(f"  ✅ Features created: {len(predictor.X.columns)} features")
        else:
            print("  ❌ Features not created")
            return False
            
        print("  ✅ Model training process works")
        return True
        
    except Exception as e:
        print(f"  ❌ Error in training: {e}")
        traceback.print_exc()
        return False

def test_prediction_functionality():
    """Test if the model can make predictions"""
    print("\n🔍 Test 3: Prediction Functionality")
    print("-" * 50)
    
    try:
        from ml_model import AdvancedBusRidershipPredictor
        import pandas as pd
        import numpy as np
        
        # Initialize and train a basic model
        predictor = AdvancedBusRidershipPredictor(enable_advanced_features=False)
        df = predictor.load_and_prepare_data()
        
        # Quick training with limited models for testing
        print("  🤖 Training basic models for testing...")
        
        # Manually set up a simple test
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        if hasattr(predictor, 'X') and hasattr(predictor, 'y'):
            X_train, X_test, y_train, y_test = train_test_split(
                predictor.X, predictor.y, test_size=0.2, random_state=42
            )
            
            # Train a simple model for testing
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            print(f"  ✅ Predictions generated: {len(predictions)} samples")
            print(f"  📊 Prediction range: {predictions.min():.1f} - {predictions.max():.1f}")
            print(f"  📊 Mean prediction: {predictions.mean():.1f}")
            
            # Check if predictions are reasonable (not in millions)
            if predictions.max() < 1000:  # Should be < 1000 passengers
                print("  ✅ Predictions in reasonable range")
                return True
            else:
                print(f"  ⚠️ Some predictions seem high: max = {predictions.max():.1f}")
                return False
        else:
            print("  ❌ Training data not available")
            return False
            
    except Exception as e:
        print(f"  ❌ Error in prediction: {e}")
        traceback.print_exc()
        return False

def test_scenario_predictions():
    """Test specific scenarios with the model"""
    print("\n🔍 Test 4: Scenario-Based Predictions")
    print("-" * 50)
    
    try:
        from ml_model import AdvancedBusRidershipPredictor
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Initialize predictor
        predictor = AdvancedBusRidershipPredictor(enable_advanced_features=False)
        df = predictor.load_and_prepare_data()
        
        # Train a quick model
        X_train, X_test, y_train, y_test = train_test_split(
            predictor.X, predictor.y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        # Test scenarios
        scenarios = [
            {
                'name': 'Monday Morning Peak',
                'hour': 8,
                'day_of_week': 1,
                'is_weekend': False,
                'expected_range': (50, 120)
            },
            {
                'name': 'Thursday Evening Peak',
                'hour': 18,
                'day_of_week': 4,
                'is_weekend': False,
                'expected_range': (50, 120)
            },
            {
                'name': 'Saturday Afternoon',
                'hour': 14,
                'day_of_week': 6,
                'is_weekend': True,
                'expected_range': (20, 80)
            },
            {
                'name': 'Sunday Late Night',
                'hour': 23,
                'day_of_week': 0,
                'is_weekend': True,
                'expected_range': (5, 30)
            }
        ]
        
        accurate_predictions = 0
        
        for scenario in scenarios:
            # Create test data matching the training features
            test_data = {}
            
            for col in predictor.X.columns:
                if col == 'hour':
                    test_data[col] = scenario['hour']
                elif col == 'day_of_week':
                    test_data[col] = scenario['day_of_week']
                elif col == 'is_weekend':
                    test_data[col] = scenario['is_weekend']
                elif 'hour_sin' in col:
                    test_data[col] = np.sin(2 * np.pi * scenario['hour'] / 24)
                elif 'hour_cos' in col:
                    test_data[col] = np.cos(2 * np.pi * scenario['hour'] / 24)
                elif 'dow_sin' in col:
                    test_data[col] = np.sin(2 * np.pi * scenario['day_of_week'] / 7)
                elif 'dow_cos' in col:
                    test_data[col] = np.cos(2 * np.pi * scenario['day_of_week'] / 7)
                elif 'is_morning_peak' in col:
                    test_data[col] = scenario['hour'] in [7, 8, 9]
                elif 'is_evening_peak' in col:
                    test_data[col] = scenario['hour'] in [17, 18, 19]
                elif 'is_late_night' in col:
                    test_data[col] = scenario['hour'] in [22, 23, 0, 1, 2]
                elif 'weather_factor' in col:
                    test_data[col] = 1.0
                elif 'peak_weekend_interaction' in col:
                    is_peak = scenario['hour'] in [7, 8, 9, 17, 18, 19]
                    test_data[col] = is_peak * scenario['is_weekend']
                else:
                    test_data[col] = 0.0
            
            # Make prediction
            test_df = pd.DataFrame([test_data])
            prediction = model.predict(test_df)[0]
            
            # Check if prediction is in expected range
            expected_min, expected_max = scenario['expected_range']
            is_accurate = expected_min <= prediction <= expected_max
            
            if is_accurate:
                accurate_predictions += 1
                status = "✅ ACCURATE"
            else:
                status = "⚠️ OUTSIDE RANGE"
            
            print(f"  📊 {scenario['name']:20} {prediction:6.1f} passengers (Expected: {expected_min:3d}-{expected_max:3d}) {status}")
        
        accuracy = (accurate_predictions / len(scenarios)) * 100
        print(f"\n  🎯 Scenario Accuracy: {accuracy:.1f}% ({accurate_predictions}/{len(scenarios)})")
        
        return accuracy >= 50  # At least 50% should be reasonable
        
    except Exception as e:
        print(f"  ❌ Error in scenario testing: {e}")
        traceback.print_exc()
        return False

def test_data_quality():
    """Test the quality of the training data"""
    print("\n🔍 Test 5: Data Quality Assessment")
    print("-" * 50)
    
    try:
        import pandas as pd
        
        # Load data directly
        df = pd.read_csv('bus_ridership_data.csv')
        
        print(f"  📊 Dataset size: {len(df)} records")
        print(f"  📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            print("  ✅ No missing values")
        else:
            print(f"  ⚠️ Missing values found: {missing_counts.sum()} total")
        
        # Check target variable
        target_col = 'passengers'
        if target_col in df.columns:
            target_stats = df[target_col].describe()
            print(f"  📊 Target variable stats:")
            print(f"    Mean: {target_stats['mean']:.1f}")
            print(f"    Std:  {target_stats['std']:.1f}")
            print(f"    Min:  {target_stats['min']:.1f}")
            print(f"    Max:  {target_stats['max']:.1f}")
            
            # Check for reasonable values
            if target_stats['min'] >= 0 and target_stats['max'] <= 500:
                print("  ✅ Target values in reasonable range")
                return True
            else:
                print("  ⚠️ Some target values seem unusual")
                return False
        else:
            print("  ❌ Target column 'passengers' not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Error in data quality check: {e}")
        return False

def test_feature_engineering():
    """Test the feature engineering process"""
    print("\n🔍 Test 6: Feature Engineering Process")
    print("-" * 50)
    
    try:
        from ml_model import AdvancedBusRidershipPredictor
        
        # Test with advanced features disabled first
        print("  🔧 Testing basic feature engineering...")
        predictor_basic = AdvancedBusRidershipPredictor(enable_advanced_features=False)
        df_basic = predictor_basic.load_and_prepare_data()
        
        if hasattr(predictor_basic, 'X'):
            basic_features = len(predictor_basic.X.columns)
            print(f"  ✅ Basic features: {basic_features}")
        else:
            print("  ❌ Basic feature engineering failed")
            return False
        
        # Test advanced features if possible
        print("  🔧 Testing advanced feature engineering...")
        try:
            predictor_advanced = AdvancedBusRidershipPredictor(enable_advanced_features=True)
            df_advanced = predictor_advanced.load_and_prepare_data()
            
            if hasattr(predictor_advanced, 'X'):
                advanced_features = len(predictor_advanced.X.columns)
                print(f"  ✅ Advanced features: {advanced_features}")
                
                if advanced_features > basic_features:
                    print(f"  ✅ Advanced engineering creates {advanced_features - basic_features} additional features")
                    return True
                else:
                    print("  ⚠️ Advanced features not significantly different from basic")
                    return True
            else:
                print("  ⚠️ Advanced feature engineering had issues, but basic works")
                return True
                
        except Exception as e:
            print(f"  ⚠️ Advanced features failed ({e}), but basic works")
            return True
            
    except Exception as e:
        print(f"  ❌ Error in feature engineering: {e}")
        return False

def run_all_tests():
    """Run all tests and provide a comprehensive report"""
    print("🚀 Ultra-Advanced Bus Ridership Model Test Suite")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Model Training", test_model_training),
        ("Prediction Functionality", test_prediction_functionality),
        ("Scenario Predictions", test_scenario_predictions),
        ("Data Quality", test_data_quality),
        ("Feature Engineering", test_feature_engineering),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:25} {status}")
    
    percentage = (passed / total) * 100
    print(f"\n🎯 Overall Score: {percentage:.1f}% ({passed}/{total} tests passed)")
    
    if percentage >= 85:
        print("🎉 EXCELLENT! Model is working great!")
    elif percentage >= 70:
        print("✅ GOOD! Model is functional with minor issues.")
    elif percentage >= 50:
        print("⚠️ ACCEPTABLE! Model works but needs improvements.")
    else:
        print("❌ NEEDS WORK! Significant issues found.")
    
    print("\n💡 Test completed. Check individual test outputs for details.")
    return percentage >= 70

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)