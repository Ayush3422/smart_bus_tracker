import requests
import json
from datetime import datetime

def quick_api_test():
    """Quick test of all API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🚀 Quick Backend API Test")
    print("=" * 30)
    
    # Test each endpoint
    endpoints = [
        ("/api/status", "System Status"),
        ("/api/buses", "Real-time Buses"),
        ("/api/predictions/RT001", "ML Predictions"),
        ("/api/optimize/RT001", "Schedule Optimization"),
        ("/api/analytics", "Analytics Dashboard"),
        ("/api/model/info", "Model Information")
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        try:
            print(f"\n📡 Testing {description}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {description}: SUCCESS")
                
                # Show key data
                if endpoint == "/api/status":
                    print(f"   Model Loaded: {data.get('model_loaded', 'Unknown')}")
                    print(f"   Model Type: {data.get('model_type', 'Unknown')}")
                
                elif endpoint == "/api/buses":
                    buses = data.get('buses', [])
                    print(f"   Active Buses: {len(buses)}")
                    if buses:
                        print(f"   Sample: {buses[0]['bus_id']} has {buses[0]['passengers']} passengers")
                
                elif "predictions" in endpoint:
                    predictions = data.get('predictions', [])
                    print(f"   Predictions: {len(predictions)} hours")
                    if predictions:
                        next_pred = predictions[0]
                        print(f"   Next Hour: {next_pred['predicted_passengers']} passengers")
                
                elif "optimize" in endpoint:
                    opt = data.get('schedule_optimization', {})
                    print(f"   Current Demand: {data.get('current_demand', 'Unknown')}")
                    print(f"   Optimized Frequency: {opt.get('optimized_frequency', 'Unknown')} min")
                
                elif endpoint == "/api/analytics":
                    summary = data.get('system_summary', {})
                    print(f"   Active Buses: {summary.get('active_buses', 'Unknown')}")
                    print(f"   Avg Passengers: {summary.get('avg_current_passengers', 'Unknown')}")
                
                elif endpoint == "/api/model/info":
                    print(f"   Model Name: {data.get('model_name', 'Unknown')}")
                    print(f"   Features: {data.get('features_count', 'Unknown')}")
                
                results.append((description, "PASS", None))
            else:
                print(f"❌ {description}: HTTP {response.status_code}")
                results.append((description, "FAIL", f"HTTP {response.status_code}"))
                
        except Exception as e:
            print(f"❌ {description}: ERROR - {e}")
            results.append((description, "FAIL", str(e)))
    
    # Summary
    print("\n" + "=" * 30)
    print("📋 QUICK TEST SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL ENDPOINTS WORKING!")
        print("🚀 Backend is ready for frontend development!")
    else:
        print("⚠️ Some endpoints failed:")
        for desc, status, error in results:
            if status == "FAIL":
                print(f"   ❌ {desc}: {error}")
    
    return passed == total

if __name__ == "__main__":
    quick_api_test()