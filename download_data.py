
import requests
import pandas as pd
import zipfile
import os
import numpy as np
from datetime import datetime, timedelta
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_realistic_gtfs_data():
    """Create realistic GTFS data structure based on Delhi bus system"""
    print("� Creating realistic GTFS data structure...")
    
    # Create GTFS directory
    os.makedirs("gtfs_data", exist_ok=True)
    
    # Create routes.txt based on real Delhi bus routes
    routes_data = [
        {"route_id": "DTC_01", "route_short_name": "1", "route_long_name": "Red Fort - ISBT Kashmere Gate", "route_type": 3},
        {"route_id": "DTC_34", "route_short_name": "34", "route_long_name": "Nehru Place - Dhaula Kuan", "route_type": 3},
        {"route_id": "DTC_52", "route_short_name": "52", "route_long_name": "Anand Vihar - Central Secretariat", "route_type": 3},
        {"route_id": "DTC_181", "route_short_name": "181", "route_long_name": "Mundka - New Delhi Railway Station", "route_type": 3},
        {"route_id": "DTC_269", "route_short_name": "269", "route_long_name": "Rohini Sec-18 - Sarai Kale Khan", "route_type": 3},
        {"route_id": "DTC_300", "route_short_name": "300", "route_long_name": "Dwarka Sec-21 - Connaught Place", "route_type": 3},
        {"route_id": "DTC_402", "route_short_name": "402", "route_long_name": "Badarpur - Red Fort", "route_type": 3},
        {"route_id": "DTC_505", "route_short_name": "505", "route_long_name": "Ghaziabad - ISBT Kashmere Gate", "route_type": 3},
        {"route_id": "DTC_615", "route_short_name": "615", "route_long_name": "Najafgarh - New Delhi Railway Station", "route_type": 3},
        {"route_id": "DTC_728", "route_short_name": "728", "route_long_name": "Yamuna Vihar - Connaught Place", "route_type": 3},
    ]
    
    routes_df = pd.DataFrame(routes_data)
    routes_df.to_csv("gtfs_data/routes.txt", index=False)
    
    # Create stops.txt with major Delhi locations
    stops_data = [
        {"stop_id": "DTC_STOP_001", "stop_name": "Red Fort", "stop_lat": 28.6562, "stop_lon": 77.2410},
        {"stop_id": "DTC_STOP_002", "stop_name": "Connaught Place", "stop_lat": 28.6315, "stop_lon": 77.2167},
        {"stop_id": "DTC_STOP_003", "stop_name": "New Delhi Railway Station", "stop_lat": 28.6431, "stop_lon": 77.2197},
        {"stop_id": "DTC_STOP_004", "stop_name": "ISBT Kashmere Gate", "stop_lat": 28.6667, "stop_lon": 77.2167},
        {"stop_id": "DTC_STOP_005", "stop_name": "Nehru Place", "stop_lat": 28.5478, "stop_lon": 77.2536},
        {"stop_id": "DTC_STOP_006", "stop_name": "Dwarka Sector 21", "stop_lat": 28.5521, "stop_lon": 77.0590},
        {"stop_id": "DTC_STOP_007", "stop_name": "Anand Vihar", "stop_lat": 28.6469, "stop_lon": 77.3150},
        {"stop_id": "DTC_STOP_008", "stop_name": "Rohini Sector 18", "stop_lat": 28.7197, "stop_lon": 77.1050},
        {"stop_id": "DTC_STOP_009", "stop_name": "Badarpur", "stop_lat": 28.4958, "stop_lon": 77.2982},
        {"stop_id": "DTC_STOP_010", "stop_name": "Central Secretariat", "stop_lat": 28.6139, "stop_lon": 77.2090},
    ]
    
    stops_df = pd.DataFrame(stops_data)
    stops_df.to_csv("gtfs_data/stops.txt", index=False)
    
    # Create agency.txt
    agency_data = [
        {"agency_id": "DTC", "agency_name": "Delhi Transport Corporation", 
         "agency_url": "http://dtc.delhi.gov.in", "agency_timezone": "Asia/Kolkata"}
    ]
    agency_df = pd.DataFrame(agency_data)
    agency_df.to_csv("gtfs_data/agency.txt", index=False)
    
    print(f"✅ Created realistic GTFS structure:")
    print(f"  📍 {len(stops_df)} stops (major Delhi locations)")
    print(f"  🚌 {len(routes_df)} routes (real DTC route numbers)")
    print(f"  🏢 1 agency (Delhi Transport Corporation)")
    
    return "gtfs_data/"

def create_sample_ridership_data():
    """Create realistic ridership data based on GTFS structure"""
    print("📊 Creating sample ridership data...")
    
    # Read actual routes from GTFS
    try:
        routes_df = pd.read_csv("gtfs_data/routes.txt")
        stops_df = pd.read_csv("gtfs_data/stops.txt")
        
        # Generate realistic ridership patterns
        np.random.seed(42)
        # ...existing code...
    except Exception as e:
        print(f"⚠️  Error with GTFS data: {e}")
        print("📊 Creating synthetic data instead...")
        
        # Fallback: Create completely synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2025-08-01', end='2025-08-31', freq='H')
        
        routes = ['RT001', 'RT002', 'RT003', 'RT004', 'RT005']
        data = []
        
        for date in dates:
            for route in routes:
                hour = date.hour
                base = 30
                
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    base *= 2.5
                if date.weekday() >= 5:
                    base *= 0.6
                
                passengers = max(1, int(base + np.random.normal(0, 8)))
                
                data.append({
                    'timestamp': date,
                    'route_id': route,
                    'route_name': f'Route {route[-1]}',
                    'passengers': passengers,
                    'hour': hour,
                    'day_of_week': date.weekday(),
                    'is_weekend': date.weekday() >= 5,
                    'weather_factor': round(np.random.uniform(0.8, 1.2), 2)
                })
        
        df = pd.DataFrame(data)
        df.to_csv('bus_ridership_data.csv', index=False)
        print(f"✅ Created synthetic ridership data: {len(df)} records")
        return df

if __name__ == "__main__":
    # Create realistic GTFS data structure
    try:
        gtfs_path = create_realistic_gtfs_data()
        print(f"✅ GTFS data created at: {gtfs_path}")
    except Exception as e:
        print(f"⚠️ GTFS creation failed: {e}")
        print("📊 Proceeding with synthetic data generation...")
    
    # Create ridership data
    ridership_df = create_sample_ridership_data()
    print("🎉 Data preparation complete!")
