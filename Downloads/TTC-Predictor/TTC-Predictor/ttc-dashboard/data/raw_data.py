# data/raw_data.py
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os
import numpy as np

class TTCDataCollector:
    def __init__(self):
        self.base_url = "https://ckan0.cf.opendata.inter.sandbox-toronto.ca/api/3/action/"
        self.delay_data_url = "https://ckan0.cf.opendata.inter.sandbox-toronto.ca/dataset/ttc-subway-delay-data"
        
    def fetch_delay_data(self):
        """Fetch TTC subway delay data from Toronto Open Data"""
        try:
            # Get the dataset metadata
            dataset_url = f"{self.base_url}package_show?id=ttc-subway-delay-data"
            response = requests.get(dataset_url)
            
            if response.status_code == 200:
                dataset_info = response.json()
                resources = dataset_info['result']['resources']
                
                # Get the most recent CSV file
                csv_resources = [r for r in resources if r['format'].upper() == 'CSV']
                if csv_resources:
                    latest_resource = max(csv_resources, key=lambda x: x['last_modified'])
                    csv_url = latest_resource['url']
                    
                    print(f"Downloading delay data from: {csv_url}")
                    df = pd.read_csv(csv_url)
                    return df
                else:
                    print("No CSV resources found")
                    return None
            else:
                print(f"Failed to fetch dataset info: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching delay data: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample delay data for development"""
        print("Creating sample data for development...")
        
        # Generate sample data that mimics real TTC delay patterns
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='H')
        
        data = []
        stations = ['BLOOR-YONGE', 'UNION', 'ST GEORGE', 'KING', 'QUEEN', 'DUNDAS', 
                   'COLLEGE', 'WELLESLEY', 'ROSEDALE', 'SUMMERHILL', 'EGLINTON']
        lines = ['YU', 'BD', 'SHP', 'SRT']
        
        for i, date in enumerate(dates[:5000]):  # Limit to 5000 records for demo
            # More delays during rush hours
            hour = date.hour
            is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
            delay_prob = 0.3 if is_rush_hour else 0.1
            
            if np.random.random() < delay_prob:
                delay_minutes = np.random.exponential(3) + 1  # Most delays are short
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Time': date.strftime('%H:%M'),
                    'Day': date.strftime('%A'),
                    'Station': np.random.choice(stations),
                    'Code': np.random.choice(['MUSAN', 'TUSC', 'MURS', 'EUNT', 'PUSC']),
                    'Min Delay': int(delay_minutes),
                    'Min Gap': int(delay_minutes * 1.5),
                    'Bound': np.random.choice(['N', 'S', 'E', 'W']),
                    'Line': np.random.choice(lines),
                    'Vehicle': f"{np.random.randint(5000, 6000)}"
                })
        
        df = pd.DataFrame(data)
        return df
    
    def save_data(self, df, filename='ttc_delays.csv'):
        """Save the collected data"""
        if df is not None:
            filepath = os.path.join('data', filename)
            df.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return filepath
        return None
    
    def get_weather_data(self, start_date, end_date):
        """Get weather data for the date range (simplified version)"""
        # In a real implementation, you'd use Environment Canada API or OpenWeather
        # For now, we'll create mock weather data
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        weather_data = []
        
        for date in dates:
            # Simulate weather patterns
            temp = 15 + 10 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 5)
            precipitation = max(0, np.random.exponential(2) - 1)
            
            weather_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Temperature': round(temp, 1),
                'Precipitation': round(precipitation, 1),
                'Weather_Condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow'], 
                                                       p=[0.4, 0.3, 0.2, 0.1])
            })
        
        return pd.DataFrame(weather_data)

def main():
    collector = TTCDataCollector()
    
    print("Fetching TTC delay data...")
    delay_df = collector.fetch_delay_data()
    
    if delay_df is not None:
        # Save delay data
        delay_file = collector.save_data(delay_df, 'ttc_delays.csv')
        
        # Get date range from delay data
        delay_df['Date'] = pd.to_datetime(delay_df['Date'])
        start_date = delay_df['Date'].min()
        end_date = delay_df['Date'].max()
        
        print(f"Getting weather data for {start_date} to {end_date}")
        weather_df = collector.get_weather_data(start_date, end_date)
        weather_file = collector.save_data(weather_df, 'weather_data.csv')
        
        print("\nData collection complete!")
        print(f"Delay records: {len(delay_df)}")
        print(f"Weather records: {len(weather_df)}")
        
        # Show sample data
        print("\nSample delay data:")
        print(delay_df.head())
        
    else:
        print("Failed to collect data")

if __name__ == "__main__":
    main()