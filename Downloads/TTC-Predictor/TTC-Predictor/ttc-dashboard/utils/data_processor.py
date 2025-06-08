# utils/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TTCDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_merge_data(self, delay_file='data/ttc_delays.csv', weather_file='data/weather_data.csv'):
        """Load and merge delay and weather data"""
        print("Loading delay data...")
        delay_df = pd.read_csv(delay_file)
        
        print("Loading weather data...")
        weather_df = pd.read_csv(weather_file)
        
        # Merge on date
        merged_df = pd.merge(delay_df, weather_df, on='Date', how='left')
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        print("Cleaning data...")
        
        # Convert date and time columns
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
        
        # Handle missing values
        numeric_columns = ['Min Delay', 'Min Gap', 'Temperature', 'Precipitation']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Remove extreme outliers (delays > 60 minutes are rare)
        if 'Min Delay' in df.columns:
            df = df[df['Min Delay'] <= 60]
        
        # Fill categorical missing values
        categorical_columns = ['Station', 'Code', 'Bound', 'Line', 'Weather_Condition']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Create features for machine learning"""
        print("Engineering features...")
        
        # Temporal features
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Rush hour indicators
        df['IsRushHour'] = ((df['Hour'].between(7, 9)) | (df['Hour'].between(17, 19))).astype(int)
        df['IsMorningRush'] = df['Hour'].between(7, 9).astype(int)
        df['IsEveningRush'] = df['Hour'].between(17, 19).astype(int)
        
        # Season features
        df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                                      9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Weather impact features
        if 'Temperature' in df.columns:
            df['IsExtremeTemp'] = ((df['Temperature'] < -10) | (df['Temperature'] > 30)).astype(int)
            df['TempCategory'] = pd.cut(df['Temperature'], 
                                     bins=[-np.inf, 0, 10, 20, np.inf], 
                                     labels=['Cold', 'Cool', 'Mild', 'Warm'])
        
        if 'Precipitation' in df.columns:
            df['HasPrecipitation'] = (df['Precipitation'] > 0).astype(int)
            df['HeavyPrecipitation'] = (df['Precipitation'] > 5).astype(int)
        
        # Station and line features
        if 'Station' in df.columns:
            # Major interchange stations
            major_stations = ['BLOOR-YONGE', 'ST GEORGE', 'UNION', 'SHEPPARD-YONGE']
            df['IsMajorStation'] = df['Station'].isin(major_stations).astype(int)
        
        # Historical delay features (rolling averages)
        if 'Min Delay' in df.columns:
            df = df.sort_values(['Station', 'Date', 'Time'])
            df['AvgDelay_7days'] = df.groupby('Station')['Min Delay'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
            df['DelayCount_7days'] = df.groupby('Station')['Min Delay'].rolling(7, min_periods=1).count().reset_index(0, drop=True)
        
        # Target variable categories
        if 'Min Delay' in df.columns:
            df['DelayCategory'] = pd.cut(df['Min Delay'], 
                                       bins=[0, 2, 5, 10, np.inf], 
                                       labels=['Minor', 'Moderate', 'Major', 'Severe'])
            df['IsSignificantDelay'] = (df['Min Delay'] > 5).astype(int)
        
        print(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, categorical_columns=None, fit=True):
        """Encode categorical features"""
        if categorical_columns is None:
            categorical_columns = ['Station', 'Code', 'Bound', 'Line', 'Weather_Condition', 
                                 'Season', 'TempCategory', 'DelayCategory']
        
        encoded_df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    encoded_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(self.label_encoders[col].classes_)
                        df_values = df[col].astype(str)
                        df_values = df_values.apply(lambda x: x if x in unique_values else 'Unknown')
                        encoded_df[f'{col}_encoded'] = self.label_encoders[col].transform(df_values)
        
        return encoded_df
    
    def prepare_features_for_ml(self, df, target_column='Min Delay'):
        """Prepare final feature set for machine learning"""
        
        # Select features for ML
        feature_columns = [
            'Hour', 'Month', 'DayOfWeek', 'DayOfYear', 'IsWeekend',
            'IsRushHour', 'IsMorningRush', 'IsEveningRush',
            'Temperature', 'Precipitation', 'IsExtremeTemp', 'HasPrecipitation',
            'HeavyPrecipitation', 'IsMajorStation', 'AvgDelay_7days', 'DelayCount_7days'
        ]
        
        # Add encoded categorical features
        encoded_columns = [col for col in df.columns if col.endswith('_encoded')]
        feature_columns.extend(encoded_columns)
        
        # Filter to only existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Prepare feature matrix
        X = df[feature_columns].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Prepare target variable
        y = None
        if target_column in df.columns:
            y = df[target_column].copy()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Selected features: {feature_columns}")
        
        return X, y, feature_columns
    
    def process_pipeline(self, delay_file='data/ttc_delays.csv', weather_file='data/weather_data.csv'):
        """Complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Load and merge data
        df = self.load_and_merge_data(delay_file, weather_file)
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features for ML
        X, y, feature_columns = self.prepare_features_for_ml(df)
        
        print("Data processing pipeline complete!")
        
        return X, y, feature_columns, df

def main():
    processor = TTCDataProcessor()
    
    try:
        X, y, features, processed_df = processor.process_pipeline()
        
        # Save processed data
        processed_df.to_csv('data/processed_ttc_data.csv', index=False)
        
        print(f"\nProcessed data saved!")
        print(f"Features: {len(features)}")
        print(f"Samples: {len(X)}")
        print(f"Target variable stats:")
        if y is not None:
            print(f"  Mean delay: {y.mean():.2f} minutes")
            print(f"  Max delay: {y.max():.2f} minutes")
            print(f"  Std delay: {y.std():.2f} minutes")
        
    except FileNotFoundError:
        print("Data files not found. Please run the data collector first.")
        print("Run: python data/raw_data.py")

if __name__ == "__main__":
    main()