# models/delay_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TTCDelayPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_performance = {}
        
    def load_processed_data(self, filepath='data/processed_ttc_data.csv'):
        """Load the processed data"""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded processed data: {df.shape}")
            return df
        except FileNotFoundError:
            print("Processed data not found. Please run data processing first.")
            return None
    
    def prepare_data_for_training(self, df, target_column='Min Delay'):
        """Prepare data for model training"""
        
        # Feature columns (excluding target and non-predictive columns)
        exclude_columns = [
            'Date', 'Time', 'Day', 'Vehicle', target_column,
            'DelayCategory', 'Min Gap'  # These are derived from target
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle categorical encoded columns
        numeric_features = []
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        X = df[numeric_features].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"Training features: {X.shape[1]}")
        print(f"Training samples: {X.shape[0]}")
        
        return X, y, numeric_features
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        print("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                eval_metric='rmse'
            )
        }
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.models[name] = model
            self.model_performance[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df
        
        return X_test, y_test
    
    def optimize_best_model(self, X, y):
        """Optimize the best performing model"""
        print("Optimizing best model...")
        
        # Find best model based on R2 score
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda x: self.model_performance[x]['R2'])
        
        print(f"Best model: {best_model_name}")
        
        # Hyperparameter tuning for Random Forest (usually performs well)
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update best model
            optimized_model = grid_search.best_estimator_
            y_pred = optimized_model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.models['Optimized Random Forest'] = optimized_model
            self.model_performance['Optimized Random Forest'] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"Optimized model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
            print(f"Best parameters: {grid_search.best_params_}")
    
    def create_delay_classifier(self, X, y):
        """Create a classification model for delay severity"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score
        
        # Create delay categories
        y_cat = pd.cut(y, bins=[0, 2, 5, 10, np.inf], 
                      labels=['Minor', 'Moderate', 'Major', 'Severe'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=0.2, random_state=42
        )
        
        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['Delay Classifier'] = classifier
        self.model_performance['Delay Classifier'] = {
            'Accuracy': accuracy,
            'Classification Report': classification_report(y_test, y_pred)
        }
        
        print(f"Delay Classifier Accuracy: {accuracy:.3f}")
    
    def save_models(self, model_dir='models/'):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"{model_dir}{name.replace(' ', '_').lower()}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, f"{model_dir}scaler.joblib")
        
        # Save performance metrics
        import json
        perf_data = {}
        for name, metrics in self.model_performance.items():
            perf_data[name] = {k: v for k, v in metrics.items() 
                              if k not in ['predictions', 'actual']}
        
        with open(f"{model_dir}model_performance.json", 'w') as f:
            json.dump(perf_data, f, indent=2)
    
    def plot_results(self):
        """Create visualizations of model performance"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        models = list(self.model_performance.keys())
        mae_scores = [self.model_performance[m]['MAE'] for m in models if 'MAE' in self.model_performance[m]]
        r2_scores = [self.model_performance[m]['R2'] for m in models if 'R2' in self.model_performance[m]]
        
        axes[0, 0].bar(models[:len(mae_scores)], mae_scores)
        axes[0, 0].set_title('Mean Absolute Error by Model')
        axes[0, 0].set_ylabel('MAE (minutes)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models[:len(r2_scores)], r2_scores)
        axes[0, 1].set_title('R² Score by Model')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Prediction vs Actual (best model)
        best_model = max(self.model_performance.keys(), 
                        key=lambda x: self.model_performance[x].get('R2', 0))
        
        if 'predictions' in self.model_performance[best_model]:
            actual = self.model_performance[best_model]['actual']
            predicted = self.model_performance[best_model]['predictions']
            
            axes[1, 0].scatter(actual, predicted, alpha=0.5)
            axes[1, 0].set_title('Actual vs Predicted (Best Model)')
            axes[1, 0].set_xlabel('Actual Delay (min)')
            axes[1, 0].set_ylabel('Predicted Delay (min)')

if __name__ == "__main__":
    predictor = TTCDelayPredictor()
    df = predictor.load_processed_data('data/processed_ttc_data.csv')
    if df is not None:
        X, y, features = predictor.prepare_data_for_training(df)
        predictor.train_models(X, y)
        predictor.optimize_best_model(X, y)
        predictor.save_models('models/trained/')
        print("\nModel training and export complete!\n")
    else:
        print("Processed data not found. Please run the data processor first.")