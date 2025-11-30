"""
Tabular Model Training for Fish Freshness Prediction
XGBoost Regression for freshness_days_remaining prediction
Uses synthetic dataset: tamil_nadu_fish_dataset_50k.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FishTabularTrainer:
    """XGBoost trainer for fish freshness prediction"""
    
    def __init__(self, dataset_path='tamil_nadu_fish_dataset_50k.csv'):
        self.dataset_path = dataset_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the synthetic dataset"""
        print("📊 Loading synthetic fish dataset...")
        
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded {len(df):,} records from {self.dataset_path}")
        except FileNotFoundError:
            print(f"❌ Dataset not found: {self.dataset_path}")
            print("Please ensure you have generated the synthetic dataset first!")
            return None, None, None, None
        
        # Display dataset info
        print(f"\n📋 Dataset Overview:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['date_of_catch'].min()} to {df['date_of_catch'].max()}")
        
        # Target variable
        target = 'freshness_days'
        if target not in df.columns:
            print(f"❌ Target column '{target}' not found!")
            return None, None, None, None
        
        print(f"\n🎯 Target Variable Analysis:")
        print(f"  Min freshness: {df[target].min():.2f} days")
        print(f"  Max freshness: {df[target].max():.2f} days")
        print(f"  Mean freshness: {df[target].mean():.2f} days")
        print(f"  Std freshness: {df[target].std():.2f} days")
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Select features for training
        feature_columns = [
            'fish_species_encoded', 'weight_of_fish_g', 'height_of_fish_cm',
            'storage_temp', 'storage_type_encoded', 'area_temp',
            'time_since_catch_hours', 'distance_km', 'price_per_kg',
            'fisher_location_encoded', 'buyer_location_encoded',
            'catch_day_of_week', 'catch_month', 'catch_hour',
            'temp_freshness_interaction', 'weight_storage_interaction'
        ]
        
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing feature columns: {missing_cols}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns]
        y = df[target]
        
        self.feature_names = feature_columns
        
        print(f"\n🔧 Features selected: {len(feature_columns)}")
        print(f"  Features: {feature_columns}")
        
        return X, y, df, feature_columns
    
    def engineer_features(self, df):
        """Create additional features from existing data"""
        print("🔧 Engineering features...")
        
        # Encode categorical variables
        categorical_columns = ['fish_species', 'storage_type', 'fisher_location', 'buyer_location']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {len(le.classes_)} unique values")
        
        # Date/time features
        if 'date_of_catch' in df.columns:
            df['date_of_catch'] = pd.to_datetime(df['date_of_catch'])
            df['catch_day_of_week'] = df['date_of_catch'].dt.dayofweek
            df['catch_month'] = df['date_of_catch'].dt.month
            df['catch_hour'] = df['date_of_catch'].dt.hour
            print("  Added temporal features: day_of_week, month, hour")
        
        # Interaction features (important for freshness prediction)
        if 'storage_temp' in df.columns and 'freshness_days' in df.columns:
            # Temperature-freshness interaction (key predictor)
            df['temp_freshness_interaction'] = df['storage_temp'] * df['time_since_catch_hours'] if 'time_since_catch_hours' in df.columns else df['storage_temp']
        
        if 'weight_of_fish_g' in df.columns and 'storage_temp' in df.columns:
            # Weight-storage interaction (larger fish may preserve differently)
            df['weight_storage_interaction'] = df['weight_of_fish_g'] * (1 / (df['storage_temp'] + 1))
        
        print(f"  Added interaction features")
        
        return df
    
    def train_xgboost_model(self, X, y, test_size=0.2, optimize_hyperparams=True):
        """Train XGBoost regression model"""
        print("\n🚀 Training XGBoost model...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        
        if optimize_hyperparams:
            print("  🔍 Optimizing hyperparameters...")
            
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Grid search with cross-validation
            xgb_regressor = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
            grid_search = GridSearchCV(
                xgb_regressor, param_grid, 
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            print(f"  ✅ Best parameters: {grid_search.best_params_}")
            print(f"  ✅ Best CV score: {-grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            best_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            best_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Evaluate model
        train_metrics = self.calculate_metrics(y_train, y_train_pred, "Training")
        test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")
        
        # Feature importance
        self.plot_feature_importance(best_model, X.columns)
        
        # Save model and preprocessors
        self.save_model_artifacts(best_model, X.columns)
        
        return best_model, {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_names': list(X.columns),
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    def train_comparison_models(self, X, y):
        """Train comparison models (RandomForest)"""
        print("\n🌲 Training comparison model (Random Forest)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Predictions
        rf_pred = rf_model.predict(X_test)
        rf_metrics = self.calculate_metrics(y_test, rf_pred, "Random Forest")
        
        return rf_model, rf_metrics
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n📊 {model_name} Metrics:")
        print(f"  RMSE: {rmse:.4f} days")
        print(f"  MAE:  {mae:.4f} days")
        print(f"  R²:   {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse
        }
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importances - XGBoost Model')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance plot saved as 'feature_importance.png'")
        
        # Print top features
        print(f"\n🔝 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('Actual Freshness Days')
        plt.ylabel('Predicted Freshness Days')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score as text
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        filename = f'{model_name.lower().replace(" ", "_")}_predictions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Prediction plot saved as '{filename}'")
    
    def analyze_residuals(self, y_true, y_pred, model_name):
        """Analyze prediction residuals"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0,0].scatter(y_pred, residuals, alpha=0.5)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted')
        axes[0,0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0,1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(x=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Residuals')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot of Residuals')
        axes[1,0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[1,1].scatter(y_true, residuals, alpha=0.5)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Actual Values')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Residuals vs Actual')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis')
        plt.tight_layout()
        
        filename = f'{model_name.lower().replace(" ", "_")}_residuals.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Residual analysis saved as '{filename}'")
    
    def save_model_artifacts(self, model, feature_names):
        """Save model and preprocessing artifacts"""
        
        # Save XGBoost model
        model.save_model('fish_freshness_xgboost.json')
        
        # Save using pickle as backup
        with open('fish_freshness_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save preprocessors
        with open('tabular_label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open('tabular_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names and metadata
        model_metadata = {
            'feature_names': list(feature_names),
            'label_encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()},
            'model_type': 'XGBoost',
            'target_variable': 'freshness_days',
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_names)
        }
        
        with open('tabular_model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print("✅ Model artifacts saved:")
        print("  - fish_freshness_xgboost.json (XGBoost native format)")
        print("  - fish_freshness_model.pkl (Pickle format)")
        print("  - tabular_label_encoders.pkl (Label encoders)")
        print("  - tabular_scaler.pkl (Feature scaler)")
        print("  - tabular_model_metadata.json (Metadata)")
    
    def predict_freshness(self, model, species, weight, storage_temp, hours_since_catch, **kwargs):
        """Make freshness prediction for new sample"""
        
        # Create feature vector (simplified for demo)
        feature_dict = {
            'fish_species_encoded': self.label_encoders['fish_species'].transform([species])[0] if species in self.label_encoders['fish_species'].classes_ else 0,
            'weight_of_fish_g': weight,
            'storage_temp': storage_temp,
            'time_since_catch_hours': hours_since_catch,
            'height_of_fish_cm': kwargs.get('height', 20),  # Default values
            'distance_km': kwargs.get('distance', 50),
            'area_temp': kwargs.get('area_temp', 28),
            'price_per_kg': kwargs.get('price', 300)
        }
        
        # Add encoded features with defaults
        for col in self.feature_names:
            if col not in feature_dict:
                feature_dict[col] = 0  # Default for missing features
        
        # Create feature array
        X_new = np.array([feature_dict[col] for col in self.feature_names]).reshape(1, -1)
        
        # Predict
        prediction = model.predict(X_new)[0]
        
        return max(0.1, prediction)  # Ensure positive prediction


def main():
    """Main training pipeline for tabular model"""
    print("🐟 FISH FRESHNESS TABULAR MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = FishTabularTrainer('tamil_nadu_fish_dataset_50k.csv')
    
    # Load and preprocess data
    X, y, df, feature_names = trainer.load_and_preprocess_data()
    
    if X is None:
        return
    
    # Train XGBoost model
    xgb_model, xgb_results = trainer.train_xgboost_model(X, y, optimize_hyperparams=True)
    
    # Train comparison model
    rf_model, rf_metrics = trainer.train_comparison_models(X, y)
    
    # Generate visualizations
    trainer.plot_predictions_vs_actual(
        xgb_results['y_test'], 
        xgb_results['y_test_pred'], 
        'XGBoost'
    )
    
    trainer.analyze_residuals(
        xgb_results['y_test'],
        xgb_results['y_test_pred'],
        'XGBoost'
    )
    
    # Save final results
    final_results = {
        'training_completed': True,
        'model_path': 'fish_freshness_model.pkl',
        'xgboost_metrics': xgb_results['test_metrics'],
        'random_forest_metrics': rf_metrics,
        'feature_count': len(feature_names),
        'training_samples': len(X),
        'best_model': 'XGBoost' if xgb_results['test_metrics']['r2'] > rf_metrics['r2'] else 'Random Forest'
    }
    
    with open('tabular_training_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n🎉 TABULAR MODEL TRAINING COMPLETED!")
    print(f"✅ Best model: {final_results['best_model']}")
    print(f"✅ XGBoost R²: {xgb_results['test_metrics']['r2']:.4f}")
    print(f"✅ XGBoost RMSE: {xgb_results['test_metrics']['rmse']:.4f} days")
    print(f"✅ Random Forest R²: {rf_metrics['r2']:.4f}")
    
    # Demo prediction
    print(f"\n🧪 Demo Prediction:")
    sample_prediction = trainer.predict_freshness(
        xgb_model, 
        species='Indian Mackerel',
        weight=200,
        storage_temp=5,
        hours_since_catch=12
    )
    print(f"  Species: Indian Mackerel, 200g, 5°C storage, 12 hours old")
    print(f"  Predicted freshness: {sample_prediction:.2f} days remaining")

if __name__ == "__main__":
    main()
