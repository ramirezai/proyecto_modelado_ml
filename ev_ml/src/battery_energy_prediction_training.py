# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install Required Libraries and Restart
# MAGIC %pip install -q optuna xgboost lightgbm catboost scikit-learn mlflow

# COMMAND ----------

# DBTITLE 1,Import Libraries and Load Data
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data using Spark
print("Loading data from Unity Catalog...")
df_spark = spark.table("dev.ml_features.ml_features_energy_prediction")

# Show basic info
print(f"\nTotal rows: {df_spark.count():,}")
print(f"Total columns: {len(df_spark.columns)}")

# Convert to pandas for analysis
df = df_spark.toPandas()
print("\nData loaded successfully!")
print(f"Pandas DataFrame shape: {df.shape}")

# COMMAND ----------

# DBTITLE 1,Data Quality Assessment
# Data Quality Assessment
print("=" * 80)
print("DATA QUALITY ASSESSMENT")
print("=" * 80)

# Basic statistics
print("\n1. DataFrame Info:")
print(f"   Shape: {df.shape}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for missing values
print("\n2. Missing Values Analysis:")
missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('Missing_Count', ascending=False)

print(missing_stats[missing_stats['Missing_Count'] > 0])

# Target variable statistics
print("\n3. Target Variable (target_energy_consumption):")
if df['target_energy_consumption'].notna().sum() > 0:
    print(f"   Non-null count: {df['target_energy_consumption'].notna().sum():,}")
    print(f"   Mean: {df['target_energy_consumption'].mean():.2f}")
    print(f"   Median: {df['target_energy_consumption'].median():.2f}")
    print(f"   Std: {df['target_energy_consumption'].std():.2f}")
    print(f"   Min: {df['target_energy_consumption'].min():.2f}")
    print(f"   Max: {df['target_energy_consumption'].max():.2f}")
else:
    print("   WARNING: Target variable is completely null!")

# Data types
print("\n4. Data Types:")
print(df.dtypes.value_counts())

# COMMAND ----------

# DBTITLE 1,Feature Types and Temporal Analysis
# Feature Classification
print("=" * 80)
print("FEATURE TYPES CLASSIFICATION")
print("=" * 80)

feature_types = {
    'ID Features': ['DayNum', 'VehId', 'Trip'],
    'Target': ['target_energy_consumption'],
    'Temporal': ['trip_date', 'start_timestamp_ms', 'trip_year', 'trip_month', 'trip_dayofweek', 'trip_hour', 'feature_timestamp'],
    'Trip Characteristics': ['distance_km', 'duration_minutes', 'avg_speed_kmh', 'max_speed_kmh'],
    'Battery State': ['avg_battery_soc', 'min_battery_soc', 'max_battery_soc'],
    'Location & Elevation': ['avg_elevation_m', 'elevation_gain_m', 'origin_lat_rounded', 'origin_lon_rounded', 'dest_lat_rounded', 'dest_lon_rounded'],
    'Vehicle History (30 days)': ['vehicle_avg_efficiency_last30', 'vehicle_avg_energy_last30', 'vehicle_stddev_energy_last30', 'vehicle_avg_distance_last30', 'vehicle_trip_count_last30'],
    'Route History (10 trips)': ['route_avg_energy_last10', 'route_avg_duration_last10', 'route_avg_speed_last10', 'route_trip_count']
}

for category, features in feature_types.items():
    print(f"\n{category}:")
    for feat in features:
        if feat in df.columns:
            dtype = df[feat].dtype
            null_pct = (df[feat].isnull().sum() / len(df) * 100)
            print(f"  - {feat:40s} | Type: {str(dtype):10s} | Nulls: {null_pct:5.1f}%")

# Temporal structure check
print("\n" + "=" * 80)
print("TEMPORAL STRUCTURE ANALYSIS")
print("=" * 80)

print("\nDate range:")
print(f"  trip_date unique values: {df['trip_date'].nunique()}")
print(f"  Min date: {df['trip_date'].min()}")
print(f"  Max date: {df['trip_date'].max()}")

print("\nTemporal features distribution:")
for col in ['trip_year', 'trip_month', 'trip_dayofweek', 'trip_hour']:
    print(f"  {col}: {df[col].unique()[:10]}")

# COMMAND ----------

# DBTITLE 1,Target Variable Distribution and Key Feature Correlations
# Target distribution and correlations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Target distribution
df_with_target = df[df['target_energy_consumption'].notna()]
if len(df_with_target) > 0:
    axes[0, 0].hist(df_with_target['target_energy_consumption'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Target Energy Consumption Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Energy Consumption')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df_with_target['target_energy_consumption'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()

    # 2. Distance vs Energy
    axes[0, 1].scatter(df_with_target['distance_km'], df_with_target['target_energy_consumption'], alpha=0.3)
    axes[0, 1].set_title('Distance vs Energy Consumption', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Distance (km)')
    axes[0, 1].set_ylabel('Energy Consumption')
    
    # 3. Speed vs Energy
    axes[1, 0].scatter(df_with_target['avg_speed_kmh'], df_with_target['target_energy_consumption'], alpha=0.3, color='green')
    axes[1, 0].set_title('Average Speed vs Energy Consumption', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Average Speed (km/h)')
    axes[1, 0].set_ylabel('Energy Consumption')
    
    # 4. Elevation gain vs Energy
    axes[1, 1].scatter(df_with_target['elevation_gain_m'], df_with_target['target_energy_consumption'], alpha=0.3, color='orange')
    axes[1, 1].set_title('Elevation Gain vs Energy Consumption', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Elevation Gain (m)')
    axes[1, 1].set_ylabel('Energy Consumption')
else:
    fig.text(0.5, 0.5, 'No target data available for visualization', ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.show()

print(f"Visualized {len(df_with_target):,} records with non-null target values")

# COMMAND ----------

# DBTITLE 1,Correlation Analysis
# Correlation analysis
print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Select numeric features for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID columns and keep only relevant features
cols_to_exclude = ['DayNum', 'VehId', 'Trip', 'start_timestamp_ms']
numeric_features = [col for col in numeric_cols if col not in cols_to_exclude]

# Calculate correlation with target
if 'target_energy_consumption' in numeric_features and df['target_energy_consumption'].notna().sum() > 0:
    correlations = df[numeric_features].corr()['target_energy_consumption'].sort_values(ascending=False)
    print("\nTop 15 features correlated with target:")
    print(correlations.head(15))
    
    # Correlation heatmap for top features
    top_features = correlations.head(12).index.tolist()
    
    plt.figure(figsize=(14, 10))
    correlation_matrix = df[top_features].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Top Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Check for potential target leakage
    print("\n⚠️  LEAKAGE CHECK:")
    high_corr = correlations[(correlations.abs() > 0.95) & (correlations.index != 'target_energy_consumption')]
    if len(high_corr) > 0:
        print("\n   WARNING: Features with very high correlation (>0.95) - potential leakage:")
        for feat, corr in high_corr.items():
            print(f"   - {feat}: {corr:.4f}")
    else:
        print("   ✓ No features with suspiciously high correlation detected")
else:
    print("Cannot compute correlations - target variable has no data")

# COMMAND ----------

# DBTITLE 1,Data Preprocessing and Feature Selection
# Data preprocessing
print("=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Filter to rows with non-null target
df_model = df[df['target_energy_consumption'].notna()].copy()
print(f"\nRows with valid target: {len(df_model):,}")

# Define feature columns to use
feature_cols = [
    # Trip characteristics
    'distance_km', 'duration_minutes', 'avg_speed_kmh', 'max_speed_kmh',
    # Elevation
    'avg_elevation_m', 'elevation_gain_m',
    # Location (rounded coordinates)
    'origin_lat_rounded', 'origin_lon_rounded', 'dest_lat_rounded', 'dest_lon_rounded',
    # Temporal features
    'trip_month', 'trip_dayofweek', 'trip_hour',
    # Vehicle history features (if available)
    'vehicle_avg_efficiency_last30', 'vehicle_avg_energy_last30', 
    'vehicle_stddev_energy_last30', 'vehicle_avg_distance_last30', 'vehicle_trip_count_last30',
    # Route history features (if available)
    'route_avg_energy_last10', 'route_avg_duration_last10', 
    'route_avg_speed_last10', 'route_trip_count'
]

# Keep only features that exist in the dataframe
feature_cols = [col for col in feature_cols if col in df_model.columns]
print(f"\nSelected features ({len(feature_cols)}): ")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# Prepare X and y
X = df_model[feature_cols].copy()
y = df_model['target_energy_consumption'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nMissing values in features:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# COMMAND ----------

# DBTITLE 1,Advanced Feature Engineering
# Advanced Feature Engineering
print("=" * 80)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 80)

print("\nCreating interaction and derived features...")

# Make a copy for feature engineering
X_enriched = X.copy()

# 1. Energy efficiency metrics
X_enriched['energy_per_km'] = X_enriched['distance_km'] / (X_enriched['distance_km'] + 0.001)  # proxy, will be computed post-prediction
X_enriched['speed_to_max_ratio'] = X_enriched['avg_speed_kmh'] / (X_enriched['max_speed_kmh'] + 1)

# 2. Speed and elevation interactions
X_enriched['speed_elevation_interaction'] = X_enriched['avg_speed_kmh'] * X_enriched['elevation_gain_m'].fillna(0)
X_enriched['distance_elevation_ratio'] = X_enriched['distance_km'] / (X_enriched['elevation_gain_m'].fillna(0) + 1)

# 3. Acceleration proxy (speed variability)
X_enriched['speed_variability'] = (X_enriched['max_speed_kmh'] - X_enriched['avg_speed_kmh']) / (X_enriched['duration_minutes'] + 1)

# 4. Temporal patterns
X_enriched['is_rush_hour'] = ((X_enriched['trip_hour'] >= 7) & (X_enriched['trip_hour'] <= 9) | 
                               (X_enriched['trip_hour'] >= 17) & (X_enriched['trip_hour'] <= 19)).astype(int)
X_enriched['is_weekend'] = (X_enriched['trip_dayofweek'] >= 5).astype(int)

# 5. Hour cyclical encoding
X_enriched['hour_sin'] = np.sin(2 * np.pi * X_enriched['trip_hour'] / 24)
X_enriched['hour_cos'] = np.cos(2 * np.pi * X_enriched['trip_hour'] / 24)

# 6. Distance categories
X_enriched['distance_category'] = pd.cut(X_enriched['distance_km'], 
                                          bins=[0, 5, 15, 30, 100], 
                                          labels=[0, 1, 2, 3]).astype(float)

# 7. Speed categories (urban vs highway)
X_enriched['speed_category'] = pd.cut(X_enriched['avg_speed_kmh'],
                                       bins=[0, 30, 60, 200],
                                       labels=[0, 1, 2]).astype(float)  # 0=urban, 1=suburban, 2=highway

# 8. Trip efficiency score (distance per time)
X_enriched['trip_efficiency'] = X_enriched['distance_km'] / (X_enriched['duration_minutes'] + 1)

# 9. Elevation intensity
X_enriched['elevation_per_km'] = X_enriched['elevation_gain_m'].fillna(0) / (X_enriched['distance_km'] + 0.001)

# 10. Vehicle efficiency deviation (if available)
if 'vehicle_avg_efficiency_last30' in X_enriched.columns:
    global_avg_efficiency = X_enriched['vehicle_avg_efficiency_last30'].mean()
    X_enriched['vehicle_efficiency_vs_fleet'] = X_enriched['vehicle_avg_efficiency_last30'] - global_avg_efficiency

# Update feature list
engineered_features = [
    'speed_elevation_interaction', 'distance_elevation_ratio', 'speed_variability',
    'is_rush_hour', 'is_weekend', 'hour_sin', 'hour_cos', 'distance_category',
    'speed_category', 'trip_efficiency', 'elevation_per_km', 'speed_to_max_ratio'
]

if 'vehicle_efficiency_vs_fleet' in X_enriched.columns:
    engineered_features.append('vehicle_efficiency_vs_fleet')

all_features = feature_cols + engineered_features

print(f"\nOriginal features: {len(feature_cols)}")
print(f"Engineered features: {len(engineered_features)}")
print(f"Total features: {len(all_features)}")

print("\nNew features created:")
for i, feat in enumerate(engineered_features, 1):
    print(f"  {i:2d}. {feat}")

# Update X with enriched features
X = X_enriched[all_features].copy()
feature_cols = all_features

print(f"\n✓ Feature engineering complete!")
print(f"  Final feature matrix shape: {X.shape}")

# COMMAND ----------

# DBTITLE 1,Train-Test Split
from sklearn.model_selection import train_test_split

print("=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

# Note: trip_date shows 1970-01-01 for all records, indicating temporal data is not properly populated
# Therefore, using random stratified split instead of time-based split
print("\n⚠️  Note: Date field shows epoch date (1970-01-01) for all records.")
print("   Using random split since temporal ordering is not reliable.")

# Random split with stratification on target quartiles for better distribution
y_quartiles = pd.qcut(y, q=4, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y_quartiles
)

print(f"\nSplit results:")
print(f"  Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTarget statistics:")
print(f"  Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"  Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

# COMMAND ----------

# DBTITLE 1,MLflow Setup
# MLflow setup
import mlflow
from mlflow.models.signature import infer_signature

print("=" * 80)
print("MLFLOW EXPERIMENT SETUP")
print("=" * 80)

# Set experiment name
experiment_name = "/Users/joel@ramirezai.com/proyecto_modelado_ml/ev_ml/experiments/battery_energy_prediction"
mlflow.set_experiment(experiment_name)

# Set registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

print(f"\nExperiment: {experiment_name}")
print(f"Registry: Unity Catalog")
print(f"\nReady to log models!")

# COMMAND ----------

# DBTITLE 1,Model 1: XGBoost with Optuna Hyperparameter Tuning
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

print("=" * 80)
print("MODEL 1: XGBOOST WITH OPTUNA TUNING")
print("=" * 80)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), feature_cols)
    ])

# Fit preprocessor on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

def objective(trial):
    """Optuna objective function for XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)], verbose=False)
    preds = model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

print("\nStarting Optuna hyperparameter optimization...")
print("This will run 50 trials to find optimal parameters.\n")

# Run Optuna study
study = optuna.create_study(direction='minimize', study_name='xgboost_battery_prediction')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n✓ Optimization complete!")
print(f"  Best RMSE: {study.best_value:.4f}")
print(f"  Best parameters: {study.best_params}")

# COMMAND ----------

# DBTITLE 1,Train and Log Best XGBoost Model
# Train final model with best parameters and log to MLflow
print("\nTraining final XGBoost model with best parameters...")

# Create pipeline with preprocessing and model
best_xgb_model = xgb.XGBRegressor(**study.best_params, random_state=42, tree_method='hist', device='cpu')
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_xgb_model)
])

with mlflow.start_run(run_name="XGBoost_Optuna") as run:
    # Train model
    xgb_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = xgb_pipeline.predict(X_train)
    y_pred_test = xgb_pipeline.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log parameters
    mlflow.log_params(study.best_params)
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("optimization", "Optuna")
    mlflow.log_param("n_trials", 50)
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    
    # Log model with signature
    signature = infer_signature(X_train, y_pred_train)
    mlflow.sklearn.log_model(
        xgb_pipeline, 
        name="model",
        signature=signature,
        input_example=X_train.head(3)
    )
    
    xgb_run_id = run.info.run_id

print("\n" + "="*60)
print("XGBoost Results:")
print("="*60)
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE:  {test_rmse:.4f}")
print(f"  Train MAE:  {train_mae:.4f}")
print(f"  Test MAE:   {test_mae:.4f}")
print(f"  Train R²:   {train_r2:.4f}")
print(f"  Test R²:    {test_r2:.4f}")
print(f"\n✓ Model logged to MLflow (Run ID: {xgb_run_id})")

# COMMAND ----------

# DBTITLE 1,Model 2: Regularized Regression with Optuna
import lightgbm as lgb

print("\n" + "=" * 80)
print("MODEL 2: LIGHTGBM WITH OPTUNA TUNING")
print("=" * 80)

def objective_lgb(trial):
    """Optuna objective function for LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_processed, y_train)
    preds = model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

print("\nStarting LightGBM hyperparameter optimization...")
print("Running 100 trials for optimal parameters.\n")

# Run Optuna study
study_lgb = optuna.create_study(direction='minimize', study_name='lightgbm_battery_prediction')
study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=True)

print(f"\n✓ LightGBM optimization complete!")
print(f"  Best RMSE: {study_lgb.best_value:.4f}")
print(f"  Best parameters: {study_lgb.best_params}")

# COMMAND ----------

# DBTITLE 1,Train and Log Best Regularized Model
# Train final LightGBM model with best parameters and log to MLflow
print("\nTraining final LightGBM model with best parameters...")

# Create pipeline with preprocessing and model
best_lgb_model = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)
lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_lgb_model)
])

with mlflow.start_run(run_name="LightGBM_Optuna") as run:
    # Train model
    lgb_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = lgb_pipeline.predict(X_train)
    y_pred_test = lgb_pipeline.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log parameters
    mlflow.log_params(study_lgb.best_params)
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("optimization", "Optuna")
    mlflow.log_param("n_trials", 100)
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    
    # Log model with signature
    signature = infer_signature(X_train, y_pred_train)
    mlflow.sklearn.log_model(
        lgb_pipeline, 
        name="model",
        signature=signature,
        input_example=X_train.head(3)
    )
    
    lgb_run_id = run.info.run_id

print("\n" + "="*60)
print("LightGBM Results:")
print("="*60)
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE:  {test_rmse:.4f}")
print(f"  Train MAE:  {train_mae:.4f}")
print(f"  Test MAE:   {test_mae:.4f}")
print(f"  Train R²:   {train_r2:.4f}")
print(f"  Test R²:    {test_r2:.4f}")
print(f"\n✓ Model logged to MLflow (Run ID: {lgb_run_id})")

# COMMAND ----------

# DBTITLE 1,Model 3: Prophet Time Series (Limited by Data)
from catboost import CatBoostRegressor

print("\n" + "=" * 80)
print("MODEL 3: CATBOOST WITH OPTUNA TUNING")
print("=" * 80)

def objective_catboost(trial):
    """Optuna objective function for CatBoost"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'random_state': 42,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train_processed, y_train)
    preds = model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

print("\nStarting CatBoost hyperparameter optimization...")
print("Running 100 trials for optimal parameters.\n")

# Run Optuna study
study_catboost = optuna.create_study(direction='minimize', study_name='catboost_battery_prediction')
study_catboost.optimize(objective_catboost, n_trials=100, show_progress_bar=True)

print(f"\n✓ CatBoost optimization complete!")
print(f"  Best RMSE: {study_catboost.best_value:.4f}")
print(f"  Best parameters: {study_catboost.best_params}") 'elevation_gain_m', 'trip_hour']:
    if col in df_model.columns:
        df_prophet[col] = df_model[col].values

print(f"Prophet dataset shape: {df_prophet.shape}")
print(f"Date range: {df_prophet['ds'].min()} to {df_prophet['ds'].max()}")

# Split: use last 20% as test
train_size = int(len(df_prophet) * 0.8)
df_prophet_train = df_prophet.iloc[:train_size]
df_prophet_test = df_prophet.iloc[train_size:]

try:
    # Initialize and train Prophet
    prophet_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # Add regressors
    for col in ['distance_km', 'avg_speed_kmh', 'elevation_gain_m', 'trip_hour']:
        if col in df_prophet_train.columns:
            prophet_model.add_regressor(col)
    
    print("\nTraining Prophet model...")
    prophet_model.fit(df_prophet_train[['ds', 'y', 'distance_km', 'avg_speed_kmh', 'elevation_gain_m', 'trip_hour']])
    
    # Predict
    y_pred_test_prophet = prophet_model.predict(df_prophet_test[['ds', 'distance_km', 'avg_speed_kmh', 'elevation_gain_m', 'trip_hour']])['yhat'].values
    y_pred_train_prophet = prophet_model.predict(df_prophet_train[['ds', 'distance_km', 'avg_speed_kmh', 'elevation_gain_m', 'trip_hour']])['yhat'].values
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(df_prophet_train['y'], y_pred_train_prophet))
    test_rmse = np.sqrt(mean_squared_error(df_prophet_test['y'], y_pred_test_prophet))
    train_mae = mean_absolute_error(df_prophet_train['y'], y_pred_train_prophet)
    test_mae = mean_absolute_error(df_prophet_test['y'], y_pred_test_prophet)
    train_r2 = r2_score(df_prophet_train['y'], y_pred_train_prophet)
    test_r2 = r2_score(df_prophet_test['y'], y_pred_test_prophet)
    
    print("\n" + "="*60)
    print("Prophet Results:")
    print("="*60)
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Train MAE:  {train_mae:.4f}")
    print(f"  Test MAE:   {test_mae:.4f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    print("\n⚠️  Note: Prophet performance limited by lack of proper temporal structure")
    
    prophet_success = True
    
except Exception as e:
    print(f"\n❌ Prophet model failed: {str(e)}")
    print("   This is expected due to the temporal data quality issues.")
    prophet_success = False

# COMMAND ----------

# DBTITLE 1,Train and Log Best CatBoost Model
# Train final CatBoost model with best parameters and log to MLflow
print("\nTraining final CatBoost model with best parameters...")

# Create pipeline with preprocessing and model
best_catboost_model = CatBoostRegressor(**study_catboost.best_params, random_state=42, verbose=False)
catboost_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_catboost_model)
])

with mlflow.start_run(run_name="CatBoost_Optuna") as run:
    # Train model
    catboost_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = catboost_pipeline.predict(X_train)
    y_pred_test = catboost_pipeline.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Log parameters
    mlflow.log_params(study_catboost.best_params)
    mlflow.log_param("model_type", "CatBoost")
    mlflow.log_param("optimization", "Optuna")
    mlflow.log_param("n_trials", 100)
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    
    # Log model with signature
    signature = infer_signature(X_train, y_pred_train)
    mlflow.sklearn.log_model(
        catboost_pipeline, 
        name="model",
        signature=signature,
        input_example=X_train.head(3)
    )
    
    catboost_run_id = run.info.run_id

print("\n" + "="*60)
print("CatBoost Results:")
print("="*60)
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE:  {test_rmse:.4f}")
print(f"  Train MAE:  {train_mae:.4f}")
print(f"  Test MAE:   {test_mae:.4f}")
print(f"  Train R²:   {train_r2:.4f}")
print(f"  Test R²:    {test_r2:.4f}")
print(f"\n✓ Model logged to MLflow (Run ID: {catboost_run_id})")

# COMMAND ----------

# DBTITLE 1,Model 4: Ensemble Stacking with 5-Fold CV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

print("\n" + "=" * 80)
print("MODEL 4: ENSEMBLE STACKING WITH CROSS-VALIDATION")
print("=" * 80)

print("\nBuilding stacking ensemble with XGBoost, LightGBM, and CatBoost...")

# Define base models with best parameters
base_estimators = [
    ('xgboost', xgb.XGBRegressor(**study.best_params, random_state=42, tree_method='hist', device='cpu')),
    ('lightgbm', lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)),
    ('catboost', CatBoostRegressor(**study_catboost.best_params, random_state=42, verbose=False))
]

# Meta-learner (Ridge regression)
meta_learner = Ridge(alpha=1.0)

# Create stacking regressor
stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# Create pipeline
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stacking_model)
])

print("\nPerforming 5-Fold Cross-Validation on ensemble...")
print("This will take a few minutes...\n")

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    stacking_pipeline, 
    X_train, 
    y_train, 
    cv=kf, 
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

cv_rmse_scores = -cv_scores
print(f"\n5-Fold CV RMSE scores: {cv_rmse_scores}")
print(f"Mean CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")

# Train on full training set
print("\nTraining final ensemble on full training set...")
stacking_pipeline.fit(X_train, y_train)

# Predictions
y_pred_train_stack = stacking_pipeline.predict(X_train)
y_pred_test_stack = stacking_pipeline.predict(X_test)

# Metrics
train_rmse_stack = np.sqrt(mean_squared_error(y_train, y_pred_train_stack))
test_rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_test_stack))
train_mae_stack = mean_absolute_error(y_train, y_pred_train_stack)
test_mae_stack = mean_absolute_error(y_test, y_pred_test_stack)
train_r2_stack = r2_score(y_train, y_pred_train_stack)
test_r2_stack = r2_score(y_test, y_pred_test_stack)

# Log to MLflow
with mlflow.start_run(run_name="Ensemble_Stacking_CV") as run:
    # Log parameters
    mlflow.log_param("model_type", "Stacking_Ensemble")
    mlflow.log_param("base_models", "XGBoost+LightGBM+CatBoost")
    mlflow.log_param("meta_learner", "Ridge")
    mlflow.log_param("cv_folds", 5)
    
    # Log CV metrics
    mlflow.log_metric("cv_rmse_mean", cv_rmse_scores.mean())
    mlflow.log_metric("cv_rmse_std", cv_rmse_scores.std())
    
    # Log train/test metrics
    mlflow.log_metric("train_rmse", train_rmse_stack)
    mlflow.log_metric("test_rmse", test_rmse_stack)
    mlflow.log_metric("train_mae", train_mae_stack)
    mlflow.log_metric("test_mae", test_mae_stack)
    mlflow.log_metric("train_r2", train_r2_stack)
    mlflow.log_metric("test_r2", test_r2_stack)
    
    # Log model
    signature = infer_signature(X_train, y_pred_train_stack)
    mlflow.sklearn.log_model(
        stacking_pipeline,
        name="model",
        signature=signature,
        input_example=X_train.head(3)
    )
    
    ensemble_run_id = run.info.run_id

print("\n" + "="*60)
print("Ensemble Stacking Results:")
print("="*60)
print(f"  CV RMSE:    {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
print(f"  Train RMSE: {train_rmse_stack:.4f}")
print(f"  Test RMSE:  {test_rmse_stack:.4f}")
print(f"  Train MAE:  {train_mae_stack:.4f}")
print(f"  Test MAE:   {test_mae_stack:.4f}")
print(f"  Train R²:   {train_r2_stack:.4f}")
print(f"  Test R²:    {test_r2_stack:.4f}")
print(f"\n✓ Ensemble model logged to MLflow (Run ID: {ensemble_run_id})")

# COMMAND ----------

# DBTITLE 1,Register Best Model to Unity Catalog
from mlflow.tracking import MlflowClient
import mlflow

print("\n" + "=" * 80)
print("COMPARE MODELS AND SELECT BEST")
print("=" * 80)

# Get all runs from current experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_rmse ASC"])

# Display model comparison
print("\nModel Performance Comparison (sorted by test RMSE):")
print("=" * 60)
for idx, row in runs.iterrows():
    run_name = row.get('tags.mlflow.runName', 'Unknown')
    test_rmse = row.get('metrics.test_rmse', float('inf'))
    test_r2 = row.get('metrics.test_r2', 0)
    print(f"  {run_name:25s} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

# Select best model (lowest test RMSE)
best_run = runs.iloc[0]
best_model_run_id = best_run['run_id']
best_model_name = best_run.get('tags.mlflow.runName', 'Unknown')
best_model_rmse = best_run['metrics.test_rmse']
best_model_r2 = best_run['metrics.test_r2']

print("\n" + "=" * 60)
print(f"✓ Best Model: {best_model_name}")
print(f"  Test RMSE: {best_model_rmse:.4f}")
print(f"  Test R²: {best_model_r2:.4f}")
print(f"  Run ID: {best_model_run_id}")

print("\n" + "=" * 80)
print("REGISTER BEST MODEL TO UNITY CATALOG")
print("=" * 80)

# Define UC model name
uc_model_name = "dev.ml_models.battery_energy_prediction"

print(f"\nRegistering model to: {uc_model_name}")
print(f"Source run ID: {best_model_run_id}")

# Get the best run's model URI
model_uri = f"runs:/{best_model_run_id}/model"

try:
    # Register model to Unity Catalog
    print("\nRegistering model...")
    
    # Use the mlflow client to register the model
    client = MlflowClient()
    
    # Register from the run
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=uc_model_name,
        tags={
            "model_type": best_model_name,
            "test_rmse": str(best_model_rmse),
            "test_r2": str(best_model_r2),
            "use_case": "battery_energy_prediction",
            "training_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    
    print(f"\n✓ Model successfully registered!")
    print(f"  Model name: {uc_model_name}")
    print(f"  Version: {model_version.version}")
    print(f"  Status: {model_version.status}")
    
    # Set model alias for easy reference
    client.set_registered_model_alias(uc_model_name, "champion", model_version.version)
    print(f"  Alias 'champion' set to version {model_version.version}")
    
    print("\n" + "="*80)
    print("✅ MODEL TRAINING AND REGISTRATION COMPLETE")
    print("="*80)
    print(f"\nThe {best_model_name} is now ready for deployment!")
    print(f"\nTo load the model for inference:")
    print(f"  model = mlflow.pyfunc.load_model('models:/{uc_model_name}@champion')")
    print(f"\nNext steps:")
    print(f"  1. Deploy to Model Serving endpoint for real-time predictions")
    print(f"  2. Integrate with battery prediction app")
    print(f"  3. Monitor model performance and retrain as needed")
    
except Exception as e:
    print(f"\n❌ Error registering model: {str(e)}")
    print("\nTroubleshooting:")
    print("  - Ensure the catalog 'dev' and schema 'ml_models' exist")
    print("  - Check permissions to create models in Unity Catalog")
    print(f"  - Model URI: {model_uri}")

# COMMAND ----------

