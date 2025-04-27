"""
Script to train and evaluate standard regression models (Linear, Ridge, Lasso) for multi-output prediction.

This provides baseline performance metrics for comparison with more complex models like XGBoost.
Implements multiple standard regression approaches while maintaining the same data pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
INPUT_DIR = "ML_Data/processed_clusters"  # Read from standard processed data
OUTPUT_DIR = "ML_Data/model_results_multioutput/standard_regression"  # Save in a subfolder
ROAD_TO_PROCESS = "M60"

# Multiple targets configuration
TARGETS = ['total_traffic_flow', 'speed_reduction']
TARGET_NAMES = {
    'total_traffic_flow': 'Traffic Flow',
    'speed_reduction': 'Speed Reduction (mph)'
}

# Features for training (including interaction features)
FEATURES = [
    'hour',
    'is_weekend',
    'is_school_holiday',
    'weather_code',
    'is_pre_match',
    'is_post_match',
    'is_united_match',
    'is_city_match',
    'hour_weekend',           # Interaction: hour × is_weekend
    'hour_school_holiday',    # Interaction: hour × is_school_holiday
]

# Model configurations
MODELS = {
    'linear': LinearRegression(),
    'ridge': Ridge(alpha=1.0),  # Default alpha
    'lasso': Lasso(alpha=1.0)   # Default alpha
}

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

def get_feature_dir_name():
    """Create directory name from feature list."""
    return '_'.join(FEATURES)

def get_output_path(direction, model_name):
    """Construct output path."""
    return Path(OUTPUT_DIR) / ROAD_TO_PROCESS / direction / model_name / get_feature_dir_name()

def load_data(direction):
    """Load train, validation, and test data for a specific direction."""
    base_path = Path(INPUT_DIR)
    
    # Construct file paths
    train_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_train.csv"
    val_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_val.csv"
    test_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_test.csv"
    
    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def prepare_features_and_targets(df):
    """Prepare features and multiple targets for training."""
    # Check if all required features and targets exist
    missing_features = [f for f in FEATURES if f not in df.columns]
    missing_targets = [t for t in TARGETS if t not in df.columns]
    
    if missing_features:
        raise ValueError(f"Missing features in data: {', '.join(missing_features)}")
    if missing_targets:
        raise ValueError(f"Missing targets in data: {', '.join(missing_targets)}")
    
    # Extract features and targets
    X = df[FEATURES].copy()
    y = df[TARGETS].copy()
    
    return X, y

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model performance with detailed metrics."""
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics for each target
    metrics = {}
    for i, target in enumerate(TARGETS):
        metrics[target] = {
            'rmse': np.sqrt(mean_squared_error(y[target], y_pred[:, i])),
            'mae': mean_absolute_error(y[target], y_pred[:, i]),
            'r2': r2_score(y[target], y_pred[:, i]),
            'predictions': y_pred[:, i]
        }
    
    # Print metrics
    print(f"\n{Colors.BLUE}{dataset_name} Metrics:{Colors.RESET}")
    for target in TARGETS:
        print(f"  {target}:")
        print(f"    RMSE: {metrics[target]['rmse']:.2f}")
        print(f"    MAE:  {metrics[target]['mae']:.2f}")
        print(f"    R²:   {metrics[target]['r2']:.3f}")
    
    return metrics

def plot_actual_vs_predicted(y_true, y_pred, output_path, target_name):
    """Create actual vs predicted scatter plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted - {target_name}')
    plt.tight_layout()
    plt.savefig(output_path / f'actual_vs_predicted_{target_name.lower().replace(" ", "_")}.png')
    plt.close()

def train_and_evaluate_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, output_path):
    """Train and evaluate a specific model."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"\n{Colors.CYAN}{timestamp} Training {model_name.upper()} model...{Colors.RESET}")
    
    # Create and train model
    model = MultiOutputRegressor(MODELS[model_name])
    model.fit(X_train, y_train)
    
    # Evaluate on all sets
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Create plots directory
    plots_dir = output_path / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Create plots for each target
    for target in TARGETS:
        plot_actual_vs_predicted(
            y_test[target],
            test_metrics[target]['predictions'],
            plots_dir,
            TARGET_NAMES[target]
        )
    
    # Save metrics
    metrics = {
        'train': {target: {k: v for k, v in m.items() if k != 'predictions'} 
                 for target, m in train_metrics.items()},
        'validation': {target: {k: v for k, v in m.items() if k != 'predictions'} 
                      for target, m in val_metrics.items()},
        'test': {target: {k: v for k, v in m.items() if k != 'predictions'} 
                for target, m in test_metrics.items()}
    }
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"{Colors.GREEN}✓ Results saved to {output_path}{Colors.RESET}")
    
    return model, metrics

def process_direction(direction):
    """Process one direction with all regression models."""
    print(f"\n{Colors.MAGENTA}Processing {ROAD_TO_PROCESS} {direction.upper()} for Standard Regression{Colors.RESET}")
    print(f"{Colors.BLUE}Models: {', '.join(MODELS.keys())}{Colors.RESET}")
    print("-" * 50)
    
    try:
        # Load data
        train_df, val_df, test_df = load_data(direction)
        X_train, y_train = prepare_features_and_targets(train_df)
        X_val, y_val = prepare_features_and_targets(val_df)
        X_test, y_test = prepare_features_and_targets(test_df)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate each model
        results = {}
        for model_name in MODELS:
            # Create output directory
            output_path = get_output_path(direction, model_name)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Train and evaluate
            model, metrics = train_and_evaluate_model(
                model_name,
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
                output_path
            )
            results[model_name] = metrics
        
        return results
    
    except Exception as e:
        print(f"{Colors.RED}Error processing {direction}: {str(e)}{Colors.RESET}")
        return None

def main():
    """Main function to train and evaluate standard regression models."""
    print(f"{Colors.BOLD}Starting Standard Regression Model Training and Evaluation{Colors.RESET}")
    print("=" * 70)
    
    # Get available directions
    base_path = Path(INPUT_DIR)
    direction_files = list(base_path.glob(f"{ROAD_TO_PROCESS}_*_cluster_train.csv"))
    
    if not direction_files:
        print(f"{Colors.RED}No processed files found for {ROAD_TO_PROCESS} in {INPUT_DIR}{Colors.RESET}")
        return
    
    directions = sorted(str(f.name).split('_')[1] for f in direction_files)
    print(f"{Colors.BLUE}Found directions: {', '.join(directions)}{Colors.RESET}")
    
    # Process each direction
    all_results = {}
    for direction in directions:
        results = process_direction(direction)
        if results:
            all_results[direction] = results
    
    # Print final summary
    print(f"\n{Colors.MAGENTA}=== Standard Regression Summary ==={Colors.RESET}")
    for direction, results in all_results.items():
        print(f"\n{Colors.BLUE}{direction.upper()}{Colors.RESET}")
        for model_name, metrics in results.items():
            print(f"\n  {model_name.upper()}:")
            for target in TARGETS:
                test_metrics = metrics['test'][target]
                print(f"    {target}:")
                print(f"      RMSE: {test_metrics['rmse']:.2f}")
                print(f"      R²:   {test_metrics['r2']:.3f}")

if __name__ == "__main__":
    main() 