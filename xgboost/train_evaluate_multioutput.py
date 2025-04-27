#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#THIS RUNS THE FULL THING


"""
Script to train and evaluate multi-output XGBoost model.

This script trains and evaluates a multi-output regression model that predicts 
both total_traffic_flow and speed_reduction simultaneously.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# Configuration
INPUT_DIR = "ML_Data/processed_clusters"  # Read from standard processed data
OUTPUT_DIR = "Output/ML"  # Changed to new output directory
ROAD_TO_PROCESS = "M60"

# Output naming configuration
OUTPUT_NAME = "all_3_targets"  # Match the name in tune_multioutput_optuna.py
USE_FEATURE_DIR_NAME = False

# Multiple targets configuration
TARGETS = ['total_traffic_flow', 'speed_reduction', 'normalized_time']
TARGET_NAMES = {
    'total_traffic_flow': 'Traffic Flow',
    'speed_reduction': 'Speed Reduction (mph)',
    'normalized_time': 'Journey Time (min/mile)'  # Added new target name
}

# Features for training (including interaction features)
FEATURES = [
    'hour',
    'is_weekend',
    'is_school_holiday',
    #'weather_code',
    #'is_pre_match',
    #'is_post_match',
    #'is_united_match',
    #'is_city_match',
    'hour_weekend',           # Interaction: hour × is_weekend
    'hour_school_holiday',    # Interaction: hour × is_school_holiday
]

# Base XGBoost parameters
BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,
    # 'early_stopping_rounds': 50,  # Not compatible with MultiOutputRegressor
    'n_estimators': 300,             # Fixed value since we can't use early stopping
    'min_child_weight': 3,           # Default value
    'gamma': 0,                      # Default value
    'subsample': 0.9,                # Slight subsampling to prevent overfitting
    'colsample_bytree': 0.9          # Slight feature sampling to prevent overfitting
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
    if USE_FEATURE_DIR_NAME:
        # Only use active (non-commented) features
        active_features = [f for f in FEATURES if not f.startswith('#')]
        return '_'.join(active_features)
    else:
        return OUTPUT_NAME

def get_output_path(direction):
    """Construct output path."""
    # Simplified path: ML_Output/direction/output_name
    return Path(OUTPUT_DIR) / direction / get_feature_dir_name()

def get_shared_params_path():
    """Construct path for shared parameters."""
    # Simplified path: Output/shared_params/output_name.json
    return Path(OUTPUT_DIR) / "shared_params" / f"{get_feature_dir_name()}.json"

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
    """Prepare features and multiple targets for training/evaluation."""
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

def train_multioutput_model(X_train, y_train_multioutput, X_val, y_val_multioutput, best_params):
    """Train multi-output XGBoost model with best parameters."""
    print(f"{Colors.CYAN}⚡ Training multi-output model...{Colors.RESET}")
    
    # Create base XGBoost regressor with best parameters from tuning
    final_params = {**BASE_PARAMS, **best_params}
    
    # Remove early_stopping_rounds as it's not compatible with MultiOutputRegressor
    if 'early_stopping_rounds' in final_params:
        del final_params['early_stopping_rounds']
    
    # Create base model
    base_model = xgb.XGBRegressor(**final_params)
    
    # Create MultiOutputRegressor
    model = MultiOutputRegressor(base_model)
    
    # Train the model
    # Note: MultiOutputRegressor doesn't support eval_set directly
    # It trains a separate model for each target
    model.fit(X_train, y_train_multioutput)
    
    print(f"{Colors.GREEN}✓ Model training complete{Colors.RESET}")
    
    return model

def evaluate_multioutput_model(model, X, y_multioutput, dataset_name):
    """Evaluate multi-output model with detailed metrics for each target."""
    # Make predictions
    y_pred_multioutput = model.predict(X)
    
    # Initialize dictionaries to store metrics
    metrics = {
        'rmse': {},
        'mae': {},
        'r2': {},
        'predictions': {},
        'residuals': {}
    }
    
    print(f"\n{Colors.BLUE}{dataset_name} Metrics:{Colors.RESET}")
    
    # Calculate metrics for each target
    for i, target in enumerate(TARGETS):
        y_true = y_multioutput[target]
        y_pred = y_pred_multioutput[:, i]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Store metrics
        metrics['rmse'][target] = rmse
        metrics['mae'][target] = mae
        metrics['r2'][target] = r2
        metrics['predictions'][target] = y_pred
        metrics['residuals'][target] = residuals
        
        print(f"\n{Colors.CYAN}{TARGET_NAMES[target]}:{Colors.RESET}")
        print(f"{'':>4}RMSE: {rmse:.2f}")
        print(f"{'':>4}MAE:  {mae:.2f}")
        print(f"{'':>4}R²:   {r2:.3f}")
    
    return metrics

def get_available_directions():
    """Detect available directions for the road from the processed files."""
    base_path = Path(INPUT_DIR)
    direction_files = list(base_path.glob(f"{ROAD_TO_PROCESS}_*_cluster_train.csv"))
    
    if not direction_files:
        print(f"{Colors.RED}No processed files found for {ROAD_TO_PROCESS} in {INPUT_DIR}{Colors.RESET}")
        return []
    
    directions = []
    for file in direction_files:
        # Extract direction from filename (e.g., M60_clockwise_cluster_train.csv -> clockwise)
        direction = str(file.name).split('_')[1]
        directions.append(direction)
    
    return sorted(directions)

def process_direction(direction):
    """Process one direction with training and evaluation for multi-output regression."""
    print(f"\n{Colors.MAGENTA}Processing {ROAD_TO_PROCESS} {direction.upper()} for Multi-Output Regression{Colors.RESET}")
    print(f"{Colors.BLUE}Targets: {', '.join(TARGETS)}{Colors.RESET}")
    active_features = [f for f in FEATURES if not f.startswith('#')]
    print(f"{Colors.BLUE}Active Features: {', '.join(active_features)}{Colors.RESET}")
    print("-" * 50)
    
    try:
        # Load data
        print(f"{Colors.CYAN}⚡ Loading data...{Colors.RESET}")
        train_df, val_df, test_df = load_data(direction)
        X_train, y_train = prepare_features_and_targets(train_df)
        X_val, y_val = prepare_features_and_targets(val_df)
        X_test, y_test = prepare_features_and_targets(test_df)
        
        print(f"{Colors.GREEN}✓ Data loaded with shapes:{Colors.RESET}")
        print(f"{'':>4}X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"{'':>4}X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"{'':>4}X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Create output directory for results
        output_path = get_output_path(direction)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Look for shared parameters
        shared_params_path = get_shared_params_path()
        print(f"\n{Colors.CYAN}Looking for parameters in:{Colors.RESET}")
        print(f"{'':>4}Shared parameters: {shared_params_path}")
        
        # Load parameters - exit if not found
        if not shared_params_path.exists():
            print(f"\n{Colors.RED}ERROR: No tuned parameters found at {shared_params_path}{Colors.RESET}")
            print(f"{Colors.RED}Please run tune_multioutput_optuna.py first to generate parameters.{Colors.RESET}")
            return None
            
        # Load the parameters
        with open(shared_params_path, 'r') as f:
            best_params = json.load(f)
        print(f"{Colors.GREEN}✓ Successfully loaded parameters from: {shared_params_path}{Colors.RESET}")
        print(f"{Colors.CYAN}Parameters: {best_params}{Colors.RESET}")
        
        # Train model
        model = train_multioutput_model(X_train, y_train, X_val, y_val, best_params)
        
        # Evaluate model
        train_metrics = evaluate_multioutput_model(model, X_train, y_train, "Training")
        val_metrics = evaluate_multioutput_model(model, X_val, y_val, "Validation")
        test_metrics = evaluate_multioutput_model(model, X_test, y_test, "Test")
        
        # Save metrics
        all_metrics = {
            'train': {target: {k: float(train_metrics[k][target]) 
                            for k in ['rmse', 'mae', 'r2']}
                    for target in TARGETS},
            'validation': {target: {k: float(val_metrics[k][target]) 
                                for k in ['rmse', 'mae', 'r2']}
                        for target in TARGETS},
            'test': {target: {k: float(test_metrics[k][target]) 
                          for k in ['rmse', 'mae', 'r2']}
                  for target in TARGETS},
            # Add target info for visualization
            'targets': [
                {'name': t, 'display_name': TARGET_NAMES[t]} for t in TARGETS
            ]
        }
        
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        print(f"\n{Colors.CYAN}Metrics Summary:{Colors.RESET}")
        for target in TARGETS:
            print(f"\n{Colors.BLUE}Target: {TARGET_NAMES[target]}{Colors.RESET}")
            print(f"{'':>4}Test RMSE: {test_metrics['rmse'][target]:.2f}")
            print(f"{'':>4}Test MAE:  {test_metrics['mae'][target]:.2f}")
            print(f"{'':>4}Test R²:   {test_metrics['r2'][target]:.3f}")
        
        # === Visualization code removed here ===
        # Instead, save all outputs needed for visualization
        # Save X_train, X_test, y_test, y_pred_test as CSV or npy
        X_train.to_csv(output_path / 'X_train.csv', index=False)
        X_test.to_csv(output_path / 'X_test.csv', index=False)
        y_test.to_csv(output_path / 'y_test.csv', index=False)
        y_pred_test = model.predict(X_test)
        np.save(output_path / 'y_pred_test.npy', y_pred_test)
        # Add a comment: Visualization should be done in a separate script
        print(f"{Colors.GREEN}✓ Data for visualization saved to {output_path}{Colors.RESET}")

        # === Save trained model and statistics for dashboard integration ===
        def save_multioutput_xgb_models(model, output_path):
            for i, estimator in enumerate(model.estimators_):
                target = TARGETS[i]
                model_file = output_path / f"model_{target}.json"
                estimator.save_model(str(model_file))
                print(f"{Colors.CYAN}Saved XGBoost model for target '{target}' to {model_file}{Colors.RESET}")

        def save_feature_stats(X_train, output_path):
            stats = {}
            for col in X_train.columns:
                stats[col] = {
                    'min': float(X_train[col].min()),
                    'max': float(X_train[col].max()),
                    'mean': float(X_train[col].mean()),
                    'std': float(X_train[col].std())
                }
            stats_file = output_path / 'feature_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"{Colors.CYAN}Saved feature statistics to {stats_file}{Colors.RESET}")

        def save_prediction_stats(y_true, y_pred, output_path):
            stats = {}
            for i, target in enumerate(TARGETS):
                pred = y_pred[:, i]
                stats[target] = {
                    'min': float(np.min(pred)),
                    'max': float(np.max(pred)),
                    'mean': float(np.mean(pred)),
                    'std': float(np.std(pred)),
                    'p05': float(np.percentile(pred, 5)),
                    'p95': float(np.percentile(pred, 95))
                }
            stats_file = output_path / 'prediction_ranges.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"{Colors.CYAN}Saved prediction statistics to {stats_file}{Colors.RESET}")

        # Save models
        save_multioutput_xgb_models(model, output_path)
        # Save feature stats
        save_feature_stats(X_train, output_path)
        # Save prediction stats (using test predictions)
        save_prediction_stats(y_test, y_pred_test, output_path)
        # === END ===

        # Return metrics summary
        return {direction: {'test': test_metrics}}
    
    except Exception as e:
        print(f"{Colors.RED}Error processing {direction}: {str(e)}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to train and evaluate multi-output regression for all directions."""
    print(f"{Colors.BOLD}Starting Multi-Output XGBoost Training and Evaluation for {ROAD_TO_PROCESS}{Colors.RESET}")
    print("=" * 70)
    
    directions = get_available_directions()
    if not directions:
        print(f"{Colors.RED}No valid directions found for {ROAD_TO_PROCESS}. Exiting.{Colors.RESET}")
        return
    
    print(f"{Colors.BLUE}Found directions: {', '.join(directions)}{Colors.RESET}")
    results = {}
    
    for direction in directions:
        metrics = process_direction(direction)
        if metrics:
            results.update(metrics)
    
    print(f"\n{Colors.MAGENTA}=== Multi-Output Evaluation Summary ==={Colors.RESET}")
    for direction, direction_metrics in results.items():
        print(f"\n{Colors.BLUE}{direction.upper()}{Colors.RESET}")
        for target in TARGETS:
            if 'test' in direction_metrics and target in direction_metrics['test']:
                target_metrics = direction_metrics['test'][target]
                print(f"{'':>4}Target: {TARGET_NAMES[target]}")
                print(f"{'':>4}Test R²: {target_metrics['r2']:.3f}")
                print(f"{'':>4}Test RMSE: {target_metrics['rmse']:.2f}")
                print(f"{'':>4}Test MAE: {target_metrics['mae']:.2f}")

if __name__ == "__main__":
    main() 