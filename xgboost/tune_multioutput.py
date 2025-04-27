"""
Script to tune hyperparameters for multi-output XGBoost model.

This script tunes hyperparameters for a multi-output regression model that predicts 
both total_traffic_flow and speed_reduction simultaneously.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# Configuration
INPUT_DIR = "ML_Data/processed_clusters"  # Read from standard processed data
OUTPUT_DIR = "ML_Data/model_results_multioutput"  # Save results to a new directory
ROAD_TO_PROCESS = "M60"

# New configuration for shared parameter tuning
TUNE_SINGLE_DIRECTION = True  # Set to True to tune only one direction and share parameters
TUNE_DIRECTION = "clockwise"  # The direction to use for tuning when TUNE_SINGLE_DIRECTION=True

# Multiple targets configuration
TARGETS = ['total_traffic_flow', 'speed_reduction']

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

# Parameter grid for tuning
PARAM_GRID = {
    'estimator__max_depth': [3, 5, 7],        # Tree depth
    'estimator__learning_rate': [0.01, 0.1],  # Learning rate
}

# Base XGBoost parameters
BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'n_estimators': 300,             # Fixed value since we use early stopping
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
    return '_'.join(FEATURES)

def get_output_path(direction):
    """Construct output path."""
    # When using shared parameters, save to a special "shared_params" directory
    if TUNE_SINGLE_DIRECTION and direction != TUNE_DIRECTION:
        # For non-tuning directions, point to the shared parameters
        direction_part = "shared_params"
    else:
        direction_part = direction
    
    # Create path: road/direction/multioutput_prediction/feature_combo
    return Path(OUTPUT_DIR) / ROAD_TO_PROCESS / direction_part / "multioutput_prediction" / get_feature_dir_name()

def load_data(direction):
    """Load train and validation data for a specific direction."""
    base_path = Path(INPUT_DIR)
    
    # Construct file paths
    train_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_train.csv"
    val_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_val.csv"
    
    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    return train_df, val_df

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

def tune_multioutput_hyperparameters(X_train, y_train_multioutput, X_val, y_val_multioutput):
    """Perform hyperparameter tuning using GridSearchCV with MultiOutputRegressor."""
    print(f"\n{Colors.CYAN}⚡ Tuning hyperparameters for multi-output model...{Colors.RESET}")
    
    # Combine train and validation for cross-validation
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train_multioutput, y_val_multioutput])
    
    # Create base XGBoost regressor
    # Note: early_stopping_rounds can't be used with MultiOutputRegressor + GridSearchCV
    tuning_params = BASE_PARAMS.copy()
    if 'early_stopping_rounds' in tuning_params:
        del tuning_params['early_stopping_rounds']
    
    base_model = xgb.XGBRegressor(**tuning_params)
    
    # Wrap with MultiOutputRegressor
    multi_output_model = MultiOutputRegressor(base_model)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=multi_output_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring='neg_root_mean_squared_error',  # This will be averaged across all targets
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    print(f"{Colors.CYAN}This may take a while as MultiOutputRegressor trains a separate model for each target...{Colors.RESET}")
    grid_search.fit(X_combined, y_combined)
    
    print(f"{Colors.GREEN}✓ Best RMSE (averaged): {-grid_search.best_score_:.2f}{Colors.RESET}")
    
    # The parameters are prefixed with 'estimator__' due to MultiOutputRegressor
    # Remove the prefix to make parameters compatible with standard XGBoost
    best_params = {}
    for key, value in grid_search.best_params_.items():
        if key.startswith('estimator__'):
            # Extract the parameter name without the 'estimator__' prefix
            param_name = key.split('estimator__')[1]
            best_params[param_name] = value
        else:
            best_params[key] = value
    
    return best_params

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
    """Process one direction with hyperparameter tuning for multi-output regression."""
    print(f"\n{Colors.MAGENTA}Processing {ROAD_TO_PROCESS} {direction.upper()} for Multi-Output Regression{Colors.RESET}")
    print(f"{Colors.BLUE}Targets: {', '.join(TARGETS)}{Colors.RESET}")
    print(f"{Colors.BLUE}Features: {', '.join(FEATURES)}{Colors.RESET}")
    print("-" * 50)
    
    try:
        # Load data
        print(f"{Colors.CYAN}⚡ Loading data...{Colors.RESET}")
        train_df, val_df = load_data(direction)
        X_train, y_train = prepare_features_and_targets(train_df)
        X_val, y_val = prepare_features_and_targets(val_df)
        
        print(f"{Colors.GREEN}✓ Data loaded with shapes:{Colors.RESET}")
        print(f"{'':>4}X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"{'':>4}X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # Create output directory
        output_path = get_output_path(direction)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature and target lists for reference
        with open(output_path / 'model_info.txt', 'w') as f:
            f.write(f"Targets: {', '.join(TARGETS)}\n")
            f.write("Features:\n")
            for feature in FEATURES:
                f.write(f"- {feature}\n")
        
        # Tune hyperparameters
        best_params = tune_multioutput_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Save best parameters
        params_file = output_path / 'best_params.json'
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print(f"{Colors.GREEN}✓ Best parameters saved to {params_file}{Colors.RESET}")
        
        return best_params
    
    except Exception as e:
        print(f"{Colors.RED}Error processing {direction}: {str(e)}{Colors.RESET}")
        return None

def main():
    """Main function to tune hyperparameters for multi-output regression for all directions."""
    print(f"{Colors.BOLD}Starting Multi-Output XGBoost Hyperparameter Tuning for {ROAD_TO_PROCESS}{Colors.RESET}")
    print("=" * 70)
    
    directions = get_available_directions()
    if not directions:
        print(f"{Colors.RED}No valid directions found for {ROAD_TO_PROCESS}. Exiting.{Colors.RESET}")
        return
    
    print(f"{Colors.BLUE}Found directions: {', '.join(directions)}{Colors.RESET}")
    
    if TUNE_SINGLE_DIRECTION:
        if TUNE_DIRECTION not in directions:
            print(f"{Colors.RED}Specified tuning direction '{TUNE_DIRECTION}' not found. Available directions: {', '.join(directions)}{Colors.RESET}")
            return
        
        print(f"{Colors.CYAN}Using single-direction tuning with '{TUNE_DIRECTION}' as the reference.{Colors.RESET}")
        
        # Only tune the specified direction
        best_params = process_direction(TUNE_DIRECTION)
        
        if best_params:
            # Create shared parameters directory
            shared_path = Path(OUTPUT_DIR) / ROAD_TO_PROCESS / "shared_params" / "multioutput_prediction" / get_feature_dir_name()
            shared_path.mkdir(parents=True, exist_ok=True)
            
            # Save the parameters to the shared location
            params_file = shared_path / 'best_params.json'
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
            
            print(f"{Colors.GREEN}✓ Shared parameters saved to {params_file}{Colors.RESET}")
            
            # Print summary
            print(f"\n{Colors.MAGENTA}=== Multi-Output Tuning Summary (Shared Parameters) ==={Colors.RESET}")
            print(f"\n{Colors.BLUE}Tuned using {TUNE_DIRECTION.upper()}{Colors.RESET}")
            print(f"{'':>4}Targets: {', '.join(TARGETS)}")
            print(f"{'':>4}Features: {', '.join(FEATURES)}")
            for param, value in best_params.items():
                print(f"{'':>4}{param}: {value}")
    else:
        # Original behavior: tune each direction separately
        results = {}
        for direction in directions:
            best_params = process_direction(direction)
            results[direction] = best_params
        
        # Print summary
        print(f"\n{Colors.MAGENTA}=== Multi-Output Tuning Summary ==={Colors.RESET}")
        for direction, params in results.items():
            if params:
                print(f"\n{Colors.BLUE}{direction.upper()}{Colors.RESET}")
                print(f"{'':>4}Targets: {', '.join(TARGETS)}")
                print(f"{'':>4}Features: {', '.join(FEATURES)}")
                for param, value in params.items():
                    print(f"{'':>4}{param}: {value}")

if __name__ == "__main__":
    main() 