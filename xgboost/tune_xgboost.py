#LITTLE ONE


import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

# Configuration
INPUT_DIR = "ML_Data/processed_clusters"
OUTPUT_DIR = "ML_Data/model_results_tuned"
ROAD_TO_PROCESS = "M60"

# New configuration for shared parameter tuning
TUNE_SINGLE_DIRECTION = True  # Set to True to tune only one direction and share parameters
TUNE_DIRECTION = "clockwise"  # The direction to use for tuning when TUNE_SINGLE_DIRECTION=True

# Features and target configuration
TARGET = 'total_traffic_flow'  # Can be 'journey_delay' or 'traffic_flow_value'

# Features for training - Updated to include match features and interaction features
FEATURES = [
    'hour',           # Time of day
    'is_weekend',     # NEW: Binary flag (weekend/bank holiday or not)
    'is_school_holiday', # NEW: Binary flag (school holiday/Xmas or not)
    'weather_code',   # Weather conditions
    'is_pre_match',    # Binary: 1 if in pre-match window, 0 otherwise
    'is_post_match',   # Binary: 1 if in post-match window, 0 otherwise
    'is_united_match', # Binary: 1 if Man United match, 0 otherwise
    'is_city_match',   # Binary: 1 if Man City match, 0 otherwise
    'hour_weekend',    # Interaction: hour × is_weekend
    'hour_school_holiday', # Interaction: hour × is_school_holiday
    # Optional: 'is_no_match_period', 'is_no_match_team' (if not dropped in B3)
]

# Parameter grid for tuning
PARAM_GRID = {
    'max_depth': [3, 5, 7],          # Tree depth - most important parameter
    'learning_rate': [0.01, 0.1],    # Learning rate - second most important
}

# Base XGBoost parameters (parameters not being tuned)
BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'early_stopping_rounds': 50,
    # Set reasonable default values for other parameters
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
    """Construct output path including target and features."""
    # When using shared parameters, save to a special "shared_params" directory
    if TUNE_SINGLE_DIRECTION and direction != TUNE_DIRECTION:
        # For non-tuning directions, point to the shared parameters
        direction_part = "shared_params"
    else:
        direction_part = direction
    
    # Create path: road/direction/target_prediction/feature_combo
    return Path(OUTPUT_DIR) / ROAD_TO_PROCESS / direction_part / f"{TARGET}_prediction" / get_feature_dir_name()

def load_data(direction):
    """Load train and validation data for a specific direction."""
    base_path = Path(INPUT_DIR)
    
    # Construct file paths
    train_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_train.csv"
    val_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_val.csv"
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    return train_df, val_df

def prepare_features(df):
    """Prepare features for training."""
    X = df[FEATURES].copy()
    y = df[TARGET]
    return X, y

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Perform hyperparameter tuning using GridSearchCV."""
    print(f"\n{Colors.CYAN}⚡ Tuning hyperparameters...{Colors.RESET}")
    
    # Combine train and validation for cross-validation
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    # Use base params *without* early stopping for GridSearchCV
    tuning_params = BASE_PARAMS.copy()
    if 'early_stopping_rounds' in tuning_params:
        del tuning_params['early_stopping_rounds']
        
    # Create XGBoost regressor for tuning
    model = xgb.XGBRegressor(**tuning_params)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_combined, y_combined)
    
    print(f"{Colors.GREEN}✓ Best RMSE: {-grid_search.best_score_:.2f}{Colors.RESET}")
    
    return grid_search.best_params_

def get_available_directions():
    """Detect available directions for the road from the processed files."""
    base_path = Path(INPUT_DIR)
    direction_files = list(base_path.glob(f"{ROAD_TO_PROCESS}_*_cluster_train.csv"))
    
    if not direction_files:
        print(f"{Colors.RED}No processed files found for {ROAD_TO_PROCESS}{Colors.RESET}")
        return []
    
    directions = []
    for file in direction_files:
        direction = str(file.name).split('_')[1]
        directions.append(direction)
    
    return sorted(directions)

def process_direction(direction):
    """Process one direction with hyperparameter tuning."""
    print(f"\n{Colors.MAGENTA}Processing {ROAD_TO_PROCESS} {direction.upper()}{Colors.RESET}")
    print(f"{Colors.BLUE}Target: {TARGET}{Colors.RESET}")
    print(f"{Colors.BLUE}Features: {', '.join(FEATURES)}{Colors.RESET}")
    print("-" * 50)
    
    print(f"{Colors.CYAN}⚡ Loading data...{Colors.RESET}")
    train_df, val_df = load_data(direction)
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    print(f"{Colors.GREEN}✓ Data loaded{Colors.RESET}")
    
    # Create output directory with feature combination name
    output_path = get_output_path(direction)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature list for reference
    with open(output_path / 'feature_list.txt', 'w') as f:
        f.write(f"Target: {TARGET}\n")
        f.write("Features:\n")
        for feature in FEATURES:
            f.write(f"- {feature}\n")
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Save best parameters
    params_file = output_path / 'best_params.json'
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"{Colors.GREEN}✓ Best parameters saved to {params_file}{Colors.RESET}")
    
    return best_params

def main():
    """Main function to tune hyperparameters for all directions."""
    print(f"{Colors.BOLD}Starting XGBoost Hyperparameter Tuning for {ROAD_TO_PROCESS}{Colors.RESET}")
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
            shared_path = Path(OUTPUT_DIR) / ROAD_TO_PROCESS / "shared_params" / f"{TARGET}_prediction" / get_feature_dir_name()
            shared_path.mkdir(parents=True, exist_ok=True)
            
            # Save the parameters to the shared location
            params_file = shared_path / 'best_params.json'
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
            
            print(f"{Colors.GREEN}✓ Shared parameters saved to {params_file}{Colors.RESET}")
            
            # Print summary
            print(f"\n{Colors.MAGENTA}=== Tuning Summary (Shared Parameters) ==={Colors.RESET}")
            print(f"\n{Colors.BLUE}Tuned using {TUNE_DIRECTION.upper()}{Colors.RESET}")
            print(f"{'':>4}Target: {TARGET}")
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
        print(f"\n{Colors.MAGENTA}=== Tuning Summary ==={Colors.RESET}")
        for direction, params in results.items():
            print(f"\n{Colors.BLUE}{direction.upper()}{Colors.RESET}")
            print(f"{'':>4}Target: {TARGET}")
            print(f"{'':>4}Features: {', '.join(FEATURES)}")
            for param, value in params.items():
                print(f"{'':>4}{param}: {value}")

if __name__ == "__main__":
    main() 