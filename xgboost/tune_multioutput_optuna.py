"""
Script to tune hyperparameters for multi-output XGBoost model using Optuna.

This script uses Bayesian optimization to find optimal parameters for a multi-output 
regression model that predicts both total_traffic_flow and speed_reduction simultaneously.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import optuna
from tqdm import tqdm
import logging
from datetime import datetime

# Silence Optuna info logging
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)
# (Optional) Silence XGBoost logs further
os.environ['XGBOOST_VERBOSITY'] = '0'

# Configuration
INPUT_DIR = "ML_Data/processed_clusters"  # Read from standard processed data
OUTPUT_DIR = "Output/ML"  # Changed to new output directory
ROAD_TO_PROCESS = "M60"

# Output naming configuration
OUTPUT_NAME = "all_3_targets"  # Change this to name your output directory
USE_FEATURE_DIR_NAME = False  # Set to True to use old behavior (directory named after features)

# Optimization configuration
N_TRIALS = 50  # Number of optimization trials
TIMEOUT = None  # Optionally set timeout in seconds (None for no timeout)

# New configuration for shared parameter tuning
TUNE_SINGLE_DIRECTION = True  # Set to True to tune only one direction and share parameters
TUNE_DIRECTION = "clockwise"  # The direction to use for tuning when TUNE_SINGLE_DIRECTION=True

# Multiple targets configuration
TARGETS = ['total_traffic_flow', 'speed_reduction', 'normalized_time']
TARGET_NAMES = {
    'total_traffic_flow': 'Traffic Flow',
    'speed_reduction': 'Speed Reduction (mph)',
    'normalized_time': 'Journey Time (min/mile)'
}

# Target-specific scaling factors (to normalize RMSE across different scales)
TARGET_SCALE_HINTS = {
    'total_traffic_flow': 1000,  # typical flow values are in hundreds/thousands
    'speed_reduction': 10,       # typical speed reductions are 0-30
    'normalized_time': 1         # typical journey times are 1-5 minutes/mile
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

# Base XGBoost parameters (parameters not being tuned)
BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,
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

def create_objective(X_train, y_train, X_val, y_val):
    """Create Optuna objective function for hyperparameter optimization."""
    
    # Calculate standard deviation of each target for normalization
    target_stds = {target: y_val[target].std() for target in TARGETS}
    
    # Store initial values for progress tracking
    initial_rmse = {}
    for i, target in enumerate(TARGETS):
        # Train a simple model to get baseline RMSE
        simple_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        simple_model.fit(X_train, y_train[target])
        y_pred = simple_model.predict(X_val)
        initial_rmse[target] = np.sqrt(mean_squared_error(y_val[target], y_pred))
    
    def objective(trial):
        # Suggest values for hyperparameters
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }
        
        # Combine with base parameters
        final_params = {**BASE_PARAMS, **param}
        final_params['verbosity'] = 0
        final_params['n_jobs'] = 1
        
        # Create base model
        base_model = xgb.XGBRegressor(**final_params)
        
        # Create and train multi-output model
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        
        # Predict and calculate normalized RMSE for each target
        y_pred = model.predict(X_val)
        
        # Calculate raw and normalized RMSE for each target
        rmse_scores = {}
        normalized_rmse_scores = {}
        relative_improvement = {}
        
        for i, target in enumerate(TARGETS):
            raw_rmse = np.sqrt(mean_squared_error(y_val[target], y_pred[:, i]))
            # Normalize by both standard deviation and scale hint
            normalized_rmse = (raw_rmse / target_stds[target]) / TARGET_SCALE_HINTS[target]
            rmse_scores[target] = raw_rmse
            normalized_rmse_scores[target] = normalized_rmse
            
            # Calculate improvement over initial RMSE
            improvement = (initial_rmse[target] - raw_rmse) / initial_rmse[target] * 100
            relative_improvement[target] = improvement
        
        # Store raw RMSE scores and improvements as trial user attributes for logging
        trial.set_user_attr('raw_rmse_scores', rmse_scores)
        trial.set_user_attr('improvements', relative_improvement)
        
        # Return average normalized RMSE (this is what Optuna will minimize)
        avg_normalized_rmse = np.mean(list(normalized_rmse_scores.values()))
        return avg_normalized_rmse
    
    return objective

def optimize_hyperparameters(direction):
    """Run Optuna optimization for hyperparameters."""
    print(f"\n{Colors.MAGENTA}Processing {ROAD_TO_PROCESS} {direction.upper()} for Multi-Output Regression{Colors.RESET}")
    print(f"{Colors.BLUE}Optimizing for {len(TARGETS)} targets: {', '.join(TARGETS)}{Colors.RESET}")
    
    try:
        # Load data
        train_df, val_df = load_data(direction)
        X_train, y_train = prepare_features_and_targets(train_df)
        X_val, y_val = prepare_features_and_targets(val_df)
        
        # Create shared parameters directory
        shared_path = Path(OUTPUT_DIR) / "shared_params"
        shared_path.mkdir(parents=True, exist_ok=True)
        
        # Get parameter file path based on active features
        params_file = get_shared_params_path()
        
        print(f"\n{Colors.CYAN}Will save parameters to:{Colors.RESET}")
        print(f"{'':>4}Shared parameters: {params_file}")
        
        # Create and run study
        study_name = f"{ROAD_TO_PROCESS}_{direction}_multioutput"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize"
        )
        
        # Create objective
        objective = create_objective(X_train, y_train, X_val, y_val)
        
        # Progress tracking
        best_value = float('inf')
        best_trial_number = 0
        patience = 10  # Stop if no improvement in the last 10 trials
        
        def progress_callback(study, trial):
            nonlocal best_value, best_trial_number
            current_trial = len(study.trials)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            
            # Get raw RMSE scores and improvements from trial user attributes
            raw_rmse_scores = trial.user_attrs.get('raw_rmse_scores', {})
            improvements = trial.user_attrs.get('improvements', {})
            
            if study.best_value < best_value:
                best_value = study.best_value
                best_trial_number = current_trial
                print(f"\n{timestamp} Trial [{current_trial}/{N_TRIALS}] {Colors.GREEN}New Best Score!{Colors.RESET}")
            else:
                print(f"\n{timestamp} Trial [{current_trial}/{N_TRIALS}]")
            
            # Print metrics for each target
            for target in TARGETS:
                rmse = raw_rmse_scores.get(target, 0)
                imp = improvements.get(target, 0)
                print(f"{'':>4}{target}: RMSE = {rmse:.2f} (Improvement: {imp:+.1f}%)")
            
            # Early stopping check
            if current_trial - best_trial_number >= patience:
                print(f"{Colors.YELLOW}\nEarly stopping: No improvement in the last {patience} trials.{Colors.RESET}")
                study.stop()
        
        # Run optimization
        print(f"\n{Colors.CYAN}Starting optimization with {N_TRIALS} trials...{Colors.RESET}")
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            callbacks=[progress_callback],
            show_progress_bar=False
        )
        
        # Get best parameters
        best_params = study.best_params
        
        # Save best parameters to shared location
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print(f"\n{Colors.GREEN}✓ Best parameters saved to {params_file}{Colors.RESET}")
        
        # Print final summary
        print(f"\n{Colors.MAGENTA}=== Final Results ==={Colors.RESET}")
        best_trial = study.best_trial
        raw_rmse_scores = best_trial.user_attrs.get('raw_rmse_scores', {})
        improvements = best_trial.user_attrs.get('improvements', {})
        
        for target in TARGETS:
            print(f"\n{Colors.BLUE}{target}:{Colors.RESET}")
            print(f"{'':>4}Best RMSE: {raw_rmse_scores.get(target, 0):.2f}")
            print(f"{'':>4}Improvement: {improvements.get(target, 0):+.1f}%")
        
        return best_params
    
    except Exception as e:
        print(f"{Colors.RED}Error processing {direction}: {str(e)}{Colors.RESET}")
        return None

def main():
    """Main function to tune hyperparameters for multi-output regression."""
    print(f"{Colors.BOLD}Starting Multi-Output XGBoost Hyperparameter Tuning with Optuna{Colors.RESET}")
    print("=" * 70)
    print(f"{Colors.BLUE}Features being used: {', '.join(FEATURES)}{Colors.RESET}")
    
    # Get available directions
    base_path = Path(INPUT_DIR)
    direction_files = list(base_path.glob(f"{ROAD_TO_PROCESS}_*_cluster_train.csv"))
    
    if not direction_files:
        print(f"{Colors.RED}No processed files found for {ROAD_TO_PROCESS} in {INPUT_DIR}{Colors.RESET}")
        return
    
    directions = sorted(str(f.name).split('_')[1] for f in direction_files)
    print(f"{Colors.BLUE}Found directions: {', '.join(directions)}{Colors.RESET}")
    
    if TUNE_SINGLE_DIRECTION:
        if TUNE_DIRECTION not in directions:
            print(f"{Colors.RED}Specified tuning direction '{TUNE_DIRECTION}' not found.{Colors.RESET}")
            return
        
        print(f"{Colors.CYAN}Using single-direction tuning with '{TUNE_DIRECTION}' as reference.{Colors.RESET}")
        
        # Only tune the specified direction
        best_params = optimize_hyperparameters(TUNE_DIRECTION)
        
        if best_params:
            # Create shared parameters directory
            shared_path = get_shared_params_path()
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the parameters to the shared location
            with open(shared_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            
            print(f"{Colors.GREEN}✓ Shared parameters saved to {shared_path}{Colors.RESET}")
    else:
        # Tune each direction separately
        results = {}
        for direction in directions:
            best_params = optimize_hyperparameters(direction)
            results[direction] = best_params
        
        # Print summary
        print(f"\n{Colors.MAGENTA}=== Multi-Output Tuning Summary ==={Colors.RESET}")
        for direction, params in results.items():
            if params:
                print(f"\n{Colors.BLUE}{direction.upper()}{Colors.RESET}")
                for param, value in params.items():
                    print(f"{'':>4}{param}: {value}")

if __name__ == "__main__":
    main() 