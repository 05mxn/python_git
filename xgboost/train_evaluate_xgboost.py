#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#RUN THE FULL THING


import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap

# Configuration
INPUT_DIR = "ML_Data/processed_clusters"
OUTPUT_DIR = "ML_Data/model_results_tuned"
ROAD_TO_PROCESS = "M60"

# Features and target configuration
TARGET = 'total_traffic_flow'  # Can be 'journey_delay' or 'traffic_flow_value'

# Features for training - Updated to include match features and interaction features
FEATURES = [
    #'hour',           # Time of day
    #'is_weekend',     # NEW: Binary flag (weekend/bank holiday or not)
    #'is_school_holiday', # NEW: Binary flag (school holiday/Xmas or not)
    'weather_code',   # Weather conditions
    #'is_pre_match',    # Binary: 1 if in pre-match window, 0 otherwise
    #'is_post_match',   # Binary: 1 if in post-match window, 0 otherwise
    #'is_united_match', # Binary: 1 if Man United match, 0 otherwise
    #'is_city_match',   # Binary: 1 if Man City match, 0 otherwise
    #'hour_weekend',    # Interaction: hour × is_weekend
    #'hour_school_holiday', # Interaction: hour × is_school_holiday
    # Optional: 'is_no_match_period', 'is_no_match_team' (if not dropped in B3)
]

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
    # Create path: road/direction/target_prediction/feature_combo
    return Path(OUTPUT_DIR) / ROAD_TO_PROCESS / direction / f"{TARGET}_prediction" / get_feature_dir_name()

def get_shared_params_path():
    """Construct path for shared parameters."""
    # Create path: road/shared_params/target_prediction/feature_combo
    return Path(OUTPUT_DIR) / ROAD_TO_PROCESS / "shared_params" / f"{TARGET}_prediction" / get_feature_dir_name()

def load_data(direction):
    """Load train, validation, and test data for a specific direction."""
    base_path = Path(INPUT_DIR)
    
    # Construct file paths
    train_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_train.csv"
    val_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_val.csv"
    test_path = base_path / f"{ROAD_TO_PROCESS}_{direction}_cluster_test.csv"
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def prepare_features(df):
    """Prepare features for training."""
    X = df[FEATURES].copy()
    y = df[TARGET]
    return X, y

def train_model(X_train, y_train, X_val, y_val, best_params):
    """Train XGBoost model with best parameters."""
    print(f"{Colors.CYAN}⚡ Training final model...{Colors.RESET}")
    
    # Combine base params with best params
    final_params = {**BASE_PARAMS, **best_params}
    
    # Create and train model
    model = xgb.XGBRegressor(**final_params)
    
    # Train with minimal output
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print(f"{Colors.GREEN}✓ Model training complete{Colors.RESET}")
    return model

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model with detailed metrics and analysis."""
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate residuals
    residuals = y - y_pred
    
    print(f"\n{Colors.BLUE}{dataset_name} Metrics:{Colors.RESET}")
    print(f"{'':>4}RMSE: {rmse:.2f}")
    print(f"{'':>4}MAE:  {mae:.2f}")
    print(f"{'':>4}R²:   {r2:.3f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'residuals': residuals
    }

def analyze_residuals(metrics, output_path):
    """Analyze and plot residuals distribution."""
    residuals = metrics['residuals']
    
    plt.figure(figsize=(12, 6))
    
    # Create subplot for residuals distribution
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel(f'Residual Error ({TARGET})')
    
    # Create subplot for Q-Q plot
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(output_path / 'residuals_analysis.png')
    plt.close()

def plot_feature_importance(model, X_train, output_path):
    """Plot feature importance with SHAP values."""
    # Create plots directory
    plots_dir = output_path / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Create and save SHAP summary plot
    plt.figure(figsize=(10, 8))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / 'shap_importance.png')
    plt.close()
    
    # Create and save feature importance plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='gain')
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png')
    plt.close()

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

def plot_actual_vs_predicted(y_true, y_pred, output_path):
    """Create scatter plot of actual vs predicted values."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Traffic Flow')
    plt.xlabel('Actual Traffic Flow')
    plt.ylabel('Predicted Traffic Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² value to plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'actual_vs_predicted.png')
    plt.close()

def plot_24h_pattern(X, y_true, y_pred, output_path):
    """Create 24-hour traffic pattern comparison plot."""
    # Create DataFrame with hour, actual, and predicted values
    df = pd.DataFrame({
        'hour': X['hour'],
        'actual': y_true,
        'predicted': y_pred
    })
    
    # Calculate hourly averages
    hourly_avg = df.groupby('hour').agg({
        'actual': 'mean',
        'predicted': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual and predicted patterns
    plt.plot(hourly_avg['hour'], hourly_avg['actual'], 
             'b-', label='Actual', marker='o')
    plt.plot(hourly_avg['hour'], hourly_avg['predicted'], 
             'g--', label='Predicted', marker='s')
    
    plt.title('24-Hour Traffic Pattern: Actual vs Predicted')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Traffic Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks for each hour
    plt.xticks(range(24))
    
    plt.tight_layout()
    plt.savefig(output_path / '24h_pattern.png')
    plt.close()

def process_direction(direction):
    """Process one direction with model training and evaluation."""
    print(f"\n{Colors.MAGENTA}Processing {ROAD_TO_PROCESS} {direction.upper()}{Colors.RESET}")
    print(f"{Colors.BLUE}Target: {TARGET}{Colors.RESET}")
    print(f"{Colors.BLUE}Features: {', '.join(FEATURES)}{Colors.RESET}")
    print("-" * 50)
    
    try:
        # Load data
        print(f"{Colors.CYAN}⚡ Loading data...{Colors.RESET}")
        train_df, val_df, test_df = load_data(direction)
        X_train, y_train = prepare_features(train_df)
        X_val, y_val = prepare_features(val_df)
        X_test, y_test = prepare_features(test_df)
        print(f"{Colors.GREEN}✓ Data loaded with shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}{Colors.RESET}")
        
        # Create output directory with feature combination name
        output_path = get_output_path(direction)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # First check if direction-specific params exist
        params_file = output_path / 'best_params.json'
        
        # If direction-specific params don't exist, check for shared params
        if not params_file.exists():
            shared_params_path = get_shared_params_path() / 'best_params.json'
            
            if shared_params_path.exists():
                print(f"{Colors.YELLOW}No direction-specific parameters found. Using shared parameters from {shared_params_path}{Colors.RESET}")
                params_file = shared_params_path
            else:
                print(f"{Colors.YELLOW}Warning: No parameters found at {params_file} or {shared_params_path}. Using default parameters.{Colors.RESET}")
                best_params = {'max_depth': 5, 'learning_rate': 0.1}
        
        # Load parameters if they exist
        if params_file.exists():
            with open(params_file, 'r') as f:
                best_params = json.load(f)
            print(f"{Colors.CYAN}Loaded parameters: {best_params}{Colors.RESET}")
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, best_params)
        
        # Evaluate
        print(f"{Colors.CYAN}⚡ Evaluating model...{Colors.RESET}")
        train_metrics = evaluate_model(model, X_train, y_train, "Train")
        val_metrics = evaluate_model(model, X_val, y_val, "Val")
        test_metrics = evaluate_model(model, X_test, y_test, "Test")
        
        # Save results
        print(f"{Colors.CYAN}⚡ Saving results...{Colors.RESET}")
        model.save_model(str(output_path / 'model.json'))
        
        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'best_params': best_params,
            'features': FEATURES,
            'target': TARGET
        }
        
        # Save metrics as JSON for better readability
        with open(output_path / 'metrics.json', 'w') as f:
            metrics_json = {k: {k2: v2.item() if hasattr(v2, 'item') else v2 
                              for k2, v2 in v.items() if k2 not in ['predictions', 'residuals']}
                           if isinstance(v, dict) else v
                           for k, v in metrics.items()}
            json.dump(metrics_json, f, indent=4)
        
        # Generate existing plots
        analyze_residuals(test_metrics, output_path)
        plot_feature_importance(model, X_train, output_path)
        plot_actual_vs_predicted(y_test, test_metrics['predictions'], output_path)
        plot_24h_pattern(X_test, y_test, test_metrics['predictions'], output_path)
        
        print(f"{Colors.GREEN}✓ Results saved{Colors.RESET}")
        
        return metrics
    except Exception as e:
        print(f"{Colors.RED}Error processing {direction}: {e}{Colors.RESET}")
        return None

def main():
    """Main function to train and evaluate models for all directions."""
    print(f"{Colors.BOLD}Starting XGBoost Training and Evaluation for {ROAD_TO_PROCESS}{Colors.RESET}")
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
            results[direction] = metrics
    
    print(f"\n{Colors.MAGENTA}=== Final Summary ==={Colors.RESET}")
    for direction, metrics in results.items():
        print(f"\n{Colors.BLUE}{direction.upper()}{Colors.RESET}")
        print(f"{'':>4}Target: {TARGET}")
        print(f"{'':>4}Features: {', '.join(FEATURES)}")
        print(f"{'':>4}Test RMSE: {metrics['test']['rmse']:.2f}")
        print(f"{'':>4}Test R²: {metrics['test']['r2']:.3f}")

if __name__ == "__main__":
    main() 