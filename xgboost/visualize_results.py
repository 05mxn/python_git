import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import r2_score
import sys

# ========== COLORS FOR TERMINAL OUTPUT ==========
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

# ========== CONFIGURATION ==========
OUTPUT_DIR = "Output/ML"  # Base output directory
OUTPUT_NAME = "all_3_targets"

# ========== UTILITY FUNCTIONS ==========
def get_output_path(direction):
    return Path(OUTPUT_DIR) / direction / OUTPUT_NAME

def load_metrics_and_targets(direction):
    output_path = get_output_path(direction)
    metrics_path = output_path / 'metrics.json'
    if not metrics_path.exists():
        print(f"{Colors.RED}[ERROR] metrics.json not found in {output_path}{Colors.RESET}")
        sys.exit(1)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    # Extract targets info
    if 'targets' not in metrics:
        print(f"{Colors.RED}[ERROR] 'targets' key not found in metrics.json. Cannot determine targets for visualization.{Colors.RESET}")
        sys.exit(1)
    targets_info = metrics['targets']
    if not isinstance(targets_info, list) or not all('name' in t and 'display_name' in t for t in targets_info):
        print(f"{Colors.RED}[ERROR] 'targets' in metrics.json is malformed. Each target must have 'name' and 'display_name'.{Colors.RESET}")
        sys.exit(1)
    return metrics, targets_info, output_path

def load_outputs(direction, targets_info):
    output_path = get_output_path(direction)
    X_train = pd.read_csv(output_path / 'X_train.csv')
    X_test = pd.read_csv(output_path / 'X_test.csv')
    y_test = pd.read_csv(output_path / 'y_test.csv')
    y_pred_test = np.load(output_path / 'y_pred_test.npy')
    return X_train, X_test, y_test, y_pred_test, output_path

def load_models(direction, targets_info):
    output_path = get_output_path(direction)
    models = []
    for target in targets_info:
        model_file = output_path / f"model_{target['name']}.json"
        if not model_file.exists():
            print(f"{Colors.RED}[ERROR] Model file {model_file} not found.{Colors.RESET}")
            sys.exit(1)
        model = xgb.XGBRegressor()
        model.load_model(str(model_file))
        models.append(model)
    return models

def ensure_plots_dir(output_path):
    plots_dir = output_path / 'plots'
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

# ========== VISUALIZATION FUNCTIONS ==========
def analyze_residuals(y_true, y_pred, output_path, target, target_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title(f'{target_name} - Residuals Distribution')
    plt.xlabel(f'Residual Error ({target_name})')
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{target_name} - Q-Q Plot')
    plt.tight_layout()
    plt.savefig(output_path / f'{target}_residuals_analysis.png')
    plt.close()

def plot_feature_importance(models, X_train, output_path, targets_info):
    plots_dir = ensure_plots_dir(output_path)
    for i, (target, model) in enumerate(zip(targets_info, models)):
        target_name = target['display_name']
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, importance_type='gain')
        plt.title(f'Feature Importance - {target_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{target['name']}_feature_importance.png")
        plt.close()
        try:
            plt.figure(figsize=(10, 8))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, show=False)
            plt.title(f'SHAP Feature Importance - {target_name}')
            plt.tight_layout()
            plt.savefig(plots_dir / f"{target['name']}_shap_importance.png")
            plt.close()
        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not generate SHAP plot for {target_name}: {str(e)}{Colors.RESET}")

def plot_actual_vs_predicted(y_true_df, y_pred, output_path, targets_info):
    for i, target in enumerate(targets_info):
        y_true = y_true_df[target['name']]
        y_pred_i = y_pred[:, i]
        target_name = target['display_name']
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred_i, alpha=0.5)
        min_val = min(y_true.min(), y_pred_i.min())
        max_val = max(y_true.max(), y_pred_i.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        r2 = r2_score(y_true, y_pred_i)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.title(f'Actual vs Predicted - {target_name}')
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"{target['name']}_actual_vs_predicted.png")
        plt.close()

def plot_24h_pattern(X_test, y_true_df, y_pred, output_path, targets_info):
    for i, target in enumerate(targets_info):
        y_true = y_true_df[target['name']].values
        y_pred_i = y_pred[:, i]
        target_name = target['display_name']
        hours = X_test['hour'].values
        df_pattern = pd.DataFrame({'hour': hours, 'actual': y_true, 'predicted': y_pred_i})
        hourly_pattern = df_pattern.groupby('hour').mean()
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_pattern.index, hourly_pattern['actual'], 'o-', label='Actual')
        plt.plot(hourly_pattern.index, hourly_pattern['predicted'], 's-', label='Predicted')
        plt.title(f'24-Hour Pattern - {target_name}')
        plt.xlabel('Hour of Day')
        plt.ylabel(target_name)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{target['name']}_24h_pattern.png")
        plt.close()

def plot_weekend_weekday_pattern(X_test, y_true_df, y_pred, output_path, targets_info):
    if 'is_weekend' not in X_test.columns:
        print(f"{Colors.YELLOW}'is_weekend' not in features, skipping weekend/weekday plots.{Colors.RESET}")
        return
    for i, target in enumerate(targets_info):
        y_true = y_true_df[target['name']].values
        y_pred_i = y_pred[:, i]
        target_name = target['display_name']
        df_pattern = pd.DataFrame({
            'hour': X_test['hour'].values,
            'is_weekend': X_test['is_weekend'].values,
            'actual': y_true,
            'predicted': y_pred_i
        })
        grouped_pattern = df_pattern.groupby(['hour', 'is_weekend']).mean().reset_index()
        grouped_std = df_pattern.groupby(['hour', 'is_weekend']).std().reset_index()
        grouped_count = df_pattern.groupby(['hour', 'is_weekend']).count().reset_index()
        grouped_pattern['se'] = grouped_std['actual'] / np.sqrt(grouped_count['actual'])
        weekday_data = grouped_pattern[grouped_pattern['is_weekend'] == 0]
        weekend_data = grouped_pattern[grouped_pattern['is_weekend'] == 1]
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.errorbar(weekday_data['hour'], weekday_data['actual'], yerr=weekday_data['se'], fmt='o-', color='blue', label='Actual', capsize=3)
        plt.plot(weekday_data['hour'], weekday_data['predicted'], 's-', color='lightblue', label='Predicted')
        plt.title(f'Weekday 24-Hour Pattern - {target_name}')
        plt.xlabel('Hour of Day')
        plt.ylabel(target_name)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.errorbar(weekend_data['hour'], weekend_data['actual'], yerr=weekend_data['se'], fmt='o-', color='red', label='Actual', capsize=3)
        plt.plot(weekend_data['hour'], weekend_data['predicted'], 's-', color='lightcoral', label='Predicted')
        plt.title(f'Weekend 24-Hour Pattern - {target_name}')
        plt.xlabel('Hour of Day')
        plt.ylabel(target_name)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{target['name']}_weekend_weekday_pattern.png")
        plt.close()
        plt.figure(figsize=(14, 7))
        plt.errorbar(weekday_data['hour'], weekday_data['actual'], yerr=weekday_data['se'], fmt='o-', color='blue', label='Weekday (Actual)', capsize=3)
        plt.plot(weekday_data['hour'], weekday_data['predicted'], 's--', color='lightblue', label='Weekday (Predicted)')
        plt.errorbar(weekend_data['hour'], weekend_data['actual'], yerr=weekend_data['se'], fmt='o-', color='red', label='Weekend (Actual)', capsize=3)
        plt.plot(weekend_data['hour'], weekend_data['predicted'], 's--', color='lightcoral', label='Weekend (Predicted)')
        plt.title(f'Weekend vs Weekday 24-Hour Pattern - {target_name}')
        plt.xlabel('Hour of Day')
        plt.ylabel(target_name)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{target['name']}_weekend_weekday_combined.png")
        plt.close()

# ========== MAIN SCRIPT ==========
def main():
    print(f"{Colors.CYAN}Current working directory: {os.getcwd()}{Colors.RESET}")
    print(f"{Colors.CYAN}Looking for ML output in: {OUTPUT_DIR}{Colors.RESET}")
    print(f"{Colors.CYAN}Subdirectories found:{Colors.RESET}")
    for d in Path(OUTPUT_DIR).iterdir():
        print(f"  {Colors.BLUE}{d}{Colors.RESET}")
        if (d / OUTPUT_NAME).exists():
            print(f"    {Colors.GREEN}-> Contains {OUTPUT_NAME}{Colors.RESET}")
    base_path = Path(OUTPUT_DIR)
    directions = [d.name for d in base_path.iterdir() if (d / OUTPUT_NAME).exists()]
    print(f"{Colors.CYAN}Found directions: {directions}{Colors.RESET}")
    for direction in directions:
        print(f"\n{Colors.MAGENTA}Processing direction: {direction}{Colors.RESET}")
        metrics, targets_info, output_path = load_metrics_and_targets(direction)
        X_train, X_test, y_test, y_pred_test, output_path = load_outputs(direction, targets_info)
        models = load_models(direction, targets_info)
        # Residuals analysis
        for i, target in enumerate(targets_info):
            analyze_residuals(y_test[target['name']], y_pred_test[:, i], output_path, target['name'], target['display_name'])
        # Feature importance
        plot_feature_importance(models, X_train, output_path, targets_info)
        # Actual vs predicted
        plot_actual_vs_predicted(y_test, y_pred_test, output_path, targets_info)
        # 24h pattern
        plot_24h_pattern(X_test, y_test, y_pred_test, output_path, targets_info)
        # Weekend/weekday pattern
        plot_weekend_weekday_pattern(X_test, y_test, y_pred_test, output_path, targets_info)
        print(f"{Colors.GREEN}Visualizations saved to {output_path}{Colors.RESET}")

if __name__ == "__main__":
    main() 