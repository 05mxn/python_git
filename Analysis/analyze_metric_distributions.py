import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Output and data directories
TYPICAL_AGG_DIR = 'ML_Data/typical_aggregates'
OUTPUT_DIR = 'Output/Exploration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Terminal color definitions
class Colors:
    RESET = '\033[0m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'

def print_color(msg, color=Colors.RESET, bold=False):
    style = Colors.BOLD if bold else ''
    print(f"{color}{style}{msg}{Colors.RESET}")

def get_metric_columns(df):
    # Return all columns ending with _mean (typical metrics)
    return [col for col in df.columns if col.endswith('_mean')]

def summarize_metric(series):
    # Compute summary statistics for a pandas Series
    return {
        'min': series.min(),
        'max': series.max(),
        'mean': series.mean(),
        'std': series.std(),
        '5th_percentile': series.quantile(0.05),
        '25th_percentile': series.quantile(0.25),
        '50th_percentile': series.quantile(0.5),
        '75th_percentile': series.quantile(0.75),
        '95th_percentile': series.quantile(0.95),
        'count': series.count()
    }

def save_summary_table(summary_dict, output_path):
    df = pd.DataFrame(summary_dict).T
    df.to_csv(output_path)
    print_color(f"Saved summary table: {output_path}", Colors.GREEN)

def plot_metric(series, metric_name, direction, output_dir):
    plt.figure(figsize=(10, 4))
    sns.histplot(series.dropna(), bins=30, kde=True, color='skyblue')
    plt.title(f"Histogram of {metric_name} ({direction})")
    plt.xlabel(metric_name)
    plt.ylabel('Count')
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"{metric_name}_{direction}_hist.png")
    plt.savefig(hist_path)
    plt.close()
    print_color(f"Saved histogram: {hist_path}", Colors.CYAN)

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=series.dropna(), color='orange')
    plt.title(f"Boxplot of {metric_name} ({direction})")
    plt.xlabel(metric_name)
    plt.tight_layout()
    box_path = os.path.join(output_dir, f"{metric_name}_{direction}_box.png")
    plt.savefig(box_path)
    plt.close()
    print_color(f"Saved boxplot: {box_path}", Colors.CYAN)

def main():
    print_color("Starting metric distribution analysis...", Colors.CYAN, bold=True)
    all_files = list(Path(TYPICAL_AGG_DIR).glob('*.parquet'))
    if not all_files:
        print_color(f"No parquet files found in {TYPICAL_AGG_DIR}", Colors.RED, bold=True)
        return
    for file_path in all_files:
        print_color(f"Processing {file_path.name}", Colors.YELLOW, bold=True)
        df = pd.read_parquet(file_path)
        direction = 'unknown'
        if 'clockwise' in file_path.name:
            direction = 'clockwise'
        elif 'anticlockwise' in file_path.name:
            direction = 'anticlockwise'
        metrics = get_metric_columns(df)
        summary_dict = {}
        for metric in metrics:
            print_color(f"  Analyzing {metric}", Colors.CYAN)
            summary = summarize_metric(df[metric])
            summary_dict[metric] = summary
            plot_metric(df[metric], metric, direction, OUTPUT_DIR)
        # Save summary table for this file
        summary_path = os.path.join(OUTPUT_DIR, f"summary_{file_path.stem}.csv")
        save_summary_table(summary_dict, summary_path)
    print_color("Analysis complete!", Colors.GREEN, bold=True)

if __name__ == '__main__':
    main() 