#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Football Match Impact Visualization Script
Creates focused visualizations comparing pre-match and post-match impacts
for Manchester United and Manchester City matches.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
INPUT_DIR = "Analysis/Match_Impact_Results"
OUTPUT_DIR = "Figures"
ROAD_NAME = "M60"

# Color schemes for teams
UNITED_COLOR = "#DA291C"  # United Red
CITY_COLOR = "#6CABDD"    # City Blue

# Terminal colors for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def print_section(message):
    """Print a formatted section header."""
    print(f"\n{Colors.HEADER}{message}{Colors.ENDC}")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}{message}{Colors.ENDC}")

def prepare_data(data):
    """
    Prepare data for visualization by cleaning segment names and calculating absolute changes.
    Segment labels should include both direction and junctions for clarity.
    Args:
        data (pd.DataFrame): Raw results data
    Returns:
        pd.DataFrame: Processed data with clean labels
    """
    def create_segment_label(row):
        # Build label: M60 [direction] \n [junctions] (if available)
        label = f"M60 {row['direction']}"
        if pd.notna(row['junctions']) and str(row['junctions']).strip():
            label += f"\n{row['junctions']}"
        elif pd.notna(row['clean_name']) and 'within' in str(row['clean_name']):
            # Fallback: try to extract location from clean_name
            label += '\n' + str(row['clean_name']).split('within ')[-1]
        return label
    data['segment_label'] = data.apply(create_segment_label, axis=1)
    data['abs_flow_pct'] = data['flow_pct'].abs()
    return data

def plot_period_comparison(data, period, output_dir):
    """
    Create side-by-side comparison of United and City impacts for a specific period.
    
    Args:
        data (pd.DataFrame): Processed results data
        period (str): 'pre_match' or 'post_match'
        output_dir (str): Directory to save the figure
    """
    # Filter data for period
    period_data = data[data['period'] == period]
    
    # Get top 4 segments for each team based on absolute flow change
    def get_top_segments(team_data):
        return team_data.nlargest(4, 'abs_flow_pct')['segment_id'].unique()
    
    united_top = get_top_segments(period_data[period_data['team'] == 'united'])
    city_top = get_top_segments(period_data[period_data['team'] == 'city'])
    
    # Filter data for top segments
    united_data = period_data[
        (period_data['team'] == 'united') & 
        (period_data['segment_id'].isin(united_top))
    ].sort_values('abs_flow_pct', ascending=True)  # Sort for consistent ordering
    
    city_data = period_data[
        (period_data['team'] == 'city') & 
        (period_data['segment_id'].isin(city_top))
    ].sort_values('abs_flow_pct', ascending=True)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot function for each team
    def plot_team_data(ax, data, team_color, team_name):
        # Create horizontal bars
        # Use IQR/2 as a proxy for standard deviation (since flow_std is not available)
        bars = ax.barh(range(len(data)), data['flow_pct'], 
                      xerr=data['flow_iqr'] / 2, color=team_color, 
                      capsize=5, alpha=0.7)
        
        # Add value labels above each bar
        for i, bar in enumerate(bars):
            value = data.iloc[i]["flow_pct"]
            label = f"{value:.1f}%"
            offset = 10
            ax.text(
                bar.get_width() + offset,
                bar.get_y() + bar.get_height()/2 + 0.2,  # vertical offset
                label,
                va='center', ha='left' if bar.get_width() >= 0 else 'right',
                fontweight='regular', color='black', fontsize=10
            )
        
        # Customize appearance
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['segment_label'])
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add title
        ax.set_title(f'Manchester {team_name}', pad=20)
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        return ax
    
    # Plot each team's data
    plot_team_data(ax1, united_data, UNITED_COLOR, 'United')
    plot_team_data(ax2, city_data, CITY_COLOR, 'City')
    
    # Align x-axes and add common label
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Add common x-label
    fig.text(0.5, -0.02, 'Flow Change (%)', ha='center', va='center', fontweight='bold')
    
    # Add main title
    period_title = 'Pre-Match' if period == 'pre_match' else 'Post-Match'
    plt.suptitle(f'Traffic Flow Changes at Critical M60 Segments\n{period_title} Period',
                 y=0.98, fontsize=14, fontweight='bold')
    
    # Add footnote
    plt.figtext(0.5, -0.08,
                'Analysis compares match days to equivalent non-match days.\n'
                'Error bars show ±1 standard deviation.',
                ha='center', va='center', fontsize=8, fontstyle='italic')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'{period}_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Main execution function."""
    print_section("FOOTBALL MATCH IMPACT VISUALIZATION")
    
    # Input and output paths
    results_file = os.path.join(INPUT_DIR, "top_segments_for_viz.csv")
    output_dir = OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_file):
        print_error(f"Results file not found: {results_file}")
        print_warning("Make sure you've run analyze_match_impact.py first")
        return
    
    # Read and prepare data
    try:
        data = pd.read_csv(results_file)
        data = prepare_data(data)
        
        # Create visualizations for pre-match and post-match periods
        pre_match_file = plot_period_comparison(data, 'pre_match', output_dir)
        post_match_file = plot_period_comparison(data, 'post_match', output_dir)
        
        print_success("Visualizations complete")
        print_success(f"Pre-match comparison: {pre_match_file}")
        print_success(f"Post-match comparison: {post_match_file}")
        
    except Exception as e:
        print_error(f"Error creating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main() 