import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set style for all plots
plt.style.use('default')  # Using default matplotlib style

# Constants
DATA_DIR = "ML_Data/transformed_features"
OUTPUT_DIR = "Output/network_distribution_results"
ROAD_NAME = "M60"  # We're focusing on M60

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NetworkDistributionAnalyzer:
    def __init__(self, data_dir=DATA_DIR, road_name=ROAD_NAME):
        """Initialize the analyzer with data directory and road name."""
        self.data_dir = data_dir
        self.road_name = road_name
        self.data = None
        self.grouped_data = {}
        
        # Create output subdirectories
        self.plot_dir = os.path.join(OUTPUT_DIR, 'plots')
        self.stats_dir = os.path.join(OUTPUT_DIR, 'statistics')
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Set up consistent colors for conditions
        self.colors = {
            'weekday': '#1f77b4',    # blue
            'weekend': '#2ca02c',    # green
            'school_holiday': '#ff7f0e'  # orange
        }
        
        print(f"\033[36mInitializing Network Distribution Analysis for {road_name}\033[0m")
    
    def load_and_prepare_data(self):
        """Load and prepare data from processed clusters."""
        print(f"\033[35mLoading data from {self.data_dir}...\033[0m")
        
        # Find all relevant CSV files
        data_files = list(Path(self.data_dir).glob(f"{self.road_name}*.csv"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found for {self.road_name}")
        
        # Load and combine all files
        dfs = []
        for file in data_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        
        # Print available columns for debugging
        print("\033[33mAvailable columns in data:\033[0m")
        for col in self.data.columns:
            print(f"  - {col}")
        
        # Calculate normalized journey time if not already present
        if 'normalized_time' not in self.data.columns and 'fused_travel_time' in self.data.columns and 'link_length' in self.data.columns:
            print("\033[36mCalculating normalized journey times...\033[0m")
            # Convert link length to miles
            self.data['link_length_miles'] = self.data['link_length'] / 1609.34
            # Calculate normalized time (minutes per mile)
            self.data['normalized_time'] = np.where(
                self.data['link_length_miles'] > 0,
                (self.data['fused_travel_time'] / 60) / self.data['link_length_miles'],
                np.nan
            ).round(2)
            # Clean up intermediate column
            self.data.drop(columns=['link_length_miles'], inplace=True)
        
        # Select relevant columns that exist in the data
        relevant_columns = [
            'ntis_link_number',  # segment identifier
            'hour',
            'is_weekend',
            'is_school_holiday',
            'total_traffic_flow',
            'speed_reduction',
            'normalized_time',  # Added normalized journey time
            'direction'  # might be useful for analysis
        ]
        
        # Keep only relevant columns that exist in the data
        self.data = self.data[[col for col in relevant_columns if col in self.data.columns]]
        
        # Remove any match-related periods if those columns exist
        if 'is_pre_match' in self.data.columns:
            self.data = self.data[~self.data['is_pre_match']]
        if 'is_post_match' in self.data.columns:
            self.data = self.data[~self.data['is_post_match']]
            
        print(f"\033[32mLoaded {len(self.data):,} records\033[0m")
        
    def group_data(self):
        """Group data by conditions (weekday, weekend, school holiday) and hour."""
        print("\033[35mGrouping data by conditions...\033[0m")
        
        # Create groups
        conditions = {
            'weekday': (self.data['is_weekend'] == 0) & (self.data['is_school_holiday'] == 0),
            'weekend': self.data['is_weekend'] == 1
        }
        
        # Add school holiday condition if we have enough data
        if 'is_school_holiday' in self.data.columns:
            school_holiday_data = self.data[self.data['is_school_holiday'] == 1]
            if len(school_holiday_data) > 0:
                conditions['school_holiday'] = self.data['is_school_holiday'] == 1
        
        # Group data for each condition
        for condition_name, condition_mask in conditions.items():
            condition_data = self.data[condition_mask]
            
            # Group by hour and calculate statistics for each segment
            hourly_groups = condition_data.groupby(['hour', 'ntis_link_number']).agg({
                'total_traffic_flow': ['mean', 'std', 'count'],
                'speed_reduction': ['mean', 'std', 'count'],
                'normalized_time': ['mean', 'std', 'count']  # Added normalized time
            }).reset_index()
            
            self.grouped_data[condition_name] = hourly_groups
            
            print(f"\033[32mProcessed {condition_name}: {len(condition_data):,} records\033[0m")
    
    def analyze_distributions(self):
        """Analyze the distribution of traffic across segments for each hour and condition."""
        print("\033[35mAnalyzing distributions...\033[0m")
        
        # Calculate network-wide statistics for each hour and condition
        self.hourly_stats = {}
        
        for condition, data in self.grouped_data.items():
            print(f"\033[36mAnalyzing {condition} patterns...\033[0m")
            
            # Calculate hourly statistics across all segments
            hourly_stats = []
            for hour in range(24):
                hour_data = data[data['hour'] == hour]
                
                # Get flow statistics
                flow_mean = hour_data[('total_traffic_flow', 'mean')].mean()
                flow_std = hour_data[('total_traffic_flow', 'mean')].std()
                
                # Get speed reduction statistics
                speed_mean = hour_data[('speed_reduction', 'mean')].mean()
                speed_std = hour_data[('speed_reduction', 'mean')].std()
                
                # Get normalized time statistics
                time_mean = hour_data[('normalized_time', 'mean')].mean()
                time_std = hour_data[('normalized_time', 'mean')].std()
                
                hourly_stats.append({
                    'hour': hour,
                    'flow_mean': flow_mean,
                    'flow_std': flow_std,
                    'speed_mean': speed_mean,
                    'speed_std': speed_std,
                    'time_mean': time_mean,
                    'time_std': time_std,
                    'n_segments': len(hour_data)
                })
            
            self.hourly_stats[condition] = pd.DataFrame(hourly_stats)
            
            # Save statistics to file
            stats_file = os.path.join(self.stats_dir, f"{condition}_hourly_stats.csv")
            self.hourly_stats[condition].to_csv(stats_file, index=False)
            print(f"\033[32mSaved statistics to {stats_file}\033[0m")
    
    def create_visualizations(self):
        """Create visualizations of the distributions."""
        print("\033[35mCreating visualizations...\033[0m")
        
        # 1. Traffic Flow Throughout the Day
        plt.figure(figsize=(12, 6))
        for condition in self.hourly_stats:
            stats = self.hourly_stats[condition]
            plt.plot(stats['hour'], stats['flow_mean'], 
                    color=self.colors[condition], 
                    label=condition.capitalize(),
                    linewidth=2)
            # Add confidence intervals
            plt.fill_between(stats['hour'],
                           stats['flow_mean'] - stats['flow_std'],
                           stats['flow_mean'] + stats['flow_std'],
                           color=self.colors[condition],
                           alpha=0.2)
        
        plt.title('Average Traffic Flow Throughout the Day\nwith ±1 Standard Deviation', pad=20)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Traffic Flow')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'traffic_flow_daily_pattern.png'))
        plt.close()
        
        # 2. Speed Reduction Throughout the Day
        plt.figure(figsize=(12, 6))
        for condition in self.hourly_stats:
            stats = self.hourly_stats[condition]
            plt.plot(stats['hour'], stats['speed_mean'],
                    color=self.colors[condition],
                    label=condition.capitalize(),
                    linewidth=2)
            # Add confidence intervals
            plt.fill_between(stats['hour'],
                           stats['speed_mean'] - stats['speed_std'],
                           stats['speed_mean'] + stats['speed_std'],
                           color=self.colors[condition],
                           alpha=0.2)
        
        plt.title('Average Speed Reduction Throughout the Day\nwith ±1 Standard Deviation', pad=20)
        plt.xlabel('Hour of Day')
        plt.ylabel('Speed Reduction (mph)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'speed_reduction_daily_pattern.png'))
        plt.close()
        
        # 3. Normalized Journey Time Throughout the Day
        plt.figure(figsize=(12, 6))
        for condition in self.hourly_stats:
            stats = self.hourly_stats[condition]
            plt.plot(stats['hour'], stats['time_mean'],
                    color=self.colors[condition],
                    label=condition.capitalize(),
                    linewidth=2)
            # Add confidence intervals
            plt.fill_between(stats['hour'],
                           stats['time_mean'] - stats['time_std'],
                           stats['time_mean'] + stats['time_std'],
                           color=self.colors[condition],
                           alpha=0.2)
        
        plt.title('Average Normalized Journey Time Throughout the Day\nwith ±1 Standard Deviation', pad=20)
        plt.xlabel('Hour of Day')
        plt.ylabel('Journey Time (minutes/mile)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'normalized_time_daily_pattern.png'))
        plt.close()
        
        # 4. Box plots for each hour and condition
        for condition in self.grouped_data:
            data = self.grouped_data[condition]
            
            # Traffic Flow Distribution
            plt.figure(figsize=(15, 6))
            plt.boxplot([data[data['hour'] == h][('total_traffic_flow', 'mean')] 
                        for h in range(24)],
                       positions=range(24),
                       widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=self.colors[condition], alpha=0.5))
            
            plt.title(f'Distribution of Traffic Flow Across Segments\n{condition.capitalize()}', pad=20)
            plt.xlabel('Hour of Day')
            plt.ylabel('Traffic Flow')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'traffic_flow_distribution_{condition}.png'))
            plt.close()
            
            # Normalized Journey Time Distribution
            plt.figure(figsize=(15, 6))
            plt.boxplot([data[data['hour'] == h][('normalized_time', 'mean')] 
                        for h in range(24)],
                       positions=range(24),
                       widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=self.colors[condition], alpha=0.5))
            
            plt.title(f'Distribution of Normalized Journey Times Across Segments\n{condition.capitalize()}', pad=20)
            plt.xlabel('Hour of Day')
            plt.ylabel('Journey Time (minutes/mile)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'normalized_time_distribution_{condition}.png'))
            plt.close()
        
        print(f"\033[32mSaved visualizations to {self.plot_dir}\033[0m")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.load_and_prepare_data()
        self.group_data()
        self.analyze_distributions()
        self.create_visualizations()
        print("\033[32mAnalysis complete!\033[0m")

if __name__ == "__main__":
    # Create analyzer instance
    analyzer = NetworkDistributionAnalyzer()
    
    # Run analysis
    analyzer.run_analysis() 