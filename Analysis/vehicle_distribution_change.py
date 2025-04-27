import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set style for all plots
plt.style.use('default')  # Using default matplotlib style

# Constants
DATA_DIR = "ML_Data/transformed_features"
OUTPUT_DIR = "Output/vehicle_distribution_results"
ROAD_NAME = "M60"  # We're focusing on M60

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VehicleDistributionAnalyzer:
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
        
        # Set up consistent colors for conditions and vehicle types
        self.condition_colors = {
            'weekday': '#1f77b4',    # blue
            'weekend': '#2ca02c',    # green
            'school_holiday': '#ff7f0e'  # orange
        }
        
        # Colors for vehicle types
        self.vehicle_colors = {
            'traffic_1_pct': '#1f77b4',     # blue (cars)
            'traffic_2_pct': '#2ca02c',      # green (vans)
            'traffic_3_pct': '#ff7f0e',     # orange (medium trucks)
            'traffic_4_pct': '#d62728'       # red (large trucks)
        }
        
        # Vehicle type labels for plotting
        self.vehicle_labels = {
            'traffic_1_pct': '< 5.2m (Cars)',
            'traffic_2_pct': '5.2-6.6m (Vans)',
            'traffic_3_pct': '6.6-11.6m (Medium Trucks)',
            'traffic_4_pct': '> 11.6m (Large Trucks)'
        }
        
        # Raw count column names for absolute numbers
        self.count_columns = {
            'traffic_1_pct': 'traffic_flow_value1',
            'traffic_2_pct': 'traffic_flow_value2',
            'traffic_3_pct': 'traffic_flow_value3',
            'traffic_4_pct': 'traffic_flow_value4'
        }
        
        print(f"\033[36mInitializing Vehicle Distribution Analysis for {road_name}\033[0m")
    
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
            
            # Data validation
            print(f"\n\033[36mValidating data in {file.name}:\033[0m")
            
            # Check for missing values
            missing_values = df[['traffic_flow_value1', 'traffic_flow_value2', 
                               'traffic_flow_value3', 'traffic_flow_value4', 
                               'total_traffic_flow']].isnull().sum()
            if missing_values.any():
                print(f"\033[33mWarning: Found missing values:\n{missing_values}\033[0m")
            
            # Check for zero total flow
            zero_flow = (df['total_traffic_flow'] == 0).sum()
            if zero_flow > 0:
                print(f"\033[33mWarning: Found {zero_flow} records with zero total flow\033[0m")
            
            # Validate percentages sum to approximately 100%
            flow_sum = df[['traffic_flow_value1', 'traffic_flow_value2', 
                          'traffic_flow_value3', 'traffic_flow_value4']].sum(axis=1)
            total_mismatch = (abs(flow_sum - df['total_traffic_flow']) > 1).sum()
            if total_mismatch > 0:
                print(f"\033[33mWarning: Found {total_mismatch} records where sum of vehicle types doesn't match total\033[0m")
            
            # Print sample statistics
            print("\n\033[36mSample statistics:\033[0m")
            stats = df[['traffic_flow_value1', 'traffic_flow_value2', 
                       'traffic_flow_value3', 'traffic_flow_value4']].describe()
            print(stats)
            
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        
        # Additional validation on combined data
        print("\n\033[36mValidating combined data:\033[0m")
        
        # Check for unreasonable values
        for col in ['traffic_flow_value1', 'traffic_flow_value2', 
                   'traffic_flow_value3', 'traffic_flow_value4']:
            outliers = self.data[self.data[col] > self.data['total_traffic_flow']].shape[0]
            if outliers > 0:
                print(f"\033[33mWarning: Found {outliers} records where {col} exceeds total flow\033[0m")
        
        # Group by hour and print average distribution
        hourly_avg = self.data.groupby('hour')[['traffic_flow_value1', 'traffic_flow_value2', 
                                               'traffic_flow_value3', 'traffic_flow_value4']].mean()
        print("\n\033[36mAverage hourly distribution:\033[0m")
        print(hourly_avg)
        
        print(f"\033[32mLoaded {len(self.data):,} records\033[0m")
    
    def group_data(self):
        """Group data by conditions (weekday, weekend, school holiday) and hour."""
        print("\033[35mGrouping data by conditions...\033[0m")
        
        # Create groups
        conditions = {
            'weekday': (self.data['is_weekend'] == 0) & (self.data['is_school_holiday'] == 0),
            'weekend': self.data['is_weekend'] == 1,
            'school_holiday': self.data['is_school_holiday'] == 1
        }
        
        # Vehicle type columns (percentages and counts)
        pct_columns = list(self.vehicle_colors.keys())
        count_columns = list(self.count_columns.values())
        
        # Group data for each condition
        for condition_name, condition_mask in conditions.items():
            condition_data = self.data[condition_mask]
            
            # Group by hour and calculate statistics
            hourly_groups = condition_data.groupby('hour').agg({
                **{col: ['mean', 'std', 'count'] for col in pct_columns},
                **{col: ['mean', 'std', 'count'] for col in count_columns}
            }).reset_index()
            
            self.grouped_data[condition_name] = hourly_groups
            
            print(f"\033[32mProcessed {condition_name}: {len(condition_data):,} records\033[0m")
    
    def create_visualizations(self):
        """Create visualizations of the vehicle type distributions."""
        print("\033[35mCreating visualizations...\033[0m")
        
        # 1. Daily Pattern Plot (Stacked Area) for each condition
        for condition in self.grouped_data:
            plt.figure(figsize=(12, 6))
            data = self.grouped_data[condition]
            
            # Prepare data for stacking
            y_data = [data[(col, 'mean')] for col in self.count_columns.values()]
            labels = [self.vehicle_labels[pct_col] for pct_col in self.vehicle_colors.keys()]
            colors = list(self.vehicle_colors.values())
            
            plt.stackplot(data['hour'], y_data, labels=labels, colors=colors, alpha=0.7)
            
            plt.title(f'Vehicle Type Distribution Throughout the Day\n{condition.capitalize()}', pad=20)
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Vehicles')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'daily_distribution_{condition}.png'))
            plt.close()
        
        # 2. Percentage Distribution Plot
        for condition in self.grouped_data:
            plt.figure(figsize=(12, 6))
            data = self.grouped_data[condition]
            
            # Plot percentage for each vehicle type
            for pct_col in self.vehicle_colors.keys():
                # Get the data
                mean_values = data[(pct_col, 'mean')]
                std_values = data[(pct_col, 'std')]
                counts = data[(pct_col, 'count')]
                
                # Clean the data
                # Remove any invalid values (NaN, inf, etc.)
                valid_mask = ~(mean_values.isna() | std_values.isna() | counts.isna())
                mean_values = mean_values[valid_mask]
                std_values = std_values[valid_mask]
                counts = counts[valid_mask]
                hours = data['hour'][valid_mask]
                
                # Calculate standard error (with minimum count protection)
                counts = counts.clip(lower=1)  # Prevent division by zero
                std_error = std_values / np.sqrt(counts)
                
                # Calculate confidence intervals (clip to reasonable range)
                lower_bound = np.clip(mean_values - 1.96 * std_error, 0, 100)
                upper_bound = np.clip(mean_values + 1.96 * std_error, 0, 100)
                
                # Sort by hour to ensure proper line connection
                sort_idx = np.argsort(hours)
                hours = hours.iloc[sort_idx]
                mean_values = mean_values.iloc[sort_idx]
                lower_bound = lower_bound.iloc[sort_idx]
                upper_bound = upper_bound.iloc[sort_idx]
                
                # Plot with proper handling of gaps
                plt.plot(hours, 
                        mean_values,
                        color=self.vehicle_colors[pct_col],
                        label=self.vehicle_labels[pct_col],
                        linewidth=2)
                
                # Add confidence intervals
                plt.fill_between(hours,
                               lower_bound,
                               upper_bound,
                               color=self.vehicle_colors[pct_col],
                               alpha=0.2)
            
            plt.title(f'Vehicle Type Percentages Throughout the Day\n{condition.capitalize()}', pad=20)
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage of Total Flow')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim(0, 100)
            plt.xlim(0, 23)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'percentage_distribution_{condition}.png'))
            plt.close()
        
        # 3. Box plots for each vehicle type by hour
        for pct_col, count_col in self.count_columns.items():
            plt.figure(figsize=(15, 6))
            
            # Create box plots for each hour
            data_to_plot = [self.data[self.data['hour'] == h][count_col] 
                           for h in range(24)]
            
            plt.boxplot(data_to_plot,
                       positions=range(24),
                       widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=self.vehicle_colors[pct_col], alpha=0.5))
            
            plt.title(f'Distribution of {self.vehicle_labels[pct_col]} Throughout the Day', pad=20)
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Vehicles')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'hourly_boxplot_{count_col}.png'))
            plt.close()
        
        # 4. Overall Vehicle Type Distribution (Pie Chart)
        plt.figure(figsize=(10, 8))
        total_counts = [self.data[col].sum() for col in self.count_columns.values()]
        plt.pie(total_counts,
                labels=[self.vehicle_labels[pct_col] for pct_col in self.vehicle_colors.keys()],
                colors=[self.vehicle_colors[pct_col] for pct_col in self.vehicle_colors.keys()],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Overall Distribution of Vehicle Types', pad=20)
        plt.axis('equal')
        plt.savefig(os.path.join(self.plot_dir, 'overall_distribution_pie.png'))
        plt.close()
        
        print(f"\033[32mSaved visualizations to {self.plot_dir}\033[0m")
    
    def analyze_distributions(self):
        """Analyze the distribution of vehicle types and save statistics."""
        print("\033[35mAnalyzing distributions...\033[0m")
        
        # Calculate statistics for each condition
        stats_data = []
        
        for condition, data in self.grouped_data.items():
            print(f"\033[36mAnalyzing {condition} patterns...\033[0m")
            
            # Calculate hourly statistics for each vehicle type
            for hour in range(24):
                hour_data = data[data['hour'] == hour]
                
                stats = {
                    'condition': condition,
                    'hour': hour
                }
                
                # Add statistics for percentages and counts
                for pct_col, count_col in self.count_columns.items():
                    stats.update({
                        f'{count_col}_mean': hour_data[(count_col, 'mean')].iloc[0],
                        f'{count_col}_std': hour_data[(count_col, 'std')].iloc[0],
                        f'{pct_col}_mean': hour_data[(pct_col, 'mean')].iloc[0],
                        f'{pct_col}_std': hour_data[(pct_col, 'std')].iloc[0]
                    })
                
                stats_data.append(stats)
        
        # Convert to DataFrame and save
        stats_df = pd.DataFrame(stats_data)
        stats_file = os.path.join(self.stats_dir, 'vehicle_type_stats.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"\033[32mSaved statistics to {stats_file}\033[0m")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.load_and_prepare_data()
        self.group_data()
        self.analyze_distributions()
        self.create_visualizations()
        print("\033[32mAnalysis complete!\033[0m")

if __name__ == "__main__":
    # Create analyzer instance
    analyzer = VehicleDistributionAnalyzer()
    
    # Run analysis
    analyzer.run_analysis() 