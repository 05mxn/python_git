#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import numpy as np

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

# Configuration
INPUT_DIR = "ML_Data/clustered_main_carriageway"
OUTPUT_DIR = "ML_Data/transformed_features"
ROAD_TO_PROCESS = "M60"  # Change this to process different roads (e.g., "M60", "M67", etc.)

# Constants
KMH_TO_MPH = 0.621371  # Conversion factor from km/h to mph
FREE_FLOW_SPEED = 70  # mph, typical motorway speed limit

def transform_features(file_path):
    """Transform features in a single cluster file for machine learning preparation."""
    try:
        print(f"\n{Colors.BLUE}Processing {os.path.basename(file_path)}{Colors.RESET}")
        
        # Load data
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # --- One-Hot Encode Match Features ---
        print(f"  {Colors.CYAN}One-hot encoding match features...{Colors.RESET}")
        if 'match_period' in df.columns and 'match_team' in df.columns:
            # Ensure the columns are treated as categoricals
            df['match_period'] = df['match_period'].astype('category')
            df['match_team'] = df['match_team'].astype('category')
            
            # Perform one-hot encoding
            df = pd.get_dummies(df, columns=['match_period'], prefix='is', drop_first=False)
            df = pd.get_dummies(df, columns=['match_team'], prefix='is', drop_first=False)
            
            # Rename columns for clarity and consistency (e.g., remove spaces)
            df.rename(columns={
                'is_pre_match': 'is_pre_match', 
                'is_post_match': 'is_post_match',
                'is_no_match_period': 'is_no_match_period', # Keep track of no match period if needed
                'is_Man United': 'is_united_match', 
                'is_Man City': 'is_city_match',
                'is_no_match_team': 'is_no_match_team' # Keep track of no match team if needed
            }, inplace=True)
            
            print(f"    {Colors.GREEN}Encoded match features: {list(df.filter(like='is_').columns)}{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}Match columns not found. Skipping encoding.{Colors.RESET}")
        # ------------------------------------

        # Create temporal features
        df['hour'] = df['datetime'].dt.hour
        
        # Convert speed to mph and round to 2 decimal places
        df['speed_mph'] = (df['fused_average_speed'] * KMH_TO_MPH).round(2)
        
        # Calculate vehicle type percentages
        print(f"  {Colors.CYAN}Calculating vehicle type percentages...{Colors.RESET}")
        total_flow = df[[f'traffic_flow_value{i}' for i in range(1, 5)]].sum(axis=1)
        for i in range(1, 5):
            col_name = f'traffic_flow_value{i}'
            pct_col_name = f'traffic_{i}_pct'
            df[pct_col_name] = (df[col_name] / total_flow * 100).round(2)
        
        print(f"    {Colors.GREEN}Added percentage columns: {[f'traffic_{i}_pct' for i in range(1, 5)]}{Colors.RESET}")
        
        # Calculate speed reduction (new target variable)
        df['speed_reduction'] = (FREE_FLOW_SPEED - df['speed_mph']).clip(lower=0).round(2)
        
        # Calculate normalized journey time (minutes per mile)
        # Convert travel time to minutes and length to miles
        # Ensure link_length is not zero or NaN to avoid division errors
        df['link_length_miles'] = df['link_length'] / 1609.34
        df['normalized_time'] = np.where(
            df['link_length_miles'] > 0,
            (df['fused_travel_time'] / 60) / df['link_length_miles'],
            np.nan  # Assign NaN if link length is zero or invalid
        ).round(2)
        
        # Calculate free-flow time (minutes per mile at 70 mph)
        FREE_FLOW_TIME = (60 / FREE_FLOW_SPEED)  # minutes per mile at 70 mph
        
        # Calculate delay (extra minutes per mile compared to free-flow)
        df['journey_delay'] = (df['normalized_time'] - FREE_FLOW_TIME).clip(lower=0).round(2)
        
        # Drop intermediate columns if not needed
        if 'link_length_miles' in df.columns: df.drop(columns=['link_length_miles'], inplace=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save transformed data
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_transformed.csv")
        df.to_csv(output_path, index=False)
        
        print(f"{Colors.GREEN}Successfully transformed and saved:{Colors.RESET}")
        print(f"  Rows: {len(df)}")
        print(f"  Target variable 1: speed_reduction (from {FREE_FLOW_SPEED} mph)")
        print(f"  Target variable 2: journey_delay (extra minutes/mile above {FREE_FLOW_TIME:.2f} min/mile)")
        
        return True
        
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Input file not found: {file_path}{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error processing {os.path.basename(file_path)}: {str(e)}{Colors.RESET}")
        return False

def process_road_clusters():
    """Process all cluster files for the specified road."""
    print(f"{Colors.MAGENTA}Starting feature transformation for {ROAD_TO_PROCESS}...{Colors.RESET}")
    
    # Get all cluster files for the specified road
    input_path = Path(INPUT_DIR)
    cluster_files = list(input_path.glob(f"{ROAD_TO_PROCESS}*_cluster.csv"))
    
    if not cluster_files:
        print(f"{Colors.YELLOW}No cluster files found for {ROAD_TO_PROCESS}{Colors.RESET}")
        return
    
    print(f"{Colors.BLUE}Found {len(cluster_files)} cluster files to process{Colors.RESET}")
    
    # Process each cluster file
    successful = 0
    failed = 0
    
    for file_path in cluster_files:
        if transform_features(file_path):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{Colors.MAGENTA}=== Processing Summary ==={Colors.RESET}")
    print(f"{Colors.GREEN}Successfully transformed: {successful} files{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}Failed to transform: {failed} files{Colors.RESET}")
    print(f"\nTransformed data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_road_clusters() 