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

# Configuration
BASE_DATA_DIR = "Merged Traffic Data"
OUTPUT_DIR = "Final Data"
LARGE_GAP_THRESHOLD = 6  # hours

# Columns to sum
SUM_COLUMNS = [
    'total_traffic_flow',
    'traffic_flow_value1',
    'traffic_flow_value2',
    'traffic_flow_value3',
    'traffic_flow_value4'
]

# Columns to average
AVG_COLUMNS = [
    'fused_travel_time',
    'fused_average_speed'
]

def process_segment_file(file_path, output_path):
    """Process a single segment file to create hourly aggregated data"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert datetime to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract hour and group by it
        df['hour'] = df['datetime'].dt.floor('H')
        
        # Count number of 15-min intervals per hour
        completeness = df.groupby('hour').size().reset_index(name='data_completeness')
        
        # Aggregate data
        agg_dict = {col: 'sum' for col in SUM_COLUMNS}
        agg_dict.update({col: 'mean' for col in AVG_COLUMNS})
        
        # For all other columns, take the first value
        other_cols = [col for col in df.columns if col not in SUM_COLUMNS + AVG_COLUMNS + ['datetime', 'hour']]
        agg_dict.update({col: 'first' for col in other_cols})
        
        # Perform aggregation
        hourly_data = df.groupby('hour').agg(agg_dict).reset_index()
        
        # Merge completeness info
        hourly_data = hourly_data.merge(completeness, on='hour')
        
        # Identify gaps
        hourly_data['time_diff'] = hourly_data['hour'].diff().dt.total_seconds() / 3600
        
        # Remove rows that are part of large gaps (>LARGE_GAP_THRESHOLD hours)
        hourly_data = hourly_data[hourly_data['time_diff'].fillna(0) <= LARGE_GAP_THRESHOLD]
        
        # Set is_interpolated flag where completeness < 4
        hourly_data['is_interpolated'] = (hourly_data['data_completeness'] < 4).astype(int)
        
        # Clean up and prepare final dataset
        hourly_data = hourly_data.rename(columns={'hour': 'datetime'})
        hourly_data = hourly_data.drop('time_diff', axis=1)
        
        # Save to output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        hourly_data.to_csv(output_path, index=False)
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}Error processing {file_path}: {str(e)}{Colors.RESET}")
        return False

def process_all_roads():
    """Process all road segments in the base directory"""
    print(f"{Colors.BOLD}Starting hourly data aggregation...{Colors.RESET}")
    
    # Get all road directories
    road_dirs = [d for d in Path(BASE_DATA_DIR).iterdir() if d.is_dir()]
    
    total_segments = 0
    processed_segments = 0
    
    for road_dir in road_dirs:
        road_name = road_dir.name
        print(f"\nProcessing {Colors.YELLOW}{road_name}{Colors.RESET}")
        
        # Create output directory for this road
        output_road_dir = Path(OUTPUT_DIR) / road_name
        
        # Process each segment file
        segment_files = list(road_dir.glob("*.csv"))
        total_segments += len(segment_files)
        
        for file_path in segment_files:
            output_path = output_road_dir / file_path.name
            if process_segment_file(file_path, output_path):
                processed_segments += 1
    
    # Print summary
    print(f"\n{Colors.GREEN}Processing complete:{Colors.RESET}")
    print(f"Successfully processed {processed_segments} out of {total_segments} segments")

if __name__ == "__main__":
    process_all_roads() 