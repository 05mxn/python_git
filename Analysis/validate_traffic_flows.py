#!/usr/bin/env python3

"""
Traffic Flow Data Validator

This script analyzes CSV files containing traffic flow data to validate data integrity.
It checks for several types of inconsistencies in vehicle count data:

1. Validation Checks:
   - Rows where individual vehicle counts exceed total flow
   - Rows where sum of vehicle types doesn't match total flow
   - Rows with zero total flow
   - Missing required columns

2. Output:
   - Number and percentage of invalid rows
   - Sample of problematic rows
   - Summary statistics for each file
   
3. Usage:
   - Set TARGET_DIR to the directory containing CSV files to analyze
   - Set ANALYZE_ALL_FILES to True to check all files, False to check one random file
   - Set EXCLUDED_TERMS to modify which terms in filenames will cause files to be excluded
   - Color-coded output for easy reading:
     * Green: Valid data
     * Yellow: Warnings
     * Red: Invalid data
     * Blue/Cyan: Information
     * Magenta: File names
"""

import os
import pandas as pd
from pathlib import Path
import random

# Configuration
TARGET_DIR = "ML_Data/transformed_features"  # Change this to analyze different directories
ANALYZE_ALL_FILES = False  # Set to True to analyze all files, False for one random file
EXCLUDED_TERMS = [  # Add terms here to exclude files containing these strings (case-insensitive)
    'connector',
    'link',
    'roundabout',
    'entry',
    'exit',
    'onramp',
    'offramp',
    'onramp',
    'offramp',
    'onramp',
    'slip',
]

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

def is_valid_file(file_path):
    """Check if the file should be included in analysis."""
    name_lower = file_path.name.lower()
    return not any(term.lower() in name_lower for term in EXCLUDED_TERMS)

def validate_traffic_flows(directory):
    """Validate traffic flow data in CSV files within the specified directory."""
    print(f"{Colors.BLUE}Analyzing traffic flow data in: {directory}{Colors.RESET}\n")
    print(f"{Colors.YELLOW}Excluding files containing these terms: {', '.join(EXCLUDED_TERMS)}{Colors.RESET}\n")
    
    # Find all CSV files in directory and filter out excluded files
    all_csv_files = list(Path(directory).glob("*.csv"))
    csv_files = [f for f in all_csv_files if is_valid_file(f)]
    
    if not csv_files:
        print(f"{Colors.RED}No valid CSV files found in directory.{Colors.RESET}")
        if all_csv_files:
            print(f"{Colors.YELLOW}Found {len(all_csv_files)} CSV files but all were excluded due to naming rules.{Colors.RESET}")
        return
    
    # Select files to analyze
    if ANALYZE_ALL_FILES:
        files_to_analyze = csv_files
        print(f"{Colors.CYAN}Found {len(csv_files)} valid CSV files to analyze.{Colors.RESET}\n")
    else:
        files_to_analyze = [random.choice(csv_files)]
        print(f"{Colors.CYAN}Randomly selected file: {files_to_analyze[0].name}{Colors.RESET}\n")
    
    # Process each file
    for file_path in files_to_analyze:
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['total_traffic_flow'] + [f'traffic_flow_value{i}' for i in range(1, 5)]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"{Colors.YELLOW}Skipping {file_path.name} - Missing columns: {', '.join(missing_columns)}{Colors.RESET}\n")
                continue
            
            print(f"{Colors.MAGENTA}Analyzing: {file_path.name}{Colors.RESET}")
            print(f"Total rows: {len(df):,}")
            
            # Check for invalid total flow
            zero_flow = (df['total_traffic_flow'] == 0).sum()
            if zero_flow > 0:
                print(f"{Colors.YELLOW}Found {zero_flow:,} rows with zero total flow ({zero_flow/len(df)*100:.1f}%){Colors.RESET}")
            
            # Check each vehicle type
            invalid_counts = {}
            for i in range(1, 5):
                col = f'traffic_flow_value{i}'
                invalid_mask = df[col] > df['total_traffic_flow']
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    invalid_counts[col] = invalid_count
            
            if invalid_counts:
                print(f"\n{Colors.RED}Found rows where vehicle counts exceed total flow:{Colors.RESET}")
                for col, count in invalid_counts.items():
                    print(f"  - {col}: {count:,} rows ({count/len(df)*100:.1f}%)")
                
                # Sample of invalid rows if any found
                print(f"\n{Colors.YELLOW}Sample of invalid rows:{Colors.RESET}")
                # Fix: Create a mask for rows where any vehicle count exceeds total flow
                invalid_mask = pd.Series(False, index=df.index)
                for i in range(1, 5):
                    col = f'traffic_flow_value{i}'
                    invalid_mask = invalid_mask | (df[col] > df['total_traffic_flow'])
                
                invalid_rows = df[invalid_mask]
                sample = invalid_rows.head(3)
                for _, row in sample.iterrows():
                    print(f"  Total flow: {row['total_traffic_flow']}")
                    for i in range(1, 5):
                        col = f'traffic_flow_value{i}'
                        print(f"    {col}: {row[col]}")
                    print()
            else:
                print(f"{Colors.GREEN}No invalid vehicle counts found!{Colors.RESET}")
            
            # Check sum of vehicle types vs total
            flow_sum = df[[f'traffic_flow_value{i}' for i in range(1, 5)]].sum(axis=1)
            sum_mismatch = (abs(flow_sum - df['total_traffic_flow']) > 1).sum()
            
            if sum_mismatch > 0:
                print(f"\n{Colors.YELLOW}Found {sum_mismatch:,} rows where sum of vehicle types doesn't match total ({sum_mismatch/len(df)*100:.1f}%){Colors.RESET}")
            
            print("\n" + "-"*80 + "\n")
            
        except Exception as e:
            print(f"{Colors.RED}Error processing {file_path.name}: {str(e)}{Colors.RESET}\n")

if __name__ == "__main__":
    validate_traffic_flows(TARGET_DIR) 