#!/usr/bin/env python3
import os
import csv
import re
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Color definitions for terminal output
class Colors:
    # Text styles
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Text colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

# Configuration
BASE_DIR = "A57/2024"  # Base directory containing the original data
# Extract the road name from the BASE_DIR (assumes format "RoadName/Year")
ROAD_NAME = os.path.basename(os.path.dirname(BASE_DIR)) if os.path.dirname(BASE_DIR) else "A57"
# New merged directory structure: "Merged Traffic Data/[road name]"
MERGED_DIR = os.path.join("Merged Traffic Data", ROAD_NAME)

# Columns to drop
COLUMNS_TO_DROP = [
    'ntis_model_version', 
    'flow_quality', 
    'profile_travel_time', 
    'profile_traffic_flow', 
    'quality_index'
]

def convert_to_snake_case(header):
    """
    Convert a header string to snake_case
    
    Args:
        header (str): The original header string
        
    Returns:
        str: The header converted to snake_case
    """
    # First, strip any leading/trailing whitespace
    header = header.strip()
    
    # Replace spaces with underscores
    header = re.sub(r'\s+', '_', header)
    
    # Convert to lowercase
    header = header.lower()
    
    # Replace any non-alphanumeric chars (except underscores) with underscores
    header = re.sub(r'[^a-z0-9_]', '_', header)
    
    # Replace multiple consecutive underscores with a single one
    header = re.sub(r'_+', '_', header)
    
    # Remove leading/trailing underscores
    header = header.strip('_')
    
    return header

def round_to_15min(time_str):
    """
    Round a time string to the nearest 15-minute interval
    
    Args:
        time_str (str): Time in format 'HH:MM:SS'
        
    Returns:
        str: Rounded time in format 'HH:MM:00'
    """
    try:
        # Parse the time string
        t = datetime.strptime(time_str, '%H:%M:%S')
        
        # Calculate minutes since midnight
        minutes = t.hour * 60 + t.minute
        
        # Round to nearest 15 minutes
        rounded_minutes = round(minutes / 15) * 15
        
        # Convert back to hours and minutes
        rounded_hour = rounded_minutes // 60
        rounded_min = rounded_minutes % 60
        
        # Handle edge case of rounding to 24:00
        if rounded_hour == 24:
            rounded_hour = 0
        
        # Format as string (setting seconds to 00)
        return f"{rounded_hour:02d}:{rounded_min:02d}:00"
    except Exception:
        # Return the original string if parsing fails
        return time_str

def parse_coordinates(coord_str):
    """
    Parse coordinate string in format "lat ; lon" to separate values
    
    Args:
        coord_str (str): Coordinate string in format "lat ; lon"
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if parsing fails
    """
    if pd.isna(coord_str) or not isinstance(coord_str, str):
        return None, None
        
    try:
        # Split by semicolon and remove whitespace
        parts = [p.strip() for p in coord_str.split(';')]
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
        return None, None
    except Exception:
        return None, None

def process_csv_file(file_path):
    """
    Process a CSV file to:
    1. Convert headers to snake_case
    2. Round time to nearest 15 minutes
    3. Combine date and time into datetime
    4. Drop specified columns
    5. Convert data types
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First, read the CSV with pandas and convert headers to snake_case
        df = pd.read_csv(file_path)
        df.columns = [convert_to_snake_case(col) for col in df.columns]
        
        # Now continue with processing
        if 'local_date' not in df.columns or 'local_time' not in df.columns:
            print(f"{Colors.YELLOW}Warning: Required columns not found in {os.path.basename(file_path)}{Colors.RESET}")
            print(f"Available columns: {', '.join(df.columns)}")
            return False
        
        # Round the time to nearest 15 minutes
        df['local_time'] = df['local_time'].apply(round_to_15min)
        
        # Create a datetime column by combining date and time
        df['datetime'] = pd.to_datetime(df['local_date'] + ' ' + df['local_time'])
        
        # Drop original date and time columns
        df = df.drop(['local_date', 'local_time'], axis=1)
        
        # Parse start node coordinates
        if 'start_node_coordinates' in df.columns:
            df[['start_latitude', 'start_longitude']] = df['start_node_coordinates'].apply(
                lambda x: pd.Series(parse_coordinates(x))
            )
            df = df.drop('start_node_coordinates', axis=1)
        
        # Parse end node coordinates
        if 'end_node_coordinates' in df.columns:
            df[['end_latitude', 'end_longitude']] = df['end_node_coordinates'].apply(
                lambda x: pd.Series(parse_coordinates(x))
            )
            df = df.drop('end_node_coordinates', axis=1)
        
        # Reorder to put datetime first
        cols = df.columns.tolist()
        cols.remove('datetime')
        cols = ['datetime'] + cols
        df = df[cols]
        
        # Drop specified columns if they exist
        for col in COLUMNS_TO_DROP:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Convert numeric columns to appropriate types
        numeric_cols = [
            'day_type_id', 'ntis_link_number', 'link_length', 
            'fused_travel_time', 'fused_average_speed',
            'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # Try to convert to numeric, errors='coerce' sets invalid values to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with zeros
        df = df.fillna(0)
        
        # Save the processed file (overwrite original)
        df.to_csv(file_path, index=False)
        return True
    
    except Exception as e:
        print(f"{Colors.RED}Error processing {os.path.basename(file_path)}: {str(e)}{Colors.RESET}")
        return False

def process_all_csv_files(base_dir=BASE_DIR, merged_dir=MERGED_DIR):
    """
    Process all CSV files in the merged directory
    
    Args:
        base_dir (str): Base directory for the road
        merged_dir (str): Directory containing merged CSV files
    """
    # Make sure the merged directory exists
    merged_path = Path(merged_dir)
    if not merged_path.exists():
        print(f"{Colors.RED}{Colors.BOLD}ERROR: Merged folder '{merged_dir}' does not exist!{Colors.RESET}")
        return
    
    # Get all CSV files
    csv_files = list(merged_path.glob("*.csv"))
    
    if not csv_files:
        print(f"{Colors.YELLOW}No CSV files found in {merged_dir}{Colors.RESET}")
        return
    
    print(f"{Colors.CYAN}Processing {len(csv_files)} CSV files from {merged_dir}...{Colors.RESET}")
    
    # Process each file
    success_count = 0
    failed_files = []
    
    for file in csv_files:
        if process_csv_file(str(file)):
            success_count += 1
        else:
            failed_files.append(file.name)
    
    # Print summary
    print(f"\n{Colors.CYAN}{Colors.BOLD}=== Summary ==={Colors.RESET}")
    print(f"Total files processed: {len(csv_files)}")
    print(f"{Colors.GREEN}Successfully processed: {success_count}{Colors.RESET}")
    
    if failed_files:
        print(f"{Colors.RED}Failed: {len(failed_files)}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Files that failed processing:{Colors.RESET}")
        for file in failed_files:
            print(f"  - {file}")
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}All files processed successfully!{Colors.RESET}")

if __name__ == "__main__":
    print(f"{Colors.MAGENTA}{Colors.BOLD}Starting CSV preprocessing...{Colors.RESET}")
    process_all_csv_files()
    print(f"{Colors.GREEN}{Colors.BOLD}Done!{Colors.RESET}") 