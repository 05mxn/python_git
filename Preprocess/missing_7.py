#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'

# Base directory for data
BASE_DATA_DIR = "Merged Traffic Data"

# Define traffic flow column patterns to check
FLOW_COLUMNS = ['traffic_flow', 'total_traffic_flow', 'traffic_flow_value']

def discover_segment_types():
    """First pass: discover all unique segment types across files"""
    print(f"{Colors.BOLD}Discovering all unique segment types...{Colors.RESET}")
    
    unique_segment_types = set()
    carriageway_values = set()
    description_patterns = set()
    
    # Get all road directories
    road_dirs = []
    for item in os.listdir(BASE_DATA_DIR):
        road_path = os.path.join(BASE_DATA_DIR, item)
        if os.path.isdir(road_path):
            road_dirs.append(road_path)
    
    if not road_dirs:
        print(f"No road directories found in {BASE_DATA_DIR}")
        return None, None
    
    csv_file_count = 0
    
    # Process each road
    for road_dir in sorted(road_dirs):
        road_name = os.path.basename(road_dir)
        # Get all CSV files
        csv_files = list(Path(road_dir).glob("*.csv"))
        
        # Sample files (process max 50 files per road to speed up discovery)
        sample_size = min(50, len(csv_files))
        sampled_files = csv_files[:sample_size]
        
        csv_file_count += len(sampled_files)
        
        # Process each file in the sample
        for file_path in sampled_files:
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Collect unique carriageway values
                if 'carriageway' in df.columns:
                    values = df['carriageway'].dropna().unique()
                    for val in values:
                        carriageway_values.add(str(val).strip())
                
                # Collect unique description patterns
                if 'ntis_link_description' in df.columns:
                    descriptions = df['ntis_link_description'].dropna().unique()
                    for desc in descriptions:
                        desc_lower = str(desc).lower()
                        
                        # Extract key patterns from descriptions
                        if 'slip' in desc_lower:
                            description_patterns.add('slip')
                        if 'junction' in desc_lower:
                            description_patterns.add('junction')
                        if 'roundabout' in desc_lower:
                            description_patterns.add('roundabout')
                        if 'main' in desc_lower:
                            description_patterns.add('main')
                
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)} during discovery: {e}")
    
    print(f"Scanned {csv_file_count} files across {len(road_dirs)} roads")
    
    # Combine discovered types
    print("\nDiscovered segment types:")
    print(f"  {Colors.BOLD}From 'carriageway' column:{Colors.RESET}")
    for val in sorted(carriageway_values):
        print(f"    - {val}")
        unique_segment_types.add(val)
    
    print(f"  {Colors.BOLD}From 'ntis_link_description' patterns:{Colors.RESET}")
    for pattern in sorted(description_patterns):
        segment_type = f"{pattern}Road" if pattern != 'main' else 'mainCarriageway'
        print(f"    - {segment_type}")
        unique_segment_types.add(segment_type)
    
    # Always add an 'unknown' category
    unique_segment_types.add('unknown')
    
    print(f"\nTotal unique segment types discovered: {len(unique_segment_types)}")
    print("These will be used for the detailed analysis\n")
    
    return unique_segment_types, road_dirs

def analyze_segment_types():
    """Analyze which types of road segments are missing traffic flow data"""
    # First discover all segment types
    unique_segment_types, road_dirs = discover_segment_types()
    
    if not unique_segment_types or not road_dirs:
        return
    
    print(f"{Colors.BOLD}Analyzing segment types with missing traffic flow data...{Colors.RESET}")
    
    # Initialize counters for segment types
    segment_stats = defaultdict(lambda: {'total': 0, 'missing_flow': 0})
    
    start_time = time.time()
    total_files = 0
    
    # Process each road
    for road_dir in sorted(road_dirs):
        road_name = os.path.basename(road_dir)
        print(f"Analyzing {road_name}...", end="", flush=True)
        
        # Get all CSV files
        csv_files = list(Path(road_dir).glob("*.csv"))
        if not csv_files:
            print(" No files found")
            continue
        
        total_files += len(csv_files)
        
        # Process each file
        for file_path in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Determine segment type based on our discovered types
                segment_type = 'unknown'  # Default
                
                # Try to get segment type from carriageway column
                if 'carriageway' in df.columns and not df['carriageway'].empty:
                    value = str(df['carriageway'].iloc[0]).strip()
                    if not pd.isna(value) and value != 'nan' and value:
                        segment_type = value
                
                # If using the default, try link description
                if segment_type == 'unknown' and 'ntis_link_description' in df.columns and not df['ntis_link_description'].empty:
                    description = str(df['ntis_link_description'].iloc[0]).lower()
                    
                    # Extract segment type from description
                    if 'slip' in description:
                        segment_type = 'slipRoad'
                    elif 'junction' in description:
                        segment_type = 'junction'
                    elif 'roundabout' in description:
                        segment_type = 'roundabout'
                    elif 'main' in description:
                        segment_type = 'mainCarriageway'
                
                # Update total count for this segment type
                segment_stats[segment_type]['total'] += 1
                
                # Identify traffic flow columns in this file
                flow_cols = []
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in FLOW_COLUMNS):
                        flow_cols.append(col)
                
                # Check if flow data is missing (>=50% zeros in any flow column)
                has_missing_flow = False
                for col in flow_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        zero_percent = (df[col] == 0).mean() * 100
                        if zero_percent >= 50:
                            has_missing_flow = True
                            break
                
                # Update missing flow count for this segment type
                if has_missing_flow:
                    segment_stats[segment_type]['missing_flow'] += 1
                
            except Exception as e:
                print(f"\nError processing {os.path.basename(file_path)}: {e}")
        
        print(f" Done")
    
    # Calculate percentages and sort by missing rate
    segment_results = []
    for segment_type in unique_segment_types:
        stats = segment_stats[segment_type]
        if stats['total'] > 0:
            missing_percent = (stats['missing_flow'] / stats['total']) * 100
            segment_results.append({
                'segment_type': segment_type,
                'total': stats['total'],
                'missing_flow': stats['missing_flow'],
                'missing_percent': missing_percent
            })
    
    # Sort by missing percentage (highest first)
    segment_results.sort(key=lambda x: x['missing_percent'], reverse=True)
    
    # Print results
    print(f"\n{Colors.BOLD}=== Segment Type Analysis ==={Colors.RESET}")
    print(f"{'Segment Type':<20} {'Total':<8} {'Missing':<8} {'Missing %':<8}")
    print("-" * 46)
    
    for result in segment_results:
        # Choose color based on missing percentage
        if result['missing_percent'] >= 75:
            color = Colors.RED
        elif result['missing_percent'] >= 25:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN
            
        print(f"{color}{result['segment_type']:<20} {result['total']:<8} {result['missing_flow']:<8} {result['missing_percent']:.1f}%{Colors.RESET}")
    
    # Print conclusion
    print(f"\n{Colors.BOLD}=== Summary ==={Colors.RESET}")
    complete_types = [r['segment_type'] for r in segment_results if r['missing_percent'] < 25 and r['total'] >= 5]
    missing_types = [r['segment_type'] for r in segment_results if r['missing_percent'] >= 75 and r['total'] >= 5]
    
    print(f"{Colors.GREEN}Segment types with complete flow data:{Colors.RESET}")
    if complete_types:
        for segment_type in complete_types:
            print(f"  - {segment_type}")
    else:
        print("  None found (with at least 5 samples)")
    
    print(f"\n{Colors.RED}Segment types with missing flow data:{Colors.RESET}")
    if missing_types:
        for segment_type in missing_types:
            print(f"  - {segment_type}")
    else:
        print("  None found (with at least 5 samples)")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalyzed {total_files} files in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    analyze_segment_types()