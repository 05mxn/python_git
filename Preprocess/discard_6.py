# use --restore to restore from backup
# use --verbose to show detailed output 

# This script is used to discard all traffic data that is outside of the Manchester area.

import os
import pandas as pd
from pathlib import Path
import time
import shutil
import argparse
import sys

# Color definitions for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

# Manchester area polygon coordinates
MANCHESTER_POLYGON = [
    [-2.383906803032545, 53.35643233058647],
    [-2.2844199838671955, 53.35276058456604],
    [-2.1577834028942675, 53.39855439709439],
    [-2.0747205842307253, 53.417961163878886],
    [-2.0086414494866887, 53.45490425595645],
    [-2.1295617007563123, 53.590561549036835],
    [-2.2734006808037606, 53.60899768031835],
    [-2.4234124163263004, 53.58370090562224],
    [-2.495031973316337, 53.524247248597],
    [-2.4837446756599917, 53.43711930971146],
    [-2.383906803032545, 53.35643233058647]
]

# Directories
BASE_DATA_DIR = "Merged Traffic Data"
BACKUP_DIR = "Manchester_Backup"

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm
    
    Args:
        point: Tuple of (longitude, latitude)
        polygon: List of [longitude, latitude] points forming a polygon
        
    Returns:
        bool: True if point is inside or on edge of polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def is_within_manchester(start_lat, start_lon, end_lat, end_lon):
    """
    Check if a segment's coordinates fall within the Manchester polygon.
    A segment is considered within Manchester if either its start or end point is inside the polygon.
    
    Args:
        start_lat (float): Starting latitude
        start_lon (float): Starting longitude
        end_lat (float): Ending latitude
        end_lon (float): Ending longitude
        
    Returns:
        bool: True if the segment is within Manchester, False otherwise
    """
    # Check if start point is within polygon
    start_in_bounds = point_in_polygon((start_lon, start_lat), MANCHESTER_POLYGON)
    
    # If start point is within bounds, no need to check end point
    if start_in_bounds:
        return True
    
    # Check if end point is within bounds
    end_in_bounds = point_in_polygon((end_lon, end_lat), MANCHESTER_POLYGON)
    
    return end_in_bounds

def check_and_backup_csv_file(file_path, road_name):
    """Check if a CSV file's segment is within Manchester bounds and back it up if needed."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        if 'start_latitude' not in df.columns or 'start_longitude' not in df.columns or \
           'end_latitude' not in df.columns or 'end_longitude' not in df.columns:
            return True  # Keep files we can't evaluate
        
        # Get coordinates from first row
        first_row = df.iloc[0]
        start_lat = first_row['start_latitude']
        start_lon = first_row['start_longitude']
        end_lat = first_row['end_latitude']
        end_lon = first_row['end_longitude']
        
        # Check if segment is within Manchester
        if is_within_manchester(start_lat, start_lon, end_lat, end_lon):
            return True
        else:
            # Create backup before deleting
            backup_road_dir = os.path.join(BACKUP_DIR, road_name)
            os.makedirs(backup_road_dir, exist_ok=True)
            
            backup_path = os.path.join(backup_road_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            
            # Delete the file
            os.remove(file_path)
            return False
        
    except Exception as e:
        if '--verbose' in sys.argv:
            print(f"{Colors.RED}Error processing {os.path.basename(file_path)}: {str(e)}{Colors.RESET}")
        return True  # Keep files with errors to be safe

def process_road_directory(road_dir):
    """Process all CSV files in a road directory."""
    road_name = os.path.basename(road_dir)
    
    # Get all CSV files in this directory
    csv_files = list(Path(road_dir).glob("*.csv"))
    
    if not csv_files:
        if '--verbose' in sys.argv:
            print(f"{Colors.YELLOW}No CSV files found in {road_dir}{Colors.RESET}")
        return 0, 0
    
    files_kept = 0
    files_discarded = 0
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        # Show progress indicator every 10 files without newlines
        if i % 10 == 0 and i > 0 and '--verbose' not in sys.argv:
            print(".", end="", flush=True)
            
        if check_and_backup_csv_file(str(csv_file), road_name):
            files_kept += 1
        else:
            files_discarded += 1
    
    # Add a newline after progress dots
    if not '--verbose' in sys.argv and len(csv_files) > 10:
        print()
        
    return files_kept, files_discarded

def restore_from_backup():
    """Restore files from backup directory."""
    if not os.path.exists(BACKUP_DIR):
        print(f"{Colors.RED}Backup directory not found: {BACKUP_DIR}{Colors.RESET}")
        return
    
    print(f"{Colors.CYAN}Restoring files from backup...{Colors.RESET}")
    
    # Get all subdirectories (road directories) in backup
    road_dirs = [os.path.join(BACKUP_DIR, d) for d in os.listdir(BACKUP_DIR) 
                 if os.path.isdir(os.path.join(BACKUP_DIR, d))]
    
    if not road_dirs:
        print(f"{Colors.YELLOW}No road directories found in backup{Colors.RESET}")
        return
    
    total_restored = 0
    
    for backup_road_dir in road_dirs:
        road_name = os.path.basename(backup_road_dir)
        orig_road_dir = os.path.join(BASE_DATA_DIR, road_name)
        
        # Create original directory if it doesn't exist
        os.makedirs(orig_road_dir, exist_ok=True)
        
        # Get all CSV files in backup
        csv_files = list(Path(backup_road_dir).glob("*.csv"))
        
        road_restored = 0
        for csv_file in csv_files:
            orig_path = os.path.join(orig_road_dir, os.path.basename(csv_file))
            shutil.copy2(str(csv_file), orig_path)
            road_restored += 1
            total_restored += 1
        
        print(f"{Colors.GREEN}Restored {road_restored} files for {road_name}{Colors.RESET}")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Successfully restored {total_restored} files from backup!{Colors.RESET}")

def process_all_roads():
    """Process all road directories in the base data directory."""
    print(f"{Colors.MAGENTA}{Colors.BOLD}Starting Manchester area filtering...{Colors.RESET}")
    print(f"{Colors.CYAN}Using polygon with {len(MANCHESTER_POLYGON)} points to define Manchester area{Colors.RESET}")
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    # Make sure the base directory exists
    if not os.path.exists(BASE_DATA_DIR):
        print(f"{Colors.RED}Error: Base directory '{BASE_DATA_DIR}' not found{Colors.RESET}")
        return
    
    # Get all subdirectories (road directories)
    road_dirs = [os.path.join(BASE_DATA_DIR, d) for d in os.listdir(BASE_DATA_DIR) 
                 if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    
    if not road_dirs:
        print(f"{Colors.YELLOW}No road directories found in {BASE_DATA_DIR}{Colors.RESET}")
        return
    
    print(f"{Colors.CYAN}Found {len(road_dirs)} road directories{Colors.RESET}")
    start_time = time.time()
    
    # Process each road directory
    total_roads_processed = 0
    grand_total_kept = 0
    grand_total_discarded = 0
    
    for road_dir in sorted(road_dirs):
        road_name = os.path.basename(road_dir)
        print(f"{Colors.BLUE}Processing {road_name}...", end="", flush=True)
        
        kept, discarded = process_road_directory(road_dir)
        total = kept + discarded
        
        if total > 0:
            discarded_percent = (discarded / total * 100)
            print(f" Kept: {kept}, Discarded: {discarded} ({discarded_percent:.1f}%){Colors.RESET}")
            total_roads_processed += 1
            grand_total_kept += kept
            grand_total_discarded += discarded
        else:
            print(f" No files processed{Colors.RESET}")
    
    # Calculate percentages
    total_files = grand_total_kept + grand_total_discarded
    kept_percent = (grand_total_kept / total_files * 100) if total_files > 0 else 0
    discarded_percent = (grand_total_discarded / total_files * 100) if total_files > 0 else 0
    
    # Print final summary
    elapsed_time = time.time() - start_time
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}===== Manchester Area Filtering Summary ====={Colors.RESET}")
    print(f"Roads processed: {total_roads_processed}")
    print(f"Total files: {total_files}")
    print(f"{Colors.GREEN}Files kept (in Manchester): {grand_total_kept} ({kept_percent:.1f}%){Colors.RESET}")
    print(f"{Colors.YELLOW}Files discarded (outside Manchester): {grand_total_discarded} ({discarded_percent:.1f}%){Colors.RESET}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"\n{Colors.CYAN}Backup created in: {BACKUP_DIR}{Colors.RESET}")
    print(f"{Colors.CYAN}To restore: python Preprocess/discard_6.py --restore{Colors.RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter traffic data to Manchester area')
    parser.add_argument('--restore', action='store_true', help='Restore from backup')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    if args.restore:
        restore_from_backup()
    else:
        process_all_roads() 