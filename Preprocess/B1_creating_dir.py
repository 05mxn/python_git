#!/usr/bin/env python3
"""
This script organizes traffic data for machine learning by:
1. Creating directory structure for ML data (all segments and main carriageway)
2. Copying files from the "Final Data" directory to appropriate ML directories
3. Filtering files to separate all segments from main carriageway segments
"""

import os
import shutil
from pathlib import Path

# Base directories
BASE_DATA_DIR = "Final Data"
ML_DATA_DIR = "ML_Data"

# List of roads to process (can be modified as needed)
ROADS_TO_PROCESS = ["M60"]  # Add or remove roads as needed

# Subdirectories for all segments and main carriageway
ALL_SEGMENTS_DIR = os.path.join(ML_DATA_DIR, "all_segments")
MAIN_CARRIAGEWAY_DIR = os.path.join(ML_DATA_DIR, "main_carriageway")

# Color definitions for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'

def copy_files(src_dir, dest_dir, filter_func=None):
    """
    Copy files from source to destination directory for specified roads.
    
    Args:
        src_dir: Source directory
        dest_dir: Destination directory
        filter_func: Optional function to filter files
    """
    total_copied = 0
    
    for road_name in ROADS_TO_PROCESS:
        road_dir = Path(src_dir) / road_name
        if not road_dir.exists() or not road_dir.is_dir():
            print(f"{Colors.YELLOW}Warning: Directory for {road_name} not found in {src_dir}. Skipping.{Colors.RESET}")
            continue
            
        print(f"{Colors.BLUE}Processing {road_name}...{Colors.RESET}")
        
        # Create destination road directory
        dest_road_dir = Path(dest_dir) / road_name
        os.makedirs(dest_road_dir, exist_ok=True)
        
        # Copy files
        files_copied = 0
        for file in road_dir.glob("*.csv"):
            if filter_func is None or filter_func(file):
                shutil.copy(file, dest_road_dir / file.name)
                files_copied += 1
                total_copied += 1
        
        print(f"{Colors.GREEN}✓ Copied {files_copied} files for {road_name}{Colors.RESET}")
    
    return total_copied

def main():
    print(f"{Colors.MAGENTA}Starting data organization for roads: {', '.join(ROADS_TO_PROCESS)}{Colors.RESET}")
    print("=" * 70)
    
    # Create directory structure
    print(f"{Colors.BLUE}Creating directory structure...{Colors.RESET}")
    os.makedirs(ALL_SEGMENTS_DIR, exist_ok=True)
    os.makedirs(MAIN_CARRIAGEWAY_DIR, exist_ok=True)
    print(f"{Colors.GREEN}✓ Directory structure created{Colors.RESET}")
    
    # Copy all segments
    print(f"\n{Colors.BLUE}Copying all segments...{Colors.RESET}")
    all_segments_copied = copy_files(BASE_DATA_DIR, ALL_SEGMENTS_DIR)
    
    # Copy only main carriageway segments
    print(f"\n{Colors.BLUE}Copying main carriageway segments...{Colors.RESET}")
    main_carriageway_copied = copy_files(
        BASE_DATA_DIR, 
        MAIN_CARRIAGEWAY_DIR, 
        lambda f: "mainCarriageway" in f.name
    )
    
    # Print summary
    print(f"\n{Colors.MAGENTA}=== Processing Summary ==={Colors.RESET}")
    print(f"{Colors.GREEN}Total files copied to all_segments: {all_segments_copied}{Colors.RESET}")
    print(f"{Colors.GREEN}Total files copied to main_carriageway: {main_carriageway_copied}{Colors.RESET}")
    print(f"{Colors.MAGENTA}------------------------{Colors.RESET}")

if __name__ == "__main__":
    main() 