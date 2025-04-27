#!/usr/bin/env python3
import os
import re
from pathlib import Path
from collections import defaultdict

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
# New target directory structure: "Merged Traffic Data/[road name]"
TARGET_DIR = os.path.join("Merged Traffic Data", ROAD_NAME)

def find_duplicates():
    """Find duplicate files and provide a concise summary"""
    # Make sure the target directory exists
    target_path = Path(TARGET_DIR).resolve()
    if not target_path.exists():
        print(f"\n{Colors.RED}{Colors.BOLD}ERROR: Merged folder '{TARGET_DIR}' does not exist!{Colors.RESET}")
        return
    
    csv_files = list(target_path.glob("*.csv"))
    
    # Extract link numbers from filenames
    link_numbers = {}
    for file in csv_files:
        # Extract the numeric ID at the end of the filename
        match = re.search(r'(\d+)\.csv$', file.name)
        if match:
            link_id = match.group(1)
            if link_id in link_numbers:
                link_numbers[link_id].append(file.name)
            else:
                link_numbers[link_id] = [file.name]
    
    # Find duplicates (same link ID in multiple files)
    duplicates = {link: files for link, files in link_numbers.items() if len(files) > 1}
    
    # Print concise summary
    print(f"\n{Colors.CYAN}{Colors.BOLD}=== MERGED FILES ANALYSIS FOR {ROAD_NAME} ==={Colors.RESET}")
    print(f"Files found in {TARGET_DIR}: {len(csv_files)}")
    print(f"Unique link IDs found: {len(link_numbers)}")
    
    # Check if we have the expected number of merged files (24)
    expected_count = 24
    if len(csv_files) != expected_count:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNING: Found {len(csv_files)} files, expected {expected_count}{Colors.RESET}")
        print(f"{Colors.YELLOW}Difference: {abs(len(csv_files) - expected_count)} {'more' if len(csv_files) > expected_count else 'fewer'} than expected{Colors.RESET}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}Success: Found exactly {expected_count} merged files as expected{Colors.RESET}")
    
    if duplicates:
        print(f"\n{Colors.RED}{Colors.BOLD}PROBLEM IDENTIFIED: Found {len(duplicates)} duplicate link numbers in Merged folder:{Colors.RESET}")
        for link, files in duplicates.items():
            print(f"\n{Colors.RED}Link ID {link} appears in {len(files)} files:{Colors.RESET}")
            for file in files:
                print(f"  - {file}")
            
        print(f"\n{Colors.YELLOW}{Colors.BOLD}RECOMMENDATION:{Colors.RESET}")
        print(f"{Colors.YELLOW}These files represent the same road section but weren't properly merged.{Colors.RESET}")
        print(f"{Colors.YELLOW}1. Run the combine_csv.py script again{Colors.RESET}")
        print(f"{Colors.YELLOW}2. Or manually merge these duplicate files{Colors.RESET}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}No duplicate link IDs found in Merged folder - good!{Colors.RESET}")
    
    # Check for files without link IDs
    no_id_files = [file.name for file in csv_files if not re.search(r'\d+\.csv$', file.name)]
    if no_id_files:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNING: Found {len(no_id_files)} files without link IDs:{Colors.RESET}")
        for file in no_id_files:
            print(f"  - {file}")
    
    print(f"\n{Colors.GREEN}Analysis complete!{Colors.RESET}")

if __name__ == "__main__":
    print(f"{Colors.MAGENTA}{Colors.BOLD}Starting duplicate analysis...{Colors.RESET}")
    find_duplicates() 