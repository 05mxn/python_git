#!/usr/bin/env python3
import os
import csv
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

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
BASE_DIR = "A57/2024"  # Base directory containing month folders
MONTH_PATTERN = "month_*"  # Pattern to match month folders
# Extract the road name from the BASE_DIR (assumes format "RoadName/Year")
ROAD_NAME = os.path.basename(os.path.dirname(BASE_DIR)) if os.path.dirname(BASE_DIR) else "A57"
# New target directory structure: "Merged Traffic Data/[road name]"
TARGET_DIR = os.path.join("Merged Traffic Data", ROAD_NAME)

def read_csv_file(csv_path):
    """
    Read a CSV file, returning headers and data rows
    Skip empty rows at the bottom
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        tuple: (headers, data_rows) where headers is a list and data_rows is a list of lists
    """
    headers = []
    data_rows = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            
            # Read all non-empty rows
            for row in reader:
                # Skip empty rows (all values empty or only containing whitespace)
                if row and any(cell.strip() for cell in row):
                    data_rows.append(row)
        
        return headers, data_rows
    
    except Exception as e:
        print(f"{Colors.RED}Error reading {os.path.basename(csv_path)}: {str(e)}{Colors.RESET}")
        return headers, data_rows

def extract_section_id(folder_name):
    """
    Extract the numeric ID from a section folder name
    
    Args:
        folder_name (str): Section folder name
        
    Returns:
        str: Section ID or None if not found
    """
    # Extract the numeric ID at the end (e.g., "200049562" from "A57 eastbound access from A5063 entrySlipRoad 200049562")
    match = re.search(r'\s+(\d+)$', folder_name)
    if match:
        return match.group(1)
    return None

def identify_road_sections():
    """
    Identify all unique road sections across all month folders
    Group them by ID and find most common name for each ID
    
    Returns:
        dict: Map of section IDs to (most_common_name, list_of_csv_files)
    """
    base_path = Path(BASE_DIR).resolve()
    section_id_files = defaultdict(list)  # Maps ID to list of (folder_name, csv_file) tuples
    
    # Get all month directories
    month_paths = sorted(base_path.glob(MONTH_PATTERN))
    
    for month_path in month_paths:
        # Get all section folders in this month
        section_folders = [d for d in month_path.iterdir() if d.is_dir()]
        
        for section_folder in section_folders:
            section_name = section_folder.name
            section_id = extract_section_id(section_name)
            
            if section_id:
                # Find the CSV file in this section folder
                csv_files = list(section_folder.glob("*.csv"))
                
                if csv_files:
                    # There should be just one CSV file per section folder
                    section_id_files[section_id].append((section_name, str(csv_files[0])))
    
    # For each ID, find the most common name and list of CSV files
    result = {}
    for section_id, name_file_pairs in section_id_files.items():
        names = [name for name, _ in name_file_pairs]
        name_counter = Counter(names)
        most_common_name = name_counter.most_common(1)[0][0]
        csv_files = [file for _, file in name_file_pairs]
        
        result[section_id] = (most_common_name, csv_files)
    
    return result

def check_missing_days(data_rows):
    """
    Check for missing days in the data
    
    Args:
        data_rows (list): List of data rows from CSV
        
    Returns:
        list: List of missing dates
    """
    # Extract dates from data
    dates = set()
    for row in data_rows:
        if len(row) > 0:  # Make sure row has at least one element
            try:
                # Assuming date is in the first column and in format YYYY-MM-DD
                date_str = row[0].strip()
                if date_str and date_str != "Local Date":  # Skip header or empty dates
                    dates.add(date_str)
            except Exception:
                pass
    
    # Convert string dates to datetime objects
    date_objs = []
    for date_str in dates:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_objs.append(date_obj)
        except ValueError:
            pass
    
    if not date_objs:
        return []
    
    # Find min and max dates
    min_date = min(date_objs)
    max_date = max(date_objs)
    
    # Generate list of all dates in range
    all_dates = set()
    current_date = min_date
    while current_date <= max_date:
        all_dates.add(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    # Find missing dates
    missing_dates = all_dates - dates
    
    return sorted(list(missing_dates))

def combine_csv_files(section_id, section_name, csv_files, target_dir):
    """
    Combine multiple CSV files for a section
    
    Args:
        section_id (str): ID of the road section
        section_name (str): Standardized name of the road section
        csv_files (list): List of CSV files to combine
        target_dir (str): Directory to save the combined file
        
    Returns:
        tuple: (combined_file_path, missing_days_count, row_count)
    """
    headers = None
    all_data = []
    
    # Read all CSV files for this section
    for csv_file in sorted(csv_files):
        file_headers, data_rows = read_csv_file(csv_file)
        
        # Set headers from first file
        if headers is None:
            headers = file_headers
        
        # Add data rows
        all_data.extend(data_rows)
    
    # Sort data by date and time (assuming first and second columns)
    all_data.sort(key=lambda x: (x[0], x[1]) if len(x) >= 2 else (x[0], ""))
    
    # Create target file path using standardized name
    clean_section_name = section_name.replace('/', '_')
    target_file = os.path.join(target_dir, f"{clean_section_name}.csv")
    
    # Write combined data
    try:
        with open(target_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write headers
            if headers:
                writer.writerow(headers)
            # Write data
            writer.writerows(all_data)
        
        # Check for missing days
        missing_days = check_missing_days(all_data)
        missing_days_count = len(missing_days)
        
        return target_file, missing_days_count, len(all_data)
    
    except Exception as e:
        print(f"{Colors.RED}Error writing {os.path.basename(target_file)}: {str(e)}{Colors.RESET}")
        return None, 0, 0

def process_all_sections():
    """
    Process all road sections
    """
    print(f"{Colors.CYAN}Identifying road sections...{Colors.RESET}")
    section_data = identify_road_sections()
    
    print(f"{Colors.CYAN}Found {len(section_data)} unique road sections{Colors.RESET}")
    
    # Create target directory if it doesn't exist (creating parent directories if needed)
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Process each section ID
    section_results = []
    successful_sections = 0
    failed_sections = []
    
    print(f"{Colors.CYAN}Combining CSV files for {len(section_data)} sections...{Colors.RESET}")
    print(f"{Colors.CYAN}Files will be saved to: {TARGET_DIR}{Colors.RESET}")
    
    for section_id, (section_name, csv_files) in section_data.items():
        if csv_files:
            target_file, missing_days, row_count = combine_csv_files(section_id, section_name, csv_files, TARGET_DIR)
            if target_file:
                section_results.append((section_id, section_name, target_file, missing_days, row_count))
                successful_sections += 1
            else:
                failed_sections.append(section_name)
    
    # Print summary report
    print(f"\n{Colors.CYAN}{Colors.BOLD}===== Summary Report ====={Colors.RESET}")
    print(f"Total road sections: {len(section_data)}")
    print(f"{Colors.GREEN}Successfully combined: {successful_sections}{Colors.RESET}")
    
    if failed_sections:
        print(f"{Colors.RED}Failed: {len(failed_sections)}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Failed sections:{Colors.RESET}")
        for name in failed_sections:
            print(f"  - {name}")
    
    # Show missing days
    missing_data_sections = [(section_id, name, days) for section_id, name, _, days, _ in section_results if days > 0]
    if missing_data_sections:
        print(f"\n{Colors.YELLOW}Sections with missing days:{Colors.RESET}")
        for section_id, name, days in missing_data_sections:
            print(f"  {Colors.YELLOW}ID {section_id}: {name} - missing data {days} days{Colors.RESET}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Done!{Colors.RESET}")

if __name__ == "__main__":
    print(f"{Colors.MAGENTA}{Colors.BOLD}Starting to combine CSV files...{Colors.RESET}")
    process_all_sections() 