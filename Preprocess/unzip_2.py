#!/usr/bin/env python3
import os
import zipfile
import glob
from pathlib import Path

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

def extract_zip_file(zip_path, extract_to=None, delete_original=True):
    """
    Extract a zip file to the specified directory or to its parent directory
    
    Args:
        zip_path (str): Path to the zip file
        extract_to (str, optional): Path to extract to. If None, extract to the same directory.
        delete_original (bool): Whether to delete the original zip file after extraction
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    # Create a folder for extraction with the name of the zip file (without extension)
    zip_name = os.path.basename(zip_path)
    folder_name = os.path.splitext(zip_name)[0]
    extract_folder = os.path.join(extract_to, folder_name)
    
    # Create extraction folder if it doesn't exist
    os.makedirs(extract_folder, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"{Colors.GREEN}Successfully extracted: {os.path.basename(zip_path)} to {os.path.basename(extract_folder)}{Colors.RESET}")
        
        # Delete the original zip file if requested
        if delete_original:
            try:
                os.remove(zip_path)
                print(f"{Colors.YELLOW}Deleted original zip file: {os.path.basename(zip_path)}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Error deleting zip file {os.path.basename(zip_path)}: {str(e)}{Colors.RESET}")
        
        return True
    except Exception as e:
        print(f"{Colors.RED}Error extracting {os.path.basename(zip_path)}: {str(e)}{Colors.RESET}")
        return False

def process_all_zip_files():
    """
    Process all zip files in the directory structure:
    BASE_DIR/month_*/[zip files]
    """
    # Get the full path to the base directory
    base_path = Path(BASE_DIR).resolve()
    
    # Get all month directories
    month_paths = list(base_path.glob(MONTH_PATTERN))
    
    if not month_paths:
        print(f"{Colors.YELLOW}No month directories found in {base_path}{Colors.RESET}")
        return
    
    total_files = 0
    extracted_files = 0
    
    # Process each month directory
    for month_path in sorted(month_paths):
        print(f"\n{Colors.CYAN}Processing: {month_path.name}{Colors.RESET}")
        
        # Find all zip files in this month directory
        zip_files = list(month_path.glob("*.zip"))
        
        if not zip_files:
            print(f"{Colors.YELLOW}  No zip files found in {month_path.name}{Colors.RESET}")
            continue
        
        print(f"{Colors.CYAN}  Found {len(zip_files)} zip files{Colors.RESET}")
        total_files += len(zip_files)
        
        # Process each zip file
        for zip_file in zip_files:
            success = extract_zip_file(str(zip_file))
            if success:
                extracted_files += 1
    
    # Print summary
    print(f"\n{Colors.CYAN}{Colors.BOLD}===== Summary ====={Colors.RESET}")
    print(f"Total zip files found: {total_files}")
    print(f"{Colors.GREEN}Files successfully extracted: {extracted_files}{Colors.RESET}")
    
    if total_files - extracted_files > 0:
        print(f"{Colors.RED}Files failed: {total_files - extracted_files}{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}All files extracted successfully!{Colors.RESET}")

if __name__ == "__main__":
    print(f"{Colors.MAGENTA}{Colors.BOLD}Starting to extract all zip files...{Colors.RESET}")
    process_all_zip_files()
    print(f"{Colors.GREEN}{Colors.BOLD}Done!{Colors.RESET}")
