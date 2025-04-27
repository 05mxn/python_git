import pandas as pd
import numpy as np
from datetime import timedelta
import os

# Configuration
MATCH_DATA_RAW = "match_data_raw.csv"  # Input raw match data
OUTPUT_DIR = "Preprocess"           # Directory to save processed data
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "match_windows.csv")

# Time window settings
PRE_MATCH_WINDOW = timedelta(hours=2)  # 2 hours before kickoff
MATCH_DURATION = timedelta(hours=2)    # Estimated match duration (including halftime)
POST_MATCH_WINDOW = timedelta(hours=2) # 2 hours after match ends

# Define color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'

def process_match_data(file_path):
    """Loads match data, calculates time windows, and returns a DataFrame."""
    print(f"{Colors.CYAN}Processing match data from: {file_path}{Colors.RESET}")
    
    try:
        # Load the raw match data
        df = pd.read_csv(file_path)
        print(f"  {Colors.GREEN}Loaded {len(df)} match records.{Colors.RESET}")
        
        # Combine Date and Kick-off Time into a single datetime column
        # Using errors='coerce' will turn unparseable dates/times into NaT
        df['kickoff_datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Kick-off Time'], errors='coerce'
        )
        
        # Drop rows where datetime parsing failed
        original_count = len(df)
        df.dropna(subset=['kickoff_datetime'], inplace=True)
        dropped_count = original_count - len(df)
        if dropped_count > 0:
            print(f"  {Colors.YELLOW}Warning: Dropped {dropped_count} rows due to invalid date/time formats.{Colors.RESET}")

        # Calculate time windows
        df['start_pre_match'] = df['kickoff_datetime'] - PRE_MATCH_WINDOW
        df['end_pre_match'] = df['kickoff_datetime']
        
        df['start_post_match'] = df['kickoff_datetime'] + MATCH_DURATION
        df['end_post_match'] = df['start_post_match'] + POST_MATCH_WINDOW
        
        # Select and rename relevant columns
        df_windows = df[['Team', 'Fixture', 'start_pre_match', 'end_pre_match', 'start_post_match', 'end_post_match']].copy()
        df_windows.rename(columns={'Team': 'match_team', 'Fixture': 'fixture'}, inplace=True)
        
        print(f"  {Colors.GREEN}Calculated pre-match and post-match windows.{Colors.RESET}")
        return df_windows

    except FileNotFoundError:
        print(f"{Colors.RED}Error: Input file not found at {file_path}{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}An error occurred: {e}{Colors.RESET}")
        return None

def main():
    print(f"{Colors.BOLD}Starting Match Data Processing...{Colors.RESET}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    match_windows_df = process_match_data(MATCH_DATA_RAW)
    
    if match_windows_df is not None:
        # Save the processed data
        try:
            match_windows_df.to_csv(OUTPUT_FILE, index=False)
            print(f"{Colors.GREEN}Successfully saved processed match windows to: {OUTPUT_FILE}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error saving processed data: {e}{Colors.RESET}")
    else:
        print(f"{Colors.RED}Failed to process match data. Exiting.{Colors.RESET}")

if __name__ == "__main__":
    main() 