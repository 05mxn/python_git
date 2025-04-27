import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Get the root directory (one level up from Preprocess)
ROOT_DIR = SCRIPT_DIR.parent

# --- Configuration ---
INPUT_DIR = ROOT_DIR / "Seperation of Data" / "main_carriageway"
WEATHER_FILE = ROOT_DIR / "weather_raw.csv"  # Correct path relative to root
OUTPUT_DIR = ROOT_DIR / "ML_Data" / "clustered_main_carriageway"
MATCH_WINDOWS_FILE = ROOT_DIR / "Preprocess" / "match_windows.csv" # Path relative to root

# List of roads to process (can be modified as needed)
ROADS_TO_PROCESS = ["M60"]  # Add or remove roads as needed
# ---------------------

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

def determine_direction(description):
    """
    Determines direction from NTIS link description.
    Uses more precise matching to avoid partial matches.
    """
    description_lower = str(description).lower()
    
    # Check for anticlockwise first (more specific) before clockwise
    if 'anti-clockwise' in description_lower or 'anticlockwise' in description_lower:
        return 'anticlockwise'
    if 'clockwise' in description_lower and 'anti' not in description_lower and 'anti-' not in description_lower:
        return 'clockwise'
    
    # Other directions
    if 'eastbound' in description_lower:
        return 'eastbound'
    if 'westbound' in description_lower:
        return 'westbound'
    if 'northbound' in description_lower:
        return 'northbound'
    if 'southbound' in description_lower:
        return 'southbound'
    
    return 'other'  # Fallback category

def load_and_prepare_weather_data(file_path):
    """Loads and prepares weather data."""
    try:
        weather_df = pd.read_csv(file_path)
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        weather_df = weather_df.set_index('datetime')
        # Select and rename relevant weather columns
        weather_df = weather_df[['temperature_2m', 'weather_code', 'wind_speed_10m']].rename(columns={
            'temperature_2m': 'temperature',
            'wind_speed_10m': 'wind_speed'
        })
        # Resample to hourly frequency, taking the mean for temp/wind, mode for weather code
        weather_df = weather_df.resample('H').agg({
            'temperature': 'mean',
            'weather_code': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            'wind_speed': 'mean'
        }).ffill().reset_index()
        print(f"  {Colors.GREEN}Successfully loaded and prepared weather data.{Colors.RESET}")
        return weather_df
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Weather file not found at {file_path}{Colors.RESET}")
        return None
    except KeyError as e:
        print(f"{Colors.RED}Error loading weather data: Missing expected column - {e}{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}Error loading weather data: {e}{Colors.RESET}")
        return None

def load_match_windows(file_path):
    """Loads processed match windows data."""
    try:
        match_df = pd.read_csv(file_path)
        # Convert window columns to datetime
        for col in ['start_pre_match', 'end_pre_match', 'start_post_match', 'end_post_match']:
            match_df[col] = pd.to_datetime(match_df[col])
        print(f"  {Colors.GREEN}Successfully loaded match windows data.{Colors.RESET}")
        return match_df
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Match windows file not found at {file_path}{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}Error loading match windows data: {e}{Colors.RESET}")
        return None

def add_match_features(traffic_df, match_windows_df):
    """Adds match period and team features to the traffic dataframe."""
    if match_windows_df is None:
        print(f"  {Colors.YELLOW}Skipping match feature addition as match windows data is not available.{Colors.RESET}")
        # Add default columns if match data isn't loaded
        traffic_df['match_period'] = "no_match"
        traffic_df['match_team'] = "no_match"
        return traffic_df
        
    # Ensure traffic datetime is in the correct format
    traffic_df['datetime'] = pd.to_datetime(traffic_df['datetime'])
    
    # Initialize new columns with default values
    traffic_df['match_period'] = "no_match"
    traffic_df['match_team'] = "no_match"
    
    print(f"  {Colors.CYAN}Adding match features to traffic data...{Colors.RESET}")
    
    # --- Efficient Check using Time Intervals ---
    # Create interval indices for faster lookups (optional but good for performance)
    # This might require pandas >= 1.0
    # For simplicity, we'll stick to iterating through matches first.
    
    # Iterate through each match window
    for _, match_row in tqdm(match_windows_df.iterrows(), total=len(match_windows_df), desc="Processing matches"):
        # Find traffic data points within the pre-match window
        pre_match_mask = (
            (traffic_df['datetime'] >= match_row['start_pre_match']) & 
            (traffic_df['datetime'] < match_row['end_pre_match'])
        )
        traffic_df.loc[pre_match_mask, 'match_period'] = "pre_match"
        traffic_df.loc[pre_match_mask, 'match_team'] = match_row['match_team']
        
        # Find traffic data points within the post-match window
        post_match_mask = (
            (traffic_df['datetime'] >= match_row['start_post_match']) & 
            (traffic_df['datetime'] < match_row['end_post_match'])
        )
        traffic_df.loc[post_match_mask, 'match_period'] = "post_match"
        traffic_df.loc[post_match_mask, 'match_team'] = match_row['match_team']

    print(f"  {Colors.GREEN}Match features added.{Colors.RESET}")
    return traffic_df

def process_road(road_name, weather_df, match_windows_df):
    """Processes all data for a specific road."""
    print(f"\n{Colors.MAGENTA}Processing Road: {road_name}{Colors.RESET}")
    road_dir = os.path.join(INPUT_DIR, road_name)
    
    if not os.path.isdir(road_dir):
        print(f"{Colors.YELLOW}Warning: Directory not found for road {road_name}. Skipping.{Colors.RESET}")
        return

    all_files = [os.path.join(road_dir, f) for f in os.listdir(road_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"{Colors.YELLOW}Warning: No CSV files found for road {road_name}. Skipping.{Colors.RESET}")
        return

    road_data = []
    print(f"  {Colors.CYAN}Loading traffic data files...{Colors.RESET}")
    for file_path in tqdm(all_files, desc=f"Loading {road_name} files"):
        try:
            df = pd.read_csv(file_path)
            # Determine direction from description (assuming description exists)
            if 'ntis_link_description' in df.columns:
                df['direction'] = df['ntis_link_description'].apply(determine_direction)
            else:
                df['direction'] = 'unknown' # Handle missing description
            road_data.append(df)
        except Exception as e:
            print(f"{Colors.RED}Error loading file {os.path.basename(file_path)}: {e}{Colors.RESET}")
            continue # Skip this file
            
    if not road_data:
      print(f"{Colors.RED}No valid traffic data loaded for {road_name}. Skipping clustering.{Colors.RESET}")
      return

    # Combine all data for the road
    combined_df = pd.concat(road_data, ignore_index=True)
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df = combined_df.sort_values('datetime')
    
    print(f"  {Colors.GREEN}Loaded {len(combined_df)} total records for {road_name}.{Colors.RESET}")

    # Add match features
    combined_df = add_match_features(combined_df, match_windows_df)

    # Merge with weather data
    if weather_df is not None:
        print(f"  {Colors.CYAN}Merging with weather data...{Colors.RESET}")
        combined_df['datetime_hour'] = combined_df['datetime'].dt.floor('h')
        # Ensure weather_df datetime is also timezone naive if combined_df is, or vice versa
        if combined_df['datetime_hour'].dt.tz is None:
            weather_df['datetime'] = weather_df['datetime'].dt.tz_localize(None)
        
        merged_df = pd.merge(combined_df, weather_df, left_on='datetime_hour', right_on='datetime', how='left', suffixes= ('', '_weather'))
        # Drop redundant datetime columns
        merged_df = merged_df.drop(columns=['datetime_hour', 'datetime_weather'])
        print(f"  {Colors.GREEN}Weather data merged.{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}Weather data not available. Skipping merge.{Colors.RESET}")
        merged_df = combined_df # Use the combined df if weather merge fails

    # Group by direction and save clusters
    print(f"  {Colors.CYAN}Grouping by direction and saving clusters...{Colors.RESET}")
    grouped = merged_df.groupby('direction')
    saved_count = 0
    for direction, group in grouped:
        if direction == 'other' or direction == 'unknown':
            print(f"    {Colors.YELLOW}Skipping group '{direction}' as it's not a primary direction.{Colors.RESET}")
            continue
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        output_filename = f"{road_name}_{direction}_cluster.csv"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        try:
            group.to_csv(output_filepath, index=False)
            print(f"    {Colors.GREEN}Saved cluster: {output_filename} ({len(group)} rows){Colors.RESET}")
            saved_count += 1
        except Exception as e:
            print(f"    {Colors.RED}Error saving cluster {output_filename}: {e}{Colors.RESET}")
            
    if saved_count == 0:
        print(f"  {Colors.YELLOW}Warning: No valid direction clusters were saved for {road_name}.{Colors.RESET}")

def main():
    print(f"{Colors.BOLD}Starting Clustering and Weather/Match Data Merging...{Colors.RESET}")
    
    # Load weather data once
    print(f"{Colors.CYAN}Loading external data...{Colors.RESET}")
    weather_df = load_and_prepare_weather_data(WEATHER_FILE)
    match_windows_df = load_match_windows(MATCH_WINDOWS_FILE)

    # Process each specified road
    if not ROADS_TO_PROCESS:
        print(f"{Colors.YELLOW}Warning: No roads specified in ROADS_TO_PROCESS. Exiting.{Colors.RESET}")
        return
        
    for road in ROADS_TO_PROCESS:
        process_road(road, weather_df, match_windows_df)
        
    print(f"\n{Colors.BOLD}Processing complete.{Colors.RESET}")

if __name__ == "__main__":
    main()