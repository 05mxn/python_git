"""
Script to add interaction features to the traffic data.

This script reads the clustered main carriageway data, adds interaction features 
(hour × is_weekend, hour × is_school_holiday), and modifies the files in place 
to include these new features.

This script should be run after add_hierarchical_day_features.py and before 
B3_final_preprocess.py.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# --- ANSI Color Codes ---
COLOR = {
    "HEADER": "\033[95m",  # Magenta
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "WARNING": "\033[93m", # Yellow
    "FAIL": "\033[91m",    # Red
    "ENDC": "\033[0m",     # End color
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}

def colored(text, color_name):
    """Applies ANSI color codes to text."""
    code = COLOR.get(color_name.upper(), COLOR["ENDC"])
    return f"{code}{text}{COLOR['ENDC']}"

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the road to process
ROAD_TO_PROCESS = "M60" 
DIRECTIONS = ["clockwise", "anticlockwise"] 

# Define base directories
BASE_ML_DATA_DIR = "ML_Data"
# Both input and output are the same directory since we're modifying in place
INPUT_DIR = os.path.join(BASE_ML_DATA_DIR, "clustered_main_carriageway")

def add_interaction_features(df):
    """
    Adds interaction features to the DataFrame:
    - hour_weekend: hour × is_weekend
    - hour_school_holiday: hour × is_school_holiday
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'hour', 'is_weekend', 'is_school_holiday' columns.
        
    Returns:
        pd.DataFrame: DataFrame with added interaction features.
    """
    # Check if required columns exist
    required_columns = ['is_weekend', 'is_school_holiday']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        missing_cols_str = ', '.join(missing_columns)
        logging.error(colored(f"Missing required columns: {missing_cols_str}", "FAIL"))
        raise ValueError(f"Missing required columns: {missing_cols_str}")
    
    # Create a copy to avoid modifying the input DataFrame
    enhanced_df = df.copy()
    
    # Create hour column from datetime if needed
    if 'hour' not in enhanced_df.columns:
        if 'datetime' in enhanced_df.columns:
            enhanced_df['datetime'] = pd.to_datetime(enhanced_df['datetime'])
            enhanced_df['hour'] = enhanced_df['datetime'].dt.hour
            logging.debug(f"Created 'hour' column from datetime")
        else:
            logging.error(colored(f"Missing required column: datetime", "FAIL"))
            raise ValueError(f"Missing required column: datetime")
    
    # Create hour_weekend interaction (hour × is_weekend)
    enhanced_df['hour_weekend'] = enhanced_df['hour'] * enhanced_df['is_weekend']
    
    # Create hour_school_holiday interaction (hour × is_school_holiday)
    enhanced_df['hour_school_holiday'] = enhanced_df['hour'] * enhanced_df['is_school_holiday']
    
    logging.debug(f"Added interaction features: hour_weekend, hour_school_holiday")
    
    return enhanced_df

def process_cluster_file(file_path):
    """Process a single cluster file by adding interaction features."""
    try:
        logging.info(colored(f"Processing {os.path.basename(file_path)}", "BLUE"))
        
        # Load data
        df = pd.read_csv(file_path)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Add interaction features
        df = add_interaction_features(df)
        logging.info(colored("Added interaction features: hour_weekend, hour_school_holiday", "GREEN"))
        
        # Save the modified file back to the same location
        df.to_csv(file_path, index=False)
        logging.info(colored(f"Successfully updated file: {file_path}", "GREEN"))
        
        return True
        
    except FileNotFoundError:
        logging.error(colored(f"Error: Input file not found: {file_path}", "FAIL"))
        return False
    except Exception as e:
        logging.error(colored(f"Error processing {os.path.basename(file_path)}: {str(e)}", "FAIL"))
        return False

def process_all_files():
    """
    Processes all files for the specified road and directions.
    """
    # Keep track of successes and failures
    successful = 0
    failed = 0
    
    # Process each direction
    for direction in DIRECTIONS:
        logging.info(colored(f"Processing direction: {direction}", "BLUE"))
        
        # Process file for direction
        file_name = f"{ROAD_TO_PROCESS}_{direction}_cluster.csv"
        input_path = os.path.join(INPUT_DIR, file_name)
        
        if os.path.exists(input_path):
            logging.info(colored(f"Processing file: {file_name}", "CYAN"))
            
            if process_cluster_file(input_path):
                successful += 1
            else:
                failed += 1
        else:
            logging.warning(colored(f"Input file not found: {input_path}", "WARNING"))
            failed += 1
    
    # Print summary
    logging.info(colored("=== Processing Summary ===", "HEADER"))
    logging.info(colored(f"Successfully processed: {successful} files", "GREEN"))
    if failed > 0:
        logging.info(colored(f"Failed to process: {failed} files", "FAIL"))
    
    return successful, failed

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(colored("Starting script to add interaction features...", "GREEN"))
    successful, failed = process_all_files()
    logging.info(colored(f"Script finished. Processed {successful} files successfully, {failed} files failed.", "GREEN"))
    logging.info(colored(f"You can now run B3_final_preprocess.py to create the train/val/test splits.", "CYAN")) 