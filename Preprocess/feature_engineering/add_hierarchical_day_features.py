import pandas as pd
import os
import logging

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

# Define the road to process (adjust as needed)
ROAD_TO_PROCESS = "M60"
DIRECTIONS = ["clockwise", "anticlockwise"] # Adjust if other directions are needed

# Define base directories based on the project structure
BASE_ML_DATA_DIR = "ML_Data"
# --- Reads from B2 output ---
CLUSTERED_DIR = os.path.join(BASE_ML_DATA_DIR, "clustered_main_carriageway")
# --- MODIFIED: Write back to the SAME directory (OVERWRITES originals) ---
OUTPUT_DIR = CLUSTERED_DIR # Set output dir same as input dir

# Define the mapping rules based on discussion
# Original day_type_id mapping for reference:
# 0: First working day, 1-3: Normal Tue-Thu, 4: Last working day
# 5: Saturday, 6: Sunday
# 7: First day school holidays, 9: Mid-week school holidays, 11: Last day school holidays
# 12: Bank Holidays
# 13: Christmas period, 14: Christmas/New Year's Day

WEEKEND_IDS = [5, 6, 12] # Saturday, Sunday, Bank Holiday
SCHOOL_HOLIDAY_IDS = [7, 9, 11, 13, 14] # School holidays + Christmas period

# --- Helper Functions ---

def add_hierarchical_features(df):
    """
    Adds 'is_weekend' and 'is_school_holiday' columns based on 'day_type_id'.
    Drops the original 'day_type_id' column after transformation.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'day_type_id' column.

    Returns:
        pd.DataFrame: DataFrame with added hierarchical features and 'day_type_id' removed.
    """
    if 'day_type_id' not in df.columns:
        logging.error(colored("'day_type_id' column not found in the DataFrame.", "FAIL"))
        raise ValueError("'day_type_id' column is required.")

    # Apply mapping rules
    df['is_weekend'] = df['day_type_id'].apply(lambda x: 1 if x in WEEKEND_IDS else 0)
    df['is_school_holiday'] = df['day_type_id'].apply(lambda x: 1 if x in SCHOOL_HOLIDAY_IDS else 0)

    # --- ADDED: Drop the original day_type_id column ---
    df = df.drop(columns=['day_type_id'])
    logging.debug(f"Added hierarchical features and dropped 'day_type_id'.")

    return df

def process_clustered_road_data(road_name, directions):
    """
    Processes combined cluster files for a given road and its directions,
    adding hierarchical day features.

    Args:
        road_name (str): The name of the road (e.g., "M60").
        directions (list): List of directions (e.g., ["clockwise", "anticlockwise"]).
    """
    logging.info(colored(f"--- Processing Road: {road_name} (Clustered Data - Overwriting) ---", "HEADER"))

    # Ensure the output directory exists (though it should already)
    # --- MODIFIED: Use new output directory variable ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory (will overwrite): {OUTPUT_DIR}")

    for direction in directions:
        logging.info(colored(f"  Processing Direction: {direction}", "BLUE"))
        # Original B2 output filename format
        file_name = f"{road_name}_{direction}_cluster.csv"
        # --- MODIFIED: Use single directory variable for input/output ---
        file_path = os.path.join(CLUSTERED_DIR, file_name)
        # output_path remains the same as file_path now

        # --- MODIFIED: Check existence using unified file_path ---
        if os.path.exists(file_path):
            logging.info(colored(f"    Processing file (will overwrite): {file_name}", "CYAN"))
            try:
                # Load original clustered data
                # --- MODIFIED: Load using unified file_path ---
                df = pd.read_csv(file_path)
                logging.debug(f"      Loaded {file_path} with shape {df.shape}")

                # Add new features (and drop original day_type_id)
                df_enhanced = add_hierarchical_features(df.copy())

                # Save enhanced data (OVERWRITING original)
                # --- MODIFIED: Save using unified file_path ---
                df_enhanced.to_csv(file_path, index=False)
                logging.info(colored(f"      Successfully saved OVERWRITTEN data to {file_path}", "GREEN"))

                # Optional: Verify new columns and absence of old one
                logging.debug(f"      Columns after processing: {list(df_enhanced.columns)}")
                logging.debug(f"      Value counts for is_weekend: {str(df_enhanced['is_weekend'].value_counts(dropna=False))}")
                logging.debug(f"      Value counts for is_school_holiday: {str(df_enhanced['is_school_holiday'].value_counts(dropna=False))}")

            except FileNotFoundError:
                 # This case should be less likely now if B2 ran correctly
                logging.warning(colored(f"    Input file not found: {file_path}", "WARNING"))
            except KeyError as e:
                 logging.error(colored(f"    Key error processing {file_name}: {e}. Check column names.", "FAIL"))
            except Exception as e:
                logging.error(colored(f"    An error occurred processing {file_name}: {e}", "FAIL"))
        else:
             logging.warning(colored(f"    Input file skipped (does not exist): {file_path}", "WARNING"))

    logging.info(colored(f"--- Finished processing road: {road_name} (Clustered Data - Overwritten) ---", "HEADER"))

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(colored("Starting script to add hierarchical day features to clustered data...", "GREEN"))
    # --- MODIFIED: Call new processing function ---
    process_clustered_road_data(ROAD_TO_PROCESS, DIRECTIONS)
    logging.info(colored("Script finished successfully.", "GREEN")) 