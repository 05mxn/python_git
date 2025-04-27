#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'

# Configuration
INPUT_DIR = "ML_Data/transformed_features"
OUTPUT_DIR = "ML_Data/processed_clusters"
ROAD_TO_PROCESS = "M60"  # Change this to process different roads
SPLIT_METHOD = "chronological"  # Options: "chronological", "kfold", "time_series_kfold"
K_FOLDS = 5  # Number of folds for K-fold cross-validation

# Constants for chronological split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def split_chronological(df, output_prefix):
    """Split data chronologically into train, validation, and test sets."""
    print(f"  {Colors.CYAN}Performing chronological split...{Colors.RESET}")
    
    # Calculate split indices
    n = len(df)
    train_idx = int(n * TRAIN_SPLIT)
    val_idx = train_idx + int(n * VAL_SPLIT)
    
    # Check for sufficient data in splits
    min_train_size = 100
    min_val_test_size = 50
    if train_idx < min_train_size or (val_idx - train_idx) < min_val_test_size or (n - val_idx) < min_val_test_size:
        print(f"{Colors.YELLOW}Warning: Not enough data for meaningful splits (Train: {train_idx}, Val: {val_idx-train_idx}, Test: {n-val_idx}). Skipping file.{Colors.RESET}")
        return False
    
    # Split the data chronologically
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]
    
    # Save splits
    train_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_train.csv")
    val_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_val.csv")
    test_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"{Colors.GREEN}Successfully saved chronological splits:{Colors.RESET}")
    print(f"  Train: {len(train_df)} rows ({TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation: {len(val_df)} rows ({VAL_SPLIT*100:.0f}%)")
    print(f"  Test: {len(test_df)} rows ({TEST_SPLIT*100:.0f}%)")
    
    return True

def split_kfold(df, output_prefix):
    """Split data using K-fold cross-validation."""
    print(f"  {Colors.CYAN}Performing {K_FOLDS}-fold cross-validation...{Colors.RESET}")
    
    # Create output folders for each fold
    folds_dir = os.path.join(OUTPUT_DIR, f"{output_prefix}_folds")
    os.makedirs(folds_dir, exist_ok=True)
    
    # Set aside test data (last TEST_SPLIT portion)
    n = len(df)
    test_idx = int(n * (1 - TEST_SPLIT))
    test_df = df[test_idx:]
    train_val_df = df[:test_idx]
    
    # Save test set once, as it's the same for all folds
    test_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"{Colors.GREEN}Created test set with {len(test_df)} rows ({TEST_SPLIT*100:.0f}%){Colors.RESET}")
    
    # Create K-fold splitter
    kf = KFold(n_splits=K_FOLDS, shuffle=False)  # No shuffle to preserve order
    
    # Create and save each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df), 1):
        # Create fold directory
        fold_dir = os.path.join(folds_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        fold_train = train_val_df.iloc[train_idx]
        fold_val = train_val_df.iloc[val_idx]
        
        # Save fold data
        fold_train.to_csv(os.path.join(fold_dir, f"{output_prefix}_train.csv"), index=False)
        fold_val.to_csv(os.path.join(fold_dir, f"{output_prefix}_val.csv"), index=False)
        
        print(f"  Fold {fold}: Train={len(fold_train)} rows, Val={len(fold_val)} rows")
    
    print(f"{Colors.GREEN}Successfully saved {K_FOLDS} folds to {folds_dir}{Colors.RESET}")
    return True

def split_time_series_kfold(df, output_prefix):
    """Split data using TimeSeriesSplit for time series data."""
    print(f"  {Colors.CYAN}Performing Time Series {K_FOLDS}-fold cross-validation...{Colors.RESET}")
    
    # Create output folders for each fold
    folds_dir = os.path.join(OUTPUT_DIR, f"{output_prefix}_time_folds")
    os.makedirs(folds_dir, exist_ok=True)
    
    # Set aside test data (last TEST_SPLIT portion)
    n = len(df)
    test_idx = int(n * (1 - TEST_SPLIT))
    test_df = df[test_idx:]
    train_val_df = df[:test_idx]
    
    # Save test set once, as it's the same for all folds
    test_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"{Colors.GREEN}Created test set with {len(test_df)} rows ({TEST_SPLIT*100:.0f}%){Colors.RESET}")
    
    # Create TimeSeriesSplit splitter
    tscv = TimeSeriesSplit(n_splits=K_FOLDS)
    
    # Create and save each fold
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val_df), 1):
        # Create fold directory
        fold_dir = os.path.join(folds_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        fold_train = train_val_df.iloc[train_idx]
        fold_val = train_val_df.iloc[val_idx]
        
        # Save fold data
        fold_train.to_csv(os.path.join(fold_dir, f"{output_prefix}_train.csv"), index=False)
        fold_val.to_csv(os.path.join(fold_dir, f"{output_prefix}_val.csv"), index=False)
        
        print(f"  Fold {fold}: Train={len(fold_train)} rows, Val={len(fold_val)} rows")
    
    print(f"{Colors.GREEN}Successfully saved {K_FOLDS} time series folds to {folds_dir}{Colors.RESET}")
    return True

def process_file(file_path):
    """Process a single file for data splitting."""
    try:
        print(f"\n{Colors.BLUE}Processing {os.path.basename(file_path)}{Colors.RESET}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Ensure datetime is parsed correctly and sorted
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
        # Set up output path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Remove "_transformed" suffix if it exists
        output_prefix = base_name.replace('_transformed', '')
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Apply the selected splitting method
        if SPLIT_METHOD == "chronological":
            return split_chronological(df, output_prefix)
        elif SPLIT_METHOD == "kfold":
            return split_kfold(df, output_prefix)
        elif SPLIT_METHOD == "time_series_kfold":
            return split_time_series_kfold(df, output_prefix)
        else:
            print(f"{Colors.RED}Error: Unknown split method '{SPLIT_METHOD}'{Colors.RESET}")
            return False
        
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Input file not found: {file_path}{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error processing {os.path.basename(file_path)}: {str(e)}{Colors.RESET}")
        return False

def process_all_files():
    """Process all transformed files for the specified road."""
    print(f"{Colors.MAGENTA}Starting data splitting for {ROAD_TO_PROCESS} using {SPLIT_METHOD} method...{Colors.RESET}")
    
    # Get all transformed files for the specified road
    input_path = Path(INPUT_DIR)
    files = list(input_path.glob(f"{ROAD_TO_PROCESS}*_transformed.csv"))
    
    if not files:
        print(f"{Colors.YELLOW}No transformed files found for {ROAD_TO_PROCESS}{Colors.RESET}")
        return
    
    print(f"{Colors.BLUE}Found {len(files)} files to process{Colors.RESET}")
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in files:
        if process_file(file_path):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{Colors.MAGENTA}=== Processing Summary ==={Colors.RESET}")
    print(f"{Colors.GREEN}Successfully processed: {successful} files{Colors.RESET}")
    if failed > 0:
        print(f"{Colors.RED}Failed to process: {failed} files{Colors.RESET}")
    print(f"\nProcessed data saved to: {OUTPUT_DIR}")
    if SPLIT_METHOD in ["kfold", "time_series_kfold"]:
        print(f"Fold-specific data saved to: {OUTPUT_DIR}/[filename]_folds")

if __name__ == "__main__":
    process_all_files() 