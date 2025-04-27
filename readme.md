# Traffic Data Processing Project Documentation

## Project Overview

This project processes traffic data for the M60 motorway in Manchester. The data pipeline handles extraction of zipped data files, combining CSV files from multiple months, checking for duplicates, cleaning and preprocessing the data, incorporating weather and local match-day information, and preparing it for machine learning analysis. While currently focused on the M60, the architecture is designed to support expansion to other roads (e.g., M67, M61) in the future.

## Directory Structure

```
├── Final Data/                # Hourly-aggregated traffic data for each road/segment (main input for ML)
├── ML_Data/                   # Machine Learning ready data and results
│   ├── clustered_main_carriageway/  # Clustered data with weather & raw match info (output of B2, modified by day/interaction features)
│   └── processed_clusters/    # Final train/validation/test splits ready for ML (output of B4)
│   └── model_results_multioutput/ # Model results and visualizations
│   └── model_results_tuned/ # Model results for singe output model
│   └── transformed_features/ # Files ready to passed to B4 and split
├── Preprocess/                # Data processing utility scripts
│   ├── B0_match_features.py
│   ├── B1_creating_dir.py
│   ├── B2_cluster.py
│   ├── B3_feature_transform.py
│   ├── B4_data_split.py
│   ├── check_duplicates_4.py
│   ├── clean_5.py
│   ├── combine_csv_3.py
│   ├── discard_6.py
│   ├── download_files_1.py
│   ├── inter_8.py
│   ├── missing_7.py
│   ├── unzip_2.py
│   ├── day_features/
│   │   └── add_hierarchical_day_features.py
│   ├── feature_engineering/
│   │   └── add_interaction_features.py
│   └── match_windows.csv
├── xgboost/                   # XGBoost model training and evaluation scripts
│   ├── standard_regression.py
│   ├── train_evaluate_multioutput.py
│   ├── train_evaluate_xgboost.py
│   ├── tune_multioutput.py
│   ├── tune_multioutput_optuna.py
│   └── tune_xgboost.py
│   └── visualize_results.py
├── match_data_raw.csv         # Raw CSV containing football match schedules
├── weather_raw.csv            # Raw CSV containing hourly weather data
```

## Data Structure

The final processed CSV files (`processed_clusters` folder) ready for machine learning contain the following columns:

### Input Features (X):
- `hour`: Hour of day (0-23)
- `weather_code`: Weather condition code (numeric, from weather data).
- `temperature`: Temperature in Celsius (from merged weather data).
- `wind_speed`: Wind speed in m/s (from merged weather data).
- **Hierarchical Day Features:**
    - `is_weekend`: Binary flag (1 if Saturday, Sunday, or Bank Holiday; 0 otherwise). Replaces complexity of `day_type_id`.
    - `is_school_holiday`: Binary flag (1 if during defined school holidays or the Christmas period; 0 otherwise). Replaces complexity of `day_type_id`.
- **Match Features (Binary/One-Hot Encoded):**
    - `is_pre_match`: 1 if time falls within 2 hours before a Man Utd/Man City match kickoff, 0 otherwise.
    - `is_post_match`: 1 if time falls within 2 hours after estimated match end, 0 otherwise.
    - `is_united_match`: 1 if it's a Man United match period (pre or post), 0 otherwise.
    - `is_city_match`: 1 if it's a Man City match period (pre or post), 0 otherwise.
    - `is_no_match_period` (Optional, may be dropped): 1 if not pre/post match.
    - `is_no_match_team` (Optional, may be dropped): 1 if not a United/City match.
- **Interaction Features:**
    - `hour_weekend`: Interaction between hour and is_weekend (hour × is_weekend).
    - `hour_school_holiday`: Interaction between hour and is_school_holiday (hour × is_school_holiday).

#### Day Type ID Details (OBSOLETE - Replaced by `is_weekend` and `is_school_holiday`):
- ~~0: First working day of normal week~~
- ~~1: Normal working Tuesday~~
- ~~2: Normal working Wednesday~~
- ~~3: Normal working Thursday~~
- ~~4: Last working day of normal week~~
- ~~5: Saturday (excluding type 14)~~ -> `is_weekend=1`
- ~~6: Sunday (excluding type 14)~~ -> `is_weekend=1`
- ~~7: First day of school holidays~~ -> `is_school_holiday=1`
- ~~9: Middle of week - school holidays~~ -> `is_school_holiday=1`
- ~~11: Last day of week - school holidays~~ -> `is_school_holiday=1`
- ~~12: Bank Holidays (including Good Friday)~~ -> `is_weekend=1`
- ~~13: Christmas period holidays~~ -> `is_school_holiday=1`
- ~~14: Christmas Day/New Year's Day~~ -> `is_school_holiday=1`

### Target Variable (y):
- `total_traffic_flow`: Total vehicle count per hour (used in current model).
- OR `speed_reduction`: Reduction in speed from free-flow speed (70 mph), clipped to non-negative values.
- OR `journey_delay`: Calculated delay in minutes per mile compared to free-flow.

### Other Columns (Present in intermediate files or final file but not used as features/target):
- `datetime`: Timestamp of the record.
- `day_type_id`: (Present in early stages, **removed** after `add_hierarchical_day_features.py`)
- `fused_average_speed`: Original speed data.
- `speed_mph`: Raw speed in miles per hour (converted from km/h).
# ... (and potentially others depending on the stage)

## Processing Scripts

### 1. download_files_1.py (in Preprocess folder)

This script uses Selenium to download road data files from the National Highways website.

#### Key Functions:
- `download_file()`: Downloads a file from a URL to a specified path
- `process_download_link()`: Processes a download link and saves the file
- `download_zip_files()`: Navigates to a specified road's data and downloads all relevant ZIP files
- `process_roads()`: Processes multiple roads, downloading data for the specified year and months

#### Configuration:
- Roads list (e.g., `["A57(M)", "A56", "A5103", "A34", "A6", "M60", "M602", "M62", "M56"]`)
- Year to download (e.g., "2024")
- Months to download (e.g., `[1, 2, 3, 4, 5]` for January through May)

#### Output:
- Downloaded ZIP files organized by road/year/month

### 2. unzip_2.py (in Preprocess folder)

This script extracts zip files from monthly directories, creating folders with the same names as the zip files.

#### Key Functions:
- `extract_zip_file()`: Extracts a single zip file, optionally deleting the original
- `process_all_zip_files()`: Finds all zip files in monthly directories and extracts them

#### Configuration:
- `BASE_DIR`: Base directory containing month folders (default: "A57/2024")
- `MONTH_PATTERN`: Pattern to match month folders (default: "month_*")

#### Output:
- Extracted folders containing CSV files for each road section

### 3. combine_csv_3.py (in Preprocess folder)

This script combines CSV files from different months for each unique road section.

#### Key Functions:
- `read_csv_file()`: Reads a CSV file, returning headers and data rows
- `extract_section_id()`: Extracts the numeric ID from a section folder name
- `identify_road_sections()`: Identifies all unique road sections across all month folders
- `check_missing_days()`: Checks for missing days in the combined data
- `combine_csv_files()`: Combines multiple CSV files for a section
- `process_all_sections()`: Processes all road sections

#### Configuration:
- `BASE_DIR`: Base directory containing month folders (default: "A57/2024")
- `MONTH_PATTERN`: Pattern to match month folders (default: "month_*")
- `ROAD_NAME`: Name of the road extracted from the BASE_DIR
- `TARGET_DIR`: Where to save the combined files (default: "Merged Traffic Data/[road_name]")

#### Output:
- Combined CSV files in the "Merged Traffic Data/[road_name]" directory
- Summary report of processed sections and any missing days

### 4. check_duplicates_4.py (in Preprocess folder)

This script checks for duplicate files in the merged directory based on link IDs.

#### Key Functions:
- `find_duplicates()`: Finds and reports on duplicate files

#### Configuration:
- `BASE_DIR`: Base directory (default: "A57/2024")
- `ROAD_NAME`: Name of the road extracted from the BASE_DIR
- `TARGET_DIR`: Directory containing merged files (default: "Merged Traffic Data/[road_name]")

#### Output:
- Analysis report of merged files
- Warnings about duplicate link IDs or files without link IDs

### 5. clean_5.py (in Preprocess folder)

This script cleans and preprocesses the merged CSV files.

#### Key Functions:
- `convert_to_snake_case()`: Converts column headers to snake_case format
- `round_time_to_15min()`: Rounds time values to the nearest 15 minutes
- `parse_coordinates()`: Parses coordinate strings into separate lat/long fields
- `process_csv_file()`: Processes a single CSV file
- `process_all_csv_files()`: Processes all CSV files in the merged directory

#### Configuration:
- `BASE_DIR`: Base directory (default: "A57/2024")
- `ROAD_NAME`: Name of the road extracted from the BASE_DIR
- `MERGED_DIR`: Directory containing merged files (default: "Merged Traffic Data/[road_name]")

#### Transformations:
1. Converting column headers to snake_case
2. Rounding time to nearest 15 minutes
3. Combining date and time into a datetime field
4. Parsing coordinates into separate fields
5. Filling missing values with zeros
6. Converting numeric columns to appropriate types

#### Output:
- Cleaned and preprocessed CSV files (overwriting the original files)
- Summary of processed files

### 6. discard_6.py (in Preprocess folder)

This script filters data for the Manchester area.

#### Key Functions:
- `filter_data()`: Filters data based on location

#### Configuration:
- `BASE_DIR`: Base directory (default: "A57/2024")
- `ROAD_NAME`: Name of the road extracted from the BASE_DIR
- `TARGET_DIR`: Directory containing merged files (default: "Merged Traffic Data/[road_name]")

#### Output:
- Filtered CSV files in the "Merged Traffic Data/[road_name]" directory

### 7. missing_7.py (in Preprocess folder)

This script checks for missing data patterns.

#### Key Functions:
- `check_missing_data()`: Checks for missing data in the merged directory

#### Configuration:
- `BASE_DIR`: Base directory (default: "A57/2024")
- `ROAD_NAME`: Name of the road extracted from the BASE_DIR
- `TARGET_DIR`: Directory containing merged files (default: "Merged Traffic Data/[road_name]")

#### Output:
- Analysis report of missing data

### 8. inter_8.py (in Preprocess folder)

This script aggregates 15-minute traffic data to hourly intervals, handling missing data and gaps appropriately.

#### Key Functions:
- `process_segment_file()`: Processes a single segment file to create hourly aggregated data
- `process_all_roads()`: Processes all road segments in the base directory

#### Configuration:
- `BASE_DATA_DIR`: Directory containing merged traffic data (default: "Merged Traffic Data")
- `OUTPUT_DIR`: Directory for hourly aggregated data (default: "Final Data")
- `LARGE_GAP_THRESHOLD`: Threshold for identifying large data gaps (default: 6 hours)

#### Data Aggregation:
1. **Columns to Sum**:
   - total_traffic_flow
   - traffic_flow_value1 through traffic_flow_value4

2. **Columns to Average**:
   - fused_travel_time
   - fused_average_speed

3. **Columns to Keep First Value**:
   - All other columns (day_type_id, road, carriageway, etc.)

#### Quality Indicators:
- `data_completeness`: Number of 15-minute intervals present in each hour (0-4)
- `is_interpolated`: Flag indicating if any data was interpolated (0/1)

#### Output:
- Hourly aggregated CSV files in "Final Data/[road_name]/" directory
- Maintains original file naming convention
- Preserves data structure while reducing temporal resolution

### 9. B0_match_features.py (in Preprocess folder)

This script processes raw match data to create relevant time windows for analysis.

#### Key Functions:
- `process_match_data()`: Loads `match_data_raw.csv`, parses dates/times, calculates pre-match (2 hours before kickoff) and post-match (2 hours after estimated end time) windows.

#### Configuration:
- `MATCH_DATA_RAW`: Path to the input match schedule CSV.
- `OUTPUT_FILE`: Path to save the processed windows (`Preprocess/match_windows.csv`).
- Time window durations (pre-match, match duration estimate, post-match).

#### Output:
- `Preprocess/match_windows.csv` containing columns: `match_team`, `fixture`, `start_pre_match`, `end_pre_match`, `start_post_match`, `end_post_match`.

### 10. B1_creating_dir.py (in Preprocess folder)

This script creates the directory structure for machine learning data preparation (Note: May be less relevant now as `B2` and `B3` handle specific input/output dirs).

#### Key Functions:
- `copy_files()`: Copies files to appropriate directories with optional filtering.

#### Configuration:
- `BASE_DATA_DIR`: Source directory with hourly data (e.g., `Final Data`).
- `ML_DATA_DIR`, `ALL_SEGMENTS_DIR`, `MAIN_CARRIAGEWAY_DIR`.

#### Output:
- Organized directory structure under `ML_Data/`.

### 11. B2_cluster.py (in Preprocess folder)

This script clusters road segments by direction and merges the hourly traffic data with weather and match-day information.

#### Key Functions:
- `load_and_prepare_weather_data()`: Loads `weather_raw.csv`, prepares hourly weather data.
- `load_match_windows()`: Loads the processed match time windows from `Preprocess/match_windows.csv`.
- `add_match_features()`: Adds categorical `match_period` ('pre_match', 'post_match', 'no_match') and `match_team` ('Man United', 'Man City', 'no_match') columns to traffic data based on timestamps falling within match windows.
- `determine_direction()`: Identifies road direction from NTIS link descriptions.
- Merges combined hourly traffic data (now including raw match features) with hourly weather data based on the hour.
- Groups data by road and direction, saving clusters.

#### Configuration:
- `INPUT_DIR`: Directory containing hourly main carriageway segments (e.g., `Seperation of Data/main_carriageway` or `Final Data`).
- `OUTPUT_DIR`: Directory for clustered data (`ML_Data/clustered_main_carriageway`).
- `WEATHER_FILE`: Path to weather data CSV (`weather_raw.csv`).
- `MATCH_WINDOWS_FILE`: Path to processed match windows CSV (`Preprocess/match_windows.csv`).

#### Output:
- Clustered CSV files (e.g., `M60_clockwise_cluster.csv`) with merged weather and raw (categorical) match features in `ML_Data/clustered_main_carriageway/`. **These files are subsequently modified by `add_hierarchical_day_features.py`.**

### 12. add_hierarchical_day_features.py (in Preprocess/day_features folder)

This script transforms the complex `day_type_id` into simpler hierarchical binary features. It modifies the output files from `B2_cluster.py` in place.

#### Key Logic:
- Reads clustered data from `ML_Data/clustered_main_carriageway/`.
- Creates `is_weekend` column (1 for Sat, Sun, Bank Holidays; 0 otherwise).
- Creates `is_school_holiday` column (1 for school holidays & Christmas period; 0 otherwise).
- **Removes** the original `day_type_id` column.
- **Overwrites** the input files in `ML_Data/clustered_main_carriageway/` with the modified data.

#### Configuration:
- `ROAD_TO_PROCESS`: Specifies the road to process.
- `DIRECTIONS`: List of directions for the road.
- `WEEKEND_IDS`, `SCHOOL_HOLIDAY_IDS`: Lists defining the mapping from `day_type_id`.

#### Output:
- Modified CSV files in `ML_Data/clustered_main_carriageway/` containing `is_weekend` and `is_school_holiday` features instead of `day_type_id`.

### 13. add_interaction_features.py (in Preprocess/feature_engineering folder)

This script adds interaction features between temporal and day features. It modifies the output files from `add_hierarchical_day_features.py` in place.

#### Key Logic:
- Reads clustered data (with hierarchical day features) from `ML_Data/clustered_main_carriageway/`.
- Creates `hour_weekend` feature (hour × is_weekend).
- Creates `hour_school_holiday` feature (hour × is_school_holiday).
- **Overwrites** the input files in `ML_Data/clustered_main_carriageway/` with the modified data.

#### Configuration:
- `ROAD_TO_PROCESS`: Specifies the road to process.
- `DIRECTIONS`: List of directions for the road.

#### Output:
- Modified CSV files in `ML_Data/clustered_main_carriageway/` containing the additional interaction features.

### 14. B3_feature_transform.py (in Preprocess folder)

This script transforms features in the clustered data files for machine learning preparation.

#### Key Logic:
- Loads clustered data from `ML_Data/clustered_main_carriageway`.
- One-hot encodes match features (`match_period`, `match_team`) into binary columns.
- Creates temporal features such as `hour`.
- Converts speed to mph and calculates `speed_reduction` and `journey_delay`.
- Saves transformed data to `ML_Data/transformed_features`.

#### Configuration:
- `INPUT_DIR`: Directory containing clustered data (`ML_Data/clustered_main_carriageway`).
- `OUTPUT_DIR`: Directory for transformed features (`ML_Data/transformed_features`).

#### Output:
- Transformed CSV files in `ML_Data/transformed_features` containing features ready for ML.
- Save final processed data (`ML_Data/processed_clusters`).

### 15. B4_data_split.py (in Preprocess folder)

This script handles the data splitting process, including chronological, K-fold, and time-series K-fold splits.

#### Key Logic:
- Loads transformed data from `ML_Data/transformed_features`.
- Splits data into train, validation, and test sets based on the specified method (`chronological`, `kfold`, `time_series_kfold`).
- Ensures data is split in a way that maintains temporal order.

#### Configuration:
- `INPUT_DIR`: Directory containing transformed features (`ML_Data/transformed_features`).
- `OUTPUT_DIR`: Directory for processed clusters (`ML_Data/processed_clusters`).
- `ROAD_TO_PROCESS`: Specifies the road to process.
- `SPLIT_METHOD`: Method for splitting data (`chronological`, `kfold`, `time_series_kfold`).

#### Output:
- Train, validation, and test CSV files in `ML_Data/processed_clusters` ready for ML.

## Data Processing Workflow (Updated)

1. **Initial Traffic Data Processing** (Scripts `download_files_1` to `inter_8`):
    - Download, unzip, combine, clean raw 15-min data, aggregate to hourly intervals (`inter_8.py`).
2. **Feature Engineering - Matches** (Script `B0_match_features.py`):
    - Process `match_data_raw.csv` to create pre/post match time windows (`Preprocess/match_windows.csv`).
3. **Feature Engineering - Weather & Clustering** (Script `B2_cluster.py`):
   - Load hourly traffic data (output of `inter_8`).
   - Load weather data (`weather_raw.csv`) and prepare hourly weather features.
   - Load match windows.
   - Merge weather features and add *categorical* match features (`match_period`, `match_team`).
   - Cluster by direction and save intermediate files (e.g., `ML_Data/clustered_main_carriageway`).
4. **Feature Engineering - Day Type Simplification** (Script `add_hierarchical_day_features.py`):
   - Transform complex `day_type_id` into simpler binary flags (`is_weekend`, `is_school_holiday`).
   - Modify clustered data files in place.
5. **Feature Engineering - Interactions** (Script `add_interaction_features.py`):
   - Create interaction features between hour and day types (`hour_weekend`, `hour_school_holiday`).
   - Modify clustered data files in place.
6. **Final ML Preparation** (Script `B3_feature_transform.py`):
   - Load clustered data from step 5.
   - One-hot encode categorical match features into binary flags.
   - Calculate target variable(s).
   - Split into train/validation/test sets chronologically.
   - Save final processed data (`ML_Data/processed_clusters`).
7. **Model Training & Evaluation** (XGBoost Scripts):
   - **Hyperparameter tuning:**
     - Use either grid search (`tune_multioutput.py`) or Bayesian optimization (`tune_multioutput_optuna.py`) for multi-output regression.
     - Both scripts save best parameters in the same format for compatibility.
     - **Bayesian optimization script now includes:**
       - Scale-aware optimization using normalized RMSE
       - Early stopping based on normalized performance
       - Detailed progress tracking showing both normalized and raw metrics
     - Tune hyperparameters using train/validation sets.
   - Train final model using best parameters on train/validation set, evaluate on test set (`train_evaluate_multioutput.py`).

## Recent Enhancements

### Multi-Output Regression Enhancements
- Implemented a multi-output regression model using XGBoost to predict both `total_traffic_flow` and `speed_reduction` simultaneously.
- Updated the `train_evaluate_multioutput.py` script to include interaction features such as `hour_weekend` and `hour_school_holiday`.
- Added functionality to load shared parameters if direction-specific parameters are not found.
- Enhanced the script to generate additional visualizations for better insights.

### New Visualizations
- **Weekend vs. Weekday 24-Hour Pattern**: Added a new visualization to compare traffic patterns between weekends and weekdays.
  - Separate plots for weekday and weekend patterns for each target.
  - Combined plot for direct comparison of weekend and weekday patterns.
  - Helps in understanding how traffic behavior differs on weekends compared to weekdays and how well the model captures these differences.

### Performance Metrics
- Evaluated the model's performance on both training and test datasets.
- Metrics include RMSE, MAE, and R² for both `total_traffic_flow` and `speed_reduction`.
- Visualizations and metrics are saved in the `ML_Data/model_results_multioutput` directory.

### Error Fixes
- Fixed a KeyError in the summary display of the `train_evaluate_multioutput.py` script.
- Corrected the return structure to ensure proper handling of metrics in the main function.

### Optuna Bayesian Hyperparameter Tuning (Multi-Output)
- Added a new script (`tune_multioutput_optuna.py`) for hyperparameter tuning using Bayesian optimization (Optuna) for the multi-output XGBoost model.
- This replaces or complements the previous grid search approach, providing a more efficient and effective way to find optimal parameters.
- **Normalized RMSE Optimization:**
  - Implemented scale-aware optimization using normalized RMSE for multi-output targets.
  - Each target's RMSE is normalized by its standard deviation before averaging.
  - This ensures balanced optimization when targets have different scales (e.g., traffic flow ~900 vs. speed reduction ~10).
  - Prevents the larger-scale target from dominating the optimization process.
- Output is concise and user-friendly:
  - Each trial prints timestamp, trial number, and normalized RMSE.
  - Also shows raw RMSE values for each target for interpretability.
  - Makes it easy to monitor both normalized and actual performance.
- All parameter spam and unnecessary logs are suppressed for a clean terminal experience.
- The script saves the best parameters in the same format and location as before, so the main training pipeline does not require any changes.
- Recommended to use 100–300 trials for robust optimization, but this is configurable at the top of the script.
- The script supports early stopping for the optimization process: if the best normalized RMSE hasn't improved in the last N trials (default 10), the study stops automatically to save time and resources.

## XGBoost Model Output Files (Updated, June 2024)

After training and evaluating the multi-output XGBoost models, the following files are saved for each direction and feature set:

- **Trained Model Files:**
  - Each target variable has its own XGBoost model saved as:
    - `model_total_traffic_flow.json` (for total traffic flow predictions)
    - `model_speed_reduction.json` (for speed reduction predictions)
    - `model_normalized_time.json` (for journey time predictions)
  - These files can be loaded directly for making predictions on new data or for visualization.

- **Feature Statistics:**
  - `feature_stats.json` contains min, max, mean, and std for each input feature (from the training data).
  - Useful for input validation and understanding feature ranges.

- **Prediction Statistics:**
  - `prediction_ranges.json` contains min, max, mean, std, and 5th/95th percentiles for each target variable (from test set predictions).
  - Useful for displaying confidence intervals and typical value ranges in the dashboard.

- **Test Set Outputs for Visualization:**
  - `X_train.csv`, `X_test.csv`: Feature matrices for SHAP and time-based plots
  - `y_test.csv`: True target values for the test set
  - `y_pred_test.npy`: Model predictions for the test set
  - `metrics.json`: Summary metrics for all targets and splits

All files are saved in the output directory for each direction and feature set (e.g., `Output/ML/clockwise/[features]/`).

**Note:**
- The naming convention is now descriptive and matches the target variable, making it easy to identify and use the correct model file.
- This structure supports robust, modular integration with the dashboard and future prediction workflows.
- **All visualizations are now generated by a separate script, not during model training.**

## Model Training and Visualization Workflow (June 2024)

The machine learning workflow is now split into two clear, modular steps:

### 1. Model Training & Evaluation
- Run `xgboost/train_evaluate_multioutput.py` to train the model, evaluate performance, and save all outputs needed for further analysis.
- This script saves:
  - Trained model files (one per target)
  - Feature statistics
  - Test set predictions (`y_pred_test.npy`)
  - Test set true values (`y_test.csv`)
  - Feature matrices (`X_train.csv`, `X_test.csv`)
  - Metrics (`metrics.json`)
- **No plots or visualizations are generated during this step.**

### 2. Visualization
- Run `xgboost/visualize_results.py` to generate all plots and visualizations.
- This script loads the outputs from the training step and produces:
  - Residuals analysis plots
  - Feature importance and SHAP plots
  - Actual vs. predicted scatter plots
  - 24-hour pattern plots
  - Weekend vs. weekday pattern plots
- All plots are saved in the appropriate output directory for each direction.

### How to Use

- **Step 1:**
  - Run the training script:
    ```bash
    python xgboost/train_evaluate_multioutput.py
    ```
  - This will process all available directions and save outputs to `Output/ML/[direction]/[output_name]/`.

- **Step 2:**
  - Run the visualization script:
    ```bash
    python xgboost/visualize_multioutput_results.py
    ```
  - This will generate all visualizations for each direction using the saved outputs.

### Benefits
- **Modularity:** Training and visualization are independent, making the codebase easier to maintain and extend.
- **Speed:** You can re-run visualizations instantly without retraining the model.
- **Reusability:** The visualization script can be used for any compatible model outputs, including future experiments or baseline models.
- **Debugging:** Issues in training or visualization can be isolated and fixed independently.

This approach follows best practices for clean, maintainable, and scalable machine learning workflows.

## Notes

- Speed is now analyzed as reduction from free-flow (70 mph) rather than raw speed
- Feature set includes key temporal and environmental factors plus interaction features
- Data splits preserve chronological order to maintain temporal patterns
- Model complexity reduced to favor interpretability
- All scripts include colored terminal output for improved readability

## Terminal Output

All scripts include colored terminal output for improved readability:
- **Magenta background**: Road section dividers
- **Blue background**: Operation dividers
- **Green background**: Summary sections
- **Green text**: Success messages
- **Yellow text**: Warnings
- **Red text**: Error messages
- **Cyan text**: Information messages

## Preparing for Machine Learning

The cleaned data is now ready for machine learning applications, with:
- Consistent date/time formatting
- Standardized column names
- Proper data types
- Parsed geographical coordinates
- Filled missing values
- Uniform structure across all road sections

## Future Extensions

While currently focused on the M60 motorway, the system is designed to be extended to multiple roads. The architecture supports:
- Processing additional roads by updating the configuration
- Adding new preprocessing steps by extending the pipeline
- Handling different data formats through configurable parameters

## Common Data Challenges Addressed

1. **Missing Days**: The pipeline identifies and reports days with missing data
2. **Inconsistent Naming**: Standardized naming conventions for files and columns
3. **Duplicate Records**: Validation to ensure no duplicate road sections
4. **Data Type Consistency**: Conversion to appropriate data types
5. **Coordinate Parsing**: Separation of geographical coordinates for spatial analysis
6. **Time Standardization**: Rounding to consistent 15-minute intervals 

## Journey Time Normalization

The project implements a robust method for normalizing journey times to enable fair comparisons across different road segments and conditions. This normalization is crucial for analyzing the impact of various factors on travel times.

### Normalization Process

1. **Convert Link Length to Miles**
   ```python
   link_length_miles = link_length / 1609.34  # Convert meters to miles
   ```

2. **Calculate Normalized Time (Minutes per Mile)**
   ```python
   normalized_time = (fused_travel_time / 60) / link_length_miles  # Convert seconds to minutes
   ```
   - This gives us a standardized measure in minutes per mile
   - NaN values are assigned if link length is zero or invalid

3. **Calculate Free-Flow Time**
   ```python
   FREE_FLOW_SPEED = 70  # mph (motorway speed limit)
   FREE_FLOW_TIME = 60 / FREE_FLOW_SPEED  # minutes per mile at 70 mph
   ```

4. **Calculate Journey Delay**
   ```python
   journey_delay = (normalized_time - FREE_FLOW_TIME).clip(lower=0)
   ```
   - Represents extra minutes per mile compared to free-flow conditions
   - Negative delays are clipped to zero (can't be faster than free-flow)
   - Results are rounded to 2 decimal places for clarity

### Benefits of This Approach

1. **Comparability**: Normalizing to minutes per mile allows direct comparison between segments of different lengths
2. **Clear Baseline**: Using the motorway speed limit (70 mph) as free-flow speed provides a consistent reference point
3. **Intuitive Units**: Extra minutes per mile is an easily understood metric
4. **Controlled Analysis**: Removes segment length as a confounding variable when analyzing impact factors
5. **Error Handling**: Robust handling of edge cases (zero lengths, invalid data)

This normalized journey time and delay calculation is used throughout the analysis pipeline, particularly in:
- Impact analysis of football matches
- Weather effect assessment
- Time-of-day pattern analysis
- Machine learning model target variables

# Standard Regression Models Implementation

Added `standard_regression.py` to establish baseline performance metrics using simpler models:

- **Models Implemented**:
  - Linear Regression (no regularization)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)

- **Key Features**:
  - Uses identical data pipeline and features as XGBoost
  - Implements feature scaling (critical for linear models)
  - Generates actual vs. predicted plots for visual analysis
  - Saves results in `model_results_multioutput/standard_regression/`
  - Provides comprehensive metrics (RMSE, MAE, R²)

- **Purpose**:
  - Establish performance baselines
  - Compare with XGBoost to justify model complexity
  - Understand feature relationships through linear models
  - Identify if simpler models might suffice

The script maintains consistent output formatting and directory structure with the existing XGBoost implementation for easy comparison.

## XGBoost Model Architecture Details

### Recent Improvements (June 2024)

1. **Enhanced Metrics Handling**
   - Implemented robust metrics calculation and storage
   - Fixed JSON serialization issues for metrics storage
   - Added proper nesting of metrics by dataset type (train/validation/test)
   - Improved terminal output formatting for better readability

2. **Feature Set Stabilization**
   - Finalized the core feature set for analysis
   - Validated feature importance across different scenarios
   - Confirmed stability of interaction features

3. **Error Handling Improvements**
   - Added comprehensive validation for metrics calculations
   - Enhanced error messages for debugging
   - Implemented proper type checking for metrics values
   - Added safeguards for metrics dictionary structure

### Multi-Target Model Enhancement (June 2024)

1. **Expanded Target Variables**
   - Added normalized journey time as a third target variable
   - Model now simultaneously predicts:
     * `total_traffic_flow`: Vehicle count per hour
     * `speed_reduction`: Reduction from free-flow speed (70 mph)
     * `normalized_time`: Journey time in minutes per mile

2. **Dynamic Target Handling**
   - Implemented flexible architecture supporting variable number of targets
   - Each target has appropriate scale normalization:
     * Traffic flow (scale: 1000, typical range: hundreds/thousands)
     * Speed reduction (scale: 10, typical range: 0-30 mph)
     * Journey time (scale: 1, typical range: 1-5 min/mile)

3. **Enhanced Hyperparameter Tuning**
   - Balanced optimization across all target variables
   - Implemented target-specific scaling factors
   - Added baseline performance tracking
   - Calculates relative improvement for each target
   - Early stopping based on overall performance

4. **Improved Progress Tracking**
   - Real-time monitoring of all target metrics
   - Shows RMSE and improvement percentages
   - Clear visualization of optimization progress
   - Comprehensive final summary for each target

5. **Visualization Enhancements**
   - Added error bars to time-series plots
   - Separate visualizations for each target variable
   - Enhanced weekend vs. weekday pattern analysis
   - Improved readability of multi-target plots

### Output Structure Update

1. **Flexible Output Naming**
   - Added configurable output directory naming
   - Options for feature-based or custom naming
   - Consistent naming across tuning and training scripts

2. **Model Files**
```python
Output/ML/
└── [direction]/
    └── [output_name]/
        ├── model_total_traffic_flow.json
        ├── model_speed_reduction.json
        ├── model_normalized_time.json
        ├── feature_stats.json
        └── prediction_ranges.json
```

3. **Metrics Storage**
```python
metrics = {
    'train': {
        target: {
            'rmse': float,
            'mae': float,
            'r2': float
        } for target in TARGETS
    },
    'validation': {
        # Same structure as train
    },
    'test': {
        # Same structure as train
    }
}
```

### Model Training Pipeline

1. **Hyperparameter Tuning**
   ```python
   # tune_multioutput_optuna.py
   - Optimizes parameters for all targets simultaneously
   - Uses normalized RMSE for balanced optimization
   - Tracks improvement from baseline for each target
   ```

2. **Model Training**
   ```python
   # train_evaluate_multioutput.py
   - Trains separate XGBoost models for each target
   - Uses shared optimal parameters
   - Generates comprehensive visualizations
   - Saves model files and statistics
   ```

3. **Visualization Suite**
   - Target-specific actual vs. predicted plots
   - 24-hour pattern analysis for each target
   - Weekend vs. weekday comparisons
   - Error bars showing prediction uncertainty
   - Feature importance plots for each target

### Performance Metrics

For each target variable, the model tracks:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Relative improvement from baseline
- Prediction ranges (min, max, mean, std, percentiles)

This multi-target approach allows for:
1. Comprehensive traffic pattern analysis
2. Direct comparison of prediction accuracy across metrics
3. Understanding relationships between different traffic measures
4. Identification of which aspects are easier/harder to predict
5. Better insights for traffic management decisions

# New Feature: Football Match Impact Analysis & Visualization (June 2024)

- Implemented a comprehensive analysis and visualization workflow to assess the impact of Manchester United and Manchester City home matches on M60 motorway traffic.
- The pipeline includes:
  - Extraction and cleaning of hourly traffic data for all M60 segments.
  - Identification of pre-match and post-match periods for each football match using precise time windows.
  - Baseline calculation using equivalent non-match days, matched by day-of-week and hour-of-day.
  - Outlier removal and minimum sample size enforcement for robust statistics.
  - Calculation of absolute and percentage changes in traffic flow and speed for each segment and period (pre-match, post-match, by team).
  - Ranking of segments by impact magnitude and fallback logic to ensure visualization data is always produced.
  - Generation of clear, side-by-side comparison figures for United and City, showing the most impacted segments, with error bars and segment labels.
- The methodology and workflow are fully documented in `Analysis/method.md`.
- This feature provides a robust, transparent, and reproducible approach to quantifying and visualizing the real-world impact of football matches on motorway traffic.

# Interactive Traffic Visualization Dashboard (`app.py`)

## Overview

The `app.py` script now implements the **Typical Traffic Explorer**: an interactive dashboard for visualizing typical (aggregated) traffic patterns on the M60 motorway. It is designed for quick, intuitive exploration of how traffic varies by hour and day type (weekday, weekend, school holiday) across all segments and directions.

- **Data Source:**
  - Loads pre-aggregated Parquet files from `ML_Data/typical_aggregates/`.
  - These files are generated by the script `Preprocess/aggregate_typical_traffic.py`.
  - Each file contains typical statistics (mean, std, min, max, 5th/95th percentiles) for each segment, direction, hour, and day type.

## Purpose and Role

- **Pattern Exploration:**
  - Lets users explore typical (average and variability) traffic conditions for any hour and day type.
  - Useful for understanding baseline patterns, planning, and communication.
- **Simplicity:**
  - Focuses on clarity and ease of use, with a minimal, modern UI.

## Main Features

- **Aggregated Data Visualization:**
  - Visualizes typical traffic metrics (e.g., total flow, speed reduction, journey delay, vehicle type percentages) for each segment.
  - Data is grouped by segment, direction, hour, and day type (weekday, weekend, school holiday).
  - For each metric, shows mean, std, min, max, and percentiles.

- **User Interface:**
  - **Day Type Selector:** Choose between weekday, weekend, or school holiday.
  - **Hour Selector:** Pick any hour of the day (0–23).
  - **Metric Selector:** Choose which typical metric to visualize (e.g., flow, speed, delay, vehicle type %).
  - The map updates instantly to show the selected typical value for each segment, color-coded for easy comparison.

- **Map Display:**
  - Each segment is drawn as a colored line, with color representing the selected typical metric value.
  - Hovering shows a tooltip with detailed stats (mean, std, min, max, percentiles) for that segment, hour, and day type.
  - The map is centered on the M60 and uses a clean, modern basemap.

- **No Real-Time or Date Selection:**
  - The app does **not** show real-time or arbitrary date data. It is focused on typical (aggregated) patterns only.
  - No multi-road selection or Mapbox map-matching toggles are present.

## Data Aggregation Process

- The typical aggregates are generated by `Preprocess/aggregate_typical_traffic.py`:
  - Groups the processed ML-ready data by segment, direction, hour, and day type.
  - Calculates mean, std, min, max, 5th and 95th percentiles for each key metric.
  - Saves the result as Parquet files in `ML_Data/typical_aggregates/`.

- **Columns in typical aggregates include:**
  - Segment info: `ntis_link_number`, `ntis_link_description`, `direction`, `hour`, `day_type`, `start_latitude`, `start_longitude`, `end_latitude`, `end_longitude`, `link_length`
  - For each metric (e.g., `total_traffic_flow`, `speed_reduction`, `journey_delay`, `traffic_1_pct`, ...):
    - `{metric}_mean`, `{metric}_std`, `{metric}_min`, `{metric}_max`, `{metric}_<lambda_4>` (5th percentile), `{metric}_<lambda_5>` (95th percentile)

## Example Use Cases
- See how typical traffic flow or speed varies by hour and day type for any segment.
- Identify segments with high variability or unusual patterns.
- Communicate baseline traffic conditions for planning or reporting.

## Summary

The Typical Traffic Explorer provides a fast, clear, and interactive way to explore baseline motorway traffic patterns. It is ideal for analysis, communication, and understanding of typical conditions, using robust aggregated data rather than raw or real-time feeds.
