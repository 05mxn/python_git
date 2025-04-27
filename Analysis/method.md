# Match Impact Analysis Methodology

## Overview
This document details the methodology used to analyze and visualize the impact of football matches on M60 motorway traffic patterns. The analysis compares traffic conditions during match days to equivalent non-match days, focusing on both Manchester United and Manchester City home games.

## Data Processing Pipeline

### 1. Traffic Data Preparation
- Raw traffic data is collected from Highway England sensors along the M60 motorway
- Data includes:
  * Flow rates (vehicles/hour)
  * Speed measurements (mph)
  * Timestamps
  * Segment identifiers
- Each segment is identified by:
  * Direction (clockwise/anti-clockwise)
  * Junction numbers
  * Specific location details (e.g., entry/exit slips, main carriageway)

### 2. Match Impact Analysis
The analysis is performed in `analyze_match_impact.py` and follows these steps:

#### 2.1 Data Filtering
- Traffic data is filtered to focus on relevant time periods:
  * Pre-match: 2 hours before kickoff
  * Post-match: 2 hours after match end
- Data quality checks are performed:
  * Removal of missing or invalid measurements
  * Minimum sample size requirements (n ≥ 3)
  * Outlier detection using 1.5 × IQR rule

#### 2.2 Baseline Calculation
- For each segment and time period:
  * Baseline traffic patterns are calculated from non-match days
  * Same day-of-week and time-of-day as match events
  * Metrics include:
    - Median flow rates
    - Median speeds
    - Standard deviations (via IQR proxy)
    - Sample counts

#### 2.3 Impact Calculation
- For each segment, the following metrics are calculated:
  * Absolute flow difference (vehicles/hour)
  * Percentage flow change
  * Absolute speed difference (mph)
  * Percentage speed change
- Separate calculations for:
  * All matches combined
  * Manchester United matches
  * Manchester City matches

#### 2.4 Critical Segment Identification
- Segments are ranked based on:
  * Magnitude of flow changes
  * Statistical significance
  * Consistency of impact
- Top segments are identified for:
  * Pre-match period
  * Post-match period
  * Each team separately
- **Robust fallback:** If criticality metrics are missing, the top segments are selected by pre-match flow difference, ensuring the workflow always produces visualization data.

### 3. Visualization
The visualization process is handled by `visualize_match_impact.py`:

#### 3.1 Data Preparation
- Results data is loaded from `Match_Impact_Results/top_segments_for_viz.csv`
- **Segment labels are robustly extracted:**
  * Each label includes both direction (CW/ACW) and junctions (e.g., J12-J13)
  * Extraction logic is tailored to the actual segment ID format
- Absolute flow percentage changes are calculated for ranking

#### 3.2 Figure Generation
Two main comparison figures are produced:

##### Pre-Match Comparison
- Side-by-side visualization:
  * Left: Manchester United impact
  * Right: Manchester City impact
- Shows top 4 most impacted segments for each team
- Features:
  * Horizontal bars showing flow percentage changes
  * Error bars indicating ±1 IQR/2 (proxy for standard deviation)
  * Sample size annotations
  * **Clear segment labels with direction and junction info**
  * Team-specific colors (United red, City blue)

##### Post-Match Comparison
- Identical layout to pre-match comparison
- Focuses on post-match period impacts
- Same features and formatting

### 4. Output Files
The analysis produces several output files:

#### 4.1 Results Data
- `Match_Impact_Results/top_segments_for_viz.csv`:
  * Contains processed results for visualization
  * Includes all necessary metrics and segment information
  * **Always generated, even if criticality metrics are missing (fallback logic)**

#### 4.2 Visualizations
- `Figures/pre_match_comparison.png`:
  * Pre-match impact comparison between teams
- `Figures/post_match_comparison.png`:
  * Post-match impact comparison between teams

## Statistical Considerations

### Sample Sizes
- Minimum requirement of 3 samples per segment/period combination
- Sample sizes are clearly annotated on visualizations
- Larger sample sizes indicate more reliable results

### Error Representation
- IQR/2 is used as a proxy for standard deviation (error bars)
- Error bars on plots represent ±1 IQR/2
- Helps identify consistency of impacts

### Significance
- Large sample sizes (n > 30) provide more reliable estimates
- Error bars help visualize the uncertainty in measurements
- Consistent patterns across multiple segments increase confidence in findings

## Notes on Workflow Robustness
- **No weekend/weekday separation:** All matches are analyzed together, as nearly all are on weekends.
- **Label extraction is robust:** Direction and junctions are always shown in the visualization labels.
- **Visualization CSV is always generated:** Fallback logic ensures the workflow never fails due to missing criticality metrics.

## Future Improvements
Potential enhancements to consider:

1. Additional temporal analysis:
   - Weekend vs weekday comparisons (if relevant in future data)
   - Seasonal variations
   - Special event impacts

2. Enhanced statistical analysis:
   - Formal significance testing
   - Confidence interval calculations
   - Effect size measurements

3. Visualization enhancements:
   - Interactive versions for digital presentation
   - Additional context layers (e.g., stadium locations)
   - Alternative visualization formats for different aspects of the data 

# Network Distribution Analysis

## Overview
This analysis examines how traffic patterns are distributed across the entire M60 network under different conditions (weekdays, weekends, and school holidays). Unlike the match impact analysis which focuses on specific events, this analysis provides a system-level view of how traffic behaves across all segments simultaneously.

## Data Processing Pipeline

### 1. Data Preparation (`NetworkDistributionAnalyzer` class)
- Source data is loaded from processed cluster files in `ML_Data/processed_clusters/`
- Key data elements include:
  * Segment identifiers (`ntis_link_number`)
  * Traffic flow measurements (`total_traffic_flow`)
  * Speed reduction values (`speed_reduction`)
  * Normalized journey times (`normalized_time`)
  * Temporal indicators (`hour`, `is_weekend`, `is_school_holiday`)
  * Directional information (`direction`)
- Data filtering:
  * Match periods are excluded (both pre and post-match)
  * Only relevant columns are retained
  * Data quality checks are performed

### 2. Journey Time Normalization
- For each segment:
  * Link length is converted from meters to miles
  * Travel time is converted from seconds to minutes
  * Normalized time is calculated as minutes per mile
  * Edge cases (zero lengths, invalid data) are handled appropriately
- This normalization enables:
  * Fair comparison between segments of different lengths
  * Consistent measurement across the network
  * Direct comparison with free-flow conditions

### 3. Condition-based Grouping
- Traffic data is categorized into three main conditions:
  * Weekdays (non-school holidays)
  * Weekends
  * School holidays (if sufficient data available)
- For each condition:
  * Data is grouped by hour (0-23) and segment
  * Statistical measures are calculated:
    - Mean traffic flow
    - Standard deviation of flow
    - Mean speed reduction
    - Standard deviation of speed reduction
    - Mean normalized journey time
    - Standard deviation of journey time
    - Sample counts

### 4. Distribution Analysis
- For each hour and condition:
  * Network-wide statistics are computed:
    - Average flow across all segments
    - Flow variability (standard deviation)
    - Average speed reduction
    - Speed reduction variability
    - Average normalized journey time
    - Journey time variability
    - Number of active segments
  * These statistics capture how traffic is distributed across the entire network
  * Results are saved to CSV files for further analysis

### 5. Visualization Generation
Four types of visualizations are produced to represent different aspects of the network-wide patterns:

#### 5.1 Daily Pattern Plots
- Shows how average traffic flow changes throughout the day
- Features:
  * Separate lines for each condition (weekday, weekend, school holiday)
  * Confidence intervals (±1 standard deviation)
  * Clear color coding for different conditions
  * Grid lines for easy reference
  * Comprehensive labels and titles

#### 5.2 Speed Reduction Patterns
- Visualizes how speed reduction varies across the day
- Same features as traffic flow plots
- Helps identify periods of consistent congestion
- Shows how different conditions affect network performance

#### 5.3 Normalized Journey Time Patterns
- Shows how journey times (minutes/mile) vary throughout the day
- Highlights:
  * Peak travel time periods
  * Differences between weekday/weekend patterns
  * Periods of high journey time variability
  * Impact of school holidays on travel times
- Includes confidence intervals to show variability

#### 5.4 Distribution Box Plots
For each condition (weekday/weekend/school holiday):
- Traffic flow distribution across segments
- Normalized journey time distribution
- Features:
  * Box plots showing median, quartiles, and outliers
  * Hour-by-hour comparison
  * Consistent color scheme
  * Clear labeling and gridlines

## Output Structure

### 1. Statistics Files
Located in `ML_Data/network_distribution_results/statistics/`:
- `weekday_hourly_stats.csv`
- `weekend_hourly_stats.csv`
- `school_holiday_hourly_stats.csv` (if applicable)
Each file contains:
  * Hourly means and standard deviations
  * Sample sizes
  * Flow, speed, and journey time metrics

### 2. Visualization Files
Located in `ML_Data/network_distribution_results/plots/`:
- `traffic_flow_daily_pattern.png`: Combined daily patterns
- `speed_reduction_daily_pattern.png`: Speed reduction patterns
- `normalized_time_daily_pattern.png`: Journey time patterns
- Distribution plots for each condition:
  * `traffic_flow_distribution_{condition}.png`
  * `normalized_time_distribution_{condition}.png`

# Vehicle Type Features for Machine Learning

## Rationale and Value
Including vehicle type distributions as features in the machine learning model can provide valuable predictive power. Different vehicle types (cars, vans, medium trucks, large trucks) have distinct effects on traffic flow, congestion, and speed. For example, a higher proportion of large trucks may be associated with slower speeds or increased congestion, especially during off-peak hours. By capturing the composition of traffic, the model can better understand and predict network performance under varying conditions.

## Feature Calculation
- **Raw vehicle counts** are available as `traffic_flow_value1` through `traffic_flow_value4`.
- **Vehicle type percentages** are now calculated using the sum of these four values as the denominator (not the potentially unreliable `total_traffic_flow` column):
  - For each row:
    - `total_flow = traffic_flow_value1 + traffic_flow_value2 + traffic_flow_value3 + traffic_flow_value4`
    - `traffic_1_pct = traffic_flow_value1 / total_flow * 100`
    - `traffic_2_pct = traffic_flow_value2 / total_flow * 100`
    - `traffic_3_pct = traffic_flow_value3 / total_flow * 100`
    - `traffic_4_pct = traffic_flow_value4 / total_flow * 100`
- This approach ensures that the percentages always sum to 100% (except for rows where the total is zero).

## Use in Machine Learning
- **Percentages** are recommended as primary features, as they capture the composition of traffic and are robust to changes in overall volume.
- **Raw counts** can be included as secondary features if absolute load is also relevant to the prediction task.
- These features can help the model:
  - Predict speed reduction and journey time more accurately
  - Understand the impact of heavy vehicles on congestion
  - Capture temporal patterns in vehicle mix (e.g., more trucks at night)

## Example Insights from Distribution Analysis
- Large trucks are most prevalent during night and early morning hours, with a sharp drop during the day.
- Cars dominate the traffic mix, especially during daytime and weekends.
- The vehicle type mix varies significantly by hour and by day type (weekday, weekend, school holiday).

## Implementation Notes
- The calculation of vehicle type percentages was updated in the feature transformation script (`B3_feature_transform.py`) to use the sum of vehicle flows as the denominator.
- All downstream analysis and modeling scripts should use these new percentage columns for vehicle type features.
