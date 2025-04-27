#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Football Match Impact Analysis Script

This script analyzes the impact of Manchester United and Manchester City
home matches on traffic flow and speed across M60 motorway segments.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import pickle
import re

# Suppress pandas warnings
warnings.filterwarnings('ignore')

# Terminal colors for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA_BG = '\033[45m'
    BLUE_BG = '\033[44m'
    GREEN_BG = '\033[42m'

# Configuration
HOURLY_DATA_DIR = "Final Data/M60"  # Output of inter_8.py
MATCH_WINDOWS_FILE = "Preprocess/match_windows.csv"
OUTPUT_DIR = "Analysis/Match_Impact_Results"
ROAD_NAME = "M60"
# Verbosity level: 0=minimal, 1=normal, 2=verbose
VERBOSITY = 0

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_section(message):
    """Print a formatted section header."""
    print(f"\n{Colors.BLUE_BG}{message}{Colors.ENDC}")

def print_subsection(message):
    """Print a formatted subsection header."""
    if VERBOSITY >= 1:
        print(f"\n{Colors.CYAN}{message}{Colors.ENDC}")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}{message}{Colors.ENDC}")

def print_debug(message):
    """Print a debug message only if verbosity is high."""
    if VERBOSITY >= 2:
        print(message)

def clean_segment_name(segment_id, include_id=False):
    """
    Robustly extract road, direction, and junctions for visualization.
    """
    if not isinstance(segment_id, str):
        return str(segment_id)

    # Road and direction
    road = "M60" if "M60" in segment_id else ""
    direction = ""
    if "clockwise" in segment_id or "CW" in segment_id:
        direction = "CW"
    elif "anti-clockwise" in segment_id or "ACW" in segment_id:
        direction = "ACW"

    # Try to extract two junctions (e.g., J12 and J13)
    junctions = re.findall(r'J(\\d+)', segment_id)
    if len(junctions) >= 2:
        junc_str = f"J{junctions[0]}-J{junctions[1]}"
    elif len(junctions) == 1:
        # Check for roundabout or single junction
        if "roundabout" in segment_id.lower():
            junc_str = f"J{junctions[0]} Roundabout"
        else:
            junc_str = f"J{junctions[0]}"
    else:
        junc_str = ""

    # Compose name
    parts = [p for p in [road, direction, junc_str] if p]
    final_name = " ".join(parts) if parts else segment_id

    # Optionally add numeric ID
    if include_id:
        id_part = segment_id.split()[-1]
        if id_part.isdigit():
            final_name += f" ({id_part})"
    return final_name

def load_segment_data():
    """
    Load hourly traffic data for all M60 segments.
    Returns a dictionary of DataFrames, keyed by segment ID.
    """
    print_section("Loading segment data")
    
    segment_data = {}
    file_count = 0
    
    # Check if directory exists
    if not os.path.exists(HOURLY_DATA_DIR):
        print_error(f"Directory not found: {HOURLY_DATA_DIR}")
        print_warning("Make sure you've run the preprocessing pipeline through inter_8.py")
        return segment_data
    
    # Loop through all CSV files in the directory
    for filename in os.listdir(HOURLY_DATA_DIR):
        if filename.endswith('.csv'):
            file_path = os.path.join(HOURLY_DATA_DIR, filename)
            
            try:
                # Extract segment ID from filename (assuming format like "12345.csv")
                segment_id = filename.split('.')[0]
                
                # Load data
                df = pd.read_csv(file_path)
                
                # Convert datetime to pandas datetime
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                else:
                    print_warning(f"No datetime column in {filename}, creating from date and time columns")
                    # Try to create datetime from date and time columns if available
                    if 'date' in df.columns and 'time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                
                # Store in dictionary
                segment_data[segment_id] = df
                file_count += 1
                
                if file_count % 50 == 0 and VERBOSITY >= 1:
                    print(f"Loaded {file_count} segment files...")
                
            except Exception as e:
                print_error(f"Error loading {filename}: {str(e)}")
    
    print_success(f"Successfully loaded {file_count} segment files")
    return segment_data

def load_match_windows():
    """
    Load processed match time windows data.
    Returns a DataFrame with match window information.
    """
    print_section("Loading match windows")
    
    if not os.path.exists(MATCH_WINDOWS_FILE):
        print_error(f"Match windows file not found: {MATCH_WINDOWS_FILE}")
        print_warning("Make sure you've run B0_match_features.py")
        return pd.DataFrame()
    
    try:
        # Load match windows data
        match_windows = pd.read_csv(MATCH_WINDOWS_FILE)
        
        # Convert datetime columns
        datetime_columns = ['start_pre_match', 'end_pre_match', 'start_post_match', 'end_post_match']
        for col in datetime_columns:
            if col in match_windows.columns:
                match_windows[col] = pd.to_datetime(match_windows[col])
        
        print_success(f"Successfully loaded match windows with {len(match_windows)} matches")
        return match_windows
    
    except Exception as e:
        print_error(f"Error loading match windows: {str(e)}")
        return pd.DataFrame()

def merge_match_info(segment_df, match_windows):
    """
    Add match period flags to traffic data.
    
    Parameters:
    segment_df (DataFrame): Traffic data for a segment
    match_windows (DataFrame): Match time windows
    
    Returns:
    DataFrame: Traffic data with match flags added
    """
    # Create a copy to avoid modifying the original
    df = segment_df.copy()
    
    # Initialize match period columns
    df['is_pre_match'] = 0
    df['is_post_match'] = 0
    df['is_match_period'] = 0
    df['match_team'] = 'no_match'
    
    # Check each timestamp against match windows
    for _, match in match_windows.iterrows():
        # Pre-match period
        pre_match_mask = (df['datetime'] >= match['start_pre_match']) & (df['datetime'] <= match['end_pre_match'])
        if pre_match_mask.any():
            df.loc[pre_match_mask, 'is_pre_match'] = 1
            df.loc[pre_match_mask, 'is_match_period'] = 1
            df.loc[pre_match_mask, 'match_team'] = match['match_team']
        
        # Post-match period
        post_match_mask = (df['datetime'] >= match['start_post_match']) & (df['datetime'] <= match['end_post_match'])
        if post_match_mask.any():
            df.loc[post_match_mask, 'is_post_match'] = 1
            df.loc[post_match_mask, 'is_match_period'] = 1
            df.loc[post_match_mask, 'match_team'] = match['match_team']
    
    # Create team-specific flags
    df['is_united_match'] = (df['match_team'] == 'Man United').astype(int)
    df['is_city_match'] = (df['match_team'] == 'Man City').astype(int)
    
    # Add day of week and hour for baseline matching
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_of_day'] = df['datetime'].dt.hour
    
    return df

def identify_comparable_periods(df):
    """
    Find non-match periods that are comparable to match periods.
    
    Parameters:
    df (DataFrame): Traffic data with match flags
    
    Returns:
    DataFrame: The same data with a 'is_baseline' flag added
    """
    # Create a copy
    df_result = df.copy()
    
    # Initialize baseline flag
    df_result['is_baseline'] = 0
    
    # Group by day of week and hour of day
    grouped = df_result.groupby(['day_of_week', 'hour_of_day'])
    
    # For each time slot (day of week + hour of day)
    for (dow, hour), group in grouped:
        # Find records that are not in a match period and mark as baseline
        non_match_mask = (group['is_match_period'] == 0)
        
        if non_match_mask.any():
            # Get the indices of non-match records in this time slot
            baseline_indices = group[non_match_mask].index
            
            # Mark these as baseline records
            df_result.loc[baseline_indices, 'is_baseline'] = 1
    
    return df_result

def calculate_metrics_for_period(df, period_flag, team_flag=None):
    """
    Calculate median metrics and IQR for a specific period and team.

    Parameters:
    df (DataFrame): Traffic data with flags
    period_flag (str): Column name for the period flag
    team_flag (str, optional): Column name for the team flag

    Returns:
    dict: Median metrics, IQR, and count for the specified period
    """
    # Filter for the specified period
    if team_flag is None:
        period_data = df[df[period_flag] == 1]
    else:
        period_data = df[(df[period_flag] == 1) & (df[team_flag] == 1)]

    # If no data or only one data point, return NaN for IQR
    if len(period_data) < 2:
        return {
            'median_flow': period_data['total_traffic_flow'].median() if len(period_data) > 0 else np.nan,
            'median_speed': period_data['fused_average_speed'].median() if len(period_data) > 0 else np.nan,
            'flow_iqr': np.nan,
            'speed_iqr': np.nan,
            'flow_q1': np.nan,
            'flow_q3': np.nan,
            'speed_q1': np.nan,
            'speed_q3': np.nan,
            'count': len(period_data)
        }

    # Calculate median metrics and IQR
    flow_q1 = period_data['total_traffic_flow'].quantile(0.25)
    flow_q3 = period_data['total_traffic_flow'].quantile(0.75)
    speed_q1 = period_data['fused_average_speed'].quantile(0.25)
    speed_q3 = period_data['fused_average_speed'].quantile(0.75)

    metrics = {
        'median_flow': period_data['total_traffic_flow'].median(),
        'median_speed': period_data['fused_average_speed'].median(),
        'flow_iqr': flow_q3 - flow_q1,
        'speed_iqr': speed_q3 - speed_q1,
        'flow_q1': flow_q1,
        'flow_q3': flow_q3,
        'speed_q1': speed_q1,
        'speed_q3': speed_q3,
        'count': len(period_data)
    }

    return metrics

def detect_outliers(data):
    """
    Detect outliers using the 1.5 × IQR rule.
    
    Parameters:
    data (Series): Data to check for outliers
    
    Returns:
    Series: Boolean mask where True indicates an outlier
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (data < lower_bound) | (data > upper_bound)

def compute_segment_impact(segment_id, segment_df, match_windows):
    """
    Compute impact scores for a single segment.
    
    Parameters:
    segment_id (str): ID of the segment
    segment_df (DataFrame): Traffic data for the segment
    match_windows (DataFrame): Match time windows
    
    Returns:
    dict: Impact scores and metrics for the segment
    """
    print_debug(f"Analyzing segment {segment_id}")
    
    # Clean and prepare traffic data
    clean_df = prepare_traffic_data(segment_df)
    
    # Merge match information
    df = merge_match_info(clean_df, match_windows)
    
    # Find comparable baseline periods
    df = identify_comparable_periods(df)
    
    # Remove outliers from baseline data
    baseline_mask = df['is_baseline'] == 1
    if baseline_mask.any():
        flow_outliers = detect_outliers(df.loc[baseline_mask, 'total_traffic_flow'])
        df.loc[baseline_mask & flow_outliers, 'is_baseline'] = 0
    
    # Calculate baseline metrics
    baseline_metrics = calculate_metrics_for_period(df, 'is_baseline')
    
    # If no baseline data, skip this segment
    if np.isnan(baseline_metrics['median_flow']):
        print_debug(f"No baseline data for segment {segment_id}")
        return None
    
    # Calculate metrics for each period and team
    results = {
        'segment_id': segment_id,
        'baseline': baseline_metrics,
        'pre_match': {
            'all': calculate_metrics_for_period(df, 'is_pre_match'),
            'united': calculate_metrics_for_period(df, 'is_pre_match', 'is_united_match'),
            'city': calculate_metrics_for_period(df, 'is_pre_match', 'is_city_match')
        },
        'post_match': {
            'all': calculate_metrics_for_period(df, 'is_post_match'),
            'united': calculate_metrics_for_period(df, 'is_post_match', 'is_united_match'),
            'city': calculate_metrics_for_period(df, 'is_post_match', 'is_city_match')
        }
    }
    
    # Calculate impact scores (difference from baseline)
    impacts = {}
    
    # Process each period and team
    for period in ['pre_match', 'post_match']:
        for team in ['all', 'united', 'city']:
            period_metrics = results[period][team]
            
            # Skip if insufficient data
            if period_metrics['count'] < 3:  # Minimum 3 samples required
                continue
            
            # Calculate absolute and percentage differences using median
            flow_diff = period_metrics['median_flow'] - baseline_metrics['median_flow']
            flow_pct = (flow_diff / baseline_metrics['median_flow']) * 100 if baseline_metrics['median_flow'] > 0 else np.nan
            
            speed_diff = period_metrics['median_speed'] - baseline_metrics['median_speed']
            speed_pct = (speed_diff / baseline_metrics['median_speed']) * 100 if baseline_metrics['median_speed'] > 0 else np.nan
            
            # Store impact scores with IQR
            impact_key = f"{period}_{team}"
            impacts[impact_key] = {
                'flow_diff': flow_diff,
                'flow_pct': flow_pct,
                'flow_iqr': period_metrics['flow_iqr'],
                'flow_q1': period_metrics['flow_q1'],
                'flow_q3': period_metrics['flow_q3'],
                'speed_diff': speed_diff,
                'speed_pct': speed_pct,
                'speed_iqr': period_metrics['speed_iqr'],
                'speed_q1': period_metrics['speed_q1'],
                'speed_q3': period_metrics['speed_q3'],
                'sample_count': period_metrics['count']
            }
    
    # Add impact scores to results
    results['impacts'] = impacts
    
    print_debug(f"Completed impact analysis for segment {segment_id}")
    return results

def analyze_all_segments(segment_data, match_windows):
    """
    Run impact analysis for all segments.
    
    Parameters:
    segment_data (dict): Dictionary of segment DataFrames
    match_windows (DataFrame): Match time windows
    
    Returns:
    list: Impact results for all segments
    """
    print_section("Analyzing impact for all segments")
    
    all_results = []
    total_segments = len(segment_data)
    processed_count = 0
    
    # Process each segment
    for segment_id, segment_df in segment_data.items():
        # Skip if dataframe is empty
        if len(segment_df) == 0:
            continue
        
        # Compute impact for this segment
        impact_results = compute_segment_impact(segment_id, segment_df, match_windows)
        
        if impact_results is not None:
            all_results.append(impact_results)
        
        # Update progress counter
        processed_count += 1
        if processed_count % 50 == 0 and VERBOSITY >= 1:
            print(f"Processed {processed_count}/{total_segments} segments...")
    
    print_success(f"Completed impact analysis for {len(all_results)} segments")
    return all_results

def rank_segments_by_impact(all_results):
    """
    Rank segments by their impact scores.
    
    Parameters:
    all_results (list): Impact results for all segments
    
    Returns:
    dict: Rankings for different impact metrics
    """
    print_section("Ranking segments by impact")
    
    # Initialize rankings dictionary
    rankings = {}
    
    # Prepare data for ranking
    segments = []
    for result in all_results:
        segment_id = result['segment_id']
        impacts = result['impacts']
        
        for impact_key, impact_values in impacts.items():
            # Only include if we have enough samples
            if impact_values['sample_count'] < 3:
                continue
                
            segments.append({
                'segment_id': segment_id,
                'impact_type': impact_key,
                'flow_diff': impact_values['flow_diff'],
                'flow_pct': impact_values['flow_pct'],
                'speed_diff': impact_values['speed_diff'],
                'speed_pct': impact_values['speed_pct'],
                'sample_count': impact_values['sample_count']
            })
    
    # Convert to DataFrame for easier ranking
    segments_df = pd.DataFrame(segments)
    
    # Skip if no data
    if len(segments_df) == 0:
        print_warning("No segments with sufficient data for ranking")
        return rankings
    
    # Split by impact type
    for impact_type in segments_df['impact_type'].unique():
        type_df = segments_df[segments_df['impact_type'] == impact_type]
        
        # Create rankings for this impact type
        flow_ranking = type_df.sort_values('flow_diff', ascending=False).reset_index(drop=True)
        flow_pct_ranking = type_df.sort_values('flow_pct', ascending=False).reset_index(drop=True)
        speed_ranking = type_df.sort_values('speed_diff', ascending=True).reset_index(drop=True)  # Lower is worse for speed
        speed_pct_ranking = type_df.sort_values('speed_pct', ascending=True).reset_index(drop=True)
        
        # Store rankings
        rankings[f"{impact_type}_flow_diff"] = flow_ranking
        rankings[f"{impact_type}_flow_pct"] = flow_pct_ranking
        rankings[f"{impact_type}_speed_diff"] = speed_ranking
        rankings[f"{impact_type}_speed_pct"] = speed_pct_ranking
    
    # --- NEW: Criticality ranking system ---
    # Extract criticality metrics from all_results
    crit_rows = []
    for result in all_results:
        metrics = result.get('criticality_metrics', {})
        segment_id = result['segment_id']
        # Only include if all metrics are present and not NaN
        if (metrics and not any(pd.isna([metrics.get('max_flow_pct_increase'), metrics.get('freq_significant_impacts'), metrics.get('flow_pct_consistency')]))):
            crit_rows.append({
                'segment_id': segment_id,
                'max_flow_pct_increase': metrics['max_flow_pct_increase'],
                'freq_significant_impacts': metrics['freq_significant_impacts'],
                'flow_pct_consistency': metrics['flow_pct_consistency']
            })
    crit_df = pd.DataFrame(crit_rows)
    if not crit_df.empty:
        # Normalize each metric (min-max scaling)
        for col in ['max_flow_pct_increase', 'freq_significant_impacts']:
            min_val, max_val = crit_df[col].min(), crit_df[col].max()
            crit_df[col + '_norm'] = (crit_df[col] - min_val) / (max_val - min_val) if max_val > min_val else 0
        # For consistency (std), lower is better, so invert after scaling
        col = 'flow_pct_consistency'
        min_val, max_val = crit_df[col].min(), crit_df[col].max()
        if max_val > min_val:
            crit_df[col + '_norm'] = 1 - (crit_df[col] - min_val) / (max_val - min_val)
        else:
            crit_df[col + '_norm'] = 0
        # Combine into a single score (average of normalized metrics)
        crit_df['criticality_score'] = crit_df[['max_flow_pct_increase_norm', 'freq_significant_impacts_norm']].mean(axis=1)
        # Sort by score descending
        crit_df = crit_df.sort_values('criticality_score', ascending=False).reset_index(drop=True)
        # Add to rankings
        rankings['criticality_ranking'] = crit_df
    # --- END NEW ---

    print_success(f"Created {len(rankings)} different segment rankings")
    return rankings

def save_impact_data(all_results, rankings):
    """
    Save impact analysis results to CSV files and summary.
    
    Parameters:
    all_results (list): Impact results for all segments
    rankings (dict): Rankings dictionary
    """
    print_section("Saving impact analysis data")
    
    # Save detailed segment impact scores
    impact_rows = []
    for result in all_results:
        segment_id = result['segment_id']
        baseline = result['baseline']
        impacts = result['impacts']
        
        # Add cleaned segment name
        clean_name = clean_segment_name(segment_id)
        
        for impact_key, impact_values in impacts.items():
            # Parse impact key
            key_parts = impact_key.split('_')
            period = key_parts[0]
            team = key_parts[1]
            
            # Add to rows
            impact_rows.append({
                'segment_id': segment_id,
                'clean_name': clean_name,
                'period': period,
                'team': team,
                'baseline_flow': baseline['median_flow'],
                'baseline_speed': baseline['median_speed'],
                'flow_diff': impact_values['flow_diff'],
                'flow_pct': impact_values['flow_pct'],
                'flow_iqr': impact_values['flow_iqr'],
                'flow_q1': impact_values['flow_q1'],
                'flow_q3': impact_values['flow_q3'],
                'speed_diff': impact_values['speed_diff'],
                'speed_pct': impact_values['speed_pct'],
                'speed_iqr': impact_values['speed_iqr'],
                'speed_q1': impact_values['speed_q1'],
                'speed_q3': impact_values['speed_q3'],
                'sample_count': impact_values['sample_count']
            })
    
    # Convert to DataFrame and save
    impact_df = pd.DataFrame(impact_rows)
    impact_file = os.path.join(OUTPUT_DIR, "segment_impact_scores.csv")
    impact_df.to_csv(impact_file, index=False)
    print_success(f"Saved detailed impact scores to {impact_file}")
    
    # Save top segments for each metric
    for ranking_key, ranking_df in rankings.items():
        if len(ranking_df) > 0:
            # Add cleaned segment names
            ranking_df = ranking_df.copy()
            ranking_df['clean_name'] = ranking_df['segment_id'].apply(clean_segment_name)
            
            # Save top 20 segments
            top_df = ranking_df.head(20)
            ranking_file = os.path.join(OUTPUT_DIR, f"top_{ranking_key}.csv")
            top_df.to_csv(ranking_file, index=False)
            if VERBOSITY >= 1:
                print_success(f"Saved top segments for {ranking_key} to {ranking_file}")
    
    # Create summary file with better formatting
    summary_file = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("FOOTBALL MATCH IMPACT ANALYSIS SUMMARY\n")
        f.write("=====================================\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Road Analyzed: {ROAD_NAME}\n")
        f.write(f"Total Segments Analyzed: {len(all_results)}\n\n")
        
        f.write("TOP FINDINGS:\n")
        
        # Add flow impact findings with cleaner segment names
        if 'pre_match_all_flow_diff' in rankings and len(rankings['pre_match_all_flow_diff']) > 0:
            top_segment = rankings['pre_match_all_flow_diff'].iloc[0]
            clean_name = clean_segment_name(top_segment['segment_id'])
            f.write(f"- Highest Pre-Match Flow Increase: {clean_name} with {top_segment['flow_diff']:.1f} vehicles/hour increase ({top_segment['flow_pct']:.1f}%)\n")
        
        if 'post_match_all_flow_diff' in rankings and len(rankings['post_match_all_flow_diff']) > 0:
            top_segment = rankings['post_match_all_flow_diff'].iloc[0]
            clean_name = clean_segment_name(top_segment['segment_id'])
            f.write(f"- Highest Post-Match Flow Increase: {clean_name} with {top_segment['flow_diff']:.1f} vehicles/hour increase ({top_segment['flow_pct']:.1f}%)\n")
        
        # Add speed impact findings with cleaner segment names
        if 'pre_match_all_speed_diff' in rankings and len(rankings['pre_match_all_speed_diff']) > 0:
            top_segment = rankings['pre_match_all_speed_diff'].iloc[0]
            clean_name = clean_segment_name(top_segment['segment_id'])
            f.write(f"- Highest Pre-Match Speed Reduction: {clean_name} with {abs(top_segment['speed_diff']):.1f} mph reduction ({abs(top_segment['speed_pct']):.1f}%)\n")
        
        if 'post_match_all_speed_diff' in rankings and len(rankings['post_match_all_speed_diff']) > 0:
            top_segment = rankings['post_match_all_speed_diff'].iloc[0]
            clean_name = clean_segment_name(top_segment['segment_id'])
            f.write(f"- Highest Post-Match Speed Reduction: {clean_name} with {abs(top_segment['speed_diff']):.1f} mph reduction ({abs(top_segment['speed_pct']):.1f}%)\n")
        
        # Team comparison with cleaner segment names
        f.write("\nTEAM COMPARISON:\n")
        
        if 'pre_match_united_flow_diff' in rankings and 'pre_match_city_flow_diff' in rankings:
            united_top = rankings['pre_match_united_flow_diff'].iloc[0] if len(rankings['pre_match_united_flow_diff']) > 0 else None
            city_top = rankings['pre_match_city_flow_diff'].iloc[0] if len(rankings['pre_match_city_flow_diff']) > 0 else None
            
            if united_top is not None and city_top is not None:
                united_name = clean_segment_name(united_top['segment_id'])
                city_name = clean_segment_name(city_top['segment_id'])
                
                f.write(f"- Manchester United's highest impact segment: {united_name} with {united_top['flow_diff']:.1f} vehicles/hour increase\n")
                f.write(f"- Manchester City's highest impact segment: {city_name} with {city_top['flow_diff']:.1f} vehicles/hour increase\n")
                
                if united_top['flow_diff'] > city_top['flow_diff']:
                    f.write("- Manchester United matches appear to have a higher maximum impact on traffic flow\n")
                else:
                    f.write("- Manchester City matches appear to have a higher maximum impact on traffic flow\n")
                
                # Add some statistical context if available
                united_segments = rankings['pre_match_united_flow_diff'].head(5)
                city_segments = rankings['pre_match_city_flow_diff'].head(5)
                
                avg_united = united_segments['flow_diff'].mean()
                avg_city = city_segments['flow_diff'].mean()
                
                f.write(f"\n- Average flow increase (top 5 segments): United: {avg_united:.1f} vehicles/hour, City: {avg_city:.1f} vehicles/hour\n")
                if avg_united > avg_city:
                    f.write(f"  (United shows {((avg_united/avg_city)-1)*100:.1f}% higher average impact across top segments)\n")
                else:
                    f.write(f"  (City shows {((avg_city/avg_united)-1)*100:.1f}% higher average impact across top segments)\n")
        
        f.write("\nMETHODOLOGY NOTES:\n")
        f.write("- Impact is measured as the difference between match periods and comparable baseline periods\n")
        f.write("- Pre-match period is defined as 2 hours before kickoff\n")
        f.write("- Post-match period is defined as 2 hours after the estimated match end\n")
        f.write("- Comparable baseline periods are the same day of week and hour of day on non-match days\n")
        f.write("- Analysis accounts for both Manchester United and Manchester City home matches\n")
        f.write("- Segments with fewer than 3 match observations were excluded from rankings\n")
    
    print_success(f"Saved enhanced analysis summary to {summary_file}")
    
    # Save the processed results for visualization
    results_file = os.path.join(OUTPUT_DIR, "processed_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'rankings': rankings
        }, f)
    print_success(f"Saved processed results for visualization to {results_file}")

    # --- NEW: Prepare top segments data for visualization ---
    # Select top 6 segments
    top_n = 6
    # Use pre_match_all_flow_diff as fallback if criticality_ranking is missing/empty
    if 'criticality_ranking' in rankings and not rankings['criticality_ranking'].empty:
        top_segments = rankings['criticality_ranking'].head(top_n)['segment_id'].tolist()
    else:
        # fallback: use top segments by pre-match flow difference
        if 'pre_match_all_flow_diff' in rankings:
            top_segments = rankings['pre_match_all_flow_diff'].head(top_n)['segment_id'].tolist()
        else:
            top_segments = []
    # Only proceed if we have top segments
    if top_segments:
        def extract_direction_and_junctions(segment_id):
            direction = ''
            junctions = ''
            seg = segment_id.lower()
            if 'anti-clockwise' in seg:
                direction = 'ACW'
            elif 'clockwise' in seg:
                direction = 'CW'
            # Try to extract 'between Jx and Jy'
            m = re.search(r'between J(\d+) and J(\d+)', seg)
            if m:
                junctions = f"J{m.group(1)}-J{m.group(2)}"
            else:
                # Try to extract a single J number
                m = re.search(r'j(\d+)', seg)
                if m:
                    junctions = f"J{m.group(1)}"
            return direction, junctions
        # Prepare data structure
        viz_rows = []
        for result in all_results:
            if result['segment_id'] not in top_segments:
                continue
            segment_id = result['segment_id']
            clean_name = clean_segment_name(segment_id)
            direction, junctions = extract_direction_and_junctions(segment_id)
            for period in ['pre_match', 'post_match']:
                for team in ['all', 'united', 'city']:
                    impact = result['impacts'].get(f"{period}_{team}", {})
                    viz_rows.append({
                        'segment_id': segment_id,
                        'clean_name': clean_name,
                        'direction': direction,
                        'junctions': junctions,
                        'period': period,
                        'team': team,
                        'flow_diff': impact.get('flow_diff', None),
                        'flow_pct': impact.get('flow_pct', None),
                        'flow_iqr': impact.get('flow_iqr', None),
                        'flow_q1': impact.get('flow_q1', None),
                        'flow_q3': impact.get('flow_q3', None),
                        'speed_diff': impact.get('speed_diff', None),
                        'speed_pct': impact.get('speed_pct', None),
                        'speed_iqr': impact.get('speed_iqr', None),
                        'speed_q1': impact.get('speed_q1', None),
                        'speed_q3': impact.get('speed_q3', None),
                        'sample_count': impact.get('sample_count', None)
                    })
        viz_df = pd.DataFrame(viz_rows)
        viz_csv = os.path.join(OUTPUT_DIR, 'top_segments_for_viz.csv')
        viz_pkl = os.path.join(OUTPUT_DIR, 'top_segments_for_viz.pkl')
        viz_df.to_csv(viz_csv, index=False)
        with open(viz_pkl, 'wb') as f:
            pickle.dump(viz_df, f)
        print_success(f"Saved top segments data for visualization to {viz_csv} and {viz_pkl}")
    # --- END NEW ---

def classify_match_day(match_datetime):
    """
    Classify a match as weekend or weekday.
    
    Parameters:
    match_datetime (datetime): Match datetime
    
    Returns:
    str: 'weekend' or 'weekday'
    """
    return 'weekend' if match_datetime.weekday() >= 5 else 'weekday'

def prepare_match_data(match_windows):
    """
    Prepare and classify match data.
    
    Parameters:
    match_windows (DataFrame): Raw match windows data
    
    Returns:
    DataFrame: Processed match data with classifications
    """
    print_section("Preparing match data")
    
    # Convert datetime columns
    datetime_cols = ['start_pre_match', 'end_pre_match', 'start_post_match', 'end_post_match']
    for col in datetime_cols:
        match_windows[col] = pd.to_datetime(match_windows[col])
    
    # Add match classification
    match_windows['match_type'] = match_windows['start_pre_match'].apply(classify_match_day)
    
    # Add kickoff time (2 hours after pre-match start)
    match_windows['kickoff_time'] = match_windows['end_pre_match']
    
    # Group matches
    weekend_matches = match_windows[match_windows['match_type'] == 'weekend']
    weekday_matches = match_windows[match_windows['match_type'] == 'weekday']
    
    # Check minimum sample sizes
    min_matches = 5
    if len(weekend_matches) < min_matches:
        print_warning(f"Insufficient weekend matches ({len(weekend_matches)} < {min_matches})")
    if len(weekday_matches) < min_matches:
        print_warning(f"Insufficient weekday matches ({len(weekday_matches)} < {min_matches})")
    
    print_success(f"Processed {len(match_windows)} matches: {len(weekend_matches)} weekend, {len(weekday_matches)} weekday")
    return match_windows

def prepare_traffic_data(segment_df):
    """
    Prepare traffic data by filtering and cleaning.
    
    Parameters:
    segment_df (DataFrame): Raw traffic data for a segment
    
    Returns:
    DataFrame: Cleaned traffic data
    """
    # Remove interpolated data
    clean_df = segment_df[segment_df['is_interpolated'] == 0].copy()
    
    # Filter by data completeness (keep only high quality data with 3-4 readings per hour)
    clean_df = clean_df[clean_df['data_completeness'] >= 3]
    
    # Add hour of day
    clean_df['hour'] = pd.to_datetime(clean_df['datetime']).dt.hour
    
    return clean_df

def find_control_periods(match_info, traffic_df, min_control_periods=3):
    """
    Find suitable control periods for a match.
    
    Parameters:
    match_info (Series): Match information
    traffic_df (DataFrame): Traffic data
    min_control_periods (int): Minimum number of control periods required
    
    Returns:
    DataFrame: Traffic data with control period flags
    """
    match_type = match_info['match_type']
    kickoff_time = match_info['kickoff_time']
    
    # Get day type (weekend/weekday)
    control_days = traffic_df[traffic_df['day_type'] == match_type].copy()
    
    # Match time of day (±30 minutes)
    match_hour = kickoff_time.hour
    control_days['hour_match'] = (
        (control_days['hour'] >= match_hour - 0.5) & 
        (control_days['hour'] <= match_hour + 0.5)
    )
    
    # Flag control periods
    control_periods = control_days[control_days['hour_match']]['datetime'].unique()
    
    if len(control_periods) < min_control_periods:
        print_warning(f"Insufficient control periods for match on {kickoff_time.date()} ({len(control_periods)} < {min_control_periods})")
        return None
    
    return control_periods

def main():
    """Main execution function."""
    print_section("FOOTBALL MATCH IMPACT ANALYSIS")
    print(f"Road: {ROAD_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load segment data
    segment_data = load_segment_data()
    if not segment_data:
        print_error("No segment data found. Exiting.")
        return
    
    # Load match windows
    match_windows = load_match_windows()
    if match_windows.empty:
        print_error("No match windows data found. Exiting.")
        return
    
    # Analyze all segments
    all_results = analyze_all_segments(segment_data, match_windows)
    if not all_results:
        print_error("No valid segment results. Exiting.")
        return
    
    # Rank segments by impact
    rankings = rank_segments_by_impact(all_results)
    if not rankings:
        print_error("No valid rankings. Exiting.")
        return
    
    # Save impact data
    save_impact_data(all_results, rankings)
    
    print_section("ANALYSIS COMPLETE")
    print_success(f"Results saved to {OUTPUT_DIR}")
    print_success(f"Run visualize_match_impact.py to create visualizations")

if __name__ == "__main__":
    main() 