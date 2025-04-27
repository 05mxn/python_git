import os
import pandas as pd
import numpy as np
from glob import glob

# Directory containing transformed feature files
INPUT_DIR = 'ML_Data/transformed_features'
OUTPUT_DIR = 'ML_Data/typical_aggregates'

# Metrics to aggregate
METRICS = [
    'total_traffic_flow', 'speed_reduction', 'journey_delay',
    'traffic_1_pct', 'traffic_2_pct', 'traffic_3_pct', 'traffic_4_pct'
]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_day_type(row):
    if row['is_weekend'] == 1:
        return 'weekend'
    elif row['is_school_holiday'] == 1:
        return 'school_holiday'
    else:
        return 'weekday'

def aggregate_file(file_path):
    print(f'Processing {file_path}...')
    df = pd.read_csv(file_path)
    # Add day_type column
    df['day_type'] = df.apply(get_day_type, axis=1)
    # Group by segment, direction, hour, day_type
    group_cols = [
        'ntis_link_number', 'ntis_link_description', 'direction', 'hour', 'day_type',
        'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'link_length'
    ]
    grouped = df.groupby(group_cols)
    # Aggregate metrics
    agg_dict = {}
    for metric in METRICS:
        agg_dict[metric] = ['mean', 'std', 'min', 'max',
                            lambda x: np.percentile(x.dropna(), 5) if len(x.dropna()) > 0 else np.nan,
                            lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan]
    agg_df = grouped.agg(agg_dict)
    # Flatten columns
    agg_df.columns = ['_'.join([col[0], col[1] if isinstance(col[1], str) else str(i)])
                      for i, col in enumerate(agg_df.columns)]
    agg_df = agg_df.reset_index()
    # Save as parquet
    base = os.path.basename(file_path).replace('.csv', '_typical.parquet')
    out_path = os.path.join(OUTPUT_DIR, base)
    agg_df.to_parquet(out_path, index=False)
    print(f'Saved: {out_path}')

def main():
    files = glob(os.path.join(INPUT_DIR, '*.csv'))
    for file_path in files:
        aggregate_file(file_path)
    print('All files processed.')

if __name__ == '__main__':
    main() 