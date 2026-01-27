import pandas as pd
import os
import glob
import numpy as np

def read_cams_radiation_csv(file_path, header_row=42, mode='mean'):
    """
    Reads CAMS radiation CSV and returns downsampled hourly DataFrame.
    
    Args:
        file_path: Full path to the CSV file.
        header_row: Line number where actual headers start (default 68).
        mode: 'mean' for hourly average, 'instant' for only :00 timestamps.
        
    Returns:
        column_names (list), hourly DataFrame
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def extract_value(line):
        return float(line.split(':')[-1].strip())

    latitude = extract_value(lines[11])
    longitude = extract_value(lines[12])
    altitude = extract_value(lines[13])

    header_line = lines[header_row].strip()
    column_names = header_line.split(';')

    df = pd.read_csv(file_path, skiprows=header_row + 1, sep=';', names=column_names, encoding='utf-8')
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    df['timestamp'] = df['timestamp'].str.split('/').str[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.set_index('timestamp', inplace=True)

    if mode == 'mean':
        df = df.resample('1h').mean(numeric_only=True).dropna(how='all')
    elif mode == 'instant':
        df = df
    else:
        raise ValueError("Invalid mode: use 'mean' or 'instant'")

    df = df.reset_index()
    df['latitude'] = latitude
    df['longitude'] = longitude
    df['altitude'] = altitude
    df.attrs['latitude'] = latitude
    df.attrs['longitude'] = longitude
    df.attrs['altitude'] = altitude

    return column_names, df

def process_and_save_csvs(source_root, target_root, header_row=68, mode='mean'):
    """
    Walk through all CSVs in source_root, resample to hourly, and save to target_root.
    
    Args:
        source_root: Root folder with raw CSVs.
        target_root: Destination for hourly CSVs.
        header_row: Row where actual data headers begin.
        mode: 'mean' or 'instant'
    """
    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            if not filename.lower().endswith('.csv'):
                continue

            source_path = os.path.join(dirpath, filename)
            rel_dir = os.path.relpath(dirpath, source_root)
            target_dir = os.path.join(target_root, rel_dir)
            os.makedirs(target_dir, exist_ok=True)

            new_filename = os.path.splitext(filename)[0] + f'_{mode}.csv'
            target_path = os.path.join(target_dir, new_filename)

            if os.path.exists(target_path):
                print(f"⏭️ Skipped (already exists): {target_path}")
                continue

            try:
                headers, df = read_cams_radiation_csv(source_path, header_row=header_row, mode=mode)

                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(';'.join(headers) + '\n')
                    df.to_csv(f, sep=';', index=False, header=False)

                print(f"✅ {mode.upper()} saved: {target_path}")
            except Exception as e:
                print(f"❌ Failed: {source_path} — {e}")

def check_hourly_time_range_consistency(root_folder, delimiter=';'):
    """
    Scans all hourly CSVs in a folder tree, extracts start/end timestamps,
    and checks whether their time coverage is consistent.

    Returns:
        consistent (list): List of (file_path, start_time, end_time)
        inconsistent (list): List of (file_path, start_time, end_time, reason)
    """
    time_ranges = []
    inconsistent = []
    ref_start, ref_end = None, None

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith('.csv'):
                continue
            path = os.path.join(dirpath, filename)
            try:
                df = pd.read_csv(path, sep=delimiter, usecols=[0], names=['timestamp'], header=0, parse_dates=['timestamp'])
                if df.empty:
                    inconsistent.append((path, None, None, 'Empty file'))
                    continue

                start = df['timestamp'].iloc[0]
                end = df['timestamp'].iloc[-1]

                if ref_start is None:
                    ref_start, ref_end = start, end
                    time_ranges.append((path, start, end))
                else:
                    if start != ref_start or end != ref_end:
                        inconsistent.append((path, start, end, 'Time range mismatch'))
                    else:
                        time_ranges.append((path, start, end))

            except Exception as e:
                inconsistent.append((path, None, None, f'Read error: {e}'))

    return time_ranges, inconsistent


def read_cams_hourly_csv(file_path):
    # Read the file while fixing the header line and setting column names
    with open(file_path, 'r') as f:
        header_line = f.readline().lstrip('# ').strip()
        column_names = header_line.split(';') + ['latitude', 'longitude', 'altitude']

    # Read the data with corrected column names
    df = pd.read_csv(file_path, sep=';', skiprows=1, header=None, names=column_names)
    
    # Optional: parse the time column
    df['Observation period'] = pd.to_datetime(df['Observation period'])

    return df


def merge_cams_folder(folder_path, output_file):
    # Find all CSVs in folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found in the folder.")

    # Read + collect all DataFrames
    frames = [read_cams_hourly_csv(f) for f in csv_files]

    # Concatenate
    merged_df = pd.concat(frames, ignore_index=True)

    # Sort by time if useful
    merged_df = merged_df.sort_values("Observation period")

    # Write output
    merged_df.to_csv(output_file, index=False)

    return merged_df


def engineer_features(df):
    final_features = [
        'GHI',
        'Clear sky GHI',
        'Observation period',
        'longitude', 'latitude',
    ]
    return df, final_features


def build_grid_mapping(df):
    """
    Build row/col indices for grid based on lon/lat columns.
    Assumes df already has 'lon' and 'lat' (from read_cams_hourly_csv).
    """
    unique_coords = df[['longitude', 'latitude']].drop_duplicates()

    lons = sorted(unique_coords['longitude'].unique())
    # reverse=True so row 0 is the northernmost latitude, as before
    lats = sorted(unique_coords['latitude'].unique(), reverse=True)

    lon_to_idx = {lon: i for i, lon in enumerate(lons)}
    lat_to_idx = {lat: i for i, lat in enumerate(lats)}

    df['row'] = df['latitude'].map(lat_to_idx)
    df['col'] = df['longitude'].map(lon_to_idx)

    H, W = len(lats), len(lons)
    return df, (H, W)

def csv_to_npz_grid(df=None,
                    csv_path=None,
                    out_path="out.npz",
                    grid_shape=(50, 50),
                    time_col="Observation period",
                    ghi_col="GHI",
                    ghi_clear_col="Clear sky GHI",
                    row_col="row",
                    col_col="col"):
    """
    Convert CSV or DataFrame to NPZ:
      data: (T, 3, H, W) with features [GHI, clear_sky_GHI]
      time: (T,) datetime64
    """

    # 1. Load CSV if df not given
    if df is None:
        if csv_path is None:
            raise ValueError("Must provide either df= or csv_path=")
        df = pd.read_csv(csv_path)

    H, W = grid_shape

    # 2. Ensure time column is datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # 3. Build time axis
    times = np.sort(df[time_col].unique())
    T = len(times)
    time_index = {t: i for i, t in enumerate(times)}

    # 4. Allocate output array
    data = np.full((T, 2, H, W), np.nan, dtype=np.float32)

    # 5. Fill values using row[...] (NOT getattr)
    for _, row in df.iterrows():
        t = row[time_col]
        r = int(row[row_col])
        c = int(row[col_col])

        ti = time_index[t]

        data[ti, 0, r, c] = row[ghi_col]
        data[ti, 1, r, c] = row[ghi_clear_col]

    # 6. Save NPZ
    time_arr = times.astype("datetime64[ns]")
    np.savez(out_path, data=data, time=time_arr)

    return data, time_arr