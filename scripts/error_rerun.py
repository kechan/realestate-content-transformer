import argparse
import re
import subprocess
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

def load_run_entries(year, month):
    df = pd.read_csv('run_entry_table.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
    mask = ((df['timestamp'].dt.year == year) & 
            (df['timestamp'].dt.month == month) & 
            (df['prov_code'].str.len() == 2))
    return df[mask]

def get_log_filename(row):
    timestamp = row['timestamp'].strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_run_{row['run_number']}_{row['prov_code']}_{row['lang']}.log"

def extract_failed_geog_ids(log_filename):
    error_pattern = re.compile(r'\[ERROR\].*\[geog_id: ([^\]]+)\]')
    failed_geog_ids = set()

    try:
        with open(log_filename, 'r') as log_file:
            for line in log_file:
                match = error_pattern.search(line)
                if match:
                    failed_geog_ids.add(match.group(1))
    except FileNotFoundError:
        print(f"Warning: Log file '{log_filename}' not found.")

    return list(failed_geog_ids)

def rerun_failed_geog_ids(failed_geog_ids, config_file):
    for i, geog_id in enumerate(failed_geog_ids, 1):
        print(f"Rerunning for geog_id: {geog_id}")
        subprocess.run([
            "python", "run_locallogic_content_rewriter.py",
            "--config", config_file,
            "--geog_id", geog_id
        ])
        if i < len(failed_geog_ids):
            print("Waiting for 40 seconds before the next rerun...")
            time.sleep(40)

def main():
    parser = argparse.ArgumentParser(description="Process log files and rerun failed geog_ids for a specific month")
    parser.add_argument("--year", type=int, required=True, help="Year of the runs to process")
    parser.add_argument("--month", type=int, required=True, help="Month of the runs to process")
    parser.add_argument("--config", default="spot_fix_prod_config.yaml", help="Path to the config file")
    args = parser.parse_args()

    run_entries = load_run_entries(args.year, args.month)
    
    if run_entries.empty:
        print(f"No matching run entries found for {args.year}-{args.month:02d}")
        return

    all_failed_geog_ids = set()

    for _, row in run_entries.iterrows():
        log_filename = get_log_filename(row)
        print(f"Processing log file: {log_filename}")
        
        failed_geog_ids = extract_failed_geog_ids(log_filename)
        all_failed_geog_ids.update(failed_geog_ids)

    if not all_failed_geog_ids:
        print("No failed geog_ids found in the processed log files.")
        return

    print(f"Found {len(all_failed_geog_ids)} unique failed geog_ids across all processed logs.")
    rerun_failed_geog_ids(list(all_failed_geog_ids), args.config)

if __name__ == "__main__":
    main()