import argparse, sys
import pandas as pd
import re
from pathlib import Path

import pandas as pd
# Set the maximum number of rows and columns to display
pd.set_option('display.max_rows', None)  # or use an integer to specify a fixed number of rows
pd.set_option('display.max_columns', None)  # or use an integer to specify a fixed number of columns

# Optionally, you can set the width of the entire output
pd.set_option('display.width', None)  # or use an integer to specify the width

from realestate_core.common.class_extensions import *

parser = argparse.ArgumentParser(description='Parse log files for a specific run number.')
parser.add_argument('--log_filename', type=str, help='The name of the log file to parse. Once specified, rest of the arguments are ignored.')

parser.add_argument('--run_num', type=str, required=True, help='The run number to filter the log files by.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--prov_code', type=str, help='The province code to filter the log files by.')
group.add_argument('--geog_id', type=str, help='Optional geographic ID to process a specific location.')

parser.add_argument('--log_dir', type=str, default='.', help='The directory containing the log files. Defaults to the current directory.')
parser.add_argument('--max_colwidth', type=int, default=100, help='The maximum width of each column. Defaults to 100.')

args = parser.parse_args()
log_filename = args.log_filename
if log_filename:
  log_file_paths = [log_filename]
else:
  run_num = args.run_num
  prov_code = args.prov_code
  geog_id = args.geog_id
  log_dir = args.log_dir
  max_colwidth = args.max_colwidth
  # Set the maximum width of each column
  pd.set_option('display.max_colwidth', max_colwidth)  # or use an integer to specify the width

  # Assuming `path` is a Path object pointing to the directory containing the log files
  location_identifier = geog_id if geog_id else prov_code
  log_filename_regex_pattern = rf'^(\d{{8}})_(\d{{6}})_run_{run_num}_{location_identifier}_(\w+)\.log$'

  log_file_paths = Path(log_dir).lfre(log_filename_regex_pattern)

error_pattern = re.compile(
  r'.*\[ERROR\].*\[longId: ([^\]]+)\](?: \[geog_id: ([^\]]+)\])? (.+)$'
)

# Initialize a dictionary to store the data
data = {'longId': [], 'geog_id': [], 'Error': []}

for log_file_path in log_file_paths:
  try:
    # Open and read the log file
    with open(log_file_path, 'r') as log_file:
      for line in log_file:
        match = error_pattern.search(line)
        if match:
          # Extract longId, geog_id, and error message from the match
          long_id, geog_id, error_message = match.groups()

          # Append the extracted data to the lists in the dictionary
          data['longId'].append(long_id)
          data['geog_id'].append(geog_id)
          data['Error'].append(error_message)
  except Exception as e:
    print(f"An error occurred while processing file {log_file_path}: {e}", file=sys.stderr)


# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Remove duplicate rows based on longId and geog_id
df = df.drop_duplicates(subset=['longId', 'geog_id'])

# Display the DataFrame
print(df)
