from typing import List
import os
import tarfile
import re

from pathlib import Path
from realestate_content_transformer.utils.misc import archive_logs

def main():
  # Directory containing the log files
  log_directory = Path('.')

  # List of provincial codes
  provincial_codes = ["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE", "QC", "SK", "YT"]

  archive_logs(log_directory, provincial_codes)

if __name__ == '__main__':
  main()
