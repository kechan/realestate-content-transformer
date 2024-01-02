from typing import List
import os
import tarfile
import re

from pathlib import Path

def archive_logs(log_dir: Path, prov_codes: List[str]):
  for prov_code in prov_codes:
    pattern = re.compile(rf'\d{{8}}_\d{{6}}_run_(\d+)_{prov_code}_en\.log')

    # Extract log files matching the pattern
    log_files = [file for file in log_dir.iterdir() if pattern.match(file.name)]

    # Sort log files numerically by the run number
    log_files.sort(key=lambda x: int(pattern.match(x.name).group(1)))

    if log_files:
      first_run_num = pattern.match(log_files[0].name).group(1)
      last_run_num = pattern.match(log_files[-1].name).group(1)
      archive_name = log_dir / f"run_{first_run_num}_{last_run_num}_{prov_code}_en.tar.gz"
      with tarfile.open(archive_name, "w:gz") as tar:
        for log_file in log_files:
          tar.add(log_file, arcname=log_file.name)

      print(f"Archived {len(log_files)} files into {archive_name}")

      # Delete the archived files
      for log_file in log_files:
        log_file.unlink()
      print(f"Deleted {len(log_files)} archived files.")

def main():
  # Directory containing the log files
  log_directory = Path('.')

  # List of provincial codes
  provincial_codes = ["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE", "QC", "SK", "YT"]

  archive_logs(log_directory, provincial_codes)

if __name__ == '__main__':
  main()
