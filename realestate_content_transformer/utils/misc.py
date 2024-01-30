from typing import List, Tuple
import tiktoken, re, tarfile
from pathlib import Path

def num_tokens_from_string(string: str, encoding_name='cl100k_base') -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens

def archive_logs(log_dir: Path, prov_codes: List[str]):
  # print('archive logs')
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

  # archive also the geog_id runs
  geog_pattern = re.compile(rf'\d{{8}}_\d{{6}}_run_(\d+)_g(.*?)_en\.log')
  log_files = [file for file in log_dir.iterdir() if geog_pattern.match(file.name)]
  log_files.sort(key=lambda x: int(geog_pattern.match(x.name).group(1)))    # sort by run number
  # print(f"Found {len(log_files)} geog_id log files.")

  if log_files:
    first_run_num = geog_pattern.match(log_files[0].name).group(1)
    last_run_num = geog_pattern.match(log_files[-1].name).group(1)
    archive_name = log_dir / f"run_{first_run_num}_{last_run_num}_geog_en.tar.gz"
    with tarfile.open(archive_name, "w:gz") as tar:
      for log_file in log_files:
        tar.add(log_file, arcname=log_file.name)

    print(f"Archived {len(log_files)} files into {archive_name}")

    # Delete the archived files
    for log_file in log_files:
      log_file.unlink()
    print(f"Deleted {len(log_files)} archived files.")
