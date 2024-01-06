from typing import Optional, Dict, Union

from enum import Enum
import redis, json, re, ast, sys
import pandas as pd
from pathlib import Path

class ArchiveStorageType(Enum):
  REDIS = 1
  PLAIN_TEXT = 2

class ChatGPTRewriteArchiver:
  def __init__(self, storage_type: ArchiveStorageType, file_path: Optional[str] = None, redis_host: str = 'localhost', redis_port: int = 6379) -> None:
    self.storage_type = storage_type
    if self.storage_type == ArchiveStorageType.REDIS:
      self.db = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    elif self.storage_type == ArchiveStorageType.PLAIN_TEXT:
      if file_path is None:
        raise ValueError("file_path is required if storage_type is PLAIN_TEXT.")
      self.file_path = file_path

    # check if file_path exists for PLAIN_TEXT
    if self.storage_type == ArchiveStorageType.PLAIN_TEXT:
      if not Path(self.file_path).exists():
        # create file if not exists
        with open(self.file_path, 'w') as f:
          pass

    self.cached_df = self.get_all_records(return_df=True)

  def ping(self) -> str:
    if self.storage_type == ArchiveStorageType.REDIS:
      return self.db.ping()
    return "pong"

  def add_record(self, longId: str, property_type: str, version: str, user_prompt: str, chatgpt_response: str) -> None:
    if property_type not in ['LUXURY', 'CONDO', 'SEMI-DETACHED', 'TOWNHOUSE', 'INVESTMENT']:
      raise ValueError(f"Invalid property_type {property_type}.")

    key = f"rewrite:{longId}:{property_type}:{version}"
    value = {
      'version': version,
      'user_prompt': user_prompt,
      'chatgpt_response': chatgpt_response
    }
    if property_type is not None:
      value['property_type'] = property_type

    if self.storage_type == ArchiveStorageType.REDIS:  
      self.db.hmset(key, value)
    elif self.storage_type == ArchiveStorageType.PLAIN_TEXT:
      with open(self.file_path, 'a') as f:
        f.write(f"{key}|| {value}\n")   # use || as separator as this shouldnt be seen in value

  def update_response(self, longId: str, property_type: str, version: str, new_response: str) -> None:
    key = f"rewrite:{longId}:{property_type}:{version}"
    if self.db.exists(key):
      self.db.hset(key, 'chatgpt_response', new_response)
    else:
      raise KeyError("Record not found.")

  def update_prompt(self, longId: str, property_type: str, version: str, new_prompt: str) -> None:
    key = f"rewrite:{longId}:{property_type}:{version}"
    if self.db.exists(key):
      self.db.hset(key, 'user_prompt', new_prompt)
    else:
      raise KeyError("Record not found.")

  def get_record(self, longId: str, property_type: str, version: str, use_cache=False) -> Optional[Dict[str, str]]:
    '''
    # TODO: Implement a more robust method for determining the latest record when querying by version prefix.
    # Current implementation assumes the last record encountered in the dataset is the latest one, which might not 
    # always be accurate. This could lead to inconsistent results, especially if the dataset isn't strictly 
    # chronological. Consider adding logic to parse and compare date strings accurately to ensure the truly latest 
    # record is returned. This is particularly important for Redis storage, where the order of keys might not 
    # reflect their chronological order. For the plain text storage, while the last line is likely to be the latest, 
    # it's not guaranteed. A more reliable approach would be to convert date strings to datetime objects for 
    # comparison, or ensure that records are stored in a manner that maintains their chronological order.

    Returns a dictionary containing the records for the given longId, property_type, and version.

    The version can be a full date string (e.g., '20231230') or a year-month prefix (e.g., '202312').
    Or None if not found.
    '''
    # records = {}
    record = None
    version_regex = re.compile(f"^{re.escape(version)}")

    # key = f"rewrite:{longId}:{property_type}:{version}"
    if self.storage_type == ArchiveStorageType.REDIS:
      pattern = f"rewrite:{longId}:{property_type}:{version}*"
      keys = self.db.keys(pattern)
      for key in keys:
        record = self.db.hgetall(key)
        # records[key] = record

    elif self.storage_type == ArchiveStorageType.PLAIN_TEXT:
      if use_cache:
        if self.cached_df is not None and len(self.cached_df) > 0:
          filtered_df = self.cached_df.q(f"longId == '{longId}' and property_type == '{property_type}' and version.str.startswith('{version}')")
          if not filtered_df.empty:
            return filtered_df.iloc[-1].to_dict()
        else:
          return None

      else:
        key_prefix = f"rewrite:{longId}:{property_type}:"
        with open(self.file_path, 'r') as f:
          for line in f:
              if line.startswith(key_prefix):
                key, sep, record_str = line.partition('||')
                if sep and version_regex.match(key.split(':', 3)[-1]):
                  try:
                    record = ast.literal_eval(record_str.strip())
                    # records[key] = record_dict
                  except (SyntaxError, ValueError) as e:
                    print(f"Error parsing the record string: {e}", file=sys.stderr)

      return record if record else None

  def remove_record(self, longId: str, property_type: str, version: str) -> None:
    key = f"rewrite:{longId}:{property_type}:{version}"
    if self.db.exists(key):
      self.db.delete(key)
    else:
      raise KeyError("Record not found.")

  def get_all_records(self, return_df=False) -> Union[Dict[str, Dict[str, str]], pd.DataFrame]:
    records = {}

    if self.storage_type == ArchiveStorageType.REDIS:
      keys = self.db.keys('rewrite:*')      
      for key in keys:
        record = self.db.hgetall(key)
        longId, property_type, version = key.split(':', 3)[1:]
        records[f"{longId}:{property_type}:{version}"] = record

    elif self.storage_type == ArchiveStorageType.PLAIN_TEXT:
      with open(self.file_path, 'r') as f:
        for line in f:
          parts = line.split('||')
          if len(parts) == 2:
            key, record_str = parts
            longId, property_type, version = key.split(':', 3)[1:]
            try:
              record_dict = ast.literal_eval(record_str.strip())
              records[f"{longId}:{property_type}:{version}"] = record_dict
            except (SyntaxError, ValueError) as e:
              print(f"Error parsing the record string: {e}", file=sys.stderr)
    
    if return_df:
      df = pd.DataFrame.from_dict(records, orient='index')
      df.reset_index(inplace=True)
      if len(df) > 0:
        df[['longId', 'property_type', 'version']] = df['index'].str.split(':', expand=True)
        df.drop(columns=['index'], inplace=True)
        df = df[['longId', 'property_type', 'version', 'user_prompt', 'chatgpt_response']]

      return df

    return records
