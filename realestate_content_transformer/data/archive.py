from typing import Optional, Dict, Union

from enum import Enum
import redis, json, re, ast
import pandas as pd

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

  def get_record(self, longId: str, property_type: str, version: str) -> Optional[Dict[str, str]]:
    '''
    Returns a dictionary containing the record for the given longId, property_type, and version.

    Or None if not found.
    '''
    key = f"rewrite:{longId}:{property_type}:{version}"
    if self.storage_type == ArchiveStorageType.REDIS:
      return self.db.hgetall(key)
    elif self.storage_type == ArchiveStorageType.PLAIN_TEXT:
      regex = re.compile(f"^{re.escape(key)}\\|\\|\s*(.+)$")
      with open(self.file_path, 'r') as f:
        for line in f:
          match = regex.match(line)
          if match:
            record_str = match.group(1)
            # Assuming the record string is in a valid dictionary format
            try:
              record_dict = ast.literal_eval(record_str)
              return record_dict
            except (SyntaxError, ValueError) as e:
              print(f"Error parsing the record string: {e}", file=sys.stderr)
              raise e
    else:
      return None

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
      df[['longId', 'property_type', 'version']] = df['index'].str.split(':', expand=True)
      df.drop(columns=['index'], inplace=True)
      
      df = df[['longId', 'property_type', 'version', 'user_prompt', 'chatgpt_response']]
      return df

    return records
