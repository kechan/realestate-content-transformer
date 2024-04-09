from typing import Optional, Tuple
import logging, time, json, subprocess

from datetime import datetime
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from celery import Celery
from celery.result import AsyncResult

import pandas as pd

from realestate_content_transformer.data.pipeline import LocallogicContentRewriter
from realestate_spam.llm.chatgpt import LocalLogicGPTRewriter
from realestate_content_transformer.utils.misc import archive_logs

LIGHT_WEIGHT_LLM = 'gpt-3.5-turbo-0613'
LLM = 'gpt-4-1106-preview'

# celery_app = Celery('myapp', broker='pyamqp://guest@localhost//')
app = FastAPI()

class ProvinceCode(str, Enum):
  AB = "AB"
  BC = "BC"
  MB = "MB"
  NB = "NB"
  NL = "NL"
  NT = "NT"
  NS = "NS"
  NU = "NU"
  ON = "ON"
  PE = "PE"
  QC = "QC"
  SK = "SK"
  YT = "YT"

class RewriterConfig(BaseModel):
  es_host: str = Field(..., example="es_ip", description="Elasticsearch host. Default is 'localhost' if not provided.")
  es_port: int = Field(..., example=9200, description="Elasticsearch port. Default is 9201 if not provided.")
  
class RunConfig(RewriterConfig):
  yaml_path: str = Field(..., example="/home/jupyter/prod_config.yaml", description="The full path to the yaml config file.")
  prov_code: Optional[ProvinceCode] = Field(None, example="BC", description="Optional province code. Either prov_code or geog_id is required.")
  geog_id: Optional[str] = Field(None, example="g30_dxbcrsms", description="Optional geographic ID to process a specific location. If not provided, the script processes the entire province.")
  lang: str = Field('en', example='en', description='Language, Default is "en" if not provided.')  
  archiver_file: Optional[str] = Field(None, example="./archive_rewrites.txt", description='The filepath to the archiver file. Default is "./archive_rewrites.txt" if not provided.')
  force_rewrite: Optional[str] = Field(None, example=False, description='Force rewrite regardless of version')  

def get_run_entry_df(csv_file='run_entry_table.csv'):
  try:
    run_entry_df = pd.read_csv(csv_file)
  except FileNotFoundError:
    run_entry_df = pd.DataFrame(columns=['timestamp', 'run_number', 'prov_code', 'lang'])
    run_entry_df.to_csv(csv_file, index=False)
  return run_entry_df

def add_run_entry(run_entry_df, timestamp, run_number, location_identifier, lang, duration, rewrites_count, csv_file='run_entry_table.csv'):
  new_entry = pd.DataFrame({
      'timestamp': [timestamp], 
      'run_number': [run_number], 
      'prov_code': [location_identifier], 
      'lang': [lang],
      'duration': [duration],
      'rewrites_count': [rewrites_count]
  })
  run_entry_df = pd.concat([run_entry_df, new_entry], ignore_index=True)
  run_entry_df.to_csv(csv_file, index=False)

def setup_logging(log_filename, log_level=logging.INFO):
  logging.basicConfig(filename=log_filename, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def test_openai_health():
  ll_gpt_writer = LocalLogicGPTRewriter(llm_model=LIGHT_WEIGHT_LLM, available_sections=['housing'], property_type=None)
  status = ll_gpt_writer.test_openai_health()
  del ll_gpt_writer    # not to be used again, it is just for sanity checking health
  return status

def get_temp_filename():
  return "/tmp/dummy_archiver_file.txt"  

@app.get("/")
def read_root():
  return {"message": "Welcone to Local Logic Content Rewriter Service Endpoints"}

'''
@celery_app.task
def main(config: RunConfig) -> Tuple[int, float]:
  # check openai health
  # will immediately exit if not healthy and log the error
  # NOTE: within rewrite_property_types, a consecutive failure count is used to halt if there are too many consecutive failures (due mostly to openai api)
  start_time = time.time()

  if not test_openai_health():
    logging.error("OpenAI API is not healthy. Exiting...")
    return
  
  # process for run entry table
  run_entry_df = get_run_entry_df()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  location_identifier = config.geog_id if config.geog_id else config.prov_code

  # Set up logging
  log_filename = f'{timestamp}_run_{run_number}_{location_identifier}_{config.lang}.log'
  setup_logging(log_filename=log_filename, log_level=config.log_level)

  # Check for the highest run_number for the given prov_code and lang
  if config.geog_id is not None:
    filtered_df = run_entry_df[(run_entry_df['prov_code'] == config.geog_id) & (run_entry_df['lang'] == config.lang)]
  else:
    filtered_df = run_entry_df[(run_entry_df['prov_code'] == config.prov_code) & (run_entry_df['lang'] == config.lang)]

  run_number = filtered_df['run_number'].max() + 1 if not filtered_df.empty else 1

  ll_rewriter = LocallogicContentRewriter(
    es_host=config.es_host, 
    es_port=config.es_port, 
    llm_model=LLM,
    simple_append=True,
    archiver_filepath=config.archiver_file)

  use_rag = True  # this is better than placeholders 1) stronger context 2) simpler realtime content retrieval

  city_rewrites_count = 0
  property_type_rewrites_count = 0
  try:
    ll_rewriter.extract_content(prov_code=config.prov_code, geog_id=config.geog_id, incl_property_override=True)

    city_rewrites_count = ll_rewriter.rewrite_cities(prov_code=config.prov_code, geog_id=config.geog_id, lang=config.lang, use_rag=use_rag)

    for property_type in ll_rewriter.property_types:
      property_type_rewrites_count += ll_rewriter.rewrite_property_types(property_type=property_type, 
                                                                         prov_code=config.prov_code, 
                                                                         geog_id=config.geog_id, 
                                                                         lang=config.lang, 
                                                                         use_rag=use_rag, 
                                                                         force_rewrite=config.force_rewrite)

  except Exception as e:
    logging.exception("An error occurred: %s", e)

  finally:
    end_time = time.time()
    duration = (end_time - start_time) / 60   # in minutes

    add_run_entry(run_entry_df, timestamp, run_number, location_identifier, config.lang, duration, city_rewrites_count + property_type_rewrites_count)

    return city_rewrites_count + property_type_rewrites_count, duration
  
@app.get("/task/{task_id}")
def get_task_status(task_id: str):
  task = AsyncResult(task_id, app=celery_app)
  return {"task_id": str(task.id), "task_status": task.status}
'''

@app.post("/main")
def run_main(config: RunConfig) -> JSONResponse:
  # Check that either prov_code or geog_id is provided but not both
  if not (config.prov_code or config.geog_id):
    raise HTTPException(status_code=400, detail="Either prov_code or geog_id must be provided.")
  if config.prov_code and config.geog_id:
    raise HTTPException(status_code=400, detail="Both prov_code and geog_id cannot be provided at the same time.")

  # Check that archiver_file is not an existing directory
  if Path(config.archiver_file).is_dir():
    raise HTTPException(status_code=400, detail=f"The archiver_file {config.archiver_file} is a directory. Please provide a filename.")
  
  # launch the python script via OS command
  # python run_locallogic_content_rewriter.py --config prod_config.yaml

  command = ["python", "run_locallogic_content_rewriter.py", "--config", "prod_config.yaml"]
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()

  if process.returncode != 0:
    # If the subprocess failed, raise an HTTPException with the stderr
    raise HTTPException(status_code=500, detail=stderr.decode())


  return JSONResponse(content={"message": "Job launched"})

@app.get("/report/{version}")
async def generate_report(version: str, config: RewriterConfig = Depends()):
  """
  Generate a report for the given version
  version: the version of the report E.g. 202401
  """
  # print(f'host: {config.es_host}, port: {config.es_port}')
  ll_rewriter = LocallogicContentRewriter(
    es_host=config.es_host, 
    es_port=config.es_port, 
    llm_model=LLM,
    simple_append=True,
    archiver_filepath=get_temp_filename())
  
  ll_rewriter.extract_content(incl_property_override=True)

  report_df = ll_rewriter.generate_reports(version=version)

  report_json = report_df.reset_index().to_json(orient='records')
  report_obj = json.loads(report_json)
  return JSONResponse(content=report_obj)

@app.get("/geo_overrides/{longId}")
async def get_geo_overrides_doc(longId: str, config: RewriterConfig = Depends()):
  """
  Get the document from index rlp_content_geo_overrides_current for a given longId
  """
  ll_rewriter = LocallogicContentRewriter(
    es_host=config.es_host, 
    es_port=config.es_port, 
    llm_model=LLM,
    simple_append=True,
    archiver_filepath=get_temp_filename())
  
  geo_overrides_index_name = 'rlp_content_geo_overrides_current'
  doc = ll_rewriter.es_client.get(index=geo_overrides_index_name, id=longId)

  return JSONResponse(content=doc['_source'])

@app.post("/archive_logs")
async def run_archive_logs(log_dir: str):
  """
  Archive log files in the given directory.
  """
  log_dir = Path(log_dir)

  prov_codes = ["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE", "QC", "SK", "YT"]
  archive_logs(log_dir, prov_codes)

  return {"message": "Logs archiving process started."}