import sys, yaml

import pandas as pd
import argparse, logging
# from logging.handlers import TimedRotatingFileHandler

from pathlib import Path
from datetime import datetime
from realestate_content_transformer.data.pipeline import LocallogicContentRewriter
from realestate_spam.llm.chatgpt import LocalLogicGPTRewriter

LIGHT_WEIGHT_LLM = 'gpt-3.5-turbo-0613'
LLM = 'gpt-4-1106-preview'

def load_config(yaml_config_file):
  with open(yaml_config_file, 'r') as f:
    return yaml.safe_load(f)

def test_openai_health():
  ll_gpt_writer = LocalLogicGPTRewriter(llm_model=LIGHT_WEIGHT_LLM, available_sections=['housing'], property_type=None)
  return ll_gpt_writer.test_openai_health()


def main(es_host, es_port, prov_code=None, geog_id=None, lang='en', archiver_file=None, force_rewrite=False):
  # check openai health
  # will immediately exit if not healthy and log the error
  # NOTE: within rewrite_property_types, a consecutive failure count is used to halt if there are too many consecutive failures (due mostly to openai api)
  if not test_openai_health():
    logging.error("OpenAI API is not healthy. Exiting...")
    return

  ll_rewriter = LocallogicContentRewriter(
    es_host=es_host, 
    es_port=es_port, 
    llm_model=LLM,
    simple_append=True,
    archiver_filepath=archiver_file)
    # archiver_filepath=Path.home()/'tmp'/'dev_rewrites.txt')

  use_rag = True  # this is better than placeholders 1) stronger context 2) simpler realtime content retrieval

  try:
    ll_rewriter.extract_content(prov_code=prov_code, geog_id=geog_id, incl_property_override=True)

    ll_rewriter.rewrite_cities(prov_code=prov_code, geog_id=geog_id, lang=lang, use_rag=use_rag)

    for property_type in ll_rewriter.property_types:
      ll_rewriter.rewrite_property_types(property_type=property_type, prov_code=prov_code, geog_id=geog_id, lang=lang, use_rag=use_rag, force_rewrite=force_rewrite)

  except Exception as e:
    logging.exception("An error occurred: %s", e)

def rerun_to_recover(es_host, es_port, prov_code, lang, run_num, gpt_backup_version=None, archiver_file=None):
  # check openai health
  if not test_openai_health():
    logging.error("OpenAI API is not healthy. Exiting...")
    return
  
  ll_rewriter = LocallogicContentRewriter(
    es_host=es_host, 
    es_port=es_port, 
    llm_model=LLM, 
    simple_append=True,
    archiver_filepath=archiver_file)

  use_rag = True

  try:
    ll_rewriter.rerun_to_recover(run_num=run_num, prov_code=prov_code, lang=lang, use_rag=use_rag, gpt_backup_version=gpt_backup_version)
  except Exception as e:
    logging.exception("An error occurred: %s", e)

def setup_logging(log_filename, log_level):
  logging.basicConfig(
        filename=log_filename,
        level=log_level,
        format='%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s'  # Log format
  )
  
  # Note: The TimedRotatingFileHandler is commented out as it may not be needed
  # with the current log filename structure that includes a changing run_number.
  # If you decide to use a static log filename in the future, you can consider
  # uncommenting and using the TimedRotatingFileHandler.

  # # Create a timed rotating file handler
  # timed_handler = TimedRotatingFileHandler(log_filename, when="D", interval=183, backupCount=2)  # Keep 2 half-yearly logs
  # formatter = logging.Formatter('%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s')
  # timed_handler.setFormatter(formatter)

  # # Get the root logger and set the level and handler
  # logger = logging.getLogger()
  # logger.setLevel(log_level)
  # logger.addHandler(timed_handler)


if __name__ == '__main__':
  # Example python run_locallogic_content_rewriter.py --prov_code='NT'

  # Initialize argument parser
  parser = argparse.ArgumentParser(description='Run Locallogic Content Rewriter')

  parser.add_argument('--config', type=str, help='Path to the YAML configuration file')

  parser.add_argument('--es_host', help='Elasticsearch host. Default is "localhost" if not provided.')
  parser.add_argument('--es_port', type=int, help='Elasticsearch port. Default is 9201 if not provided.')

  # Creating a mutually exclusive group for prov_code and geog_id
  group = parser.add_mutually_exclusive_group(required=False)

  group.add_argument('--prov_code', type=str, choices=["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE", "QC", "SK", "YT"], 
                    help='Optional province code. Either prov_code or geog_id is required.')
  
  group.add_argument('--geog_id', type=str, 
                    help='Optional geographic ID to process a specific location. If not provided, the script processes the entire province.')


  parser.add_argument('--lang', choices=['en', 'fr'], help='Language, Default is "en" if not provided.')
  parser.add_argument('--log_level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], help='Logging Level, default is ERROR')

  parser.add_argument('--rerun', action='store_true', help='Flag to launch rerun & recovery')
  parser.add_argument('--run_num', type=int, help='The rerun number, only relevant if --rerun is true')

  parser.add_argument('--archiver_file', type=str, help='The filepath to the archiver file. Default is "./archive_rewrites.txt" if not provided.')
  parser.add_argument('--gpt_backup_version', type=str, help='The version of records in archiver_file to use for recovery if available')

  parser.add_argument('--force_rewrite', type=bool, help='Force rewrite regardless of version')

  args = parser.parse_args()
  # Load YAML config if provided
  config = {}
  if args.config:
      config = load_config(args.config)

  # Use values from config as defaults, overridden by command line arguments
  es_host = args.es_host if args.es_host is not None else config.get('es_host', 'localhost')
  es_port = args.es_port if args.es_port is not None else config.get('es_port', 9201)

  # For prov_code and geog_id, since they are mutually exclusive,
  # check if they are provided in command line args or YAML config

  prov_code = args.prov_code if args.prov_code is not None else config.get('prov_code')
  geog_id = args.geog_id if args.geog_id is not None else config.get('geog_id')

  if not (prov_code or geog_id):
    raise ValueError("Either --prov_code or --geog_id must be provided.")

  # Ensure only one of prov_code or geog_id is provided
  # if prov_code and geog_id:
  #   raise ValueError("Only one of prov_code or geog_id should be provided.")

  lang = args.lang if args.lang is not None else config.get('lang', 'en')
  log_level = args.log_level if args.log_level is not None else config.get('log_level', 'ERROR')

  rerun = args.rerun if args.rerun is not None else config.get('rerun', False)
  run_num = args.run_num if args.run_num is not None else config.get('run_num')

  archiver_file = args.archiver_file if args.archiver_file is not None else config.get('archiver_file', './archive_rewrites.txt')
  gpt_backup_version = args.gpt_backup_version if args.gpt_backup_version is not None else config.get('gpt_backup_version')

  force_rewrite = args.force_rewrite if args.force_rewrite is not None else config.get('force_rewrite', False)

  if geog_id is not None and rerun:
    print("Cannot invoke rerun & recovery for a specific location. Exiting...")
    sys.exit(1)

  if Path(archiver_file).is_dir(): 
    print(f"The archiver_file {archiver_file} is an existing dir. Please provide a filename. Exiting...")
    sys.exit(1)

  location_identifier = geog_id if geog_id else prov_code

  if not rerun: 
    csv_file = 'run_entry_table.csv'
    try:
      run_entry_df = pd.read_csv(csv_file)
    except FileNotFoundError:
      run_entry_df = pd.DataFrame(columns=['timestamp', 'run_number', 'prov_code', 'lang'])
      run_entry_df.to_csv(csv_file, index=False)

    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check for the highest run_number for the given prov_code and lang
    if geog_id is not None:
      filtered_df = run_entry_df[(run_entry_df['prov_code'] == geog_id) & (run_entry_df['lang'] == lang)]
    else:
      filtered_df = run_entry_df[(run_entry_df['prov_code'] == prov_code) & (run_entry_df['lang'] == lang)]

    run_number = filtered_df['run_number'].max() + 1 if not filtered_df.empty else 1
    # Add the new run entry to the table
    new_entry = pd.DataFrame({
      'timestamp': [timestamp], 
      'run_number': [run_number], 
      'prov_code': [location_identifier], 
      'lang': [lang]
    })
    run_entry_df = pd.concat([run_entry_df, new_entry], ignore_index=True)
    
    # Save the updated table
    run_entry_df.to_csv(csv_file, index=False)

    # Generate the log filename using timestamp, run_number, prov_code, and lang
    log_filename = f'{timestamp}_run_{run_number}_{location_identifier}_{lang}.log'

    
    setup_logging(log_filename=log_filename, log_level=log_level)

    main(es_host=es_host, es_port=es_port, prov_code=prov_code, geog_id=geog_id, lang=lang, archiver_file=archiver_file, force_rewrite=force_rewrite)
  else:
    if run_num is None: 
      print("Launching rerun & recovery...")
      print("Please provide the run_num number using --run_num")
      sys.exit(1)
    else:
      assert isinstance(run_num, int), "rerun_num must be an integer"

    csv_file = 'rerun_entry_table.csv'
    try:
      rerun_entry_df = pd.read_csv(csv_file)
    except FileNotFoundError:
      rerun_entry_df = pd.DataFrame(columns=['timestamp', 'rerun_number', 'prov_code', 'lang'])
      rerun_entry_df.to_csv(csv_file, index=False)

    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check for the highest rerun_number for the given prov_code and lang
    filtered_df = rerun_entry_df[(rerun_entry_df['prov_code'] == prov_code) & (rerun_entry_df['lang'] == lang)]
    rerun_number = filtered_df['rerun_number'].max() + 1 if not filtered_df.empty else 1
    # Add the new rerun entry to the table
    new_entry = pd.DataFrame({
      'timestamp': [timestamp], 
      'rerun_number': [rerun_number], 
      'prov_code': [prov_code], 
      'lang': [lang]
    })
    rerun_entry_df = rerun_entry_df.append(new_entry, ignore_index=True)

    # Save the updated table
    rerun_entry_df.to_csv(csv_file, index=False)

    # Generate the log filename using timestamp, rerun_number, prov_code, and lang
    log_filename = f'{timestamp}_rerun_{rerun_number}_{prov_code}_{lang}.log'
    logging.basicConfig(
        filename=log_filename,
        level=log_level,
        format='%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s'  # Log format
    )

    rerun_to_recover(
      es_host=es_host, es_port=es_port, 
      prov_code=prov_code, 
      lang=lang, 
      run_num=run_num, 
      gpt_backup_version=gpt_backup_version, 
      archiver_file=archiver_file
    )

