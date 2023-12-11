import sys

import pandas as pd
import argparse, logging
from pathlib import Path
from datetime import datetime
from realestate_content_transformer.data.pipeline import LocallogicContentRewriter
from realestate_spam.llm.chatgpt import LocalLogicGPTRewriter

def test_openai_health():
  ll_gpt_writer = LocalLogicGPTRewriter(llm_model='gpt-3.5-turbo-0613', available_sections=['housing'], property_type=None)
  return ll_gpt_writer.test_openai_health()


def main(es_host, es_port, prov_code=None, geog_id=None, lang='en', archiver_file=None):
  # check openai health
  # if not test_openai_health():
  #   logging.error("OpenAI API is not healthy. Exiting...")
  #   return

  ll_rewriter = LocallogicContentRewriter(
    es_host=es_host, 
    es_port=es_port, 
    # llm_model='gpt-4', 
    llm_model='gpt-4-1106-preview', 
    simple_append=True,
    archiver_filepath=archiver_file)
    # archiver_filepath=Path.home()/'tmp'/'dev_rewrites.txt')

  use_rag = True  # this is better than placeholders 1) stronger context 2) simpler realtime content retrieval

  try:
    ll_rewriter.extract_content(prov_code=prov_code, geog_id=geog_id)
    # ll_rewriter.extract_dataclasses(prov_code=prov_code)

    ll_rewriter.rewrite_cities(prov_code=prov_code, geog_id=geog_id, lang=lang, use_rag=use_rag)
    # ll_rewriter.rewrite_cities_using_dataclasses(simple_append=True, prov_code=prov_code, lang=lang, use_rag=True)

    supported_property_types = ll_rewriter.property_types

    for property_type in supported_property_types:
      ll_rewriter.rewrite_property_types(property_type=property_type, prov_code=prov_code, geog_id=geog_id, lang=lang, use_rag=use_rag)
      # ll_rewriter.rewrite_property_types_using_dataclasses(property_type=property_type, prov_code=prov_code, lang=lang, use_rag=True)

  except Exception as e:
    logging.exception("An error occurred: %s", e)

def rerun_to_recover(es_host, es_port, prov_code, lang, run_num, gpt_backup_version=None):
  # check openai health
  # if not test_openai_health():
  #   logging.error("OpenAI API is not healthy. Exiting...")
  #   return
  
  ll_rewriter = LocallogicContentRewriter(es_host=es_host, es_port=es_port, llm_model='gpt-4', simple_append=True)
  use_rag = True

  try:
    ll_rewriter.rerun_to_recover(run_num=run_num, prov_code=prov_code, lang=lang, use_rag=use_rag, gpt_backup_version=gpt_backup_version)
  except Exception as e:
    logging.exception("An error occurred: %s", e)



if __name__ == '__main__':
  # Example python run_locallogic_content_rewriter.py --prov_code='NT'

  # Initialize argument parser
  parser = argparse.ArgumentParser(description='Run Locallogic Content Rewriter')

  # Add arguments
  parser.add_argument('--es_host', default='localhost', help='Elasticsearch host, default is localhost')
  parser.add_argument('--es_port', type=int, default=9201, help='Elasticsearch port, default is 9201')

  # Creating a mutually exclusive group for prov_code and geog_id
  group = parser.add_mutually_exclusive_group(required=True)

  group.add_argument('--prov_code', type=str, choices=["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE", "QC", "SK", "YT"], 
                    help='Optional province code. Either prov_code or geog_id is required.')
  
  group.add_argument('--geog_id', type=str, 
                    help='Optional geographic ID to process a specific location. If not provided, the script processes the entire province.')


  parser.add_argument('--lang', default='en', choices=['en', 'fr'], help='Language, default is en')
  parser.add_argument('--log_level', default='ERROR', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], help='Logging Level, default is ERROR')

  parser.add_argument('--rerun', action='store_true', help='Flag to launch rerun & recovery')
  parser.add_argument('--run_num', type=int, help='The rerun number, only relevant if --rerun is true')

  parser.add_argument('--archiver_file', default='.', type=str, help='The filepath to the archiver file')
  parser.add_argument('--gpt_backup_version', type=str, default=None, help='The version of records in archiver_file to use for recovery if available')

  args = parser.parse_args()
  es_host = args.es_host
  es_port = args.es_port
  prov_code = args.prov_code
  geog_id = args.geog_id
  lang = args.lang
  log_level = args.log_level

  rerun = args.rerun
  run_num = args.run_num

  archiver_file = args.archiver_file
  gpt_backup_version = args.gpt_backup_version

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
    filtered_df = run_entry_df[(run_entry_df['prov_code'] == prov_code) & (run_entry_df['lang'] == lang)]
    run_number = filtered_df['run_number'].max() + 1 if not filtered_df.empty else 1
    # Add the new run entry to the table
    new_entry = pd.DataFrame({
      'timestamp': [timestamp], 
      'run_number': [run_number], 
      'prov_code': [location_identifier], 
      'lang': [lang]
    })
    run_entry_df = run_entry_df.append(new_entry, ignore_index=True)
    
    # Save the updated table
    run_entry_df.to_csv(csv_file, index=False)

    # Generate the log filename using timestamp, run_number, prov_code, and lang

    log_filename = f'{timestamp}_run_{run_number}_{location_identifier}_{lang}.log'

    logging.basicConfig(
        filename=log_filename,
        level=log_level,
        format='%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s'  # Log format
    )

    main(es_host=es_host, es_port=es_port, prov_code=prov_code, geog_id=geog_id, lang=lang, archiver_file=archiver_file)
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

    rerun_to_recover(es_host=es_host, es_port=es_port, prov_code=prov_code, lang=lang, run_num=run_num, gpt_backup_version=gpt_backup_version)

