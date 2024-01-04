import unittest, logging, os, yaml, difflib, re, tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from realestate_content_transformer.data.pipeline import LocallogicContentRewriter, LLM, LIGHT_WEIGHT_LLM
from realestate_spam.llm.chatgpt import LocalLogicGPTRewriter

from realestate_content_transformer.utils.misc import num_tokens_from_string
from realestate_content_transformer.data.archive import ChatGPTRewriteArchiver, ArchiveStorageType

# logging.basicConfig(level=logging.INFO)  # This will log messages of level INFO and above
logging.basicConfig(
  filename='TestLocallogicContentRewriter.log', 
  filemode='w', 
  level=logging.INFO, 
  # format='%(asctime)s - %(levelname)s - %(message)s'
  format='%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s'
) 
logger = logging.getLogger(__name__)

# enable SSL for local testing and dev
# PROD: ssh -N -L 9201:34.66.125.110:9200 jupyter@34.83.110.8 -i ~/.ssh/id_rsa
# UAT: ssh -N -L 9201:104.198.180.110:9200 jupyter@34.83.110.8 -i ~/.ssh/id_rsa

# How to run:
# export ES_HOST=localhost
# export ES_PORT=9201  
# python -m unittest test_LocallogicContentRewriter.py
# python -m unittest test_LocallogicContentRewriter.TestLocallogicContentRewriter.test_or_setup_extract_all_context

class TestLocallogicContentRewriter(unittest.TestCase):

  def setUp(self):
    # es_host = os.environ.get('ES_HOST')
    # es_port = os.environ.get('ES_PORT')
    try:
      with open('test_config.yaml') as config_file:
        config = yaml.safe_load(config_file)     # TODO: Add sensible defaults later in production
        es_host = config['es_host']
        es_port = config['es_port']
        data_csv_file = config['data_csv_file']

        self.llm_model = config['llm_model']

      self.archiver_filepath = 'unittest_rewrites.txt'
      self.ll_rewriter = LocallogicContentRewriter(
                                                  es_host=es_host, 
                                                  es_port=es_port, 
                                                  llm_model=self.llm_model,
                                                  simple_append=True,
                                                  archiver_filepath=self.archiver_filepath)

      # print(f'property types: {self.ll_rewriter.property_types}')
      self.using_cache_data = False
      if Path(data_csv_file).exists():
        self.ll_rewriter.geo_all_content_df = pd.read_csv(data_csv_file)
        self.ll_rewriter.longId_to_geog_id_dict = self.ll_rewriter.geo_all_content_df.set_index('longId')['geog_id'].to_dict()
        self.using_cache_data = True
    except Exception as e:
      logging.exception("An error occurred: %s", e)

    return 

  def test_or_setup_extract_all_context(self):
    data_csv_file = './data/uat/geo_all_content_df.csv'
    if not Path(data_csv_file).exists():
      self.ll_rewriter.extract_content(incl_property_override=True)
      self.ll_rewriter.geo_all_content_df.to_csv(data_csv_file, index=False)
    else:
      log_message(f'Using cached data from {data_csv_file}. No testing performed', level='info')

    

  def test_extract_content(self):
    if not self.using_cache_data:
      self.ll_rewriter.extract_content(incl_property_override=True)
    self.geo_all_content_df = self.ll_rewriter.geo_all_content_df

    # check columns
    self.assertSetEqual(
      set(self.geo_all_content_df.columns),
      {
       'geog_id', 'longId', 'city', 'citySlug', 'province', 'name',
       'en_housing', 'fr_housing', 'longId_y', 'overrides_en_housing',
       'overrides_en_version', 'overrides_luxury_en_housing',
       'overrides_condo_en_housing', 'overrides_semi_detached_en_housing',
       'overrides_townhouse_en_housing', 'overrides_investment_en_housing',
       'overrides_luxury_en_version', 'overrides_condo_en_version',
       'overrides_semi_detached_en_version', 'overrides_townhouse_en_version',
       'overrides_investment_en_version'
      }
    )

    self.geo_all_content_df.replace(np.nan, None, inplace=True)

    log_message(f'geo_all_content_df.shape: {self.geo_all_content_df.shape}', level='info')

    # verify presence of provinces
    self.assertEqual(set(self.geo_all_content_df.province.unique()), {'AB', 'BC', 'MB', 'NB', 'NL', 'NT', 'NS', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT'})
  
  def test_extact_content_single_geog_id(self):
    geog_id = 'g30_c3nfkdtg'
    try:
      with open('test_config.yaml') as config_file:
        config = yaml.safe_load(config_file)     
        es_host = config['es_host']
        es_port = config['es_port']
        
        self.llm_model = config['llm_model']

      self.archiver_filepath = 'unittest_rewrites.txt'
      ll_rewriter = LocallogicContentRewriter(
                                                es_host=es_host,
                                                es_port=es_port,
                                                llm_model=self.llm_model,
                                                simple_append=True,
                                                archiver_filepath=self.archiver_filepath)

      ll_rewriter.extract_content(geog_id=geog_id, incl_property_override=True)
      # print(ll_rewriter.geo_all_content_df)

      # ensure there's only 1 row with expected geog_id
      self.assertEqual(ll_rewriter.geo_all_content_df.shape[0], 1)
      self.assertEqual(ll_rewriter.geo_all_content_df.geog_id.values[0], geog_id)
    except Exception as e:
      logging.exception("An error occurred: %s", e)
  
  def test_rewrite_cities_by_geog_id(self):
    longId = 'pe_souris'
    geog_id = self.ll_rewriter.longId_to_geog_id_dict[longId]

    try:
      self.ll_rewriter.rewrite_cities(geog_id=geog_id)
      geo_detail_en_doc = self.ll_rewriter.es_client.get(index=self.ll_rewriter.geo_details_en_index_name, id=geog_id)['_source']
      geo_override_doc = self.ll_rewriter.es_client.get(index=self.ll_rewriter.geo_overrides_index_name, id=longId)['_source']

      version = geo_override_doc['overrides_en']['data']['version']
      self.assertEqual(version, self.ll_rewriter.version_string)

      housing = geo_detail_en_doc['data']['profiles']['housing']
      rewritten_housing = geo_override_doc['overrides_en']['data']['profiles']['housing']

      # check if rewritten housing is housing + " The average price of an MLS® real estate listing"
      # TODO: what if there's no listing for the entire city?
      diff_texts = list(difflib.ndiff(housing, rewritten_housing))
      additional_text = ''.join([line[2:] for line in diff_texts if line.startswith("+ ")])
      self.assertTrue('The average price of an MLS® real estate listing in' in additional_text)

    except Exception as e:
      self.fail(f"longId: {longId}, geog_id: {geog_id}, test failed with exception: {e}")


  def test_LocallogicContentRewriter(self):
    # no city level override for housing but at least 1 other sections
    log_message('no city level override for housing but at least 1 other sections', level='info')
    data_query = "overrides_en_housing.isnull() and en_housing.notnull()"
    self._test_LocalLogicGPTRewriter(data_query)

    # with city-level housing override 
    log_message('with city-level housing override', level='info')
    data_query = "overrides_en_housing.notnull()"  
    self._test_LocalLogicGPTRewriter(data_query)

    # missing housing section in original 
    log_message('missing housing section in original', level='info')
    data_query = "en_housing.isnull()"
    self._test_LocalLogicGPTRewriter(data_query)

    # missing all sections in original
    # log_message('missing all sections in original', level='info')
    # data_query = "en_housing.isnull() and en_transport.isnull() and en_services.isnull() and en_character.isnull()"
    # self._test_LocalLogicGPTRewriter(data_query)

    # just some random sample
    log_message('just some random sample', level='info')
    data_query = "longId.notnull()"
    self._test_LocalLogicGPTRewriter(data_query)


  def _test_LocalLogicGPTRewriter(self, data_query: str, verbose=False):
    log_message(f'data_query: {data_query}', level='info')

    sample_df = self.sample(data_query)

    geog_id = sample_df.geog_id.values[0]
    longId = sample_df.longId.values[0]
    city = sample_df.city.values[0]
    city_slug = sample_df.citySlug.values[0]
    prov_code = sample_df.province.values[0]

    housing = sample_df.en_housing.values[0]   # no need for transport, services and character for v1

    log_message(f'geog_id: {geog_id}, longId: {longId}, city: {city}, city_slug: {city_slug}, prov_code: {prov_code}', level='info')
    log_message(f'housing: {housing}', level='info')

    have_params = [True, False]
    property_types = self.ll_rewriter.property_types   # property type of None == 'city level'

    # create a city level writer to handle no params
    non_property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model,
                                                          available_sections=['housing'],
                                                          property_type=None)

    for property_type in property_types:
      property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, 
                                                        available_sections=['housing'], # transport, services and character are not property type specific
                                                        property_type=property_type)

      for use_param in have_params:

        params_dict = {'Average price on MLS®': 1200000}
        params_dict['Percentage of listings'] = 12.5

        if use_param: 
          gpt_writer = property_type_gpt_writer
        else:
          params_dict = None
          gpt_writer = non_property_type_gpt_writer

        gpt_prompt = gpt_writer.print_prompt(housing=housing, params_dict=params_dict)

        msg = f'longId: {longId}, data_query: {data_query}, property_type: {property_type}, use_param: {use_param}, '
        log_message('Setup: ' + msg, level='info')

        # assert that if a section is not null, then its tag is present in the prompt
        if housing: self.assertIn('<housing>', gpt_prompt, msg=msg+gpt_prompt)
        
        # if RAG is used, ensure the word placeholder is absent 
        self.assertNotIn('placeholder', gpt_prompt, msg=msg+gpt_prompt)

        # if there's at least 1 params, ensure these
        if params_dict is not None and len(params_dict) > 0:
          if property_type is not None:
            self.assertIn(f'rewrite text customized for {gpt_writer.property_pluralized} in the city referenced\nwhile utilizing these data points', gpt_prompt, msg=msg+gpt_prompt)
        else: # no param, then write it like city level
          self.assertNotIn('rewrite text customized for ', gpt_prompt, msg=msg+gpt_prompt)
          self.assertNotIn('these data points:', gpt_prompt, msg=msg+gpt_prompt)
          

        log_message('Successful, no assertion is triggered when this is reached.', level='info')
        # Test rewrites once openai key is available
        # rewrites = rewriter.rewrite(housing=housing, params_dict=params_dict)


    log_message("="*80, level='info')

  @unittest.skip("Technically not a test. We are running this to estimate # of tokens for each rewrite.")
  def test_token_count_estimate(self):
    '''
    This is technically not a test. We are running this to estimate the number of tokens for each rewrite.
    '''
    extra_num_tokens_from_metrics = 65
    # data_query = f"longId =='pe_stratford' and en_housing.notnull()"   # sanity check

    non_property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, available_sections=['housing'],
                                                        property_type=None)

    prov_codes, cities, property_types, num_input_tokenss, num_output_tokenss = [], [], [], [], []
    for prov_code in self.ll_rewriter.prov_codes:
      print(f'prov_code: {prov_code}')
      data_query = f"province=='{prov_code}' and en_housing.notnull()"
      sample_df = self.sample(data_query, n_sample=None)   # sample all 

      for property_type in self.ll_rewriter.property_types:
        print(f'property_type: {property_type}')
        property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, available_sections=['housing'], 
                                                          property_type=property_type)
        for i, row in sample_df.iterrows():      
          geog_id = row.geog_id
          longId = row.longId
          city = row.city
          housing = row.en_housing

          # print(f'city: {city}, property_type: {property_type}')
          # print(f'housing: {housing}')

          avg_price, pct = self.ll_rewriter.get_avg_price_and_active_pct(prov_code=prov_code, city=city, property_type=property_type)        

          if avg_price > 0:
            params_dict = {'Average price on MLS®': int(avg_price)}
            params_dict['Percentage of listings'] = pct
            gpt_prompt = property_type_gpt_writer.print_prompt(housing=housing, params_dict=params_dict)
            num_input_tokens = num_tokens_from_string(gpt_prompt)

            # estimate of output tokens = N_{token}(housing) + extra tokens from dynamic info
            num_output_tokens = num_tokens_from_string('<housing>' + housing + '</housing>') + extra_num_tokens_from_metrics
          else:
            gpt_prompt = non_property_type_gpt_writer.print_prompt(housing=housing, params_dict=None)
            num_input_tokens = num_tokens_from_string(gpt_prompt)

            num_output_tokens = num_tokens_from_string('<housing>' + housing + '</housing>')

          prov_codes.append(prov_code)
          cities.append(city)
          property_types.append(property_type)
          num_input_tokenss.append(num_input_tokens)
          num_output_tokenss.append(num_output_tokens)

          # print(f'num_input_tokens: {num_input_tokens}, num_output_tokens: {num_output_tokens}')
          # print(f'gpt prompt: {gpt_prompt}')
          # print(f'='*80)

    token_count_estimate_df = pd.DataFrame(data={
      'prov_code': prov_codes, 
      'city': cities, 
      'property_type': property_types, 
      'num_input_tokens': num_input_tokenss, 
      'num_output_tokens': num_output_tokenss})
      
    token_count_estimate_df.to_csv('token_count_estimate.csv', index=False)


  def test_no_param_rewrite(self):
    # PE's Lot probably hasn't investment 
    longId = 'pe_lot-1'
    query = f"longId=='{longId}'"
    sample_df = self.sample(query)
    geog_id = sample_df.geog_id.values[0]
    prov_code = sample_df.province.values[0]
    city = sample_df.city.values[0]

    property_type = 'INVESTMENT'
    avg_price, pct, count = self.ll_rewriter.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city, property_type=property_type)
    print(f'avg_price: {avg_price}, pct: {pct}')

    self.assertEqual(avg_price, 0.0)
    self.assertTrue(pct < 0.0)
    self.assertEqual(count, 0)
    
    try:
      self.ll_rewriter.rewrite_property_types(property_type=property_type, geog_id=geog_id, mode='mock')   # testing with no GPT
      geo_override_doc = self.ll_rewriter.es_client.get(index=self.ll_rewriter.geo_overrides_index_name, id=longId)['_source']

      version = geo_override_doc['overrides_investment_en']['data']['version']
      self.assertEqual(version, self.ll_rewriter.version_string)

      housing = geo_override_doc['overrides_en']['data']['profiles']['housing']
      rewritten_housing = geo_override_doc['overrides_investment_en']['data']['profiles']['housing']

      # print(f'housing: {housing}')
      # print(f'rewritten_housing: {rewritten_housing}')

      self.assertTrue(f'No {property_type} listing in' in rewritten_housing)
    
    except Exception as e:
      self.fail(f"longId: {longId}, geog_id: {geog_id}, test failed with exception: {e}")

  def test_with_avg_price_pct_rewrite(self):
    longId = 'pe_north-rustico'   # has semi listings
    query = f"longId=='{longId}'"
    sample_df = self.sample(query)
    geog_id = sample_df.geog_id.values[0]
    prov_code = sample_df.province.values[0]
    city = sample_df.city.values[0]

    property_type = 'SEMI-DETACHED'
    avg_price, pct, count = self.ll_rewriter.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city, property_type=property_type)
    print(f'avg_price: {avg_price}, pct: {pct}, count: {count}')

    # confirm precondition before testing
    self.assertGreater(avg_price, 0.0)
    self.assertGreater(pct, 0.0)

    try:
      self.ll_rewriter.rewrite_property_types(property_type=property_type, geog_id=geog_id, mode='mock')   # testing with no GPT
      geo_override_doc = self.ll_rewriter.es_client.get(index=self.ll_rewriter.geo_overrides_index_name, id=longId)['_source']

      version = geo_override_doc['overrides_semi_detached_en']['data']['version']
      self.assertEqual(version, self.ll_rewriter.version_string)

      housing = geo_override_doc['overrides_en']['data']['profiles']['housing']
      rewritten_housing = geo_override_doc['overrides_semi_detached_en']['data']['profiles']['housing']

      # print(f'housing: {housing}')
      # print(f'rewritten_housing: {rewritten_housing}')

      self.assertTrue(f'The average price of an MLS® real estate {property_type} listing in' in rewritten_housing)

    except Exception as e:
      self.fail(f"longId: {longId}, geog_id: {geog_id}, test failed with exception: {e}")


  def test_with_neg_pct_rewrite(self):
    """
    Test the case where the count of listings for the property type < 10 such that the pct is negative.
    The negative value indicate we skip providing % of active listings to the prompt    
    """
    longId = 'pe_linkletter'
    query = f"longId=='{longId}'"
    sample_df = self.sample(query)
    geog_id = sample_df.geog_id.values[0]
    prov_code = sample_df.province.values[0]
    city = sample_df.city.values[0]

    property_type = 'SEMI-DETACHED'
    avg_price, pct, count = self.ll_rewriter.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city, property_type=property_type)
    print(f'avg_price: {avg_price}, pct: {pct}, count: {count}')

    # confirm precondition before testing
    self.assertGreater(avg_price, 0.0)
    self.assertLess(pct, 0.0)

    try:
      self.ll_rewriter.rewrite_property_types(property_type=property_type, geog_id=geog_id, mode='mock')   # testing with no GPT
      geo_override_doc = self.ll_rewriter.es_client.get(index=self.ll_rewriter.geo_overrides_index_name, id=longId)['_source']

      version = geo_override_doc['overrides_semi_detached_en']['data']['version']
      self.assertEqual(version, self.ll_rewriter.version_string)

      housing = geo_override_doc['overrides_en']['data']['profiles']['housing']
      rewritten_housing = geo_override_doc['overrides_semi_detached_en']['data']['profiles']['housing']

      self.assertTrue(f'Too few listings' in rewritten_housing)

    except Exception as e:
      self.fail(f"longId: {longId}, geog_id: {geog_id}, test failed with exception: {e}")


  def test_error_and_recovery(self):
    longId = 'pe_north-rustico'
    geog_id = self.ll_rewriter.longId_to_geog_id_dict[longId]

    # run with rewrite_property_types first, this generates rewrites without the "recovered" marker
    for property_type in self.ll_rewriter.property_types:
      self.ll_rewriter.rewrite_property_types(property_type=property_type, geog_id=geog_id, mode='mock') 

    # we will mock an error by erasing the city attribute from the dataframe for this longId
    # which should trigger an error during call to self.ll_rewriter.rewrite_cities
    # IMPORTANT: at the end, of the text, put back the city attribute so it won't affect other tests
    orig_city = self.ll_rewriter.geo_all_content_df.q("longId==@longId").city.values[0]
    self.ll_rewriter.geo_all_content_df.loc[self.ll_rewriter.geo_all_content_df.longId==longId, 'city'] = None   # erase to trigger error

    try:
      self.ll_rewriter.rewrite_cities(geog_id=geog_id)
    except Exception as e:
      self.fail(f"longId: {longId}, geog_id: {geog_id}, test failed with exception: {e}")
    finally:
      # put back the city attribute
      self.ll_rewriter.geo_all_content_df.loc[self.ll_rewriter.geo_all_content_df.longId==longId, 'city'] = orig_city

    # rerun to recover
    # we will first taint the content of the archive file so we know the recovery is utilizing it

    with open(self.archiver_filepath, 'r') as f:
      file_contents = f.read()
    file_contents = file_contents.replace('[REPEAT housing]', '[REPEAT housing recovered]')
    pattern = r"'version': '\d{6}'"
    replacement = "'version': 'testing'"   
    file_contents = re.sub(pattern, replacement, file_contents)
    pattern = r":\d{6}\|\|"
    replacement = ":testing||"
    file_contents = re.sub(pattern, replacement, file_contents)
    with open(self.archiver_filepath, 'w') as f:
      f.write(file_contents)

    self.ll_rewriter.rerun_to_recover(
      log_filename='TestLocallogicContentRewriter.log', 
      gpt_backup_version='test',     # should also match testing
      mode='mock')

    # check if the recovered rewrites are present
    geo_override_doc = self.ll_rewriter.es_client.get(index=self.ll_rewriter.geo_overrides_index_name, id=longId)['_source']
    for k, v in geo_override_doc.items():
      # check if k if in the form of overrides_{property_type}_en using regex
      if re.match(r'overrides_\w+_en', k):
        # print(k, v)
        # print('------------------')
        self.assertTrue(v['data']['profiles']['housing'].startswith('[REPEAT housing recovered]'))

    # we should clean up by deleting the archive file
    os.remove(self.archiver_filepath)

  
  def test_archiving(self):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
      archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path=temp_file_path)

      # Add some records
      archiver.add_record(longId="test1", property_type="LUXURY", version="202311",
                          user_prompt="prompt1", chatgpt_response="response1")
      archiver.add_record(longId="test2", property_type="CONDO", version="202312",
                          user_prompt="prompt2", chatgpt_response="response2")

      # Retrieve records with exact version match
      record_exact = archiver.get_record(longId="test1", property_type="LUXURY", version="202311")
      self.assertIsNotNone(record_exact)
      self.assertEqual(record_exact['chatgpt_response'], "response1")

      # Retrieve records with version prefix match
      record_prefix = archiver.get_record(longId="test2", property_type="CONDO", version="2023")
      self.assertIsNotNone(record_prefix)
      self.assertEqual(record_prefix['chatgpt_response'], "response2")

      # Test get_all_records
      all_records = archiver.get_all_records(return_df=False)
      self.assertIsInstance(all_records, dict)
      self.assertIn('test1:LUXURY:202311', all_records)
      self.assertIn('test2:CONDO:202312', all_records)
      self.assertEqual(all_records['test1:LUXURY:202311']['chatgpt_response'], 'response1')
      self.assertEqual(all_records['test2:CONDO:202312']['chatgpt_response'], 'response2')

      # Negative test: Try to get a non-existent record
      non_existent_record = archiver.get_record(longId="nonexistent", property_type="UNKNOWN", version="209912")
      self.assertIsNone(non_existent_record)

    finally:
      # clean up 
      os.remove(temp_file_path)


  def sample(self, query: str, n_sample=1) -> pd.DataFrame:
    geo_all_content_df = self.ll_rewriter.geo_all_content_df
    if n_sample is None:   # sample all
      sample_df = geo_all_content_df.query(query)
    else:
      sample_df = geo_all_content_df.query(query).sample(n_sample)
    return sample_df

def log_message(message, level="info"):
    """
    Log the provided message.

    Parameters:
    - message (str): The message to be logged.
    - level (str): The logging level. Can be "info", "warning", "error", "debug".
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    else:
        raise ValueError("Invalid log level specified.")
    
if __name__ == '__main__':
  unittest.main()