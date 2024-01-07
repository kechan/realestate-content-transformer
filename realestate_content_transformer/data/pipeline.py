from typing import Union, Tuple, Dict, List

import time, sys, gc, random, copy, traceback
from collections import OrderedDict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as elasticsearch_exceptions
from elasticsearch.helpers import scan, bulk

import logging

from .data_models import Data, Profile, Overrides, GeoDoc, GeoDetailDoc, GeoOverridesDoc, BaseDoc
from .archive import ChatGPTRewriteArchiver, ArchiveStorageType

from realestate_core.common.utils import join_df
from realestate_core.common.class_extensions import *

from realestate_spam.llm.chatgpt import LocalLogicGPTRewriter

# Later enhancement
# 1. bulk upsert (see https://chat.openai.com/share/5038f03e-313f-4cce-a0cc-f52fad739d1e)

LIGHT_WEIGHT_LLM = 'gpt-3.5-turbo-0613'
LLM = 'gpt-4-1106-preview'

class BulkUpserter:
  def __init__(self, es_client: Elasticsearch, index_name: str, longId_to_geog_id_dict: Dict[str, str] = None):
    self.es_client = es_client
    self.index_name = index_name
    self.longId_to_geog_id_dict = longId_to_geog_id_dict

    self.pending_updates = []
    self.pending_longIds = []
    self.pending_count = 0
    self.bulk_size = 100
  
  def _handle_ES_update(self, update_script, longId):
    self.pending_longIds.append(longId)
    action = {
      "_op_type": "update",
      "_index": self.index_name,
      "_id": longId,
      "script": update_script["script"],
      "upsert": update_script["upsert"]
    }
    self.pending_updates.append(action)
    self.pending_count += 1

    if self.pending_count >= self.bulk_size:
      self._perform_bulk_update()

  def _perform_bulk_update(self):
    try:
      success, _ = bulk(self.es_client, self.pending_updates, raise_on_error=True)
      # self.log_info(f"{success} documents have been updated.")
    except Exception as e:
      self.log_error('[bulk_update] ' + str(e))

    # Clear the pending updates stuff
    self.pending_updates.clear()
    self.pending_longIds.clear()
    self.pending_count = 0

  def _flush_pending_updates(self):
    if self.pending_count > 0:
      self._perform_bulk_update()

  def log_error(self, message):
    logger = logging.getLogger(self.__class__.__name__)
    longIds = ','.join(self.pending_longIds) if len(self.pending_longIds) > 0 else None
    geog_ids = ','.join([self.longId_to_geog_id_dict.get(longId) for longId in self.pending_longIds]) if len(self.pending_longIds) > 0 else None
    # extra = {'longIds': longIds} if longIds is not None else {'longIds': 'None'}
    # message = f"[longIds: {longIds if longIds is not None else 'None'}] {message}"

    longId_part = f"[longIds: {longIds}] " if longIds is not None else ''
    geog_id_part = f"[geog_ids: {geog_ids}] " if geog_ids is not None else ''

    message = f"{longId_part}{geog_id_part}" + message
    logger.error(f'{message}')


class LocallogicContentRewriter:
  def __init__(self, es_host, es_port=9200, llm_model=LIGHT_WEIGHT_LLM, simple_append=True, 
               archiver_filepath: str = '.', archiver_host: str = 'localhost', archiver_port: int = 6379):
    '''
    es_host: Elasticsearch host
    es_port: Elasticsearch port
    llm_model: GPT model to use
    simple_append: if True, Use simple append of stat to housing (at city level), otherwise use GPT to rewrite content.
    '''
    self.es_host = es_host    
    self.es_client = Elasticsearch([f'http://{es_host}:{es_port}/'])
    if not self.es_client.ping():
      raise Exception(f'Cannot connect to Elasticsearch at {es_host}:{es_port}')
    
    self.llm_model = llm_model

    self.simple_append = simple_append

    # all relevant ES indices
    self.geo_index_name = 'rlp_content_geo_current'    #'rlp_content_geo_3'
    self.geo_details_en_index_name = 'rlp_geo_details_en'
    self.geo_details_fr_index_name = 'rlp_geo_details_fr'
    self.geo_overrides_index_name = 'rlp_content_geo_overrides_current'
    self.listing_index_name = 'rlp_listing_current'

    # all possible provinces
    self.prov_codes = ["AB", "BC", "MB", "NB", "NL", "NT", "NS", "NU", "ON", "PE", "QC", "SK", "YT"]

    # all property types
    self.property_types = ['LUXURY', 'CONDO', 'SEMI-DETACHED', 'TOWNHOUSE', 'INVESTMENT']

    # to store all content from geo, geo_details_en, geo_details_fr, and geo_overrides
    self.geo_all_content_df = pd.DataFrame()

    # to store all content as dataclasses: GeoDoc, GeoDetailDoc (2 langs), GeoOverridesDoc
    self.geo_docs, self.geo_detail_en_docs, self.geo_detail_fr_docs, self.geo_override_docs = [], [], [], []
    self.geo_detail_en_dict, self.geo_detail_fr_dict, self.geo_override_dict = {}, {}, {}

    # a convenience dictionary to map longId to geog_id
    self.longId_to_geog_id_dict = {}   # Dict[str, str] for {longId: geog_id} mapping

    # Designed for tracking consecutive openAI GPT completion failures
    self.gpt_rewrite_consecutive_failures = 0
    self.max_gpt_rewrite_consecutive_failures = 5

    # Setup archiver
    # self.archiver = ChatGPTRewriteArchiver(storage_type=ArchiveStorageType.REDIS, redis_host=archiver_host, redis_port=archiver_port)
    self.archiver = ChatGPTRewriteArchiver(storage_type=ArchiveStorageType.PLAIN_TEXT, file_path=archiver_filepath)
    try: 
      self.archiver.ping()
    except Exception as e:
      self.log_error(f'[LocallogicContentRewriter.init] Cannot connect to archiver at {archiver_host}:{archiver_port}. Error: {e}')
      self.archiver = None

    # misc setup
    self.version_string = self.get_current_version_string()
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] [Logger: %(name)s]: %(message)s')
    self.wait_sec = 0.01

    
  def extract_content(self, prov_code=None, geog_id=None, incl_property_override=False, save_intermediate_results=False, save_dir='.'):
    """
    Extract docs from ES from 4 indices (geo, geo_details_en, geo_details_fr, geo_overrides), 
    represent them as dataframes, and join them together, using geo as the driving "table",
    and populate self.geo_all_content_df.
    """

    # OVERRIDE_COLS = [
    #     'overrides_en_housing', 'overrides_en_transport', 'overrides_en_services', 'overrides_en_character',
    #     'overrides_fr_housing', 'overrides_fr_transport', 'overrides_fr_services', 'overrides_fr_character'
    # ]

    # we don't ever override 'transport', 'services', and 'character' for now.
    OVERRIDE_COLS = [
        'overrides_en_housing',
        # 'overrides_fr_housing',   # fr not needed in v1
    ]
    if incl_property_override:
      OVERRIDE_COLS.extend([f"overrides_{property_type.lower().replace('-','_')}_en_housing" for property_type in self.property_types])  # only for en for v1
    
    def process_overrides(geo_overrides_df, lang):
      # TODO: this is for non property type override, need to implement similar for property type override later.
      override_key = f'overrides_{lang}'
      if override_key in geo_overrides_df.columns:
        geo_overrides_df[override_key] = geo_overrides_df[override_key].replace(np.nan, None)

        data_df = geo_overrides_df[override_key].apply(pd.Series, dtype='object').data.apply(pd.Series, dtype='object')

        profiles_df = data_df.profiles.apply(pd.Series, dtype='object')
        profiles_df.rename(columns={0: 'zero'}, inplace=True)
        profiles_df.rename(columns={c: f'{override_key}_{c}' for c in profiles_df.columns}, inplace=True)

        if 'version' in data_df.columns:
          version_df = data_df.version
          version_df.name = f'{override_key}_version'
          geo_overrides_df = pd.concat([geo_overrides_df, profiles_df, version_df], axis=1)
        else:
          geo_overrides_df = pd.concat([geo_overrides_df, profiles_df], axis=1)
          geo_overrides_df[f'{override_key}_version'] = None
      else:
        geo_overrides_df[override_key] = None
        geo_overrides_df[f'{override_key}_version'] = None

      if incl_property_override:
        
        for property_type in self.property_types:
          property_type = property_type.lower().replace('-','_')
          override_key = f'overrides_{property_type}_{lang}'
          if override_key in geo_overrides_df.columns:
            geo_overrides_df[override_key] = geo_overrides_df[override_key].replace(np.nan, None)

            data_df = geo_overrides_df[override_key].apply(pd.Series, dtype='object').data.apply(pd.Series, dtype='object')

            profiles_df = data_df.profiles.apply(pd.Series, dtype='object')
            profiles_df.rename(columns={0: 'zero'}, inplace=True)
            profiles_df.rename(columns={c: f'{override_key}_{c}' for c in profiles_df.columns}, inplace=True)

            if 'version' in data_df.columns:
              version_df = data_df.version
              version_df.name = f'{override_key}_version'
              geo_overrides_df = pd.concat([geo_overrides_df, profiles_df, version_df], axis=1)
            else:
              geo_overrides_df = pd.concat([geo_overrides_df, profiles_df], axis=1)
              geo_overrides_df[f'{override_key}_version'] = None
          else:
            geo_overrides_df[override_key] = None
            geo_overrides_df[f'{override_key}_version'] = None

  

      return geo_overrides_df
    
    def process_geo_details(prov_code=None, geog_id=None, lang='en'):
      if geog_id is not None:
        geo_details_df = self._get_geo_details(geog_id=geog_id, lang=lang)[['name', 'profiles']]
      else:
        geo_details_df = self._get_geo_details(prov_code=prov_code, lang=lang)[['name', 'profiles']]

      geo_details_df.reset_index(inplace=True)
      profiles_df = geo_details_df.profiles.apply(pd.Series, dtype='object')
      geo_details_df = pd.concat([geo_details_df, profiles_df], axis=1)

      # drop profiles column
      geo_details_df.drop(columns=['profiles'], inplace=True)

      # drop transport, services, and character columns (not needed in v1)
      geo_details_df.drop(columns=['transport', 'services', 'character'], errors='ignore', inplace=True)

      if 'housing' not in geo_details_df.columns:
        geo_details_df['housing'] = None

      # rename prefix housing, transport, services, and character columns with language-specific prefix
      column_prefix = f"{lang}_"
      geo_details_df.rename(columns={
        'housing': f'{column_prefix}housing', 
        # 'transport': f'{column_prefix}transport', 
        # 'services': f'{column_prefix}services', 
        # 'character': f'{column_prefix}character'
        }, 
        inplace=True)
      return geo_details_df

    def prepare_geo_df(prov_code=None, geog_id=None):
      if geog_id is not None:
        geo_df = self._get_geo(geog_id=geog_id)
      else:
        geo_df = self._get_geo(prov_code=prov_code)

      geo_df.reset_index(inplace=True)

      if save_intermediate_results: 
        geo_df.to_feather(Path(save_dir)/f'{prov_code.lower()}_geo_df')
      if geo_df.empty:
        return None
      else:        
        return geo_df[['geog_id', 'longId', 'city', 'citySlug', 'province']]
    
    def prepare_geo_overrides_df(prov_code=None, geog_id=None):
      # extract geo overrides
      if geog_id is not None:
        geo_overrides_df = self._get_geo_overrides(localLogicId=geog_id, incl_property_override=incl_property_override)
      else:
        geo_overrides_df = self._get_geo_overrides(prov_code=prov_code, incl_property_override=incl_property_override)

      if not geo_overrides_df.empty:
        geo_overrides_df.reset_index(inplace=True)
        geo_overrides_df.rename(columns={'overrides:':'overrides_colon'}, inplace=True)
        if 'overrides' in geo_overrides_df.columns:
          geo_overrides_df.overrides = geo_overrides_df.overrides.replace(np.nan, None)

        # for lang in ['en', 'fr']:
        for lang in ['en']:     # only en needed for v1
          geo_overrides_df = process_overrides(geo_overrides_df, lang)

        # Replace np.NaN with None, and if a columnn is missing, create it with None.
        for c in OVERRIDE_COLS:
          if c not in geo_overrides_df.columns:
            geo_overrides_df[c] = None
          geo_overrides_df[c].replace(np.nan, None, inplace=True)

        if save_intermediate_results: 
          geo_overrides_df.to_feather(Path(save_dir)/f'{prov_code.lower()}_geo_overrides_df')

      return geo_overrides_df

    def prepare_geo_details_df(prov_code=None, geog_id=None):
      geo_details_dfs = {}
      if geog_id is not None:
        for lang in ['en', 'fr']:
          geo_details_dfs[lang] = process_geo_details(geog_id=geog_id, lang=lang)
      else:  
        for lang in ['en', 'fr']:
          geo_details_dfs[lang] = process_geo_details(prov_code=prov_code, lang=lang)

      # join geo_details_en_df and geo_details_fr_df
      # geo_details_df = join_df(geo_details_dfs['en'], geo_details_dfs['fr'][['geog_id', 'fr_housing', 'fr_transport', 'fr_services', 'fr_character']], left_on='geog_id', how='left')
      geo_details_df = join_df(geo_details_dfs['en'], geo_details_dfs['fr'][['geog_id', 'fr_housing']], left_on='geog_id', how='left')
      if save_intermediate_results:
        geo_details_df.to_feather(Path(save_dir)/f'{prov_code.lower()}_geo_details_df')

      return geo_details_df
  
    if geog_id is not None:    # do this just for 1 geog_id and ignore prov_code
      geo_all_content_dfs = []

      geo_df = prepare_geo_df(geog_id=geog_id)
      geo_details_df = prepare_geo_details_df(geog_id=geog_id)
      geo_overrides_df = prepare_geo_overrides_df(geog_id=geog_id)
      # inner join geo_df and geo_details_df (since only level 30 is needed)
      geo_all_content_df = join_df(geo_df, geo_details_df, left_on='geog_id', how='inner')
      # Further join with geo_overrides_df
      if not geo_overrides_df.empty:        
        geo_all_content_df = join_df(geo_all_content_df, geo_overrides_df[['geog_id', 'longId'] + OVERRIDE_COLS + [col for col in geo_overrides_df if 'version' in col]], 
          left_on='geog_id', how='left')
      else:
        for c in OVERRIDE_COLS: geo_all_content_df[c] = None
        geo_all_content_df['overrides_en_version'] = None

      geo_all_content_dfs.append(geo_all_content_df)
      gc.collect();

    else:  
      if prov_code is not None: 
        prov_codes = [prov_code]
      else:
        prov_codes = self.prov_codes

      geo_all_content_dfs = []
      for prov_code in prov_codes:
        print(f'Processing province: {prov_code}')

        geo_df = prepare_geo_df(prov_code=prov_code)
        if geo_df is None: continue

        geo_overrides_df = prepare_geo_overrides_df(prov_code)
        geo_details_df = prepare_geo_details_df(prov_code)      

        # inner join geo_df and geo_details_df (since only level 30 is needed)
        geo_all_content_df = join_df(geo_df, geo_details_df, left_on='geog_id', how='inner')

        # Further join with geo_overrides_df
        if not geo_overrides_df.empty:
          geo_all_content_df = join_df(geo_all_content_df, geo_overrides_df[['geog_id', 'longId'] + OVERRIDE_COLS + [col for col in geo_overrides_df if 'version' in col]], 
            left_on='geog_id', how='left')
        else:
          for c in OVERRIDE_COLS: geo_all_content_df[c] = None
          geo_all_content_df['overrides_en_version'] = None

        # replace np.nan with None
        geo_all_content_df.replace({np.nan: None}, inplace=True)

        geo_all_content_dfs.append(geo_all_content_df)

        gc.collect();

    geo_all_content_df = pd.concat(geo_all_content_dfs, axis=0)

    self.geo_all_content_df = pd.concat([self.geo_all_content_df, geo_all_content_df], axis=0, ignore_index=True)

    # Important: keep latest
    self.geo_all_content_df.drop_duplicates(subset=['geog_id'], keep='last', inplace=True, ignore_index=True)

    # (re)build longId -> geog_id dictionary
    self.longId_to_geog_id_dict = self.geo_all_content_df.set_index('longId')['geog_id'].to_dict()

    # geo_all_content_df relevant columns:
      # geog_id, longId,
      # en_housing, en_transport, en_services, en_character, fr_housing, fr_transport, fr_services, fr_character, 
      # overrides_en_housing, overrides_en_transport, overrides_en_services, overrides_en_character,
      # overrides_fr_housing, overrides_fr_transport, overrides_fr_services, overrides_fr_character


  def extract_dataclasses(self, prov_code=None, geog_id=None):
    """
    Extract docs from ES from 4 indices (geo, geo_details_en, geo_details_fr, geo_overrides) for prov_code (if provided) or all provinces
    if prov_code is None and geog_id is None. Only extract 1 if geo_id is provided.
    """

    if geog_id is not None:
      self.geo_docs.extend(self.get_geo(geog_id=geog_id))
      self.geo_detail_en_docs.extend(self.get_geo_details(geog_id=geog_id, lang='en'))
      self.geo_detail_fr_docs.extend(self.get_geo_details(geog_id=geog_id, lang='fr'))
      self.geo_override_docs.extend(self.get_geo_overrides(geog_id=geog_id))

      # Create dictionaries for quick look-up
      self.geo_detail_en_dict.update({doc.geog_id: doc for doc in self.geo_detail_en_docs})
      self.geo_detail_fr_dict.update({doc.geog_id: doc for doc in self.geo_detail_fr_docs})
      self.geo_override_dict.update({doc.longId: doc for doc in self.geo_override_docs})

    else:
      prov_codes = [prov_code] if prov_code is not None else self.prov_codes

      for prov_code in prov_codes:

        self.geo_docs.extend(self.get_geo(prov_code=prov_code))
        self.geo_detail_en_docs.extend(self.get_geo_details(prov_code=prov_code, lang='en'))
        self.geo_detail_fr_docs.extend(self.get_geo_details(prov_code=prov_code, lang='fr'))
        self.geo_override_docs.extend(self.get_geo_overrides(prov_code=prov_code))

        # Create dictionaries for quick look-up
        geo_detail_en_dict = {doc.geog_id: doc for doc in self.geo_detail_en_docs}
        self.geo_detail_en_dict.update(geo_detail_en_dict)

        geo_detail_fr_dict = {doc.geog_id: doc for doc in self.geo_detail_fr_docs}
        self.geo_detail_fr_dict.update(geo_detail_fr_dict)

        geo_override_dict = {doc.longId: doc for doc in self.geo_override_docs}
        self.geo_override_dict.update(geo_override_dict)

    # dedup
    self.geo_docs = self._dedup_dataclasses(self.geo_docs, pkey='geog_id')
    self.geo_detail_en_docs = self._dedup_dataclasses(self.geo_detail_en_docs, pkey='geog_id')
    self.geo_detail_fr_docs = self._dedup_dataclasses(self.geo_detail_fr_docs, pkey='geog_id')
    self.geo_override_docs = self._dedup_dataclasses(self.geo_override_docs, pkey='longId')
    # rebuild dictionaries
    self.geo_detail_en_dict = {doc.geog_id: doc for doc in self.geo_detail_en_docs}
    self.geo_detail_fr_dict = {doc.geog_id: doc for doc in self.geo_detail_fr_docs}
    self.geo_override_dict = {doc.longId: doc for doc in self.geo_override_docs}

    # (re)build longId -> geog_id dictionary
    self.longId_to_geog_id_dict = {doc.longId: doc.geog_id for doc in self.geo_docs}


  def _dedup_dataclasses(self, doc_list: List[BaseDoc], pkey: str) -> List[BaseDoc]:
    '''
    dedup the list of BaseDoc with the pkey and keep the last occurence 
    '''
    deduplicated_doc = OrderedDict()

    # Iterate in reverse to keep the last occurrences
    for doc in reversed(doc_list):
        key = getattr(doc, pkey)
        # Set default if the geog_id key is not already present
        deduplicated_doc.setdefault(key, doc)

    # Reverse again to maintain the original order
    dedup_doc_list = list(reversed(deduplicated_doc.values()))

    return dedup_doc_list


  def rewrite_cities(self, prov_code=None, geog_id=None, lang='en', use_rag=True):
    """
    loop through all cities (level 30) (or by prov_code if provided) and rewrite content for each city, or
    just one geog_id if provided.

    Extra stat will be used to enhance/rewrite content. Right now, only housing will have additional info 
    (will use RAG or placeholders during GPT prompting if necessary). The input are original 
    LocalLogic content (in geo_details)

    use_rag: if True, use RAG to rewrite content, otherwise use "placeholder" style. This is relevant only if simple_append is False.
    """

    gpt_writer = None
    if not self.simple_append:   # need to enlist GPT help.
      gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, 
                                         available_sections=['housing'],    #'transport', 'services', 'character'],  (not needed for v1)
                                         property_type=None)

    if geog_id is not None:
      geo_all_content_df = self.geo_all_content_df.q("geog_id == @geog_id")
    else:  
      geo_all_content_df = self.geo_all_content_df.q("province == @prov_code") if prov_code is not None else self.geo_all_content_df

    rewrite_count = 0
    for i, row in geo_all_content_df.iterrows():
      geog_id, longId = None, None
      try:
        geog_id = row.geog_id    # TODO: although this is not expected, any missing attribute will cause unhandled exception.
        longId = row.longId
        city = row.city
        city_slug = row.citySlug
        province = row.province

        housing = row.get(f'{lang}_housing', None)
        # don't need this in v1
        # transport = row.get(f'{lang}_transport', None)
        # services = row.get(f'{lang}_services', None)
        # character = row.get(f'{lang}_character', None)

        section_contents = {
          'housing': housing, 
          # 'transport': transport, 
          # 'services': services, 
          # 'character': character
        }

        succeeded = self.rewrite_city(geog_id=geog_id, longId=longId, city=city, city_slug=city_slug, prov_code=province,
                        lang=lang, 
                        gpt_writer=gpt_writer, use_rag=use_rag, 
                        **section_contents)
        
        rewrite_count += (1 if succeeded else 0)
        # Avoid overwhelming openai api, wait a little before moving on to the next geog_id
        time.sleep(self.wait_sec)
      except (AttributeError, KeyError) as e:
        self.log_error(str(e) + f'|| From rewrite_cities(...) geo_all_content_df with prov_code {prov_code}', longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)    # TODO: consider remove this, it should have been triggered without hitting openai chat completion api.
      except Exception as e:
        self.log_error(str(e), longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)

      # if i > 2: break  # TODO: remove this break

    self.log_info(f'[rewrite_cities] Finished rewriting {rewrite_count} neighbourhoods/cities successfully.')


  def rewrite_cities_using_dataclasses(self, prov_code=None, lang='en', use_rag=True):
    """
    same as rewrite_cities(...) but using dataclasses instead of pandas dataframe.
    """
    gpt_writer = None
    if not self.simple_append:   # need to enlist GPT help.
      gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, 
                                         available_sections=['housing', 'transport', 'services', 'character'], 
                                         property_type=None)
      
    # Perform "joins", geo is considered the "driving table"
    for geo_doc in self.geo_docs:
      try:
        geog_id = geo_doc.geog_id
        longId = geo_doc.longId
        city = geo_doc.city
        city_slug = geo_doc.citySlug

        geo_detail_doc = None
        if lang == 'en':
          geo_detail_doc = self.geo_detail_en_dict.get(geog_id)
        elif lang == 'fr':
          geo_detail_doc = self.geo_detail_fr_dict.get(geog_id)

        if geo_detail_doc:
          province = geo_detail_doc.data.province
          if prov_code is not None and province != prov_code: continue

          # city = geo_detail_doc.data.name

          housing = geo_detail_doc.data.profiles.housing if geo_detail_doc.data.profiles else None
          transport = geo_detail_doc.data.profiles.transport if geo_detail_doc.data.profiles else None
          services = geo_detail_doc.data.profiles.services if geo_detail_doc.data.profiles else None
          character = geo_detail_doc.data.profiles.character if geo_detail_doc.data.profiles else None

          section_contents = {'housing': housing, 'transport': transport, 'services': services, 'character': character}

          print(f'geog_id: {geog_id}, longId: {longId}, city: {city}, city_slug: {city_slug}, prov_code: {province}')
          print(f'housing: {housing}')
          print(f'transport: {transport}')
          print(f'services: {services}')
          print(f'character: {character}')

          # self.rewrite_city(geog_id=geog_id, longId=longId, city=city, city_slug=city_slug, prov_code=province,
          #               lang=lang, 
          #               gpt_writer=gpt_writer, use_rag=use_rag, 
          #               **section_contents)        
          # Avoid overwhelming openai api, wait a little before moving on to the next geog_id
          time.sleep(self.wait_sec)
        else:
          self.log_error(f'No geo_detail_{lang}_doc found', longId=longId, geog_id=geog_id)
          time.sleep(self.wait_sec)
          continue
      except (AttributeError, KeyError) as e:
        self.log_error(str(e) + f'|| From rewrite_cities_using_dataclasses(...) geo_doc with prov_code {prov_code}', longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)
      except Exception as e:
        self.log_error(str(e), longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)


  def rewrite_city(self, geog_id, longId, city, city_slug, prov_code, lang='en', gpt_writer=None, use_rag=True, **section_contents) -> bool:
    """
    Given a geog_id, longId, city, and section_contents, rewrite the content and update the geo_overrides_index_name index keyed by longId.

    prov_code: province code, e.g. 'AB' is passed in just for the case if upsert is needed. 

    simple_append: if True, simply append to housing, otherwise use GPT to rewrite content.
    use_rag: if True, use RAG to rewrite content, otherwise use "placeholder" style. This is relevant only if simple_append is False.
    section_contents: Original locallogic content. Dict of housing, transport, services, character. Assumption: if simple_append is True, 
                      only housing is really needed.

    Return: a bool to indicate if rewrite is successful or not.                      
    """

    rewrites = {}

    if self.simple_append:   
      # only Housing for v1
      housing = section_contents['housing']   # original LocalLogic content
      if housing is None: return       # no housing in original content, skip and do nothing

      if use_rag:
        avg_price, _, _ = self.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city)
        if avg_price > 1.0:
          overriden_housing = housing + f" The average price of an MLS® real estate listing in {city} is $ {avg_price:,.0f}."   #TODO: double check before real deployment run
        else:
          overriden_housing = housing    # no override, this can happen if there's no listing for the city, province.
      else:
        overriden_housing = housing + f" The average price of an MLS® real estate listing in {city} is $ [avg_price]"

      rewrites['error_message'] = None    # simple append shold never have problem.
      rewrites['housing'] = overriden_housing

    else:   # requiring GPT rewrite
      if gpt_writer is None:   # if not provided, create one on the fly.
        gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, 
                                           available_sections=['housing'],            #'transport', 'services', 'character'], not needed for v1
                                           property_type=None)  # for city level

      # housing, transport, services, character = section_contents['housing'], section_contents['transport'], section_contents['services'], section_contents['character']
      housing = section_contents['housing']

      if use_rag:
        # additional metrics to inject into prompt (using RAG)          
        avg_price, _, _ = self.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city)
        params_dict = {'Average price on MLS®': avg_price}        
      else: # use placeholder, instruct to use placeholders while adding new information
        avg_price_explanation = self.get_avg_price_explanation()
        params_dict = {'[avg_price]': avg_price_explanation}

      rewrites = gpt_writer.rewrite(params_dict=params_dict, use_rag=use_rag, 
                                    housing=housing
                                    # transport=transport, services=services, character=character
                                    )
      if self.archiver:
        messages = gpt_writer.construct_openai_user_prompt(params_dict=params_dict, use_rag=use_rag, 
                                    housing=housing, 
                                    # transport=transport, services=services, character=character
                                    )
        user_prompt = messages[0]['content']
        chatgpt_response = self._extract_format_for_archive(rewrites)
        self.archiver.add_record(longId=longId, property_type='None', version=self.version_string, user_prompt=user_prompt, chatgpt_response=chatgpt_response)

    if rewrites['error_message'] is None:
      rewrites.pop('error_message', None)   # remove error_message from rewrites dict
      
      # TODO: figure out all the fields for upsert.
      # rewrites['housing'] += f' {random.randint(1, 1000)}'   # TODO: remove this, just for testing
      update_script = {
        "script": {
            "source": f"""
                if (ctx._source.containsKey('overrides_{lang}')) {{
                    ctx._source['overrides_{lang}'].data.version = params.version;
                    ctx._source['overrides_{lang}'].data.profiles = params.profiles;
                    
                }} else {{
                  /*
                    Map profiles = new HashMap();
                    profiles.put('housing', params.profiles.housing);

                    Map data = new HashMap();
                    data.put('version', params.version);
                    data.put('profiles', profiles);

                    ctx._source['overrides_{lang}'] = new HashMap();
                    ctx._source['overrides_{lang}'].data = data;
                  */

                    def profiles = params.profiles;
                    def data = ['version': params.version, 'profiles': profiles];
                    ctx._source['overrides_{lang}'] = ['data': data];
                }}
            """,
            "params": {
                "version": self.version_string,
                "profiles": rewrites
            }
        },
        "upsert": {
            "city": city,
            "citySlug": city_slug,
            'localLogicId': [geog_id],
            'longId': longId,
            'neighbourhood': None,         # TODO: need to figure out how to get neighbourhood(Slug) if needed.
            'neighbourhoodSlug': None,
            'overrides:': {},
            "province": prov_code,                
            f"overrides_{lang}": {
                "data": {
                  "version": self.version_string,
                  "profiles": rewrites                        
                }
            }
        }
      }
    
      es_op_succeeded = self._handle_ES_update(update_script, longId)
    else:
      self.log_error('[GPT] ' + rewrites['error_message'], longId)

    return rewrites.get('error_message') is None and es_op_succeeded    # True if everything is ok


  def rewrite_property_types(
    self, 
    property_type='CONDO', prov_code=None, geog_id=None, 
    lang='en', use_rag=True, force_rewrite=False,
    mode='prod'):
    """
    Rewrites content for all cities (level 30). If prov_code or just one geog_id is provided, 
    then rewrite content for those only, and a specific property type. 
    
    (Not in v1: since transport, services, and character are property type agnostic, they don't need to be rewritten.) 

    Parameters:
        property_type (str): The type of property for which to rewrite content. 
                             Default is 'CONDO'.
        prov_code (str): The code of the province to filter rewrites. If None, 
                         rewrites are applied across all provinces.
        geog_id (str): The geographic ID to filter rewrites. If None, rewrites 
                       are applied based on prov_code or across all cities.
        lang (str): The language in which to perform rewrites. Default is 'en' (English).
        use_rag (bool): Flag to determine if Retriever-Augmented Generation (RAG) 
                        should be used in the rewriting process.
        force_rewrite (bool): If True, bypasses version checking and forces rewriting 
                              of content regardless of its current version. 
                              If False, only rewrites content with versions 
                              different from the current target version.
        mode (str): The mode of operation. Can be 'prod' for production or other 
                    values for different operational modes (e.g. 'mock' for testing).

    The method utilizes a LocalLogicGPTRewriter for generating rewritten content 
    and applies version control to manage content updates. The version checking 
    mechanism ensures that only outdated content is rewritten, enhancing efficiency. 
    The force_rewrite parameter provides flexibility for scenarios where selective 
    or complete rewrites are necessary, such as during recovery processes or 
    in cases where version integrity needs to be re-established.                    
    """

    property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model,
                                        available_sections=['housing'],    # transport, services and character are not property type specific
                                        property_type=property_type)
    non_property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model,
                                        available_sections=['housing'],
                                        property_type=None)      
    
    if geog_id is not None:
      geo_all_content_df = self.geo_all_content_df.q("geog_id == @geog_id")
    else:
      geo_all_content_df = self.geo_all_content_df.q("province == @prov_code") if prov_code is not None else self.geo_all_content_df
    
    rewrite_count = 0
    for i, row in geo_all_content_df.iterrows():
      if (i + 1) % 200 == 0:   #track progress
        self.log_info(f'Processing {i+1}th geog_id for property type {property_type}')

      if self.gpt_rewrite_consecutive_failures > self.max_gpt_rewrite_consecutive_failures:
        self.log_error(f'[rewrite_property_types] Exceeded max consecutive failures of {self.max_gpt_rewrite_consecutive_failures} for property type {property_type}. Exiting.')
        break

      geog_id, longId = None, None
      try:
        geog_id = row.geog_id
        longId = row.longId
        city = row.city
        province = row.province

        if not force_rewrite:
          # Construct the version column name based on the property type
          version_col = f'overrides_{property_type.lower().replace("-", "_")}_{lang}_version'
          doc_version = row.get(version_col)

          if doc_version == self.version_string:
            # this is such that if this method is rerun, only docs that are not at target_version will be rewritten,
            # which can happen due to failures/errors.
            self.log_info(f'[rewrite_property_types] Skipping rewrite for geog_id {geog_id} as it is already at version {self.version_string}.')
            continue

        housing = row.get(f'{lang}_housing', None)

        section_contents = {'housing': housing}

        succeeded = self.rewrite_property_type(geog_id=geog_id, longId=longId, city=city, prov_code=province,
                                              property_type=property_type,
                                              lang=lang, 
                                              non_property_type_gpt_writer=non_property_type_gpt_writer, 
                                              property_type_gpt_writer=property_type_gpt_writer,
                                              use_rag=use_rag,
                                              mode=mode,
                                              **section_contents)
        
        rewrite_count += (1 if succeeded else 0)
        # Avoid overwhelming openai api, wait a little before moving on to the next geog_id
        time.sleep(self.wait_sec)
      except (AttributeError, KeyError) as e:
        tb = traceback.format_exc()  # Get the full traceback
        self.log_error('[rewrite_property_types] ' + str(e) + f'|| From rewrite_property_types(...) with prov_code {prov_code}\n{tb}', longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)   # TODO consider remove this, it should have been triggered without hitting openai chat completion api.
      except Exception as e:
        tb = traceback.format_exc()
        self.log_error('[rewrite_property_types] ' + str(e) + '\n' + tb, longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)

      # if i > 2: break # TODO: remove this break
    self.log_info(f'[rewrite_property_types] Finished rewriting {rewrite_count} neighbourhoods/cities successfully for property type {property_type}.')


  def rewrite_property_types_using_dataclasses(self, property_type='CONDO', prov_code=None, lang='en', use_rag=True):
    """
    same as rewrite_property_types(...) but using dataclasses instead of pandas dataframe.
    """
    gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model, 
                                       available_sections=['housing'],
                                       property_type=property_type)
    
    # Perform "joins", geo is considered the "driving table"
    for geo_doc in self.geo_docs:
      try:
        geog_id = geo_doc.geog_id
        longId = geo_doc.longId
        city = geo_doc.city

        geo_detail_doc = self.geo_detail_en_dict.get(geog_id) if lang == 'en' else self.geo_detail_fr_dict.get(geog_id)

        if geo_detail_doc:
          province = geo_detail_doc.data.province
          if prov_code is not None and province != prov_code: continue  # filter only for prov_code if provided
          # city = geo_detail_doc.data.name

          geo_override_doc = self.geo_override_dict.get(geo_doc.longId)
          if geo_override_doc:
            if lang == 'en':
              if geo_override_doc.overrides_en and geo_override_doc.overrides_en.data:
                # get housing from overrides_en_housing if it exists, otherwise use en_housing
                # housing = geo_override_doc.overrides_en.data.profiles.housing if geo_override_doc.overrides_en.data.profiles else None
                if geo_override_doc.overrides_en.data.profiles.housing is not None:
                  housing = geo_override_doc.overrides_en.data.profiles.housing
                elif geo_detail_doc.data.profiles.housing is not None:
                  housing = geo_detail_doc.data.profiles.housing
                else:
                  housing = None

            elif lang == 'fr':
              if geo_override_doc.overrides_fr and geo_override_doc.overrides_fr.data:
                # housing = geo_override_doc.overrides_fr.data.profiles.housing if geo_override_doc.overrides_fr.data.profiles else None
                if geo_override_doc.overrides_fr.data.profiles.housing is not None:
                  housing = geo_override_doc.overrides_fr.data.profiles.housing
                elif geo_detail_doc.data.profiles.housing is not None:
                  housing = geo_detail_doc.data.profiles.housing
                else:
                  housing = None

            section_contents = {'housing': housing}

            self.rewrite_property_type(geog_id=geog_id, longId=longId, city=city,
                              property_type=property_type,
                              lang=lang, 
                              gpt_writer=gpt_writer, use_rag=use_rag,
                              **section_contents)
            
            # print(f'geog_id: {geog_id}, longId: {longId}, city: {city}, prov_code: {province}')
            # print(f'housing: {housing}')
          else:
            self.log_error(f'No geo_override_doc found', longId=longId, geog_id=geog_id)
            time.sleep(self.wait_sec)
            continue

        else:
          self.log_error(f'No geo_detail_{lang}_doc found', longId=longId, geog_id=geog_id)
          time.sleep(self.wait_sec)
          continue
      except (AttributeError, KeyError) as e:
        self.log_error(str(e) + f'|| From rewrite_property_types_using_dataclasses(...) geo_doc with prov_code {prov_code}', longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)
      except Exception as e:
        self.log_error(str(e), longId=longId, geog_id=geog_id)
        time.sleep(self.wait_sec)


  def rewrite_property_type(
    self, 
    geog_id, longId, city, prov_code, property_type='CONDO', lang='en', 
    non_property_type_gpt_writer=None, property_type_gpt_writer=None, 
    use_rag=True, mode='prod', **section_contents) -> bool:
    """
    Given a geog_id, longId, city, and section_contents, rewrite the content and update the geo_overrides_index_name index 
    keyed by longId.

    Return: a bool to indicate if rewrite is successful or not.
    """

    # first check if GPT rewrite of the targeted version is already in archive.
    # if found, then use it.
    if self.archiver:
      archived_rewrite = self.archiver.get_record(longId=longId, property_type=property_type, version=self.version_string, use_cache=True)
      if archived_rewrite:        
        gpt_response = archived_rewrite['chatgpt_response']
        match = re.search(r'<housing>(.+?)</housing>', gpt_response)   # extract <housing>??</housing>
        if match and match.group(1):
          rewritten_housing = match.group(1)
          es_op_succeeded = self.update_es_doc_property_override(longId=longId, housing_content=rewritten_housing, property_type=property_type, lang=lang)
          self.log_info(f'Found archived rewrite for {longId} and version {self.version_string}.', longId=longId, geog_id=geog_id)
          return es_op_succeeded
    
    if property_type_gpt_writer is None:
      property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model,
                                        available_sections=['housing'],    # transport, services and character are not property type specific
                                        property_type=property_type)
    if non_property_type_gpt_writer is None:                                      
      non_property_type_gpt_writer = LocalLogicGPTRewriter(llm_model=self.llm_model,
                                        available_sections=['housing'],
                                        property_type=None)                                      
                                               
      
    housing = section_contents['housing']
    if housing is None or len(housing) == 0:   
      # since housing is the only section for this mode, we will skip openai chat completion request.
      self.log_info(f"No original locallogic housing content found for {lang}.", longId=longId, geog_id=geog_id)
      return False  # not a rewrite 
    
    # dynamic info & metric injection (RAG is used in v1)
    if use_rag:
      avg_price, pct, _ = self.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city, property_type=property_type)    
      if avg_price > 1.0:
        params_dict = {'Average price on MLS®': int(avg_price)}
        if pct > 0.0:   # this is -1.0 if there's not enough listings for this to be statistically robust 
          params_dict['Percentage of listings'] = pct
        gpt_writer = property_type_gpt_writer
      else:
        # if avg_price or pct are zeros, possible if no listing for that city/province,  
        # then use non_property_type_gpt_writer to do rewrite without any ref to property type
        params_dict = None   # no data for RAG
        gpt_writer = non_property_type_gpt_writer

    else:  # placeholder way, instruct to use placeholders while adding new information
      avg_price_explanation = self.get_avg_price_explanation(property_type=property_type)
      pc_of_listings_explanation = self.get_pc_of_listings_explanation(property_type=property_type)
      if avg_price > 1.0:
        params_dict = {'[avg_price]': avg_price_explanation}
        params_dict['[pc_of_listings]'] = pc_of_listings_explanation
        gpt_writer = property_type_gpt_writer
      else:
        # if avg_price or pct are zeros, possible if no listing for that city/province,  
        # then use non_property_type_gpt_writer to do rewrite without any ref to property type
        params_dict = None
        gpt_writer = non_property_type_gpt_writer

    if mode == 'mock':   # For testing purpose and will not use GPT.
      avg_price, pct, _ = self.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=prov_code, city=city, property_type=property_type)
      if avg_price > 1.0:
        rewritten_housing = f"[REPEAT housing] The average price of an MLS® real estate {property_type} listing in {city} is $ {int(avg_price)}."
        if pct > 0.0:
          rewritten_housing += f" The % of active {property_type} listings in {city} is {pct:.2f}%."
        else:
          rewritten_housing += f" Too few listings in {city} to calculate % of active listtings for {property_type}."
      else:
        rewritten_housing = f"[REPEAT housing] No {property_type} listing in {city} currently to calculate metrics."

      rewrites = {'housing': rewritten_housing}
      rewrites['error_message'] = None
    elif mode == 'prod':
      rewrites = gpt_writer.rewrite(params_dict=params_dict, use_rag=use_rag, housing=housing)
    else:
      raise ValueError(f"Invalid mode: {mode}")

    # archive the rewrite (so we can reuse later if there are other errors downstream)
    if self.archiver:
      messages = gpt_writer.construct_openai_user_prompt(params_dict=params_dict, use_rag=use_rag, housing=housing)
      user_prompt = messages[0]['content']
      chatgpt_response = self._extract_format_for_archive(rewrites)

      self.archiver.add_record(longId=longId, property_type=property_type, version=self.version_string, user_prompt=user_prompt, chatgpt_response=chatgpt_response)

    # ES Update or log error
    if rewrites['error_message'] is None:
      rewrites.pop('error_message', None)   # remove error_message from rewrites dict
      es_op_succeeded = self.update_es_doc_property_override(longId=longId, housing_content=rewrites['housing'], property_type=property_type, lang=lang)
      self.gpt_rewrite_consecutive_failures = 0     # reset the counter
    else:
      self.log_error('[GPT] ' + rewrites['error_message'], longId=longId, geog_id=geog_id)
      self.gpt_rewrite_consecutive_failures += 1
    
    return rewrites.get('error_message') is None and es_op_succeeded     # True if everything is ok


  def update_es_doc_property_override(self, longId: str, housing_content: str, property_type: str = 'CONDO', lang='en'):

    property_type_lower = property_type.lower().replace('-', '_')
    update_script = {
      "script": {
        "source": f"""
            if (ctx._source.containsKey('overrides_{property_type_lower}_{lang}')) {{
                ctx._source.overrides_{property_type_lower}_{lang}.data.profiles = params.profiles;
                ctx._source.overrides_{property_type_lower}_{lang}.data.version = params.version;
            }} else {{
                def profiles = params.profiles;
                def data = ['version': params.version, 'profiles': profiles];
                ctx._source['overrides_{property_type_lower}_{lang}'] = ['data': data];
            }}
        """,
        "params": {
            "version": self.version_string,
            "profiles": {
                "housing": housing_content                
            }
        }
      }
    }
    return self._handle_ES_update(update_script, longId)


  def _handle_ES_update(self, update_script, longId) -> bool:
    """
    Given an update_script, update the geo_overrides_index_name index keyed by longId.
    Return: a bool to indicate if update is successful or not.
    """
    succeeded = False
    try:
      # commenting this out until UAT is available for testing.
      response = self.es_client.update(index=self.geo_overrides_index_name, id=longId, body=update_script)  
      # response = self.es_client.index(index=self.geo_overrides_index_name, id=longId, body=update_script["upsert"])

      # print("[Mock] response = self.es_client.update(index=self.geo_overrides_index_name, id=longId, body=update_script)")
      # print(f"update_script: {update_script}")

      # Next line is for testing only.
      # if random.random() > 0.5: raise elasticsearch_exceptions.NotFoundError(404, 'NotFoundError', 'Doc not found')

      succeeded = True
      
      self.log_info(f'[ES_update] Successful.', longId=longId)
    except elasticsearch_exceptions.NotFoundError:      
      self.log_error(f'[ES_update] Not found in {self.geo_overrides_index_name}', longId=longId)
    except KeyError as ke:      
      self.log_error('[ES_update] ' + str(ke) + f' update_script: {update_script}', longId=longId)      
    except Exception as e:
      self.log_error('[ES_update] ' + str(e) + f' update_script: {update_script}', longId=longId)

    return succeeded
      
  def log_error(self, message, longId=None, geog_id=None):
    # TODO: Add additional logging mechanism, maybe send to an external logging service, before going to production
    logger = logging.getLogger(self.__class__.__name__)
    # extra = {'longId': longId} if longId is not None else {'longId': 'None'}

    longId_part = f"[longId: {longId}] " if longId is not None else ''
    geog_id_part = f"[geog_id: {geog_id}] " if geog_id is not None else ''

    message = f"{longId_part}{geog_id_part}" + message

    logger.error(f'{message}')

  def log_info(self, message, longId=None, geog_id=None):
    logger = logging.getLogger(self.__class__.__name__)

    longId_part = f"[longId: {longId}] " if longId is not None else ''
    geog_id_part = f"[geog_id: {geog_id}] " if geog_id is not None else ''

    message = f"{longId_part}{geog_id_part}" + message
    logger.info(message)
  
  def update_geo_overrides_params(self, prov_code=None):
    '''
    NOTE: this may not be needed if RAG is used. We are not storing any metrics 
    '''
    geo_all_content_df = self.geo_all_content_df.q("province == @prov_code") if prov_code is not None else self.geo_all_content_df
    for k, row in geo_all_content_df.iterrows():
      try:
        longId = row.longId
        geog_id = row.geog_id
        province = row.province
        city = row['name']  # or row.city

        avg_price, _, _ = self.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=province, city=city)
        avg_price_desc_str = self.get_avg_price_explanation()
        pc_of_listings_desc_str = self.get_pc_of_listings_explanation()

        parameters = { "last_updated": datetime.now().date().isoformat()}

        # if avg_price is not None:
        #   parameters['avg_price'] = {"value": avg_price, "description": avg_price_desc_str}
        # if pc_of_listings is not None:
        #   parameters['pc_of_listings'] = {"value": pc_of_listings, "description": pc_of_listings_desc_str}

        property_type = None   # for city, non property type specific 
        parameters['avg_price'] = {"value": avg_price, 
                                   "description": self.get_avg_price_explanation(property_type=property_type)}
        
        for property_type in self.property_types:
          avg_price, pct, _ = self.get_avg_price_and_active_pct(geog_id=geog_id, prov_code=province, city=city, property_type=property_type)
          parameters[f'{property_type.lower()}_avg_price'] = {"value": avg_price,
                                                              "description": self.get_avg_price_explanation(property_type=property_type)}
          parameters[f'{property_type.lower()}_pc_of_listings'] = {"value": pct,
                                                                    "description": self.get_pc_of_listings_explanation(property_type=property_type)}

        self._update_geo_overrides_params(longId=longId, params=parameters)

      except (AttributeError, KeyError) as e:
        self.log_error(str(e) + f'|| From update_geo_overrides_params(...) with prov_code {prov_code}', longId)        
      except Exception as e:
        self.log_error(str(e), longId)


  def _update_geo_overrides_params(self, longId, params):
    """
    for a given longId, add param node to doc in rlp_content_geo_overrides_current index
    NOTE: no INSERT func is provided, it is assumed the doc_id (longId) already exists in the index, otherwise, ES exception will be thrown.

    #TODO: param value and desc can be abstracted by a dataclass?
    """
    update_script = {
      "script": {
        "source": "ctx._source.parameters = params.parameters;",
        "params": {
            "parameters": params
        }
      }
    }
    self._handle_ES_update(update_script, longId)

  # Methods to get dynamic info and metrics for RAG

  def get_avg_price_and_active_pct(self, geog_id, prov_code, city, property_type=None) -> Tuple[float, float, int]:
    '''
    Compute average price,  % of active listings, and # of active listings for a given geog_id, prov_code, city, w/wo property_type.
    '''

    MIN_LISTING_THRESHOLD = 10   # if total listings is less than this, we return -1 for % of active listing due to stat. robustness

    def create_base_query():
      return {
        "size": 0,  # We don't need to retrieve documents, only aggregate data
        "query": {
          "bool": {
            "must": [
              {"match": {"guid": geog_id}},
              {"match": {"city": city}},
              {"match": {"provState": prov_code}},
              
              {"match": {"transactionType": "SALE"}},
              {"match": {"listingStatus": "ACTIVE"}}
            ]
          }
        }
      }

    # avg_price_query = copy.deepcopy(base_query)
    avg_price_query = create_base_query()
    avg_price_query["aggs"] = {
      "average_price": {
        "avg": {
          "field": "price"
        }
      },
      "count_listings": {
        "value_count": {
          "field": "jumpId"
        }
      }
    }

    if property_type == 'CONDO':
      avg_price_query['query']['bool']['must'].extend([
        {"terms": {"listingType.keyword": ["CONDO", "REC"]}},
        {"match": {"searchCategoryType": 101}}
      ])
    elif property_type == 'LUXURY':
      avg_price_query['query']['bool']['must'].append({"match": {"carriageTrade": True}})
    elif property_type == 'SEMI-DETACHED':
      avg_price_query['query']['bool']['must'].append({"match": {"searchCategoryType": 104}})
    elif property_type == 'TOWNHOUSE':
      avg_price_query['query']['bool']['must'].append({"match": {"searchCategoryType": 102}})
    elif property_type == 'INVESTMENT':
      avg_price_query['query']['bool']['must'].extend([
        {"terms": {"searchCategoryType": [201, 202, 203, 204]}},
        {"match": {"listingType.keyword": "PPR"}}
      ])
    else:   # for city/neighbourhood level
      avg_price_query['query']['bool']['must'].append({"terms": {"listingType.keyword": ["RES", "CONDO", "REC"]}})

    # Query for counting total listings (excluding 'COMM')
    # total_listings_query = copy.deepcopy(base_query)
    total_listings_query = create_base_query()
    total_listings_query["query"]["bool"]["must_not"] = [
      {"term": {"listingType.keyword": "COMM"}}
    ]
    total_listings_query["aggs"] = {
      "total_listings": {
        "value_count": {
          "field": "jumpId"
        }
      }
    }

    # execute queries
    response_avg_price = self.es_client.search(index=self.listing_index_name, body=avg_price_query)
    avg_price = response_avg_price['aggregations']['average_price']['value']
    count_listings = response_avg_price['aggregations']['count_listings']['value']

    response_total_listings = self.es_client.search(index=self.listing_index_name, body=total_listings_query)
    total_listings = response_total_listings['aggregations']['total_listings']['value']

    # Calculate percentage of active listings if property_type is not None
    pc_active = None
    if property_type is not None:
      if total_listings < MIN_LISTING_THRESHOLD:
        pc_active = -1.0    # a signal this should be invalid and ignored
      else:
        pc_active = round(count_listings / total_listings * 100.0 if count_listings > 0 else 0.0, 2)

    return round(avg_price, 0) if avg_price is not None else 0.0, pc_active, count_listings

    
  def get_avg_price_explanation(self, property_type=None):
    # TODO: Do we need to put this into ES?
    if property_type is None:
      return "The average price of an MLS® real estate listing in the city."
    else:
      return f"The average price of an MLS® real estate listing for {property_type.lower()} in the city."
    
  def get_pc_of_listings_explanation(self, property_type):
    return f"Percentage of MLS® listings for {property_type.lower()} in the city."

  def get_all_listings(self, prov_code, return_df=True) -> Union[pd.DataFrame, Tuple[List[str], List[dict]]]:
    """
    Return all listings for a given prov_code. If return_df is True, return a DataFrame, 
    otherwise return a tuple of doc_ids and sources.
    
    Only these attributes: guid, city, provState, listingType, searchCategoryType, transactionType, 
    listingStatus, price are returned. Edit for more attributes if needed.
    """
    count_query = {
      "query": {
        "match": {"provState": prov_code}
      },
    }

    count = self.es_client.count(index="rlp_listing_current", body=count_query)['count']
    print(f'# of listings in {prov_code}: {count}')

    # Retrieval query for fetching listings
    retrieval_query = {
      "query": {
        "match": {"provState": prov_code}
      },
      "_source": ["guid", "city", "provState", "listingType", "searchCategoryType", "transactionType", "listingStatus", "price"]
    }

    scroller = scan(self.es_client, index="rlp_listing_current", query=retrieval_query)
    
    doc_ids, sources = [], []
    for hit in scroller:
      doc_id = hit['_id']
      source = hit['_source']
      doc_ids.append(doc_id)
      sources.append(source)

    if return_df:
      listing_df = pd.DataFrame(sources, index=doc_ids)
      listing_df.index.name = 'jumpId'

      return listing_df      
    else:
      return doc_ids, sources
    
  # Rerun and recover methods
  def rerun_to_recover(self, 
    log_filename: str = None, 
    run_num: int = None, 
    prov_code: str = None, 
    lang: str = 'en', 
    use_rag: bool = True, 
    log_dir: Path = '.', 
    gpt_backup_version=None,
    mode='prod'):
    """
    Given a run_num and prov_code, read the log file for error
    Or if a log_filename is directly given then run_num and prov_code are ignored
    """
    if log_filename is None:
      # then run_num and prov_code must be provided
      if run_num is None or prov_code is None:
        raise ValueError("Either log_filename or both run_num and prov_code must be provided.")

    if log_filename is not None:
      if not Path(log_filename).exists(): raise ValueError(f"Log file {log_filename} does not exist.")
      log_file_paths = [log_filename]
    else:
      log_filename_regex_pattern = rf'^(\d{{8}})_(\d{{6}})_run_{run_num}_{prov_code}_(\w+)\.log$'
      log_file_paths = Path(log_dir).lfre(log_filename_regex_pattern)

    if len(log_file_paths) == 0:
      self.log_error("No matching log file found.")   # this is considered an error

    error_pattern = re.compile(
      r'.*\[ERROR\].*\[longId: ([^\]]+)\] \[geog_id: ([^\]]+)\] (.+)$'
    )

    data = {'longId': [], 'geog_id': [], 'Error': []}
    for log_file_path in log_file_paths:
      self.log_info(f"Processing file {log_file_path}")
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
        self.log_error(f"An error occurred while processing file {log_file_path}: {e}")

    # Convert the dictionary to a DataFrame
    error_df = pd.DataFrame(data)
    error_df.drop_duplicates(subset=['longId', 'geog_id'], inplace=True)

    # rerun rewrite_cities and rewrite_property_types that have ERRORS
    for _, row in error_df.iterrows():
      geog_id = row.geog_id
      longId = row.longId

      self.extract_content(geog_id=geog_id)

      self.rewrite_cities(geog_id=geog_id, lang=lang, use_rag=use_rag)

      for property_type in self.property_types:
        if gpt_backup_version:
          # try to get any chatgpt_response from archive backup
          archived_rewrite = self.archiver.get_record(longId=row.longId, property_type=property_type, version=gpt_backup_version)
          if archived_rewrite:
            chatgpt_response = archived_rewrite['chatgpt_response']            
            match = re.search(r'<housing>(.+?)</housing>', chatgpt_response)    # extract <housing>??</housing>
            if match:
              rewritten_housing = match.group(1)
              update_status = self.update_es_doc_property_override(longId=longId, housing_content=rewritten_housing, property_type=property_type, lang=lang)
              if not update_status:
                self.log_error(f"[rerun_to_recover] Failed to update ES document", longId=longId, geog_id=geog_id)
              continue

        # Fall back to the regular rewrite process if no archive is available
        self.rewrite_property_types(property_type=property_type, geog_id=geog_id, lang=lang, use_rag=use_rag, mode=mode)

  # Versioning of rewrite
  def get_current_version_string(self):
    '''
    This version string should be used for all rewrites across all provinces. The expected frequency is quarterly, 
    hence the version string includes the year and the month only.
    '''
    # return datetime.now().strftime("%Y%m%d")
    return datetime.now().strftime("%Y%m")

  # Helper methods to query ES

  # (A) return a dataframe or (doc_ids, json sources) tuple
  def _get_geo_details(self, selects: List[str] = None, prov_code: str = None, name: str = None, geog_id: str = None, lang: str = 'en', return_df=True) -> Union[pd.DataFrame, Tuple[List[str], List[Dict]]]:
    # fetch geo details docs for a province, name, or lang and return them as a dataframe
    if selects is None:
      selects = ['data.name', 'data.province', 'data.profiles']   # default to these

    # check at least one of prov_code or name is not None and geog_id is None
    if prov_code is None and name is None and geog_id is None:
      raise ValueError("At least one of prov_code or name or geog_id must be not None")
    
    if lang == 'en': 
      index_name = self.geo_details_en_index_name
    elif lang == 'fr':
      index_name = self.geo_details_fr_index_name
    else:
      raise ValueError(f"Unknown {lang}")
    
     # if geog_id is provided, query by ids
    if geog_id is not None:
      body = {
          "query": {
              "ids": {
                  "values": [geog_id]
              }
          },
          "_source": selects
      }
    else:
      terms = {}
      if prov_code is not None:
        terms["data.province.keyword"] = prov_code
      if name is not None:
        terms["data.name.keyword"] = name

      filter_list = [{"term": {key: value}} for key, value in terms.items()]

      body = {
          "query": {
              "bool": {
                  "filter": filter_list
              }
          },
          "_source": selects
      }
      
    scroller = scan(self.es_client, index=index_name, query=body)
    
    doc_ids, sources = [], []
    for hit in scroller:
      doc_id = hit['_id']
      source = hit['_source']
      doc_ids.append(doc_id)
      sources.append(source)
      
    if return_df:
      geo_details_df = pd.DataFrame([src['data'] for src in sources], index=doc_ids)
      geo_details_df.index.name = 'geog_id'
      
      return geo_details_df
    else:
      return doc_ids, sources
  
  def _get_geo_overrides(self, selects: List[str] = None, prov_code: str = None, city: str = None, localLogicId: str = None, longId: str = None, return_df=True, incl_property_override=False) -> Union[pd.DataFrame, Tuple[List[str], List[Dict]]]:
    # fetch geo overrides docs and return them as a dataframe
    if selects is None:
      # default to these
      selects = ["localLogicId", "longId", "overrides:", "overrides", 
      "overrides_en", 
      # "overrides_fr",   # no french for v1
      ]

    if incl_property_override:   # only for en for now.
      selects.append("overrides_luxury_en")
      selects.append("overrides_condo_en")
      selects.append("overrides_semi_detached_en")
      selects.append("overrides_townhouse_en")
      selects.append("overrides_investment_en")

    if prov_code is None and city is None and longId is None:
      body = {"query": {"match_all": {}}, "_source": selects}
    if longId is not None:
      body = {
        "query": {
            "ids": {
                "values": [longId]
            }
        },
        "_source": selects
      }
    else:
      terms = {}
      if prov_code is not None:
        terms["province.keyword"] = prov_code
      if city is not None:
        terms["city.keyword"] = city
      if localLogicId is not None:
        terms["localLogicId.keyword"] = localLogicId

      filter_list = [{"term": {key: value}} for key, value in terms.items()]

      body = {
          "query": {
              "bool": {
                  "filter": filter_list
              }
          },
          "_source": selects
      }
      
    scroller = scan(self.es_client, index=self.geo_overrides_index_name, query=body)
    
    doc_ids, sources = [], []
    for hit in scroller:
      doc_id = hit['_id']
      source = hit['_source']
      doc_ids.append(doc_id)
      sources.append(source)
      
    locallogic_ids = [src['localLogicId'][0] if src.get('localLogicId') is not None else None for src in sources]
    
    if return_df:
      geo_overrides_df = pd.DataFrame(sources, index=locallogic_ids)
      geo_overrides_df.index.name = 'geog_id'
        
      return geo_overrides_df
    else:
      return locallogic_ids, sources
    
  def _get_geo(self, selects: List[str] = None, prov_code: str = None, city: str = None, geog_id: str = None, return_df=True) -> Union[pd.DataFrame, Tuple[List[str], List[Dict]]]:
    # fetch geo docs and return them as a dataframe

    # if prov_code is None and city is None:
    #   body = {"query": {"match_all": {}}}
     
    if selects is None:
      selects = ['longId', 'city', 'citySlug', 'province']   # default to these
    
    if geog_id is not None:   # if geog_id is provided, query by ids
      body = {
          "query": {
              "ids": {
                  "values": [geog_id]
              }
          },
          "_source": selects
      }
    else:
      terms = {"level": 30}   # only level 30 needed for rewrite

      if prov_code is not None:
        terms["province.keyword"] = prov_code
      if city is not None:
        terms["city.keyword"] = city

      filter_list = [{"term": {key: value}} for key, value in terms.items()]

      body = {
          "query": {
              "bool": {
                  "filter": filter_list
              }
          },
          "_source": selects
      }
      
    scroller = scan(self.es_client, index=self.geo_index_name, query=body)
    
    doc_ids, sources = [], []
    for hit in scroller:
      doc_id = hit['_id']
      source = hit['_source']
      doc_ids.append(doc_id)
      sources.append(source)    
    
    if return_df:
      geo_df = pd.DataFrame(sources, index=doc_ids)
      geo_df.index.name = 'geog_id'
        
      return geo_df
    else:
      return doc_ids, sources
    
  # (B) return list of dataclasses
  def get_geo(self, prov_code: str = None, geog_id: str = None) -> List[GeoDoc]:
    if geog_id is not None:
      geo_doc_ids, geo_sources = self._get_geo(geog_id=geog_id, return_df=False)
    else:  
      geo_doc_ids, geo_sources = self._get_geo(prov_code=prov_code, return_df=False)

    # add doc_id to source
    for doc_id, source in zip(geo_doc_ids, geo_sources):
      source['geog_id'] = doc_id

    geo_docs = [GeoDoc(**source) for source in geo_sources]
    return geo_docs
  
  def get_geo_details(self, prov_code: str = None, geog_id: str = None, lang: str = 'en') -> List[GeoDetailDoc]:
    if geog_id is not None:
      geo_detail_doc_ids, geo_detail_sources = self._get_geo_details(geog_id=geog_id, lang=lang, return_df=False)
    else:
      geo_detail_doc_ids, geo_detail_sources = self._get_geo_details(prov_code=prov_code, lang=lang, return_df=False)

    for geo_detail_doc_ids, geo_detail_source in zip(geo_detail_doc_ids, geo_detail_sources):
      geo_detail_source['geog_id'] = geo_detail_doc_ids

    geo_detail_docs = [
      GeoDetailDoc(
          geog_id=source['geog_id'],
          data=Data(
            name=source['data'].get('name', None),
            province=source['data'].get('province', None),
            profiles=Profile(**source['data'].get('profiles', {}))
          )
      )
      for source in geo_detail_sources
    ]

    return geo_detail_docs
  
  def get_geo_overrides(self, prov_code: str = None, geog_id: str = None) -> List[GeoOverridesDoc]:

    if geog_id is not None:
      geo_override_doc_ids, geo_override_sources = self._get_geo_overrides(localLogicId=geog_id, return_df=False)
    else:
      geo_override_doc_ids, geo_override_sources = self._get_geo_overrides(prov_code=prov_code, return_df=False)

    geo_override_docs = [
        GeoOverridesDoc(
            longId=source.get("longId", ""),
            localLogicId=source.get("localLogicId", []),
            overrides=source.get("overrides", {}),
            overrides_en=Overrides(data=Data(profiles=Profile(**source["overrides_en"]["data"]["profiles"]))) if "overrides_en" in source else None,
            overrides_fr=Overrides(data=Data(profiles=Profile(**source["overrides_fr"]["data"]["profiles"]))) if "overrides_fr" in source else None
        )
        for source in geo_override_sources
    ]

    return geo_override_docs

  # Helper methods for archiver's formatting needs
  def _extract_format_for_archive(self, data: Dict[str, str]) -> str:
    return ''.join(
      f"<{key}>{data[key]}</{key}>" for key in ['housing', 'transport', 'services', 'character'] if key in data and data[key]
    )

  # Generating report
  def generate_reports(self, version: str = None) -> pd.DataFrame:
    '''generate a report that look like this:

        total	en_housing	overrides_en_housing	overrides_luxury_en_housing	overrides_condo_en_housing	etc. etc.
    province
    PE	  112	66	56	56	56	56	

    filtered by version. The version is either an exact match, or prefix match (startswith). If not provided, then all versions are included.

    '''
    # Total
    total_records = self.geo_all_content_df.groupby('province').size().to_frame('total')

    # For calculating counts of en_housing, overrides_en_housing, overrides_{property_type}_en_housing
    section = 'en_housing'
    en_housing_notnull = self.geo_all_content_df.groupby('province')[section].apply(lambda x: x.notnull().sum()).to_frame()

    section = 'overrides_en_housing'
    if 'overrides_en_version' in self.geo_all_content_df.columns:
      if version:
        # overrides_en_housing_notnull = self.geo_all_content_df.q("overrides_en_version == @version").groupby('province', group_keys=True)[section].apply(lambda x: x.notnull().sum()).to_frame()
        overrides_en_housing_notnull = self.geo_all_content_df.q("overrides_en_version.notnull() and overrides_en_version.str.startswith(@version)").groupby('province', group_keys=True)[section].apply(lambda x: x.notnull().sum()).to_frame()
      else:
        overrides_en_housing_notnull = self.geo_all_content_df.groupby('province', group_keys=True)[section].apply(lambda x: x.notnull().sum()).to_frame()
    else:
      overrides_en_housing_notnull = pd.DataFrame(data=[None], columns=[section])

    report_df = [total_records, en_housing_notnull, overrides_en_housing_notnull]

    for property_type in self.property_types:
      property_type = property_type.lower().replace('-', '_')
      section = f'overrides_{property_type}_en_housing'
      if f'overrides_{property_type}_en_version' in self.geo_all_content_df.columns:
        if version:
          # report_df.append(self.geo_all_content_df.q(f"overrides_{property_type}_en_version == @version").groupby('province', group_keys=True)[section].apply(lambda x: x.notnull().sum()).to_frame())
          report_df.append(self.geo_all_content_df.q(f"overrides_{property_type}_en_version.notnull() and overrides_{property_type}_en_version.str.startswith(@version)").groupby('province', group_keys=True)[section].apply(lambda x: x.notnull().sum()).to_frame())
        else:
          report_df.append(self.geo_all_content_df.groupby('province', group_keys=True)[section].apply(lambda x: x.notnull().sum()).to_frame())
      else:
        report_df.append(pd.DataFrame(data=[None], columns=[section]))

    report_df = pd.concat(report_df, axis=1)
    return report_df

  def metrics_report(self, prov_code: str) -> pd.DataFrame:
    property_types = ['city', 'luxury', 'condo', 'semi', 'townhouse', 'investment']   # col friendly names
    all_data = []
    for k, row in self.geo_all_content_df.q(f"province == '{prov_code}' and en_housing.notnull()").iterrows():
      geog_id = row.geog_id
      longId = row.longId
      prov_code = row.province
      city = row.city

      avg_price, pct, count_city = self.get_avg_price_and_active_pct(geog_id=geog_id, city=city, prov_code=prov_code)

      avg_price_luxury, pct_luxury, count_luxury = self.get_avg_price_and_active_pct(geog_id=geog_id, city=city, prov_code=prov_code, property_type='LUXURY')
      avg_price_condo, pct_condo, count_condo = self.get_avg_price_and_active_pct(geog_id=geog_id, city=city, prov_code=prov_code, property_type='CONDO')
      avg_price_semi, pct_semi, count_semi = self.get_avg_price_and_active_pct(geog_id=geog_id, city=city, prov_code=prov_code, property_type='SEMI-DETACHED')
      avg_price_townhouse, pct_townhouse, count_townhouse = self.get_avg_price_and_active_pct(geog_id=geog_id, city=city, prov_code=prov_code, property_type='TOWNHOUSE')
      avg_price_investment, pct_investment, count_investment = self.get_avg_price_and_active_pct(geog_id=geog_id, city=city, prov_code=prov_code, property_type='INVESTMENT')

      avg_prices = [avg_price, avg_price_luxury, avg_price_condo, avg_price_semi, avg_price_townhouse, avg_price_investment]
      # avg_prices = [f"{price:,.0f}" if price is not None else None for price in avg_prices]

      pct_actives = [pct, pct_luxury, pct_condo, pct_semi, pct_townhouse, pct_investment]
      counts = [count_city, count_luxury, count_condo, count_semi, count_townhouse, count_investment]

      data = {'longId': longId, 'city': city, 'prov_code': prov_code}
      for i, prop_type in enumerate(property_types):
        data[f'{prop_type}_avg_price'] = avg_prices[i]
        data[f'{prop_type}_pc'] = pct_actives[i]
        data[f'{prop_type}_count'] = counts[i]

      all_data.append(data)

    metrics_df = pd.DataFrame(all_data)

    # drop city_pc since % of Active (pc) is only for per property type
    metrics_df.drop(columns=['city_pc'], inplace=True)

    return metrics_df

"""Sample of geo (rlp_content_geo_3) doc:

{'_index': 'rlp_content_geo_3',
 '_type': '_doc',
 '_id': 'g30_c3nfkdtg',
 '_version': 1,
 '_seq_no': 34322,
 '_primary_term': 4,
 'found': True,
 '_source': {'province': 'AB',
  'localLogicId': 'g30_c3nfkdtg',
  'level': 30,
  'levelName': 'city',
  'city': 'Calgary',
  'level30En': 'Calgary',
  'level30Fr': 'Calgary',
  'citySlug': 'calgary',
  'level40En': 'Greater Calgary',
  'level40Fr': 'Grand Calgary',
  'lat': 51.034841537475586,
  'lng': -114.0519905090332,
  'created': '2019-01-01T00:00:00',
  'modified': '2020-01-07T00:00:00',
  'hasEnProfile': True,
  'hasFrProfile': True,
  'path': 'ab/calgary',
  'longId': 'ab_calgary',
  'uploaded': '2022-01-10T20:22:08.849981'}}
"""

""" Sample of rlp_geo_details_en doc
{'_index': 'rlp_geo_details_en',
 '_type': '_doc',
 '_id': 'g30_c3nfkdtg',
 '_version': 9,
 '_seq_no': 8361,
 '_primary_term': 1,
 'found': True,
 '_source': {'data': {'type': 'profiles',
   'profiles': {'transport': 'The preferred transportation option in Calgary is usually driving. It is a reasonably short car ride to the closest highway from any home in Calgary, and it is very convenient to park. In contrast, the public transit network in this city is not very practical. Nonetheless, there are many rapid transit stations on the 202 | Blue Line - Saddletowne/69 Street CTrain and 201 | Red Line - Somerset - Bridlewood/Tuscany CTrain. Residents are served by over 250 bus lines, and bus stops are not very far away from most homes. Most houses for sale in this city are located in areas that are not especially well-suited for walking since few daily errands can be run without having to resort to a car. Despite the fact that it is a flat ride to most destinations, it is also challenging to travel by bicycle in Calgary as.',
    'services': 'Primary schools and daycares are easy to access on foot from most properties for sale in this city. In contrast, it can be challenging to access high schools as a pedestrian. Regarding eating, a fraction of house buyers in Calgary may be able to buy groceries by walking in one of over 400 stores, while others will need another means of transportation. There are also a limited number of choices for those who appreciate nearby restaurants and cafes.',
    'character': 'The character of Calgary is exemplified by its slower-paced atmosphere. There are over 2200 public green spaces nearby for residents to unwind in, making it very easy to reach them. This city is also very good for those who prefer a quiet atmosphere, as the streets tend to be very tranquil.',
    'housing': 'In Calgary, many dwellings are single detached homes, while small apartment buildings and townhouses are also present in the housing stock. This city offers mainly four or more bedroom and three bedroom homes. Homeowners occupy about three quarters of the units in the city whereas the remainder are rented. The homes in this city are quite new, since about one third of its dwellings were constructed after the year 2000, while the majority of the remaining buildings were constructed pre-1960 and in the 1960s.'},
   'id': 'g30_c3nfkdtg',
   'bounding_box': {'west': -114.33210920723513,
    'east': -113.83460970983437,
    'south': 50.85237302969596,
    'north': 51.20786856520588},
   'centroid': {'lng': -114.0520147990191, 'lat': 51.035025846379526},
   'intro': 'Calgary is a city located within Alberta, Canada.',
   'name': 'Calgary',
   'parent': 'g40_c3nghj4v',
   'province': 'AB'},
  'meta': {'sections': {'transport': 'Transportation',
    'services': 'Services',
    'character': 'Character',
    'housing': 'Housing'},
   'geometry': [[[51.19768, -114.16838],
     [51.19771, -114.16445],
     [51.19841, -114.16445],
     [51.1986, -114.15269],

     etc.
     etc.

     [51.08141, -114.28655],
     [51.08141, -114.28683],
     [51.08141, -114.28712],
     [51.08141, -114.2874],
     ...]],
   'bounding_box': {'north': 51.21256,
    'south': 50.84308,
    'east': -113.8583,
    'west': -114.31554},
   'bounding_box_padded': {'north': 51.249508000000006,
    'south': 50.806132,
    'east': -113.812576,
    'west': -114.361264}}}}    
"""    

""" Sample of rlp_geo_details_fr doc:
{'_index': 'rlp_geo_details_fr',
 '_type': '_doc',
 '_id': 'g30_c3nfkdtg',
 '_version': 9,
 '_seq_no': 8357,
 '_primary_term': 1,
 'found': True,
 '_source': {'data': {'type': 'profiles',
   'profiles': {'transport': "L'automobile est un très bon moyen de transport pour circuler dans la ville. Il n'est normalement pas trop ardu d'y trouver un endroit où stationner, et les accès aux autoroutes sont bien situés. Néanmoins, le service de transport collectif est peu fréquent. Cependant, on peut trouver plusieurs stations de transport rapide, qui donnent accès à 202 | Blue Line - Saddletowne/69 Street CTrain et 201 | Red Line - Somerset - Bridlewood/Tuscany CTrain. Quelques centaines de lignes d'autobus traversent la ville, et la plupart des maisons sont situées à deux pas d'un arrêt d'autobus. La majorité des propriétés se trouvent dans des endroits qui sont peu pratiques pour les piétons parce que les résidents doivent normalement utiliser la voiture pour combler leurs besoins quotidiens. La ville se prête aussi mal à la pratique de la bicyclette puisque. Toutefois, il y a relativement peu de pentes pour mettre les cyclistes au défi.",
    'services': "Il est plutôt simple de se rendre à une garderie ou une école primaire en marchant à Calgary. Cependant, puisqu'elles sont peu nombreuses, les écoles secondaires ne sont pas toujours à distance de marche. En matière d'accès à la nourriture, certains résidents pourront aisément faire leurs courses à pied à l'un des quelques 400 épiceries du secteur, tandis que d'autres devront prendre un véhicule. De plus, il y a un choix limité de cafés et de restaurants.",
    'character': "Le caractère de Calgary est défini par son ambiance calme. Un très grand nombre d'emplacements de la ville ont un très bon accès aux îlots de verdure, puisqu'on peut habituellement trouver l'un de ses 2240 espaces verts tout près de la majorité des propriétés en vente. Calgary est également très silencieuse, étant donné qu'il y a peu de pollution sonore liée à la circulation automobile.",
    'housing': "La majorité des bâtiments sont des maisons individuelles. La ville consiste principalement en des logements avec quatre chambres à coucher et en des logements avec trois chambres à coucher. Environ les trois quarts de la population de la ville possèdent leur demeure et les autres louent leur logement. Près du tiers des maisons de la ville ont été bâties après l'an 2000, tandis que le plus clair des maisons restantes ont été construites avant 1960 et dans les années 1960."},
   'id': 'g30_c3nfkdtg',
   'bounding_box': {'west': -114.33210920723513,
    'east': -113.83460970983437,
    'south': 50.85237302969596,
    'north': 51.20786856520588},
   'centroid': {'lng': -114.0520147990191, 'lat': 51.035025846379526},
   'intro': 'Calgary est une ville située en Alberta, Canada.',
   'name': 'Calgary',
   'parent': 'g40_c3nghj4v',
   'province': 'AB'},
  'meta': {'sections': {'transport': 'Transport',
    'services': 'Services',
    'character': 'Caractère',
    'housing': 'Hébergement'},
   'geometry': [[[51.19768, -114.16838],
     [51.19771, -114.16445],
     [51.19841, -114.16445],
     etc. etc.
     [51.08141, -114.28712],
     [51.08141, -114.2874],
     ...]],
   'bounding_box': {'north': 51.21256,
    'south': 50.84308,
    'east': -113.8583,
    'west': -114.31554},
   'bounding_box_padded': {'north': 51.249508000000006,
    'south': 50.806132,
    'east': -113.812576,
    'west': -114.361264}}}}
"""

"""Samples of rlp_content_geo_overrides_current doc:

{'province': 'AB',
  'city': 'Calgary',
  'citySlug': 'calgary',
  'localLogicId': ['g30_c3nfkdtg'],
  'longId': 'ab_calgary',
  'overrides:': {},
  'overrides_en': {'data': {'profiles': {'housing': "In Calgary, a significant portion of real estate for sale are standalone homes, but the city also boasts a mix of smaller apartment complexes and townhomes for sale as well. The predominant types of homes in the city have either three or more than four bedrooms. Around 75% of these properties are owned, leaving the rest for rentals. A notable feature of Calgary's housing landscape is its modernity; approximately one-third of homes were built post-2000. Meanwhile, most of the other homes date back to before 1960 or were established during the 1960s. The average price of an MLS® real estate listing in Calgary is $ 743,191."}}},
  'overrides_fr': {'data': {'profiles': {'housing': "À Calgary, une part significative de l'immobilier à vendre est constituée de maisons individuelles, mais la ville propose également un mélange de petits complexes d'appartements et de maisons de ville à vendre. Les types de maisons prédominants dans la ville ont soit trois chambres ou plus de quatre chambres. Environ 75 % de ces propriétés sont des propriétés privées, laissant le reste à la location. Un élément notable du paysage immobilier de Calgary est sa modernité ; environ un tiers des maisons ont été construites après 2000. Pendant ce temps, la plupart des autres maisons remontent à avant 1960 ou ont été établies pendant les années 1960. Le prix moyen d'une inscription immobilière MLS® à Calgary est 743 191 $ ."}}}}]

{'city': 'Airdrie',
  'citySlug': 'airdrie',
  'localLogicId': ['g10_c3ngtyy6'],
  'longId': 'ab_airdrie_big-springs',
  'neighbourhood': 'Big Springs',
  'neighbourhoodSlug': 'big-springs',
  'overrides': {'indexPage': True},
  'province': 'AB'},

"""