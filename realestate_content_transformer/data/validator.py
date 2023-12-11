from typing import Dict, List, Optional, Union

from pathlib import Path

from .pipeline import LocallogicContentRewriter

class RewriteValidator:
  def __init__(self, es_host: str, es_port=9200, working_dir: Path = '.') -> None:
    cheap_dummy_llm_model = 'gpt-3.5-turbo-0613'
    self.ll_rewriter = LocallogicContentRewriter(es_host=self.es_host, 
                                                 es_port=self.es_port,
                                                 llm_model=cheap_dummy_llm_model
                                                 )
    
    # TODO: better manage how save_dir is done, just use local for now
    self.ll_rewriter.extract_content(save_intermediate_results=True, save_dir='.')

  def validate_overrides(self):
    property_types = self.ll_rewriter.property_types
    geo_all_content_df = self.ll_rewriter.geo_all_content_df

    # Looking at housing for lang='en' for now.
    geo_all_content_df[['geog_id', 'longId', 'citySlug', 'en_housing', 'overrides_en_housing']]  # overrides_condo_en_housing
    # TODO: continue more validation here

    # Looking at housing for lang='en' for all property types
    property_override_cols = [f'overrides_{property_type.lower()}_en' for property_type in property_types]

    geo_all_content_df[['geog_id', 'longId', 'citySlug', 'en_housing'] + property_override_cols]
    # TODO: continue more validation here

    
