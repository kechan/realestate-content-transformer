from typing import Dict, List, Optional, Union

from pathlib import Path

from .pipeline import LocallogicContentRewriter, LLM, LIGHT_WEIGHT_LLM

class RewriteValidator:
  def __init__(self, es_host: str, es_port=9200, working_dir: Path = '.') -> None:
    self.es_host = es_host
    self.es_port = es_port
    cheap_dummy_llm_model = LIGHT_WEIGHT_LLM
    self.ll_rewriter = LocallogicContentRewriter(es_host=self.es_host, 
                                                 es_port=self.es_port,
                                                 llm_model=cheap_dummy_llm_model,
                                                 archiver_filepath=working_dir/'rewrites.txt',
                                                 )
    
    # TODO: better manage how save_dir is done, just use local for now
    self.ll_rewriter.extract_content(save_intermediate_results=True, save_dir=working_dir)

  def validate_overrides(self):
    property_types = self.ll_rewriter.property_types
    geo_all_content_df = self.ll_rewriter.geo_all_content_df

    # Looking at housing for lang='en' for now.
    geo_all_content_df.q("en_housing.notnull()")[['geog_id', 'longId', 'en_housing', 'overrides_en_housing']]
    # TODO: continue more validation here

    # Looking at housing for lang='en' for all property types
    property_override_cols = [f"overrides_{property_type.lower().replace('-', '_')}_en_housing" for property_type in property_types]

    for property_override_col in property_override_cols:
      geo_all_content_df.q(f"{property_override_col}.notnull()")[['geog_id', 'longId', property_override_col]]

    # TODO: continue more validation here

    
