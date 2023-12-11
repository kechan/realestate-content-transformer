from typing import Dict, List, Optional, Union

import json
from dataclasses import dataclass, field, asdict

@dataclass
class BaseDoc:
  def prettify(self):
    return json.dumps(asdict(self), indent=4, sort_keys=True, default=str)

@dataclass
class GeoDoc(BaseDoc):
  geog_id: str
  longId: str
#   province: Optional[str] = None
  city: Optional[str] = None
  citySlug: Optional[str] = None
#   localLogicId: Optional[List[str]] = field(default_factory=list)

@dataclass
class Profile(BaseDoc):
  housing: Optional[str] = None
  transport: Optional[str] = None
  services: Optional[str] = None
  character: Optional[str] = None
  

@dataclass
class Data(BaseDoc):
  name: Optional[str] = None
  province: Optional[str] = None
  profiles: Profile = field(default_factory=Profile)

@dataclass
class Overrides(BaseDoc):
  data: Data = field(default_factory=Data)
  
@dataclass
class GeoDetailDoc(BaseDoc):
  geog_id: str
  data: Data = field(default_factory=Data)
  # province: Optional[str] = None
  # profiles: Optional[Profile] = None
  # name: Optional[str] = None

@dataclass
class GeoOverridesDoc(BaseDoc):
  longId: str
  localLogicId: Optional[List[str]] = field(default_factory=list)

  overrides: Optional[Dict[str, Union[bool, str]]] = field(default_factory=dict)

  overrides_en: Optional[Overrides] = None #= field(default_factory=Overrides)
  overrides_fr: Optional[Overrides] = None  #= field(default_factory=Overrides)
