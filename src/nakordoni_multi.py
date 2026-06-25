import os
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from pydantic import BaseModel

logger = logging.getLogger("nakordoni_multi")

class NakordoniMultiUpdateInfo(BaseModel):
    found: bool
    timestamp: Optional[int] = None
    datetime: Optional[str] = None
    timezone: Optional[str] = None
    age_seconds: Optional[int] = None
    age_minutes: Optional[int] = None
    freshness: Optional[str] = None
    source: Optional[str] = None
    source_category: Optional[str] = None
    source_label_en: Optional[str] = None
    queue_now: Optional[int] = None

class NakordoniMultiQueue(BaseModel):
    found: bool
    queue_now: Optional[int] = None
    wait_min: Optional[int] = None
    tmin: Optional[int] = None
    tpercar: Optional[int] = None
    age_min: Optional[int] = None
    freshness: Optional[str] = None
    source: Optional[str] = None
    name: Optional[str] = None
    updated_at: Optional[str] = None

class NakordoniMultiDataNode(BaseModel):
    update_info: Optional[NakordoniMultiUpdateInfo] = None
    queue: Optional[NakordoniMultiQueue] = None

class NakordoniMultiMeta(BaseModel):
    ppids_requested: int
    include: List[str]
    units_consumed: int
    lang: str

class NakordoniMultiUsage(BaseModel):
    limit: int
    used: int
    reset: str

class NakordoniMultiResponse(BaseModel):
    ok: bool
    api_version: str
    product: str
    attribution: str
    meta: NakordoniMultiMeta
    data: Dict[str, NakordoniMultiDataNode]
    usage: Optional[NakordoniMultiUsage] = None

def fetch_nakordoni_multi_data(ppids: List[str], includes: List[str] = None) -> Dict[str, NakordoniMultiDataNode]:
    """
    Pulls border queue data from Nakordoni multi API using the NAKORDONI_API_KEY env var.
    Accepts a list of string ppids (e.g. ['id_345', 'id_344']).
    Returns a dictionary of NakordoniMultiDataNode objects keyed by their ppid.
    """
    if not ppids:
        return {}

    api_key = os.getenv("NAKORDONI_API_KEY")
    if not api_key:
        logger.error("Failed to load Nakordoni API key: NAKORDONI_API_KEY is not set in the .env file.")
        return {}

    if includes is None:
        includes = ["queue", "update-info"]

    ppids_str = ",".join(ppids)
    include_str = ",".join(includes)
    url = f"https://nakordoni.eu/api/v1/data/multi?ppids={ppids_str}&include={include_str}&lang=en"

    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    })
    
    try:
        with urllib.request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)
            parsed_response = NakordoniMultiResponse(**data)
            
            if not parsed_response.ok:
                logger.error("Nakordoni multi API returned ok=false")
                return {}
                
            return parsed_response.data
            
    except urllib.error.URLError as e:
        logger.error(f"Error fetching Nakordoni multi data: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error parsing Nakordoni multi data: {e}")
        return {}
