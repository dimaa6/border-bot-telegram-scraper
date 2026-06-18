import os
import tomllib
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from pydantic import BaseModel
from config_matrix import ConfigMatrix

logger = logging.getLogger("nakordoni_client")

class NakordoniCheckpoint(BaseModel):
    ppid: str
    name: str
    border: int
    border_name: str
    queue: Optional[int] = None
    wait_min: Optional[int] = None
    traffic_status: Optional[str] = None
    updated_at: Optional[str] = None
    age_min: Optional[int] = None
    source_url: Optional[str] = None

class NakordoniData(BaseModel):
    ok: bool
    origin: int
    origin_name: str
    destinations: List[int]
    destination_names: List[str]
    crossing_type: int
    crossing_type_label: str
    count: int
    checkpoints: List[NakordoniCheckpoint]

class NakordoniResponse(BaseModel):
    ok: bool
    api_version: str
    product: str
    attribution: str
    data: NakordoniData

def fetch_nakordoni_data() -> Dict[str, NakordoniCheckpoint]:
    """
    Pulls border queue data from Nakordoni API using the API key from scraper_config.toml.
    Returns a dictionary of NakordoniCheckpoint objects keyed by their ppid.
    """
    config_path = os.path.join("config", "scraper_config.toml")
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        api_key = config_data["nakordoni"]["api_key"]
    except Exception as e:
        logger.error(f"Failed to load Nakordoni API key from config: {e}")
        return {}

    url = "https://nakordoni.eu/api/v1/data/border/1/all/4"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    })
    
    try:
        with urllib.request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)
            parsed_response = NakordoniResponse(**data)
            
            if not parsed_response.ok or not parsed_response.data.ok:
                logger.error("Nakordoni API returned ok=false")
                return {}
                
            return {cp.ppid: cp for cp in parsed_response.data.checkpoints}
            
    except urllib.error.URLError as e:
        logger.error(f"Error fetching Nakordoni data: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error parsing Nakordoni data: {e}")
        return {}

def match_checkpoint_with_nakordoni(config_matrix: ConfigMatrix, nakordoni_data: Dict[str, NakordoniCheckpoint]) -> Dict[str, Optional[NakordoniCheckpoint]]:
    """
    Matches the inbound and outbound ppids from the config_matrix
    against the fetched Nakordoni data.
    """
    nakordoni_mapping = config_matrix.nakordoni
    
    inbound_id = nakordoni_mapping.car.inbound_id if nakordoni_mapping and nakordoni_mapping.car else None
    outbound_id = nakordoni_mapping.car.outbound_id if nakordoni_mapping and nakordoni_mapping.car else None
    
    return {
        "INBOUND": nakordoni_data.get(inbound_id) if inbound_id else None,
        "OUTBOUND": nakordoni_data.get(outbound_id) if outbound_id else None
    }