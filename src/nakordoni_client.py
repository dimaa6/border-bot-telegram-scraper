import os
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from supabase import Client
from config_matrix import ConfigMatrix
from supabase_client import get_active_country_prefixes

logger = logging.getLogger("nakordoni_client")

UKRAINE_BORDER_ID = 1
CAR_CROSSING_TYPE = 4

# Two-letter checkpoint_id prefix -> Nakordoni border id, per Nakordoni's v4 migration notice.
COUNTRY_BORDER_IDS = {
    "PL": 2,
    "SK": 3,
    "HU": 4,
    "RO": 5,
    "MD": 6,
}

class NakordoniCheckpoint(BaseModel):
    model_config = {"populate_by_name": True}

    ppid: str
    name: str
    border: int
    border_name: str
    queue: Optional[int] = None
    wait_min: Optional[int] = None
    traffic_status: Optional[str] = Field(default=None, alias="wait_status")
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

def _fetch_border_checkpoints(origin: int, destination: int, api_key: str) -> List[NakordoniCheckpoint]:
    """Issues a single v4 border call for one origin/destination pair and returns its checkpoints."""
    url = f"https://nakordoni.eu/api/v4/data/border/{origin}/{destination}/{CAR_CROSSING_TYPE}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    })

    try:
        with urllib.request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)
            parsed_response = NakordoniResponse.model_validate(data)

            if not parsed_response.ok or not parsed_response.data.ok:
                logger.error(f"Nakordoni API returned ok=false for border/{origin}/{destination}/{CAR_CROSSING_TYPE}")
                return []

            checkpoints = parsed_response.data.checkpoints
            logger.info(f"Nakordoni border/{origin}/{destination}/{CAR_CROSSING_TYPE} returned {len(checkpoints)} checkpoint(s).")
            return checkpoints

    except urllib.error.URLError as e:
        logger.error(f"Error fetching Nakordoni data for border/{origin}/{destination}/{CAR_CROSSING_TYPE}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing Nakordoni data for border/{origin}/{destination}/{CAR_CROSSING_TYPE}: {e}")
        return []

def fetch_nakordoni_data(supabase: Client) -> Dict[str, NakordoniCheckpoint]:
    """
    Pulls border queue data from Nakordoni API (v4) using the NAKORDONI_API_KEY env var.

    v4 dropped the destination=all wildcard and is no longer bidirectional, so this issues
    one outbound (Ukraine -> country) and one inbound (country -> Ukraine) call per neighbouring
    country derived from the active, non-closed checkpoints in Supabase config, then merges
    every checkpoint returned into a single dictionary keyed by ppid.
    """
    api_key = os.getenv("NAKORDONI_API_KEY")
    if not api_key:
        logger.error("Failed to load Nakordoni API key: NAKORDONI_API_KEY is not set in the .env file.")
        return {}

    prefixes = get_active_country_prefixes(supabase)
    border_ids = {COUNTRY_BORDER_IDS[prefix] for prefix in prefixes if prefix in COUNTRY_BORDER_IDS}

    if not border_ids:
        logger.error("No known neighbouring countries found among active checkpoints.")
        return {}

    result: Dict[str, NakordoniCheckpoint] = {}
    for border_id in sorted(border_ids):
        outbound_checkpoints = _fetch_border_checkpoints(UKRAINE_BORDER_ID, border_id, api_key)
        inbound_checkpoints = _fetch_border_checkpoints(border_id, UKRAINE_BORDER_ID, api_key)
        for cp in outbound_checkpoints + inbound_checkpoints:
            result[cp.ppid] = cp

    return result

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