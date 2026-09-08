import os
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from config_matrix import ConfigMatrix

logger = logging.getLogger("nakordoni_client")

UKRAINE_BORDER_ID = 1
CAR_CROSSING_TYPE = 4
INTER_CALL_DELAY_SECONDS = 10

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

class NakordoniClient:
    """
    Lazily pulls border queue data from Nakordoni API (v4) using the NAKORDONI_API_KEY env var.

    v4 dropped the destination=all wildcard and is no longer bidirectional, so a full outbound
    (Ukraine -> country) plus inbound (country -> Ukraine) pair of calls is only made the first
    time a checkpoint for that country is encountered; the merged result is cached per border id
    for the rest of the run so later checkpoints of the same country reuse it. A short delay is
    kept between the outbound and inbound calls of a pair to stay clear of Nakordoni's rate limit.
    """

    def __init__(self):
        self._cache: Dict[int, Dict[str, NakordoniCheckpoint]] = {}

    def get_country_data(self, border_id: int) -> Dict[str, NakordoniCheckpoint]:
        if border_id in self._cache:
            return self._cache[border_id]

        api_key = os.getenv("NAKORDONI_API_KEY")
        if not api_key:
            logger.error("Failed to load Nakordoni API key: NAKORDONI_API_KEY is not set in the .env file.")
            return {}

        outbound_checkpoints = _fetch_border_checkpoints(UKRAINE_BORDER_ID, border_id, api_key)
        time.sleep(INTER_CALL_DELAY_SECONDS)
        inbound_checkpoints = _fetch_border_checkpoints(border_id, UKRAINE_BORDER_ID, api_key)

        merged = {cp.ppid: cp for cp in outbound_checkpoints + inbound_checkpoints}
        self._cache[border_id] = merged
        return merged

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