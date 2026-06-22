from typing import Optional
from pydantic import BaseModel

class NakordoniCarMapping(BaseModel):
    inbound_id: Optional[str] = None
    outbound_id: Optional[str] = None

class NakordoniMapping(BaseModel):
    car: Optional[NakordoniCarMapping] = None

class AIHeuristics(BaseModel):
    inbound_throughput: Optional[int] = None
    outbound_throughput: Optional[int] = None
    territory_capacity: Optional[int] = None  # Physical car slots inside checkpoint territory
    landmark_rules: Optional[str] = None

class ConfigMatrix(BaseModel):
    nakordoni: Optional[NakordoniMapping] = None
    ai_heuristics: Optional[AIHeuristics] = None