from typing import Optional
from pydantic import BaseModel

class NakordoniCarMapping(BaseModel):
    inbound_id: Optional[str] = None
    outbound_id: Optional[str] = None

class NakordoniMapping(BaseModel):
    car: Optional[NakordoniCarMapping] = None

class AIHeuristics(BaseModel):
    throughput: Optional[int] = None
    landmark_rules: Optional[str] = None

class ConfigMatrix(BaseModel):
    nakordoni: Optional[NakordoniMapping] = None
    ai_heuristics: Optional[AIHeuristics] = None