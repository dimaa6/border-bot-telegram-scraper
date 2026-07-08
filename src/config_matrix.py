from pydantic import BaseModel

class NakordoniCarMapping(BaseModel):
    inbound_id: str | None = None
    outbound_id: str | None = None

class NakordoniMapping(BaseModel):
    car: NakordoniCarMapping | None = None

class LandmarkRules(BaseModel):
    inbound: dict[str, int] | None = None
    outbound: dict[str, int] | None = None

class AIHeuristics(BaseModel):
    inbound_throughput: int | None = None
    outbound_throughput: int | None = None
    territory_capacity: int | None = None  # Physical car slots inside checkpoint territory
    landmark_rules: LandmarkRules | None = None
    landmark_mapping: dict[str, list[str]] | None = None
    segment_mode: str | None = None

class ConfigMatrix(BaseModel):
    nakordoni: NakordoniMapping | None = None
    ai_heuristics: AIHeuristics | None = None
