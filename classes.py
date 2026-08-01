from pydantic import BaseModel
from typing import List


class MatchRequest(BaseModel):
    messages: List[str]


class Candidate(BaseModel):
    sku: str
    confidence: float


class MatchResult(BaseModel):
    message: str
    status: str
    candidates: List[Candidate]


class MatchResponse(BaseModel):
    results: List[MatchResult]
