"""Cascade 6-1-6 — Pydantic request/response models."""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    anchor: str
    k: int = 5
    steps: int = 6


class NeighborItem(BaseModel):
    token: str = ""
    count: int = 0
    symbol: str = ""
    position: int = 0


class PredictResponse(BaseModel):
    source: str = "cascade_616"
    anchor: str = ""
    predictions: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None


class ChainResponse(BaseModel):
    source: str = "cascade_616"
    anchor: str = ""
    direction: str = ""
    chain: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None


class ContextResponse(BaseModel):
    source: str = "cascade_616"
    anchor: str = ""
    symbol: str = ""
    display: str = ""
    category: str = ""
    total_count: int = 0
    positions: Dict[str, List[Dict[str, Any]]] = {}
    error: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    version: str = ""
    enabled: bool = False
    cells: int = 0
    store_path: str = ""
    error: Optional[Dict[str, Any]] = None
