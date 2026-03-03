"""
Pydantic models for LexIR REST endpoints.
"""

from typing import Optional
from pydantic import BaseModel


class FIRInput(BaseModel):
    """Schema for incoming FIR data (REST + WebSocket)."""
    fir_id: str = "UNKNOWN"
    date: Optional[str] = None
    complainant_name: Optional[str] = None
    accused_names: list = []
    victim_name: Optional[str] = None
    incident_description: str = ""
    victim_impact: str = ""
    evidence: str = ""
    location: str = ""
    police_station: str = ""


class FIRPdfRequest(BaseModel):
    """Request body for PDF generation — FIR input + optional analysis output."""
    fir: dict
    analysis: Optional[dict] = None
