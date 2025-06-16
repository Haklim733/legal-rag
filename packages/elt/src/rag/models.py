"""
Data models for the knowledge graph.
"""
from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field

class SupremeCourtCase(BaseModel):
    """Represents a U.S. Supreme Court case in the knowledge graph.
    
    Attributes:
        dcid: Unique identifier for the case
        name: Name of the case (e.g., "Roe v. Wade")
        description: Brief description of the case
        date_decided: Date the case was decided (YYYY-MM-DD format)
        citation: Official citation (e.g., "410 U.S. 113")
        parties: Parties involved in the case
        decision_direction: Direction of the decision (e.g., "affirmed", "reversed")
        opinion_author: Name of the justice who wrote the majority opinion
        cites: List of case DCIDs that this case cites
    """
    dcid: str = Field(..., description="Unique identifier for the case")
    name: str = Field(..., description="Name of the case")
    description: str = Field(..., description="Brief description of the case")
    date_decided: str = Field(..., description="Date decided in YYYY-MM-DD format")
    citation: Optional[str] = Field(None, description="Official case citation")
    parties: Optional[str] = Field(None, description="Parties involved in the case")
    decision_direction: Optional[str] = Field(None, description="Direction of the decision")
    opinion_author: Optional[str] = Field(None, description="Author of the majority opinion")
    cites: List[str] = Field(default_factory=list, description="List of cited case DCIDs")

    class Config:
        json_schema_extra = {
            "example": {
                "dcid": "dcid:case1",
                "name": "Roe v. Wade",
                "description": "Landmark decision on abortion rights",
                "date_decided": "1973-01-22",
                "citation": "410 U.S. 113",
                "parties": "Jane Roe, et al. v. Henry Wade",
                "decision_direction": "reversed",
                "opinion_author": "Harry Blackmun",
                "cites": ["dcid:case2"]
            }
        }
