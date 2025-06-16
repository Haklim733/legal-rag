from __future__ import annotations

import uuid
from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

class LegalEntity(BaseModel):
    """Base class for all legal entities in the knowledge graph."""
    id: UUID = Field(default_factory=uuid.uuid4)
    entity_type: str = Field(..., description="The specific type of the legal entity.")
    name: Optional[str] = None
    description: Optional[str] = None

    class Config:
        use_enum_values = True

class Relationship(BaseModel):
    """Defines a generic relationship between two legal entities."""
    id: UUID = Field(default_factory=uuid.uuid4)
    source_id: UUID = Field(..., description="ID of the source entity")
    target_id: UUID = Field(..., description="ID of the target entity")
    relationship_type: str = Field(..., description="Type of the relationship")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    properties: Optional[dict] = None
