from __future__ import annotations

from typing import Optional, Literal
from uuid import UUID

from .base import LegalEntity

class PropertyAsset(LegalEntity):
    entity_type: Literal["PropertyAsset"] = "PropertyAsset"
    location_jurisdiction_id: Optional[UUID] = None

class LegalRight(LegalEntity):
    entity_type: Literal["LegalRight"] = "LegalRight"

class LegalObligation(LegalEntity):
    entity_type: Literal["LegalObligation"] = "LegalObligation"
