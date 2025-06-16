from __future__ import annotations

from datetime import date
from typing import Optional, Literal
from uuid import UUID

from .base import LegalEntity

class LegalCase(LegalEntity):
    entity_type: Literal["LegalCase"] = "LegalCase"
    case_number: Optional[str] = None
    court_id: Optional[UUID] = None  # Link to the court it's filed in

class LegalEvent(LegalEntity):
    entity_type: Literal["LegalEvent"] = "LegalEvent"
    event_date: Optional[date] = None
