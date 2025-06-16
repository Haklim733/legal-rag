from __future__ import annotations

from typing import Optional, Literal
from uuid import UUID

from .base import LegalEntity

# E. Court System Types
class Court(LegalEntity):
    entity_type: Literal["Court"] = "Court"
    jurisdiction_id: Optional[UUID] = None  # Link to its jurisdiction

class FederalCourt(Court):
    entity_type: Literal["FederalCourt"] = "FederalCourt"

class StateCourt(Court):
    entity_type: Literal["StateCourt"] = "StateCourt"

class AppellateCourt(Court):  # Can be Federal or State
    entity_type: Literal["AppellateCourt"] = "AppellateCourt"

class TrialCourt(Court):  # Can be Federal or State
    entity_type: Literal["TrialCourt"] = "TrialCourt"
