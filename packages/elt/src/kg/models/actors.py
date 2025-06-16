from __future__ import annotations

from typing import Optional, Literal

from .base import LegalEntity

# C. Role Types (Representing individuals or entities fulfilling these roles)
class Party(LegalEntity):
    entity_type: Literal["Party"] = "Party"

class LegalProfessional(Party):
    entity_type: Literal["LegalProfessional"] = "LegalProfessional"
    bar_number: Optional[str] = None

class Judge(LegalProfessional):
    entity_type: Literal["Judge"] = "Judge"

class Attorney(LegalProfessional):
    entity_type: Literal["Attorney"] = "Attorney"

class LitigationParty(Party):
    entity_type: Literal["LitigationParty"] = "LitigationParty"

class Plaintiff(LitigationParty):
    entity_type: Literal["Plaintiff"] = "Plaintiff"

class Defendant(LitigationParty):
    entity_type: Literal["Defendant"] = "Defendant"
