from __future__ import annotations

from typing import Literal

from .base import LegalEntity

# B. Legal Area/Topic Types (as Concepts)
class LegalConcept(LegalEntity):
    entity_type: Literal["LegalConcept"] = "LegalConcept"

class LawConcept(LegalConcept):
    entity_type: Literal["LawConcept"] = "LawConcept"

class CivilLawConcept(LawConcept):
    entity_type: Literal["CivilLawConcept"] = "CivilLawConcept"

class ContractLawConcept(CivilLawConcept):
    entity_type: Literal["ContractLawConcept"] = "ContractLawConcept"

class TortLawConcept(CivilLawConcept):
    entity_type: Literal["TortLawConcept"] = "TortLawConcept"

class CriminalLawConcept(LawConcept):
    entity_type: Literal["CriminalLawConcept"] = "CriminalLawConcept"

class ConstitutionalLawConcept(LawConcept):
    entity_type: Literal["ConstitutionalLawConcept"] = "ConstitutionalLawConcept"
