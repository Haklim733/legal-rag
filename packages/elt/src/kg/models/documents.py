from __future__ import annotations

from datetime import date
from typing import Optional, Literal

from .base import LegalEntity

# A. Document Types
class LegalDocument(LegalEntity):
    entity_type: Literal["LegalDocument"] = "LegalDocument"
    effective_date: Optional[date] = None

class Contract(LegalDocument):
    entity_type: Literal["Contract"] = "Contract"
    signed_on: Optional[date] = None

class EmploymentAgreement(Contract):
    entity_type: Literal["EmploymentAgreement"] = "EmploymentAgreement"

class LeaseAgreement(Contract):
    entity_type: Literal["LeaseAgreement"] = "LeaseAgreement"

class LitigationDocument(LegalDocument):
    entity_type: Literal["LitigationDocument"] = "LitigationDocument"

class Pleading(LitigationDocument):
    entity_type: Literal["Pleading"] = "Pleading"

class Complaint(Pleading):
    entity_type: Literal["Complaint"] = "Complaint"

class Answer(Pleading):
    entity_type: Literal["Answer"] = "Answer"

class Motion(LitigationDocument):
    entity_type: Literal["Motion"] = "Motion"

class LegislativeDocument(LegalDocument):
    entity_type: Literal["LegislativeDocument"] = "LegislativeDocument"

class Statute(LegislativeDocument):
    entity_type: Literal["Statute"] = "Statute"

class Regulation(LegislativeDocument):
    entity_type: Literal["Regulation"] = "Regulation"
