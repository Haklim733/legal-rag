# flake8: noqa
"""Pydantic models for the Legal Knowledge Graph."""

from .base import LegalEntity, Relationship

from .documents import (
    LegalDocument, Contract, EmploymentAgreement, LeaseAgreement, LitigationDocument,
    Pleading, Complaint, Answer, Motion, LegislativeDocument, Statute, Regulation
)
from .concepts import (
    LegalConcept, LawConcept, CivilLawConcept, ContractLawConcept, TortLawConcept,
    CriminalLawConcept, ConstitutionalLawConcept
)
from .actors import (
    Party, LegalProfessional, Judge, Attorney, LitigationParty, Plaintiff, Defendant
)
from .locations import (
    Jurisdiction, Country, StateProvince, CountyMunicipality
)
from .courts import (
    Court, FederalCourt, StateCourt, AppellateCourt, TrialCourt
)
from .events_cases import (
    LegalCase, LegalEvent
)
from .objects import (
    PropertyAsset, LegalRight, LegalObligation
)

from .specific_relationships import (
    HasParty, RepresentedBy, PlaintiffIn, DefendantIn, AgentOf,
    Amends, Cites, Governs, RelatesTo, IsAmendmentTo,
    OccurredOn, ResultedIn, FiledBy, LocatedIn, EffectiveDate, SignedOn,
    Defines, Grants, ImposesObligation, IsTypeOf
)

from .types import AnyLegalEntity

__all__ = [
    "LegalEntity",
    "Relationship",
    "LegalDocument",
    "Contract",
    "EmploymentAgreement",
    "LeaseAgreement",
    "LitigationDocument",
    "Pleading",
    "Complaint",
    "Answer",
    "Motion",
    "LegislativeDocument",
    "Statute",
    "Regulation",
    "LegalConcept",
    "LawConcept",
    "CivilLawConcept",
    "ContractLawConcept",
    "TortLawConcept",
    "CriminalLawConcept",
    "ConstitutionalLawConcept",
    "Party",
    "LegalProfessional",
    "Judge",
    "Attorney",
    "LitigationParty",
    "Plaintiff",
    "Defendant",
    "Jurisdiction",
    "Country",
    "StateProvince",
    "CountyMunicipality",
    "Court",
    "FederalCourt",
    "StateCourt",
    "AppellateCourt",
    "TrialCourt",
    "LegalCase",
    "LegalEvent",
    "PropertyAsset",
    "LegalRight",
    "LegalObligation",
    "HasParty",
    "RepresentedBy",
    "PlaintiffIn",
    "DefendantIn",
    "AgentOf",
    "Amends",
    "Cites",
    "Governs",
    "RelatesTo",
    "IsAmendmentTo",
    "OccurredOn",
    "ResultedIn",
    "FiledBy",
    "LocatedIn",
    "EffectiveDate",
    "SignedOn",
    "Defines",
    "Grants",
    "ImposesObligation",
    "IsTypeOf",
    "AnyLegalEntity",
]
