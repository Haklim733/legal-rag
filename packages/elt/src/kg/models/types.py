from __future__ import annotations

from typing import Union

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

AnyLegalEntity = Union[
    LegalDocument, Contract, EmploymentAgreement, LeaseAgreement, LitigationDocument,
    Pleading, Complaint, Answer, Motion, LegislativeDocument, Statute, Regulation,
    LegalConcept, LawConcept, CivilLawConcept, ContractLawConcept, TortLawConcept,
    CriminalLawConcept, ConstitutionalLawConcept,
    Party, LegalProfessional, Judge, Attorney, LitigationParty, Plaintiff, Defendant,
    Jurisdiction, Country, StateProvince, CountyMunicipality,
    Court, FederalCourt, StateCourt, AppellateCourt, TrialCourt,
    LegalCase, LegalEvent,
    PropertyAsset, LegalRight, LegalObligation
]
