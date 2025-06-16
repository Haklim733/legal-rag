from __future__ import annotations

from typing import Literal
# from uuid import UUID # Potentially for type hinting source/target if needed later

from .base import Relationship
# Import specific entity types if you want to type hint source_id/target_id, e.g.:
# from .documents import LegalDocument, Contract
# from .actors import Party, Attorney, Plaintiff, Defendant
# from .events_cases import LegalCase, LegalEvent
# from .locations import Jurisdiction
# from .objects import PropertyAsset, LegalRight, LegalObligation
# from .concepts import LegalConcept

class HasParty(Relationship):
    relationship_type: Literal["hasParty"] = "hasParty"
    # source_id: LegalDocument.id
    # target_id: Party.id

class RepresentedBy(Relationship):
    relationship_type: Literal["representedBy"] = "representedBy"
    # source_id: Party.id
    # target_id: Attorney.id

class PlaintiffIn(Relationship):
    relationship_type: Literal["plaintiffIn"] = "plaintiffIn"
    # source_id: Plaintiff.id
    # target_id: LegalCase.id

class DefendantIn(Relationship):
    relationship_type: Literal["defendantIn"] = "defendantIn"
    # source_id: Defendant.id
    # target_id: LegalCase.id

class AgentOf(Relationship):
    relationship_type: Literal["agentOf"] = "agentOf"
    # source_id: Party.id
    # target_id: Party.id

class Amends(Relationship):
    relationship_type: Literal["amends"] = "amends"
    # source_id: LegalDocument.id
    # target_id: LegalDocument.id

class Cites(Relationship):
    relationship_type: Literal["cites"] = "cites"
    # source_id: LegalDocument.id
    # target_id: LegalDocument.id # or Statute, Regulation, LegalCase

class Governs(Relationship):
    relationship_type: Literal["governs"] = "governs"
    # source_id: Contract.id
    # target_id: Party.id # Actually List[Party.id]

class RelatesTo(Relationship):
    relationship_type: Literal["relatesTo"] = "relatesTo"
    # source_id: LegalDocument.id
    # target_id: LegalDocument.id

class IsAmendmentTo(Relationship):
    relationship_type: Literal["isAmendmentTo"] = "isAmendmentTo"
    # source_id: LegalDocument.id
    # target_id: LegalDocument.id

# Relationships where target is not an entity ID are better modeled as attributes
# on the source entity itself (e.g., LegalEvent.event_date).
# If you still want a relationship object for them, their target_id would not be a UUID.

class OccurredOn(Relationship):
    relationship_type: Literal["occurredOn"] = "occurredOn"
    # source_id: LegalEvent.id
    # target_id: date # This target is not an entity ID.

class ResultedIn(Relationship):
    relationship_type: Literal["resultedIn"] = "resultedIn"
    # source_id: LegalEvent.id
    # target_id: LegalEvent.id

class FiledBy(Relationship):
    relationship_type: Literal["filedBy"] = "filedBy"
    # source_id: LegalDocument.id
    # target_id: Party.id

class LocatedIn(Relationship):
    relationship_type: Literal["locatedIn"] = "locatedIn"
    # source_id: PropertyAsset.id
    # target_id: Jurisdiction.id

class EffectiveDate(Relationship):
    relationship_type: Literal["effectiveDate"] = "effectiveDate"
    # source_id: LegalDocument.id
    # target_id: date

class SignedOn(Relationship):
    relationship_type: Literal["signedOn"] = "signedOn"
    # source_id: Contract.id
    # target_id: date

class Defines(Relationship):
    relationship_type: Literal["defines"] = "defines"
    # source_id: Union[LegalDocument.id, str] # 'term' could be a string or a specific entity
    # target_id: LegalConcept.id

class Grants(Relationship):
    relationship_type: Literal["grants"] = "grants"
    # source_id: Party.id (Party1)
    # target_id: Party.id (Party2)
    # properties: {"granted_right_id": LegalRight.id}

class ImposesObligation(Relationship):
    relationship_type: Literal["imposesObligation"] = "imposesObligation"
    # source_id: Party.id # The party on whom the obligation is imposed
    # target_id: LegalObligation.id

class IsTypeOf(Relationship):
    relationship_type: Literal["isTypeOf"] = "isTypeOf"
    # source_id: LegalConcept.id
    # target_id: LegalConcept.id
