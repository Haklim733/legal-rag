from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime # Import datetime if date fields are needed later

# It's good practice to define shared elements or enums if any are identified
# For now, we'll start with basic models.

class FolioModel(BaseModel):
    """Base model for FOLIO entities, can include common fields like id or source."""
    id: Optional[str] = Field(None, description="Unique identifier for the entity, often its URI.")
    label: Optional[str] = Field(None, description="Human-readable label for the entity.")
    # Add other common fields if identified, e.g., source_uri, description

class Court(FolioModel):
    """
    Represents a court entity from the FOLIO ontology.
    URI: https://folio.openlegalstandard.org/Court
    This is a placeholder and should be expanded with its actual properties.
    """
    name: Optional[str] = Field(None, description="Name of the court.")
    # Example properties to be discovered from FOLIO:
    # jurisdiction: Optional[str] = None # Or a 'Jurisdiction' model
    # court_level: Optional[str] = None # e.g., trial, appellate, supreme

    class Config:
        json_schema_extra = {
            "example": {
                "id": "https://folio.openlegalstandard.org/example_court_123",
                "label": "Example Circuit Court",
                "name": "Example Circuit Court",
            }
        }

class Case(FolioModel):
    """
    Represents a legal case from the FOLIO ontology.
    URI: https://folio.openlegalstandard.org/Case
    """
    case_name: Optional[str] = Field(None, alias="hasCaseName", description="The official name of the case.")
    docket_number: Optional[str] = Field(None, alias="hasDocketNumber", description="The docket number assigned to the case.")
    court: Optional[Court] = Field(None, alias="hasCourt", description="The court in which the case was heard or is pending.")
    
    # Other potential fields to be discovered from FOLIO:
    # parties: List[Party] = Field(default_factory=list)
    # decision_date: Optional[datetime] = None
    # summary: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "https://folio.openlegalstandard.org/example_case_456",
                "label": "Smith v. Jones (Example Case)",
                "case_name": "Smith v. Jones",
                "docket_number": "CV-2023-001",
                "court": {
                    "id": "https://folio.openlegalstandard.org/example_court_123",
                    "label": "Example Circuit Court",
                    "name": "Example Circuit Court"
                }
            }
        }
        # If you use aliases for field names that don't match Pydantic/Python conventions,
        # you might need:
        # populate_by_name = True 
        # to allow Pydantic to use the alias for both serialization and deserialization if needed.
        # However, for model creation, using Pythonic names and mapping during data ingestion
        # (e.g. via alias) is common.

# TODO: Define other models like Party, LegalDocument, Jurisdiction, etc.
# TODO: Discover and add more properties to existing models based on FOLIO.
# TODO: Define Enums for controlled vocabularies if found in FOLIO (e.g., case types, roles).