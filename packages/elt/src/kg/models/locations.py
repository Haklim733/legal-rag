from __future__ import annotations

from typing import Optional, Literal

from .base import LegalEntity

# D. Jurisdiction Types
class Jurisdiction(LegalEntity):
    entity_type: Literal["Jurisdiction"] = "Jurisdiction"

class Country(Jurisdiction):
    entity_type: Literal["Country"] = "Country"
    country_code: Optional[str] = None  # Example: US, CA

class StateProvince(Jurisdiction):
    entity_type: Literal["StateProvince"] = "StateProvince"
    state_code: Optional[str] = None  # Example: CA, NY

class CountyMunicipality(Jurisdiction):
    entity_type: Literal["CountyMunicipality"] = "CountyMunicipality"
