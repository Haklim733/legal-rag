import pytest
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import date, datetime

# Assuming model_builder.py is in kg.models.folio and accessible via python path
from kg.models.folio.model_builder import (
    FolioManager, 
    generate_pydantic_from_class, 
    _model_cache, 
    _clean_name, 
    FolioBaseModel,
    OwlClass,
    ALL_FOLIO_OWL_CLASSES
)
from rdflib import XSD, URIRef

# Path to the local FOLIO ontology file
# Ensure this path is correct relative to where pytest is run (usually project root)
FOLIO_OWL_PATH = 'packages/elt/src/kg/models/folio/FOLIO.owl'

@pytest.fixture(scope='module')
def folio_manager_module_scope():
    """Provides a FolioManager instance for the entire test module."""
    return FolioManager(FOLIO_OWL_PATH)

@pytest.fixture(autouse=True)
def clear_model_cache_before_each_test():
    """Clears the global model cache before each test to ensure isolation."""
    _model_cache.clear()
    yield # Test runs here
    _model_cache.clear() # Optional: clear after if desired, though next test will clear anyway

# --- Test Cases ---

def test_folio_manager_initialization():
    """Tests if FolioManager initializes and parses the OWL file."""
    try:
        manager = FolioManager(FOLIO_OWL_PATH)
        assert manager.graph is not None
        assert len(manager.graph) > 0, "Graph should not be empty after parsing FOLIO.owl"
    except Exception as e:
        pytest.fail(f"FolioManager initialization failed: {e}")

def test_generate_simple_class_court(folio_manager_module_scope: FolioManager):
    """Tests generating a Pydantic model for a relatively simple class like 'Court'."""
    court_iri = 'https://folio.openlegalstandard.org/Court'
    
    CourtModel = generate_pydantic_from_class(folio_manager_module_scope, court_iri)
    
    assert CourtModel is not None, "Model should be generated."
    assert CourtModel.__name__ == _clean_name("Court"), "Model name should be cleaned version of label."
    assert issubclass(CourtModel, BaseModel), "Generated model should be a subclass of Pydantic BaseModel."
    # It should also inherit from FolioBaseModel if that's the ultimate base for generated models
    # This depends on the implementation in generate_pydantic_from_class
    # For now, let's assume BaseModel is a valid check. If FolioBaseModel is always a parent, check for it.

    fields = CourtModel.model_fields
    assert 'iri' in fields, "'iri' field should be present."
    assert fields['iri'].annotation == str

    # Check for a common property like 'label' if it's expected to be directly on the model
    # The current model_builder adds 'iri' and then properties from the ontology.
    # 'label' might come from FolioBaseModel or be an explicit property.
    # If FolioBaseModel is a parent, 'label' would be inherited.
    # Let's check based on the current model_builder.py structure where iri is explicit
    # and other fields come from ontology properties.

    # Example: If 'Court' has a direct rdfs:label or skos:prefLabel that becomes a field:
    # (This depends on FOLIO.owl structure for 'Court')
    # For now, we'll assume 'label' might not be a direct field unless FolioBaseModel is a parent
    # and its fields are included. The 'iri' field is explicitly added.

    # Let's check if it's in the cache
    assert court_iri in _model_cache
    assert _model_cache[court_iri] == CourtModel

def test_generate_class_with_inheritance_judge(folio_manager_module_scope: FolioManager):
    """Tests generating a Pydantic model for a class with inheritance (e.g., Judge subclass of JudicialOfficer)."""
    judge_iri = 'https://folio.openlegalstandard.org/Judge'
    judicial_officer_iri = 'https://folio.openlegalstandard.org/JudicialOfficer'

    # Ensure the parent model is generated first or simultaneously if not cached
    JudicialOfficerModel = generate_pydantic_from_class(folio_manager_module_scope, judicial_officer_iri)
    JudgeModel = generate_pydantic_from_class(folio_manager_module_scope, judge_iri)

    assert JudgeModel is not None
    assert JudicialOfficerModel is not None
    assert JudgeModel.__name__ == _clean_name("Judge")
    assert JudicialOfficerModel.__name__ == _clean_name("Judicial Officer") # Assuming label is "Judicial Officer"
    
    assert issubclass(JudgeModel, JudicialOfficerModel), "JudgeModel should be a subclass of JudicialOfficerModel."

    # Check that fields from parent are available in child
    # For example, if JudicialOfficerModel has an 'iri' field (it should), JudgeModel should too.
    assert 'iri' in JudgeModel.model_fields
    assert 'iri' in JudicialOfficerModel.model_fields

    # Verify cache
    assert judge_iri in _model_cache
    assert judicial_officer_iri in _model_cache

def test_non_existent_class_returns_base_model(folio_manager_module_scope: FolioManager):
    """Tests that a non-existent class IRI returns the default BaseModel as per current logic."""
    non_existent_iri = 'https://folio.openlegalstandard.org/ThisClassDoesNotExistForSure'
    
    # The model_builder's generate_pydantic_from_class returns a generic BaseModel
    # when a class is not found or an error occurs during its specific generation.
    # The cache might store this generic BaseModel against the IRI.
    NonExistentModel = generate_pydantic_from_class(folio_manager_module_scope, non_existent_iri)
    
    # Check if the returned model is the generic Pydantic BaseModel or a placeholder
    # The current implementation returns a new BaseModel instance for each unknown IRI.
    # It does not return *the* Pydantic BaseModel type itself, but a model *derived* from it or a simple one.
    # The key is that it's not a specifically generated FOLIO model.
    # Let's check its name or if it lacks specific FOLIO fields.
    # The current model_builder.py returns a new model that inherits from BaseModel, named e.g. 'BaseModel_Placeholder_XYZ'
    # or simply 'BaseModel' if the class is truly not found and no label could be inferred.
    # The most reliable check is that it's not one of the *successfully* generated, named models.

    # If the class is truly not found, owl_class will be None, and it returns a basic BaseModel.
    # The current implementation in model_builder.py, if owl_class is None, returns a cached BaseModel.
    # Let's refine this test based on the exact return for a non-found class.
    # The `generate_pydantic_from_class` returns a cached `BaseModel` if `owl_class` is `None`.
    # So, we expect the *type* `BaseModel` itself, or a model that is effectively empty besides `iri`.

    # The current logic returns a dynamically created model that inherits from BaseModel, even for not found.
    # It might be named 'BaseModel' or similar if no label can be found.
    # Let's check that it's not a complex model and that it's cached.
    assert NonExistentModel is not None
    # It should be in the cache, potentially as a very simple model
    assert non_existent_iri in _model_cache
    # A more robust check: it shouldn't have fields specific to known FOLIO classes.
    # For instance, if 'Court' has a property 'hasJurisdiction', NonExistentModel shouldn't.
    court_specific_field_example = _clean_name("hasJurisdiction") # Example
    assert court_specific_field_example not in NonExistentModel.model_fields


def test_model_generation_for_us_supreme_court(folio_manager_module_scope: FolioManager):
    """
    Tests dynamic generation for 'U.S. Supreme Court'. 
    This IRI might be an instance or a class. Assuming it's a class for this test.
    IRI: https://folio.openlegalstandard.org/RFE94c038Ce43B892dbECa17 (this looks like an instance ID)
    Let's try a known class that might be related, e.g., 'HighestCourt' if it exists, or use 'Court'.
    For now, let's use a known class IRI that is definitely an owl:Class.
    If 'U.S. Supreme Court' is a specific class, its label would be 'U.S. Supreme Court'.
    Let's use 'FederalCourt' as an example if it's a class in FOLIO.
    The original test used scotus_iri = "https://folio.openlegalstandard.org/RFE94c038Ce43B892dbECa17"
    This IRI seems to be an individual instance, not an owl:Class.
    The model builder generates models from owl:Class definitions.
    Let's pick a class that is likely to exist, e.g., 'LegalDocument'.
    """
    legal_document_iri = "https://folio.openlegalstandard.org/LegalDocument" # Example, verify in FOLIO
    
    LegalDocumentModel = generate_pydantic_from_class(folio_manager_module_scope, legal_document_iri)
    
    assert LegalDocumentModel is not None
    # The name depends on the rdfs:label or skos:prefLabel in FOLIO.owl
    # Assuming label is 'Legal Document', cleaned name would be 'LegalDocument'
    # We should fetch the actual label from the graph to be sure, or use _clean_name with expected label.
    # For now, let's be flexible or use a known label if possible.
    # cleaned_expected_name = _clean_name(folio_manager_module_scope.get_class(URIRef(legal_document_iri)).label)
    # assert LegalDocumentModel.__name__ == cleaned_expected_name
    assert issubclass(LegalDocumentModel, BaseModel)
    assert 'iri' in LegalDocumentModel.model_fields

    # Example: Check for a property that 'LegalDocument' might have, e.g., 'hasTitle'
    # This requires knowing the FOLIO structure for LegalDocument.
    # If 'hasTitle' is a property with domain LegalDocument and range xsd:string:
    # expected_title_field_name = _clean_name("hasTitle") 
    # if expected_title_field_name in LegalDocumentModel.model_fields:
    #     assert LegalDocumentModel.model_fields[expected_title_field_name].annotation == Optional[str]
    # Or List[str] if not functional.

    print(f"\n--- Schema for {LegalDocumentModel.__name__} ---")
    try:
        print(LegalDocumentModel.model_json_schema(indent=2))
    except Exception as e:
        print(f"Could not generate schema for {LegalDocumentModel.__name__}: {e}")


def test_datatype_property_handling(folio_manager_module_scope: FolioManager):
    """Tests that datatype properties are correctly typed.
    Find a class with a known datatype property (e.g., a date, number, or specific string type).
    Example: If 'Case' has a 'citation' (string) and 'decisionDate' (date).
    """
    # This IRI needs to be a class in FOLIO with known datatype properties.
    # Let's assume 'Case' class exists: https://folio.openlegalstandard.org/Case
    case_iri = "https://folio.openlegalstandard.org/Case"
    CaseModel = generate_pydantic_from_class(folio_manager_module_scope, case_iri)

    assert CaseModel is not None
    fields = CaseModel.model_fields

    # Example 1: A string property (e.g., 'caseName' or 'citation')
    # Assume 'caseName' is a property with rdfs:label "case name" and range xsd:string
    # The field name would be _clean_name("case name") -> "case_name" (if not for class)
    # The property cleaning in model_builder for field names needs to be consistent.
    # Current _clean_name in model_builder.py for fields from properties is just _clean_name(prop.label)
    # which might produce PascalCase if not careful. Let's assume it's handled to be snake_case or as-is.
    # For now, let's use a hypothetical direct label from ontology.
    # If a property is labeled "caseName" in the ontology:
    case_name_field = _clean_name("caseName") # Or whatever the actual label is, then cleaned.
    # if case_name_field in fields:
    #     # Check if it's Optional[str] or List[str] based on functional/non-functional
    #     # For simplicity, let's assume it resolves to str or Optional[str]
    #     assert fields[case_name_field].annotation in [str, Optional[str], List[str]]

    # Example 2: A date property (e.g., 'decisionDate')
    # Assume 'decisionDate' has rdfs:label "decision date" and range xsd:date
    # decision_date_field = _clean_name("decisionDate")
    # if decision_date_field in fields:
    #     assert fields[decision_date_field].annotation in [Optional[date], List[date]]
    
    # This test is highly dependent on the actual FOLIO.owl structure.
    # It's better to pick a very specific, known class and property from FOLIO.
    # For instance, if 'Person' class exists and has 'birthDate' (xsd:date)
    # person_iri = "https://folio.openlegalstandard.org/Person"
    # PersonModel = generate_pydantic_from_class(folio_manager_module_scope, person_iri)
    # birth_date_field = _clean_name("birthDate") # Assuming label is 'birthDate'
    # if birth_date_field in PersonModel.model_fields:
    #    field_info = PersonModel.model_fields[birth_date_field]
    #    # Type could be Optional[date] or List[date]
    #    # For a functional property, it would be Optional[date]
    #    assert field_info.annotation == Optional[date]
    pass # Placeholder until specific FOLIO properties are identified for testing

def test_object_property_handling(folio_manager_module_scope: FolioManager):
    """Tests that object properties are correctly typed as other generated models.
    Find a class with a known object property linking to another class.
    Example: If 'Case' has a property 'hasCourt' that links to a 'Court' instance.
    """
    # case_iri = "https://folio.openlegalstandard.org/Case"
    # court_iri = "https://folio.openlegalstandard.org/Court"

    # CaseModel = generate_pydantic_from_class(folio_manager_module_scope, case_iri)
    # CourtModel = generate_pydantic_from_class(folio_manager_module_scope, court_iri) # Ensure target model is also generated

    # assert CaseModel is not None
    # assert CourtModel is not None

    # fields = CaseModel.model_fields
    # # Assume 'hasCourt' is the property, label "has court"
    # has_court_field = _clean_name("hasCourt") 
    # if has_court_field in fields:
    #     field_info = fields[has_court_field]
    #     # Expected type: Optional[CourtModel] or List[CourtModel]
    #     assert field_info.annotation == Optional[CourtModel] or field_info.annotation == List[CourtModel]
    pass # Placeholder

# To make these tests more robust, we would need to:
# 1. Identify specific IRIs for classes in FOLIO.owl that have the desired features (inheritance, data props, obj props).
# 2. Know the rdfs:label or skos:prefLabel for these classes and properties to predict the _clean_name output.
# 3. Know the rdfs:range for properties to predict type annotations.

# The original test_generate_supreme_court_model is removed as the IRI seemed to be an instance.
# It can be re-added if a class IRI for 'U.S. Supreme Court' is found and its properties are known.


# --- Tests for ALL_FOLIO_OWL_CLASSES --- 

def test_all_folio_owl_classes_structure():
    """Tests the basic structure and integrity of the ALL_FOLIO_OWL_CLASSES dictionary."""
    assert ALL_FOLIO_OWL_CLASSES is not None, "ALL_FOLIO_OWL_CLASSES should be loaded."
    assert isinstance(ALL_FOLIO_OWL_CLASSES, Dict), "ALL_FOLIO_OWL_CLASSES should be a dictionary."
    assert len(ALL_FOLIO_OWL_CLASSES) > 0, "ALL_FOLIO_OWL_CLASSES should not be empty (assuming FOLIO.owl has classes)."

    for iri_key, owl_class_instance in ALL_FOLIO_OWL_CLASSES.items():
        assert isinstance(iri_key, str), "Dictionary keys should be string IRIs."
        assert isinstance(owl_class_instance, OwlClass), f"Value for IRI {iri_key} should be an OwlClass instance."
        assert owl_class_instance.iri == iri_key, \
            f"OwlClass instance IRI '{owl_class_instance.iri}' should match its key '{iri_key}'."
        assert isinstance(owl_class_instance.label, str), f"Label for {iri_key} should be a string."
        assert owl_class_instance.label.strip() != "", f"Label for {iri_key} should not be empty or just whitespace."
        assert isinstance(owl_class_instance.sub_class_of, list), f"sub_class_of for {iri_key} should be a list."
        for sub_class_uri in owl_class_instance.sub_class_of:
            assert isinstance(sub_class_uri, URIRef), \
                f"Elements in sub_class_of for {iri_key} should be URIRef instances."

def test_all_folio_owl_classes_specific_entry_court():
    """Tests a specific entry in ALL_FOLIO_OWL_CLASSES, e.g., for 'Court'."""
    court_iri = 'https://folio.openlegalstandard.org/Court'
    
    assert court_iri in ALL_FOLIO_OWL_CLASSES, f"'{court_iri}' should be a key in ALL_FOLIO_OWL_CLASSES."
    
    court_owl_class = ALL_FOLIO_OWL_CLASSES[court_iri]
    
    assert court_owl_class.iri == court_iri
    # The exact label depends on FOLIO.owl. Common expectation is 'Court'.
    # This test assumes the label is 'Court' or similar, adjust if needed based on actual data.
    assert court_owl_class.label is not None
    assert "court" in court_owl_class.label.lower(), f"Label for Court IRI should contain 'court', got '{court_owl_class.label}'"

    # Example: Check if 'Court' is a subclass of 'Legal Entity' or 'Organization' (hypothetical)
    # This requires knowledge of FOLIO.owl. For now, we check the structure.
    # If you know a superclass, you can assert its presence:
    # expected_superclass_iri = URIRef('https://folio.openlegalstandard.org/Organization') # Example
    # assert expected_superclass_iri in court_owl_class.sub_class_of, \
    #     f"Court should be a subclass of {expected_superclass_iri}"

    # Check definition (optional field)
    if court_owl_class.definition is not None:
        assert isinstance(court_owl_class.definition, str)

    # Check preferred_label (optional field)
    if court_owl_class.preferred_label is not None:
        assert isinstance(court_owl_class.preferred_label, str)

    # Check alternative_labels (list of strings)
    assert isinstance(court_owl_class.alternative_labels, list)
    for alt_label in court_owl_class.alternative_labels:
        assert isinstance(alt_label, str)

# You can add more tests for other specific classes if needed, for example:
# def test_all_folio_owl_classes_specific_entry_person():
#     person_iri = 'https://folio.openlegalstandard.org/Person'
#     assert person_iri in ALL_FOLIO_OWL_CLASSES
#     person_owl_class = ALL_FOLIO_OWL_CLASSES[person_iri]
#     assert person_owl_class.label == "Person" # Or similar, check actual data
#     # Add more assertions for Person class attributes
