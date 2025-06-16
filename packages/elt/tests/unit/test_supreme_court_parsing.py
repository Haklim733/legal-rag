import pytest
from datetime import date
from uuid import UUID

# Assuming your models are accessible via 'kg.models'
# Adjust the import path if your project structure is different
# For example, if 'packages' is in PYTHONPATH or you're running from 'etl' directory:
from kg.models import (
    LegalCase, Plaintiff, Defendant, Attorney, Judge, Court, LegalDocument, 
    Contract, Pleading, Complaint, Motion, Statute, Regulation,
    Relationship, RepresentedBy, PlaintiffIn, DefendantIn, Cites
)

# --- Mock PDF Parsing and NLP --- 
def mock_parse_supreme_court_pdf(pdf_content_mock: str) -> dict:
    """
    Simulates parsing a PDF and extracting structured information.
    In a real scenario, this would involve PDF text extraction and complex NLP.
    """
    # This is highly simplified. A real output would be much more complex
    # and would likely involve lists of entities and relationships.
    if "Wonderland v. Hatter" in pdf_content_mock:
        return {
            "case_details": {
                "name": "Wonderland v. Hatter",
                "case_number": "SC-123-2023",
                "decision_date": date(2023, 10, 26),
                "court_name": "Supreme Court of ExampleLand",
            },
            "plaintiffs": [
                {"name": "Alice Wonderland", "type": "Plaintiff"}
            ],
            "defendants": [
                {"name": "Mad Hatter", "type": "Defendant"}
            ],
            "judges": [
                {"name": "Queen of Hearts", "role": "Authoring Judge"}
            ],
            "attorneys": [
                {"name": "Cheshire Cat", "represents": "Alice Wonderland", "role": "Plaintiff Attorney"},
                {"name": "March Hare", "represents": "Mad Hatter", "role": "Defendant Attorney"}
            ],
            "cited_documents": [
                {"name": "Precedent Case Alpha", "type": "LegalCase", "citation": "123 U.S. 456"},
                {"name": "Statute of Nonsense", "type": "Statute", "citation": "1 E.L.C. § 789"}
            ],
            "opinion_text_summary": "The court found in favor of the plaintiff..."
        }
    return {}

# --- Pytest Test Case --- 

@pytest.fixture
def parsed_data_fixture():
    """Provides mock parsed data for tests."""
    mock_pdf_text = "Case: Wonderland v. Hatter. Date: 2023-10-26. Opinion by Judge Queen of Hearts."
    return mock_parse_supreme_court_pdf(mock_pdf_text)

def test_map_parsed_data_to_models(parsed_data_fixture):
    parsed_data = parsed_data_fixture
    if not parsed_data: # Handle case where mock parsing returns empty
        assert False, "Mock parsing failed to return data"

    entities = {}
    relationships = []

    # 1. Create Court Entity
    court_name = parsed_data["case_details"]["court_name"]
    court = Court(name=court_name, entity_type="Court")
    entities[court_name] = court

    # 2. Create LegalCase Entity (representing the main Supreme Court opinion document)
    case_details = parsed_data["case_details"]
    main_case_doc = LegalCase(
        name=case_details["name"],
        entity_type="LegalCase", # Or a more specific 'OpinionDocument' if defined
        case_number=case_details["case_number"],
        # decision_date=case_details["decision_date"], # Add if model supports
        court_id=court.id
    )
    entities[main_case_doc.name] = main_case_doc

    # 3. Create Party Entities (Plaintiffs, Defendants)
    for p_data in parsed_data["plaintiffs"]:
        plaintiff = Plaintiff(name=p_data["name"], entity_type="Plaintiff")
        entities[plaintiff.name] = plaintiff
        # Add PlaintiffIn relationship
        relationships.append(PlaintiffIn(
            source_id=plaintiff.id,
            target_id=main_case_doc.id,
            relationship_type="plaintiffIn"
        ))

    for d_data in parsed_data["defendants"]:
        defendant = Defendant(name=d_data["name"], entity_type="Defendant")
        entities[defendant.name] = defendant
        # Add DefendantIn relationship
        relationships.append(DefendantIn(
            source_id=defendant.id,
            target_id=main_case_doc.id,
            relationship_type="defendantIn"
        ))

    # 4. Create Judge Entities
    for j_data in parsed_data["judges"]:
        judge = Judge(name=j_data["name"], entity_type="Judge")
        entities[judge.name] = judge
        # Could add an 'authoredBy' relationship if defined:
        # relationships.append(Relationship(
        #     source_id=judge.id, 
        #     target_id=main_case_doc.id, 
        #     relationship_type="authoredOpinion"
        # ))

    # 5. Create Attorney Entities and Representation Relationships
    for a_data in parsed_data["attorneys"]:
        attorney = Attorney(name=a_data["name"], entity_type="Attorney")
        entities[attorney.name] = attorney
        represented_party_name = a_data["represents"]
        if represented_party_name in entities:
            represented_party_id = entities[represented_party_name].id
            relationships.append(RepresentedBy(
                source_id=represented_party_id, # Party is source
                target_id=attorney.id,          # Attorney is target
                relationship_type="representedBy"
            ))

    # 6. Create Cited Document Entities and Cites Relationships
    for cited_doc_data in parsed_data["cited_documents"]:
        doc_type = cited_doc_data["type"]
        cited_doc = None
        if doc_type == "LegalCase":
            cited_doc = LegalCase(name=cited_doc_data["name"], entity_type="LegalCase")
        elif doc_type == "Statute":
            cited_doc = Statute(name=cited_doc_data["name"], entity_type="Statute")
        # Add more types as needed (e.g., Regulation)
        
        if cited_doc:
            entities[cited_doc.name] = cited_doc
            relationships.append(Cites(
                source_id=main_case_doc.id, # The main opinion cites the other document
                target_id=cited_doc.id,
                relationship_type="cites"
            ))

    # --- Assertions --- 
    assert entities["Wonderland v. Hatter"].case_number == "SC-123-2023"
    assert entities["Alice Wonderland"].entity_type == "Plaintiff"
    assert entities["Mad Hatter"].entity_type == "Defendant"
    assert entities["Queen of Hearts"].entity_type == "Judge"
    assert entities["Cheshire Cat"].entity_type == "Attorney"
    assert entities["Supreme Court of ExampleLand"].entity_type == "Court"
    assert entities["Precedent Case Alpha"].entity_type == "LegalCase"
    assert entities["Statute of Nonsense"].entity_type == "Statute"

    assert len(relationships) > 0
    
    plaintiff_representation_found = any(
        r.relationship_type == "representedBy" and 
        entities["Alice Wonderland"].id == r.source_id and 
        entities["Cheshire Cat"].id == r.target_id
        for r in relationships
    )
    assert plaintiff_representation_found, "Plaintiff representation relationship not found"

    main_case_cites_precedent = any(
        r.relationship_type == "cites" and
        main_case_doc.id == r.source_id and
        entities["Precedent Case Alpha"].id == r.target_id
        for r in relationships
    )
    assert main_case_cites_precedent, "Main case citation to precedent not found"

    print(f"Successfully created {len(entities)} entities and {len(relationships)} relationships.")
    # For detailed inspection during testing:
    for name, entity in entities.items():
        print(entity.model_dump_json(indent=2))
    for rel in relationships:
        print(rel.model_dump_json(indent=2))
