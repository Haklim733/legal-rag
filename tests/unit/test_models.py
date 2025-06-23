import pytest
from src.rag.models import (
    extract_json_from_text,
    validate_rag_response,
    extract_and_validate_json,
    parse_entities_from_response,
    RAGResponse,
    Entity,
    Relationship,
    ExtractionMode,
)

TEXT = """{
    "entities": [
        {
            "entity_name": "DONALD J. TRUMP",
            "entity_type": "Person",
            "description": "President of the United States"
        },
        {
            "entity_name": "GWYNNE A. WILCOX",
            "entity_type": "Person",
            "description": "Respondent"
        },
        {
            "entity_name": "CATHY HARRIS",
            "entity_type": "Person",
            "description": "Respondent"
        },
        {
            "entity_name": "THE CHIEF JUSTICE",
            "entity_type": "Person",
            "description": "Chief Justice of the Supreme Court"
        },
        {
            "entity_name": "KAGAN, J.",
            "entity_type": "Person",
            "description": "Justice of the Supreme Court"
        },
        {
            "entity_name": "SOTOMAYOR, JUSTICE",
            "entity_type": "Person",
            "description": "Justice of the Supreme Court"
        },
        {
            "entity_name": "JACKSON, JUSTICE",
            "entity_type": "Person",
            "description": "Justice of the Supreme Court"
        },
        {
            "entity_name": "NATIONAL LABOR RELATIONS BOARD (NLRB)",
            "entity_type": "Organization",
            "description": "Executive agency"
        },
        {
            "entity_name": "MERIT SYSTEMS PROTECTION BOARD (MSPB)",
            "entity_type": "Organization",
            "description": "Executive agency"
        },
        {
            "entity_name": "CONSUMER FINANCIAL PROTECTION BUREAU",
            "entity_type": "Organization",
            "description": "Executive agency"
        },
        {
            "entity_name": "FEDERAL RESERVE'S BOARD OF GOVERNORS",
            "entity_type": "Organization",
            "description": "Independent agency"
        },
        {
            "entity_name": "FIRST AND SECOND BANKS OF THE UNITED STATES",
            "entity_type": "Organization",
            "description": "Historical entities"
        }
    ],
    "relationships": [
        {
            "src_id": "DONALD J. TRUMP",
            "tgt_id": "NATIONAL LABOR RELATIONS BOARD (NLRB)",
            "description": "President's removal of a member",
            "keywords": "removal, NLRB"
        },
        {
            "src_id": "DONALD J. TRUMP",
            "tgt_id": "MERIT SYSTEMS PROTECTION BOARD (MSPB)",
            "description": "President's removal of a member",
            "keywords": "removal, MSPB"
        },
        {
            "src_id": "THE CHIEF JUSTICE",
            "tgt_id": "DONALD J. TRUMP",
            "description": "Granting application for stay",
            "keywords": "stay, application"
        },
        {
            "src_id": "KAGAN, J.",
            "tgt_id": "DONALD J. TRUMP",
            "description": "Dissenting from the grant of the application for stay",
            "keywords": "dissent, stay"
        }
    ]
}"""


def test_json_parse():
    extracted_json = extract_json_from_text(TEXT)
    assert extracted_json is not None
    validate_rag_response(TEXT, "test")
    print(extracted_json)


def test_extract_and_validate_json():
    """Test JSON extraction and validation."""
    valid_json = '{"entities": [], "relationships": []}'
    result = extract_and_validate_json(valid_json)
    assert "entities" in result
    assert "relationships" in result


def test_parse_entities_from_response():
    """Test entity parsing."""
    data = {"entities": [{"entity_name": "Test", "entity_type": "Person"}]}
    entities = parse_entities_from_response(data)
    assert len(entities) == 1
    assert entities[0].entity_name == "Test"


def test_validate_rag_response():
    """Test validate_rag_response function."""
    # Create a proper JSON string instead of a dictionary
    response_text = """{
        "entities": [
            {
                "entity_name": "DONALD J. TRUMP",
                "entity_type": "Person",
                "description": "President of the United States"
            },
            {
                "entity_name": "GWYNNE A. WILCOX",
                "entity_type": "Person",
                "description": "Respondent"
            }
        ],
        "relationships": [
            {
                "src_id": "DONALD J. TRUMP",
                "tgt_id": "GWYNNE A. WILCOX",
                "description": "Legal dispute",
                "keywords": "case, litigation"
            }
        ]
    }"""

    query_text = "test"
    extraction_mode = "aggressive"

    result = validate_rag_response(response_text, query_text, extraction_mode)

    assert isinstance(result, RAGResponse)
    assert result.query_text == query_text
    assert result.response_text == response_text
    assert len(result.entities) == 2
    assert len(result.relationships) == 1

    # Check entity details
    assert result.entities[0].entity_name == "DONALD J. TRUMP"
    assert result.entities[0].entity_type == "Person"
    assert result.entities[1].entity_name == "GWYNNE A. WILCOX"
    assert result.entities[1].entity_type == "Person"

    # Check relationship details
    assert result.relationships[0].src_id == "DONALD J. TRUMP"
    assert result.relationships[0].tgt_id == "GWYNNE A. WILCOX"
    assert result.relationships[0].description == "Legal dispute"
    assert result.relationships[0].keywords == "case, litigation"


def test_extract_json_from_text():
    """Test JSON extraction from text."""
    text = """{
        "entities": [
            {
                "entity_name": "Test Entity",
                "entity_type": "Person",
                "description": "Test description"
            }
        ],
        "relationships": []
    }"""

    result = extract_json_from_text(text, ExtractionMode.AGGRESSIVE)
    assert isinstance(result, dict)
    assert "entities" in result
    assert "relationships" in result
    assert len(result["entities"]) == 1
    assert result["entities"][0]["entity_name"] == "Test Entity"
