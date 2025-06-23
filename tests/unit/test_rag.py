"""
Unit tests for RAG model outputs and core functionality.
"""

import pytest
from src.rag.models import RAGResponse, validate_rag_response


def test_rag_response_structure():
    """Test that RAG response has the correct structure with hardcoded data."""

    # Hardcoded test data
    test_query = "A lawyer represents a client in court proceedings."
    test_response_text = """
    {{
        "entities": [
            {{
                "entity_name": "Lawyer",
                "entity_type": "Actor/Player",
                "description": "legal professional",
                "weight": 0.85
            }},
            {{
                "entity_name": "Client",
                "entity_type": "Actor/Player",
                "description": "person seeking legal services",
                "weight": 0.80
            }},
            {{
                "entity_name": "Court",
                "entity_type": "Institution",
                "description": "legal proceeding venue",
                "weight": 0.75
            }}
        ],
        "relationships": [
            {{
                "src_id": "Lawyer",
                "tgt_id": "Client",
                "description": "legal representation",
                "keywords": "represents, legal representation",
                "weight": 0.90
            }}
        ]
    }}
    """

    try:
        # Validate the response
        response = validate_rag_response(test_response_text, test_query)

        # Test basic response structure
        assert isinstance(response, RAGResponse), "Expected RAGResponse object"
        assert response.query_text == test_query, "Expected query text to match"
        assert len(response.response_text) > 0, "Expected non-empty response text"
        assert response.total_concepts >= 0, "Expected non-negative total concepts"
        assert 0.0 <= response.weight <= 1.0, "Expected weight between 0.0 and 1.0"

        # Test concept lists
        assert isinstance(
            response.entities_found, list
        ), "Expected entities_found to be list"
        assert isinstance(
            response.relationships_found, list
        ), "Expected relationships_found to be list"
        assert isinstance(
            response.classes_found, list
        ), "Expected classes_found to be list"
        assert isinstance(
            response.properties_found, list
        ), "Expected properties_found to be list"

        # Test confidence summary
        confidence_summary = response.confidence_summary
        assert isinstance(
            confidence_summary, dict
        ), "Expected confidence_summary to be dict"
        required_keys = [
            "entities",
            "relationships",
            "classes",
            "properties",
            "overall",
        ]
        for key in required_keys:
            assert key in confidence_summary, f"Expected {key} in confidence_summary"
            assert (
                0.0 <= confidence_summary[key] <= 1.0
            ), f"Expected {key} confidence between 0.0 and 1.0"

        print(f"✅ RAG Response Structure Test Passed")
        print(f"   - Total concepts: {response.total_concepts}")
        print(f"   - Overall confidence: {confidence_summary['overall']:.3f}")
        print(f"   - Response weight: {response.weight:.3f}")

    except Exception as e:
        print(f"❌ RAG response structure test failed: {e}")
        raise


def test_ontological_concept_validation():
    """Test that individual ontological concepts are properly structured."""

    # Hardcoded test data
    test_query = "A lawyer provides legal advice to a client."
    test_response_text = """
    {{
        "entities": [
            {{
                "entity_name": "Lawyer",
                "entity_type": "Actor/Player",
                "description": "legal professional",
                "weight": 0.85
            }},
            {{
                "entity_name": "Client",
                "entity_type": "Actor/Player",
                "description": "person receiving advice",
                "weight": 0.80
            }}
        ],
        "relationships": [
            {{
                "src_id": "Lawyer",
                "tgt_id": "Client",
                "description": "gives legal advice",
                "keywords": "provides, legal advice",
                "weight": 0.90
            }}
        ]
    }}
    """

    try:
        response = validate_rag_response(test_response_text, test_query)

        # Test all concept types
        for entity in response.entities_found:
            assert isinstance(
                entity, OntologicalConcept
            ), "Expected OntologicalConcept object"
            assert entity.concept_name, "Expected non-empty concept name"
            assert entity.concept_type == "entity", "Expected entity type"
            assert entity.description, "Expected non-empty description"
            assert (
                0.0 <= entity.confidence_score <= 1.0
            ), "Expected valid confidence score"
            assert isinstance(
                entity.relationships, list
            ), "Expected relationships to be list"
            assert isinstance(
                entity.source_chunks, list
            ), "Expected source_chunks to be list"

        for rel in response.relationships_found:
            assert isinstance(
                rel, OntologicalConcept
            ), "Expected OntologicalConcept object"
            assert rel.concept_name, "Expected non-empty concept name"
            assert rel.concept_type == "relationship", "Expected relationship type"
            assert rel.description, "Expected non-empty description"
            assert 0.0 <= rel.confidence_score <= 1.0, "Expected valid confidence score"
            assert isinstance(
                rel.relationships, list
            ), "Expected relationships to be list"
            assert isinstance(
                rel.source_chunks, list
            ), "Expected source_chunks to be list"

        for cls in response.classes_found:
            assert isinstance(
                cls, OntologicalConcept
            ), "Expected OntologicalConcept object"
            assert cls.concept_name, "Expected non-empty concept name"
            assert cls.concept_type == "class", "Expected class type"
            assert cls.description, "Expected non-empty description"
            assert 0.0 <= cls.confidence_score <= 1.0, "Expected valid confidence score"
            assert isinstance(
                cls.relationships, list
            ), "Expected relationships to be list"
            assert isinstance(
                cls.source_chunks, list
            ), "Expected source_chunks to be list"

        for prop in response.properties_found:
            assert isinstance(
                prop, OntologicalConcept
            ), "Expected OntologicalConcept object"
            assert prop.concept_name, "Expected non-empty concept name"
            assert prop.concept_type == "property", "Expected property type"
            assert prop.description, "Expected non-empty description"
            assert (
                0.0 <= prop.confidence_score <= 1.0
            ), "Expected valid confidence score"
            assert isinstance(
                prop.relationships, list
            ), "Expected relationships to be list"
            assert isinstance(
                prop.source_chunks, list
            ), "Expected source_chunks to be list"

        print(f"✅ Ontological Concept Validation Test Passed")
        print(f"   - Entities: {len(response.entities_found)}")
        print(f"   - Relationships: {len(response.relationships_found)}")
        print(f"   - Classes: {len(response.classes_found)}")
        print(f"   - Properties: {len(response.properties_found)}")

        # Test source_chunks field specifically
        if response.entities_found:
            entity = response.entities_found[0]
            assert isinstance(
                entity.source_chunks, list
            ), "Expected source_chunks to be list"
            print(f"   - Entity source_chunks: {entity.source_chunks}")

    except Exception as e:
        print(f"❌ Ontological concept validation test failed: {e}")
        raise


def test_rag_response_validation_function():
    """Test the validate_rag_response function with various hardcoded inputs."""

    test_cases = [
        {
            "query": "A lawyer provides legal services.",
            "response_text": """
            {{
                "entities": [
                    {{
                        "entity_name": "Lawyer",
                        "entity_type": "Actor/Player",
                        "description": "legal professional",
                        "weight": 0.85
                    }}
                ],
                "relationships": [
                    {{
                        "src_id": "Lawyer",
                        "tgt_id": "Client",
                        "description": "gives services",
                        "keywords": "provides, services",
                        "weight": 0.90
                    }}
                ]
            }}
            """,
        },
        {
            "query": "A tenant receives an eviction notice.",
            "response_text": """
            {{
                "entities": [
                    {{
                        "entity_name": "Tenant",
                        "entity_type": "Actor/Player",
                        "description": "person renting property",
                        "weight": 0.80
                    }},
                    {{
                        "entity_name": "Eviction Notice",
                        "entity_type": "Document/Artifact",
                        "description": "legal document",
                        "weight": 0.75
                    }}
                ],
                "relationships": [
                    {{
                        "src_id": "Tenant",
                        "tgt_id": "Eviction Notice",
                        "description": "gets document",
                        "keywords": "receives, document",
                        "weight": 0.85
                    }}
                ]
            }}
            """,
        },
    ]

    try:
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Testing validation case {i+1}: {test_case['query']}")

            # Test the validation function
            validated_response = validate_rag_response(
                test_case["response_text"], test_case["query"]
            )

            # Validate the validated response
            assert isinstance(
                validated_response, RAGResponse
            ), f"Expected RAGResponse from validation case {i+1}"
            assert (
                validated_response.query_text == test_case["query"]
            ), f"Expected query text to match case {i+1}"
            assert (
                validated_response.total_concepts >= 0
            ), f"Expected non-negative total concepts case {i+1}"

            # Print results
            print(f"   ✅ Case {i+1} passed")
            print(f"   - Validated total concepts: {validated_response.total_concepts}")
            print(
                f"   - Validated overall confidence: {validated_response.confidence_summary['overall']:.3f}"
            )

        print(f"\n✅ All validation function tests passed!")

    except Exception as e:
        print(f"❌ RAG response validation function test failed: {e}")
        raise


def test_rag_response_edge_cases():
    """Test RAG response validation with edge cases."""

    try:
        # Test with empty concepts
        empty_response_text = """
        {{
            "entities": [],
            "relationships": []
        }}
        """

        response = validate_rag_response(empty_response_text, "Empty test query")
        assert (
            response.total_concepts == 0
        ), "Expected zero total concepts for empty response"
        assert (
            response.confidence_summary["overall"] == 0.0
        ), "Expected zero overall confidence for empty response"

        # Test with single concept
        single_concept_response = """
        {{
            "entities": [
                {{
                    "entity_name": "Test Entity",
                    "entity_type": "Actor/Player",
                    "description": "test entity",
                    "weight": 1.0
                }}
            ],
            "relationships": []
        }}
        """

        response = validate_rag_response(single_concept_response, "Single concept test")
        assert response.total_concepts == 1, "Expected one total concept"
        assert (
            response.confidence_summary["overall"] == 1.0
        ), "Expected perfect confidence"

        # Test with high confidence concepts
        high_confidence_response = """
        {{
            "entities": [
                {{
                    "entity_name": "High Confidence Entity",
                    "entity_type": "Actor/Player",
                    "description": "entity with high confidence",
                    "weight": 0.95
                }}
            ],
            "relationships": [
                {{
                    "src_id": "High Confidence Entity",
                    "tgt_id": "Another Entity",
                    "description": "relationship with high confidence",
                    "keywords": "high confidence",
                    "weight": 0.98
                }}
            ]
        }}
        """

        response = validate_rag_response(
            high_confidence_response, "High confidence test"
        )
        assert response.total_concepts == 2, "Expected two total concepts"
        assert (
            response.confidence_summary["overall"] > 0.9
        ), "Expected high overall confidence"

        print(f"✅ RAG Response Edge Cases Test Passed")

    except Exception as e:
        print(f"❌ RAG response edge cases test failed: {e}")
        raise


def test_rag_response_error_handling():
    """Test RAG response validation error handling."""

    try:
        # Test with invalid JSON
        invalid_json_response = "This is not valid JSON"

        with pytest.raises(
            ValueError, match="Response does not contain valid JSON structure"
        ):
            validate_rag_response(invalid_json_response, "Invalid JSON test")

        # Test with missing required fields
        incomplete_response = """
        {{
            "entities": [
                {{
                    "entity_name": "Test Entity",
                    "description": "test entity"
                }}
            ]
        }}
        """

        # This should still work as missing fields get default values
        response = validate_rag_response(incomplete_response, "Incomplete test")
        assert isinstance(
            response, RAGResponse
        ), "Expected RAGResponse even with incomplete data"

        # Test with invalid confidence scores
        invalid_confidence_response = """
        {{
            "entities": [
                {{
                    "entity_name": "Test Entity",
                    "entity_type": "Actor/Player",
                    "description": "test entity",
                    "weight": 1.5
                }}
            ],
            "relationships": []
        }}
        """

        # This should raise a validation error
        with pytest.raises(ValueError):
            validate_rag_response(
                invalid_confidence_response, "Invalid confidence test"
            )

        print(f"✅ RAG Response Error Handling Test Passed")

    except Exception as e:
        print(f"❌ RAG response error handling test failed: {e}")
        raise
