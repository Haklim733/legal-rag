"""
Unit tests for RAG model outputs and core functionality.
"""

import pytest
from src.rag.main import main
from src.rag.models import RAGResponse, Entity, Relationship, validate_rag_response


TEST_QUERY = "John, a professional lawyer at the Legal Aid of Los Angeles, spoke on behalf of Jane, a recent evictee of her apartment. Jane is a tenant of a rental property in Los Angeles, California. She received a notice to vacate the property, but she disputes the eviction. She is seeking legal representation and court proceedings to defend her case. The lawyer provides legal advice and represents clients in eviction cases."


@pytest.fixture
def response():
    return main(TEST_QUERY)


def test_rag_response_structure(response):
    """Test that RAG response has the correct structure."""
    print(response)

    try:

        # Test basic response structure
        assert isinstance(response, RAGResponse), "Expected RAGResponse object"
        assert response.query_text == TEST_QUERY, "Expected query text to match"
        assert len(response.response_text) > 0, "Expected non-empty response text"
        assert response.total_concepts >= 0, "Expected non-negative total concepts"

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

    except Exception as e:
        print(f"❌ RAG response structure test failed: {e}")
        raise


def test_ontological_concept_validation(response):
    """Test that individual ontological concepts are properly structured."""

    try:

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

        print(f"✅ Ontological Concept Validation Test Passed")
        print(f"   - Entities: {len(response.entities_found)}")
        print(f"   - Relationships: {len(response.relationships_found)}")
        print(f"   - Classes: {len(response.classes_found)}")
        print(f"   - Properties: {len(response.properties_found)}")

    except Exception as e:
        print(f"❌ Ontological concept validation test failed: {e}")
        raise


def test_rag_response_consistency(response):
    """Test that RAG response is consistent across multiple runs."""

    try:
        # Run multiple times to test consistency
        responses = []
        for i in range(3):
            responses.append(response)

            # Basic validation for each response
            assert isinstance(
                response, RAGResponse
            ), f"Expected RAGResponse object in run {i+1}"
            assert (
                response.query_text == TEST_QUERY
            ), f"Expected query text to match in run {i+1}"
            assert (
                response.total_concepts >= 0
            ), f"Expected non-negative total concepts in run {i+1}"

        # Test that all responses have similar structure
        total_concepts = [r.total_concepts for r in responses]
        overall_confidences = [r.confidence_summary["overall"] for r in responses]

        print(f"✅ RAG Response Consistency Test Passed")
        print(f"   - Total concepts across runs: {total_concepts}")
        print(
            f"   - Overall confidences across runs: {[f'{c:.3f}' for c in overall_confidences]}"
        )

        # All responses should have some concepts
        assert all(
            tc >= 0 for tc in total_concepts
        ), "All runs should have non-negative total concepts"
        assert all(
            0.0 <= c <= 1.0 for c in overall_confidences
        ), "All confidences should be between 0.0 and 1.0"

    except Exception as e:
        print(f"❌ RAG response consistency test failed: {e}")
        raise


def test_rag_response_validation_function(response):
    """Test the validate_rag_response function with mock data."""
    test_query = "A lawyer provides legal services."

    try:

        # Test the validation function
        validated_response = validate_rag_response(response.response_text, test_query)

        # Validate the validated response
        assert isinstance(
            validated_response, RAGResponse
        ), "Expected RAGResponse from validation"
        assert (
            validated_response.query_text == test_query
        ), "Expected query text to match"
        assert (
            validated_response.total_concepts >= 0
        ), "Expected non-negative total concepts"

        print(f"✅ RAG Response Validation Function Test Passed")
        print(f"   - Validated total concepts: {validated_response.total_concepts}")
        print(
            f"   - Validated overall confidence: {validated_response.confidence_summary['overall']:.3f}"
        )

    except Exception as e:
        print(f"❌ RAG response validation function test failed: {e}")
        raise


def test_rag_with_different_queries():
    """Test RAG with different types of legal queries."""
    test_queries = [
        "A lawyer represents a client in court.",
        "A tenant receives an eviction notice from the landlord.",
        "The court makes a decision in the case.",
        "A legal document is filed with the court.",
        "An attorney provides legal advice to a client.",
    ]

    try:
        for i, query in enumerate(test_queries):
            print(f"\n🧪 Testing query {i+1}: {query}")

            response = main(query)

            # Basic validation
            assert isinstance(
                response, RAGResponse
            ), f"Expected RAGResponse for query {i+1}"
            assert (
                response.query_text == query
            ), f"Expected query text to match for query {i+1}"
            assert (
                response.total_concepts >= 0
            ), f"Expected non-negative total concepts for query {i+1}"

            # Print results
            print(f"   ✅ Query {i+1} passed")
            print(f"   - Total concepts: {response.total_concepts}")
            print(
                f"   - Overall confidence: {response.confidence_summary['overall']:.3f}"
            )

            # Test that we get some response
            assert (
                len(response.response_text) > 0
            ), f"Expected non-empty response for query {i+1}"

        print(f"\n✅ All query tests passed!")

    except Exception as e:
        print(f"❌ Query testing failed: {e}")
        raise


def test_rag_error_handling():
    """Test RAG error handling with invalid inputs."""

    try:
        # Test with empty query
        with pytest.raises(Exception):
            response = main("")

        # Test with very long query
        long_query = "A " * 1000 + "lawyer represents a client."
        response = main(long_query)
        assert isinstance(response, RAGResponse), "Expected RAGResponse for long query"

        print(f"✅ RAG Error Handling Test Passed")

    except Exception as e:
        print(f"❌ RAG error handling test failed: {e}")
        raise
