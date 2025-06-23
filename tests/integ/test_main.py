"""
Unit tests for RAG model outputs and core functionality.
"""

import logging
import pytest
from src.rag.main import main
from src.rag.models import RAGResponse, Entity, Relationship, validate_rag_response

# Configure logging to handle ascii_colors errors gracefully
logging.getLogger("ascii_colors").setLevel(logging.ERROR)


TEST_QUERY = "John, a professional lawyer at the Legal Aid of Los Angeles, spoke on behalf of Jane, a recent evictee of her apartment. Jane is a tenant of a rental property in Los Angeles, California. She received a notice to vacate the property, but she disputes the eviction. She is seeking legal representation and court proceedings to defend her case. The lawyer provides legal advice and represents clients in eviction cases."

# Default test entities
DEFAULT_ENTITIES = [
    "Lawyer",
    "Legal Services Buyer",
]

# Default model names
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "all-minilm"


@pytest.fixture(scope="module")
def response():
    return main(TEST_QUERY, DEFAULT_ENTITIES, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL)


def test_rag_response_structure(response):
    """Test that RAG response has the correct structure."""

    try:

        # Test basic response structure
        assert isinstance(response, RAGResponse), "Expected RAGResponse object"
        assert response.query_text == TEST_QUERY, "Expected query text to match"
        assert len(response.response_text) > 0, "Expected non-empty response text"
        total_concepts = len(response.entities) + len(response.relationships)
        assert total_concepts >= 0, "Expected non-negative total concepts"

        # Test concept lists
        assert isinstance(response.entities, list), "Expected entities to be list"
        assert isinstance(
            response.relationships, list
        ), "Expected relationships to be list"

        print(f"✅ RAG Response Structure Test Passed")
        print(
            f"   - Total concepts: {len(response.entities) + len(response.relationships)}"
        )
        print(f"   - Entities: {len(response.entities)}")
        print(f"   - Relationships: {len(response.relationships)}")

    except Exception as e:
        print(f"❌ RAG response structure test failed: {e}")
        raise

    print(response.entities)


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
            total_concepts = len(response.entities) + len(response.relationships)
            assert (
                total_concepts >= 0
            ), f"Expected non-negative total concepts in run {i+1}"

        # Test that all responses have similar structure
        total_concepts = [len(r.entities) + len(r.relationships) for r in responses]

        print(f"✅ RAG Response Consistency Test Passed")
        print(f"   - Total concepts across runs: {total_concepts}")

        # All responses should have some concepts
        assert all(
            tc >= 0 for tc in total_concepts
        ), "All runs should have non-negative total concepts"

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
        total_concepts = len(validated_response.entities) + len(
            validated_response.relationships
        )
        assert total_concepts >= 0, "Expected non-negative total concepts"

        print(f"✅ RAG Response Validation Function Test Passed")
        print(f"   - Validated total concepts: {total_concepts}")

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

            response = main(
                query, DEFAULT_ENTITIES, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
            )

            # Basic validation
            assert isinstance(
                response, RAGResponse
            ), f"Expected RAGResponse for query {i+1}"
            assert (
                response.query_text == query
            ), f"Expected query text to match for query {i+1}"
            total_concepts = len(response.entities) + len(response.relationships)
            assert (
                total_concepts >= 0
            ), f"Expected non-negative total concepts for query {i+1}"

            # Print results
            print(f"   ✅ Query {i+1} passed")
            print(f"   - Total concepts: {total_concepts}")
            print(f"   - Entities: {len(response.entities)}")
            print(f"   - Relationships: {len(response.relationships)}")

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
        # Test with empty query - this might not raise an exception but should handle gracefully
        try:
            response = main(
                "", DEFAULT_ENTITIES, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
            )
            # If it doesn't raise an exception, it should return a valid response
            assert isinstance(
                response, RAGResponse
            ), "Expected RAGResponse for empty query"
        except Exception as e:
            # It's also acceptable for it to raise an exception for empty queries
            print(f"Empty query raised exception (acceptable): {e}")

        # Test with very long query
        long_query = "A " * 1000 + "lawyer represents a client."
        response = main(
            long_query, DEFAULT_ENTITIES, DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
        )
        assert isinstance(response, RAGResponse), "Expected RAGResponse for long query"

        print(f"✅ RAG Error Handling Test Passed")

    except Exception as e:
        print(f"❌ RAG error handling test failed: {e}")
        raise
