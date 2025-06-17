import os
import pytest
from typing import List, Tuple
from folio import FOLIO
from kg.models.folio.models import OWLClass
from dotenv import load_dotenv
from alea_llm_client.llms.models.openai_model import OpenAIModel
import json
import asyncio
from kg.models.folio.optimizer import (
    optimize_folio_data,
    chunk_folio_data,
    search_folio_chunks,
    FOLIOCache,
    optimized_folio_search,
)

# Load environment variables
load_dotenv()

# Sample test data
MOCK_EMAIL = """
Subject: Contract Review Request - Software License Agreement

Dear Legal Team,

I need your review of a software license agreement for our new cloud infrastructure project.
The vendor is proposing a 3-year term with automatic renewal clauses. Please check the 
intellectual property rights section and data privacy provisions.

The agreement includes:
- Software licensing terms
- Service level agreements
- Data processing requirements
- Confidentiality obligations

Please let me know if you need any additional information.

Best regards,
John Smith
"""


@pytest.fixture
def folio():
    """Create a FOLIO instance with OpenAI"""
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")

    # Initialize OpenAI model with proper configuration
    llm = OpenAIModel(
        model_name="gpt-4-turbo-preview",
        temperature=0.0,
        max_tokens=1000,
        api_key=api_key,
        json_mode=True,  # Enable JSON mode for structured responses
        response_format={"type": "json_object"},  # Ensure JSON responses
        system_message="You are a legal document analyzer that identifies relevant legal concepts and types.",
    )

    # Initialize FOLIO with the configured OpenAI model
    folio_instance = FOLIO(llm=llm, source_type="github")
    return folio_instance


@pytest.mark.asyncio
async def test_search_by_llm_with_email(folio):
    """Test searching specified FOLIO types in an email using OpenAI"""
    # Get all classes from FOLIO
    contract = folio.get_by_label("Contract ")[0]
    software_license = folio.get_by_label("Software License")[0]
    data_privacy = folio.get_by_label("Data Privacy")[0]

    search_set = [contract, software_license, data_privacy]

    # Perform the search
    results = await folio.search_by_llm(
        query=MOCK_EMAIL,
        search_set=search_set,
        limit=1,  # Reduced limit
        scale=1,
        include_reason=True,
    )

    # Verify the results
    assert len(results) > 0

    # Check that results are properly formatted
    for result in results:
        assert isinstance(result[0], OWLClass)
        assert isinstance(result[1], (int, float))
        assert isinstance(result[2], str)
        assert 1 <= result[1] <= 10


@pytest.mark.asyncio
async def test_search_by_llm_empty_results(folio):
    """Test handling of empty search results"""
    # Get all classes from FOLIO
    all_classes = []
    for label in [
        "Contract",
        "Software License",
        "Data Privacy",
        "Document / Artifact",
        "Actor / Player",
    ]:
        classes = folio.get_by_label(label)
        all_classes.extend(classes)

    # Perform the search with unrelated text
    results = await folio.search_by_llm(
        query="This is completely unrelated text about cooking recipes",
        search_set=all_classes,
        limit=5,
    )

    # Verify empty or low relevance results
    assert len(results) == 0 or all(score < 3 for _, score, _ in results)


@pytest.mark.asyncio
async def test_search_by_llm_parallel_search(folio):
    """Test parallel search functionality"""
    # Get all classes and their relationships
    search_sets = []

    # Get main classes
    contract = folio.get_by_label("Contract")[0]
    software_license = folio.get_by_label("Software License")[0]
    data_privacy = folio.get_by_label("Data Privacy")[0]

    # Get related classes through connections
    contract_connections = folio.find_connections(
        subject_class=contract, property_name=None, object_class=None
    )

    software_license_connections = folio.find_connections(
        subject_class=software_license, property_name=None, object_class=None
    )

    data_privacy_connections = folio.find_connections(
        subject_class=data_privacy, property_name=None, object_class=None
    )

    # Create search sets from connections
    search_sets = [
        [conn[0] for conn in contract_connections],  # Contract-related classes
        [
            conn[0] for conn in software_license_connections
        ],  # Software License-related classes
        [conn[0] for conn in data_privacy_connections],  # Data Privacy-related classes
    ]

    # Perform parallel search
    results = await folio.parallel_search_by_llm(
        query=MOCK_EMAIL,
        search_sets=search_sets,
        limit=5,
        scale=10,
        include_reason=True,
    )

    # Verify results
    assert len(results) > 0

    # Check that results are properly formatted
    for result in results:
        assert isinstance(result[0], OWLClass)
        assert isinstance(result[1], (int, float))
        assert isinstance(result[2], str)

        # Verify relevance score is within bounds
        assert 1 <= result[1] <= 10


@pytest.mark.asyncio
async def test_search_by_llm_different_scales(folio):
    """Test search with different scale values"""
    # Get all classes from FOLIO
    all_classes = []
    for label in [
        "Contract",
        "Software License",
        "Data Privacy",
        "Document / Artifact",
        "Actor / Player",
    ]:
        classes = folio.get_by_label(label)
        all_classes.extend(classes)

    scales = [5, 10, 100]

    for scale in scales:
        results = await folio.search_by_llm(
            query=MOCK_EMAIL,
            search_set=all_classes,
            limit=5,
            scale=scale,
            include_reason=True,
        )

        # Verify results
        assert len(results) > 0

        # Check that scores are within the specified scale
        for _, score, _ in results:
            assert 1 <= score <= scale


async def test_search_all_by_llm_with_email(folio):
    """Test searching all FOLIO types in an email using OpenAI"""
    # Get all FOLIO branches and their classes
    folio_branches = folio.get_folio_branches()

    # Create a flat list of all classes from all branches
    all_classes = []
    for branch_classes in folio_branches.values():
        all_classes.extend(branch_classes)

    # Perform the search with all classes
    results = await folio.search_by_llm(
        query=MOCK_EMAIL,
        search_set=all_classes,
        limit=10,  # Increased limit since we're searching all types
        scale=1,
        include_reason=True,
    )

    # Verify the results
    assert len(results) > 0

    # Check that results are properly formatted
    for result in results:
        assert isinstance(result[0], OWLClass)
        assert isinstance(result[1], (int, float))
        assert isinstance(result[2], str)
        assert 1 <= result[1] <= 10
