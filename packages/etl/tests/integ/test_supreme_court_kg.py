"""
Integration tests for the Supreme Court knowledge graph builder and store.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from kg.supreme_court import (
    SupremeCourtKG,
    save_knowledge_graph,
    load_knowledge_graph
)

# Import the new store and models
from src.rag.store import KnowledgeGraphStore, create_kg_store
from src.rag.models import SupremeCourtCase

# Sample data to mock the Data Commons API responses
SAMPLE_CASES_RESPONSE = {
    'rows': [
        {
            'cells': [
                {'name': 'case', 'value': 'dcid:case1'},
                {'name': 'name', 'value': 'Roe v. Wade'},
                {'name': 'description', 'value': 'Landmark decision on abortion rights'},
                {'name': 'dateDecided', 'value': '1973-01-22'}
            ]
        },
        {
            'cells': [
                {'name': 'case', 'value': 'dcid:case2'},
                {'name': 'name', 'value': 'Brown v. Board of Education'},
                {'name': 'description', 'value': 'Landmark decision on school desegregation'},
                {'name': 'dateDecided', 'value': '1954-05-17'}
            ]
        }
    ]
}

SAMPLE_CASE_DETAILS = {
    'data': {
        'dcid:case1': {
            'name': 'Roe v. Wade',
            'citation': '410 U.S. 113',
            'parties': 'Jane Roe, et al. v. Henry Wade',
            'decisionDirection': 'reversed',
            'opinionAuthor': 'Harry Blackmun',
            'cites': ['dcid:case2']
        },
        'dcid:case2': {
            'name': 'Griswold v. Connecticut',
            'citation': '381 U.S. 479',
            'parties': 'Griswold v. Connecticut',
            'decisionDirection': 'reversed',
            'opinionAuthor': 'William O. Douglas'
        }
    }
}

@pytest.fixture
def mock_kg_client():
    """Create a mock SupremeCourtKG client with patched API calls."""
    with patch('kg.supreme_court.requests.Session') as mock_session:
        mock_response = MagicMock()
        mock_response.json.side_effect = [
            SAMPLE_CASES_RESPONSE,  # First call for query_cases_by_year
            SAMPLE_CASE_DETAILS     # Second call for get_case_details
        ]
        mock_response.raise_for_status.return_value = None
        
        mock_session.return_value = MagicMock(
            post=MagicMock(return_value=mock_response),
            get=MagicMock(return_value=mock_response)
        )
        
        yield SupremeCourtKG()

@pytest.fixture(scope="module")
def test_cases():
    """Provide test cases for the knowledge graph store tests."""
    return [
        SupremeCourtCase(
            dcid="dcid:case1",
            name="Roe v. Wade",
            description="Landmark decision on abortion rights",
            date_decided="1973-01-22",
            citation="410 U.S. 113",
            parties="Jane Roe, et al. v. Henry Wade",
            decision_direction="reversed",
            opinion_author="Harry Blackmun",
            cites=["dcid:case2"]
        ),
        SupremeCourtCase(
            dcid="dcid:case2",
            name="Griswold v. Connecticut",
            description="Established right to privacy in marital relations",
            date_decided="1965-06-07",
            citation="381 U.S. 479",
            parties="Griswold v. Connecticut",
            decision_direction="reversed",
            opinion_author="William O. Douglas"
        )
    ]

@pytest.fixture
def kg_store(tmp_path, test_cases):
    """Create and populate a knowledge graph store for testing."""
    store = create_kg_store(
        entities=test_cases,
        entity_type=SupremeCourtCase,
        db_path=str(tmp_path / "test_kg_store"),
        id_field="dcid",
        text_fields=["name", "description", "parties", "opinion_author"],
        relationship_types={"CITES": ["cites"]}
    )
    return store

def test_save_and_load_knowledge_graph(tmp_path):
    """Test saving and loading a knowledge graph to/from a file."""
    # Create test cases
    cases = [
        SupremeCourtCase(
            dcid='dcid:case1',
            name='Test Case 1',
            description='Test description 1',
            date_decided='2023-01-01'
        ),
        SupremeCourtCase(
            dcid='dcid:case2',
            name='Test Case 2',
            description='Test description 2',
            date_decided='2023-01-02'
        )
    ]
    
    # Create a store and add cases
    store = create_kg_store(
        entities=cases,
        entity_type=SupremeCourtCase,
        db_path=str(tmp_path / "test_kg")
    )
    
    # Test saving
    output_path = tmp_path / 'test_kg.json'
    store.save(output_path)
    
    # Verify file was created
    assert output_path.exists()
    
    # Test loading
    loaded_store = KnowledgeGraphStore.load(
        path=output_path,
        entity_type=SupremeCourtCase
    )
    
    # Verify loaded data
    for case in cases:
        loaded = loaded_store.get_entity(case.dcid)
        assert loaded is not None
        assert loaded['name'] == case.name
        assert loaded['description'] == case.description

def test_query_cases_by_year(mock_kg_client):
    """Test querying cases by year with mocked API responses."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_CASES_RESPONSE
    mock_response.raise_for_status.return_value = None
    
    # Make the API call
    cases = mock_kg_client.query_cases_by_year(2023, limit=2)
    
    # Verify the results
    assert len(cases) == 2
    assert cases[0].name == 'Roe v. Wade'
    assert cases[1].name == 'Brown v. Board of Education'

class TestKnowledgeGraphStore:
    """Tests for the knowledge graph store functionality."""
    
    def test_add_and_retrieve_case(self, kg_store, test_cases):
        """Test adding and retrieving cases from the knowledge graph."""
        # Test retrieving each case by ID
        for case in test_cases:
            retrieved = kg_store.get_entity(case.dcid)
            assert retrieved is not None
            assert retrieved["name"] == case.name
            assert retrieved["description"] == case.description
    
    def test_search(self, kg_store):
        """Test searching the knowledge graph."""
        # Test search by topic
        results = kg_store.search("abortion rights", limit=1)
        assert len(results) > 0
        assert "Roe" in results[0]["name"]
        
        # Test search by justice name
        results = kg_store.search("Blackmun", limit=1)
        assert len(results) > 0
        assert "Roe" in results[0]["name"]
    
    def test_relationships(self, kg_store):
        """Test that relationships are properly stored and retrieved."""
        # Test that Roe v. Wade cites Griswold v. Connecticut
        roe = kg_store.get_entity("dcid:case1")
        assert roe is not None
        assert "cites" in roe
        assert "dcid:case2" in roe["cites"]

def test_main_integration(tmp_path, monkeypatch, mock_kg_client, capsys):
    """Test the main script functionality with mocked dependencies."""
    # Import here to avoid circular imports
    from scripts.build_supreme_court_kg import main
    
    # Mock command line arguments
    output_path = tmp_path / 'output_kg.json'
    test_args = [
        'build_supreme_court_kg.py',
        '--years', '2023',
        '--limit', '2',
        '--output', str(output_path)
    ]
    
    # Patch sys.argv
    with patch('sys.argv', test_args):
        main()
    
    # Verify the output file was created
    assert output_path.exists()
    
    # Load and verify the saved data
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Basic assertions
    assert len(data) > 0
    assert 'name' in data[0]
    assert 'dcid' in data[0]
