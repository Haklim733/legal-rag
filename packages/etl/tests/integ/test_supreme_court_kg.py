"""
Integration tests for the Supreme Court knowledge graph builder.
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
    SupremeCourtCase,
    save_knowledge_graph,
    load_knowledge_graph
)

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
            'opinionAuthor': 'Harry Blackmun'
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
    
    # Test saving
    output_path = tmp_path / 'test_kg.json'
    save_knowledge_graph(cases, output_path)
    
    # Verify file was created and has content
    assert output_path.exists()
    with open(output_path, 'r') as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]['name'] == 'Test Case 1'
        assert data[1]['name'] == 'Test Case 2'
    
    # Test loading
    loaded_cases = load_knowledge_graph(output_path)
    assert len(loaded_cases) == 2
    assert isinstance(loaded_cases[0], SupremeCourtCase)
    assert loaded_cases[0].name == 'Test Case 1'
    assert loaded_cases[1].name == 'Test Case 2'

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
        
    # Print the knowledge graph output
    print("\nKnowledge Graph Output:")
    print(json.dumps(data, indent=2))
    
    # Basic assertions
    assert len(data) > 0
    assert 'name' in data[0]
    assert 'dcid' in data[0]
    
    # Print a summary of the first case
    if data:
        first_case = data[0]
        print("\nFirst Case Summary:")
        print(f"Name: {first_case.get('name')}")
        print(f"ID: {first_case.get('dcid')}")
        print(f"Description: {first_case.get('description')}")
        print(f"Date Decided: {first_case.get('date_decided')}")
        if 'citation' in first_case:
            print(f"Citation: {first_case['citation']}")
        if 'parties' in first_case:
            print(f"Parties: {first_case['parties']}")
        if 'decision_direction' in first_case:
            print(f"Decision: {first_case['decision_direction']}")
        if 'opinion_author' in first_case:
            print(f"Opinion by: {first_case['opinion_author']}")
