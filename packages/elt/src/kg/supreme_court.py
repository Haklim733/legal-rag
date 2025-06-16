"""
Knowledge Graph module for Supreme Court cases using Data Commons.
"""
from typing import Dict, List, Optional, Any
import requests
import json
from pathlib import Path
import logging
from pydantic import BaseModel
import asyncio
from folio import FOLIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Commons API endpoint
DATA_COMMONS_API = "https://api.datacommons.org/v2beta"

class SupremeCourtCase(BaseModel):
    """Represents a Supreme Court case in the knowledge graph."""
    dcid: str  # Data Commons ID
    name: str
    description: str
    date_decided: Optional[str] = None
    citation: Optional[str] = None
    parties: Optional[str] = None
    decision_direction: Optional[str] = None  # e.g., "affirmed", "reversed"
    opinion_author: Optional[str] = None
    
    # Pydantic's model_dump(exclude_none=True) can replace the custom to_dict method.
    # If you prefer to have a method for this, you can define it as:
    # def to_dict_custom(self) -> Dict[str, Any]:
    #     return self.model_dump(exclude_none=True)

class SupremeCourtKG:
    """Knowledge graph client for Supreme Court cases using Data Commons."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the knowledge graph client.
        
        Args:
            api_key: Optional API key for Data Commons API
        """
        self.session = requests.Session()
        if api_key:
            self.session.params = {'key': api_key}
    
    def query_cases_by_year(self, year: int, limit: int = 10) -> List[SupremeCourtCase]:
        """Query Supreme Court cases by decision year.
        
        Args:
            year: The year to query cases for
            limit: Maximum number of cases to return
            
        Returns:
            List of SupremeCourtCase objects
        """
        query = """
        SELECT ?case ?name ?description ?dateDecided
        WHERE {
          ?case typeOf LegalCase .
          ?case name ?name .
          ?case description ?description .
          ?case dateDecided ?dateDecided .
          FILTER (YEAR(?dateDecided) = %d)
          FILTER(CONTAINS(LCASE(?description), "supreme court"))
        }
        LIMIT %d
        """ % (year, limit)
        
        try:
            response = self.session.post(
                f"{DATA_COMMONS_API}/query",
                json={"sparql": query}
            )
            response.raise_for_status()
            
            cases = []
            data = response.json()
            
            for row in data.get('rows', []):
                case_data = {}
                for cell in row.get('cells', []):
                    if 'value' in cell and 'name' in cell:
                        case_data[cell['name']] = cell['value']
                
                if 'case' in case_data and 'name' in case_data:
                    cases.append(SupremeCourtCase(
                        dcid=case_data['case'],
                        name=case_data['name'],
                        description=case_data.get('description', ''),
                        date_decided=case_data.get('dateDecided')
                    ))
            
            return cases
            
        except Exception as e:
            logger.error(f"Error querying Data Commons: {str(e)}")
            return []
    
    def get_case_details(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific case.
        
        Args:
            case_id: The Data Commons ID of the case
            
        Returns:
            Dictionary containing detailed case information, or None if not found
        """
        try:
            # Query for node information
            response = self.session.get(
                f"{DATA_COMMONS_API}/node",
                params={
                    'nodes': [case_id],
                    'property': '->*'  # Get all properties
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', {}).get(case_id)
            
        except Exception as e:
            logger.error(f"Error getting case details: {str(e)}")
            return None

def save_knowledge_graph(cases: List[SupremeCourtCase], output_path: Path) -> None:
    """Save the knowledge graph to a JSON file.
    
    Args:
        cases: List of SupremeCourtCase objects
        output_path: Path to save the knowledge graph
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([case.model_dump(exclude_none=True) for case in cases], f, indent=2)
    
    logger.info(f"Saved {len(cases)} cases to {output_path}")

def load_knowledge_graph(input_path: Path) -> List[SupremeCourtCase]:
    """Load a knowledge graph from a JSON file.
    
    Args:
        input_path: Path to the knowledge graph JSON file
        
    Returns:
        List of SupremeCourtCase objects
    """
    if not input_path.exists():
        return []
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [SupremeCourtCase(**case_data) for case_data in data]

def search_legal_concepts(query: str):
    """
    Searches for legal concepts using the FOLIO library.

    Args:
        query: The search query.
    """
    print(f"Initializing FOLIO client...")
    folio = FOLIO()

    print(f"Searching for concepts related to '{query}'...")
    results = folio.search_by_label(query)
    
    print("Search Results:")
    for owl_class, score in results:
        print(f"  Class: {owl_class.label}, Score: {score}")
        print(f"    Definition: {owl_class.definition}")


async def async_search_legal_concepts(query: str):
    """
    Searches for legal concepts using the FOLIO library with LLM.

    Args:
        query: The search query.
    """
    print(f"Initializing FOLIO client...")
    folio = FOLIO()

    print(f"Searching for concepts related to '{query}' with an LLM...")
    
    search_sets = [
        folio.get_areas_of_law(max_depth=2),
        folio.get_player_actors(max_depth=2),
    ]
    
    print("Search Results (LLM):")
    async for result in folio.parallel_search_by_llm(query, search_sets=search_sets):
        print(f"  {result}")


if __name__ == '__main__':
    # Example usage:
    search_legal_concepts("Constitutional Law")

    # Example async usage
    print("\n---\n")
    asyncio.run(async_search_legal_concepts("First Amendment"))
