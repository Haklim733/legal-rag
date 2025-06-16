#!/usr/bin/env python3
"""
Script to build a knowledge graph of Supreme Court cases using Data Commons.
"""
import argparse
from pathlib import Path
from typing import List

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from kg.supreme_court import (
    SupremeCourtKG,
    SupremeCourtCase,
    save_knowledge_graph,
    load_knowledge_graph
)

def main():
    parser = argparse.ArgumentParser(description='Build a knowledge graph of Supreme Court cases.')
    parser.add_argument('--years', type=int, nargs='+', default=[2020, 2021, 2022],
                       help='Years to fetch cases for')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of cases to fetch per year')
    parser.add_argument('--output', type=Path, default='data/supreme_court_kg.json',
                       help='Output file path for the knowledge graph')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Optional API key for Data Commons')
    
    args = parser.parse_args()
    
    # Initialize the knowledge graph client
    kg_client = SupremeCourtKG(api_key=args.api_key)
    
    # Collect cases from all specified years
    all_cases: List[SupremeCourtCase] = []
    
    for year in args.years:
        print(f"Fetching cases for year {year}...")
        cases = kg_client.query_cases_by_year(year=year, limit=args.limit)
        all_cases.extend(cases)
        print(f"Found {len(cases)} cases for {year}")
        
        # Enrich case details
        for case in cases:
            if case.dcid:
                details = kg_client.get_case_details(case.dcid)
                if details:
                    # Update case with additional details if available
                    case.citation = details.get('citation', case.citation)
                    case.parties = details.get('parties', case.parties)
                    case.decision_direction = details.get('decisionDirection', case.decision_direction)
                    case.opinion_author = details.get('opinionAuthor', case.opinion_author)
    
    # Save the knowledge graph
    if all_cases:
        save_knowledge_graph(all_cases, args.output)
        print(f"\nKnowledge graph saved to {args.output}")
        print(f"Total cases: {len(all_cases)}")
        
        # Print a sample of the cases
        print("\nSample cases:")
        for i, case in enumerate(all_cases[:3]):
            print(f"{i+1}. {case.name}")
            if case.citation:
                print(f"   Citation: {case.citation}")
            if case.date_decided:
                print(f"   Decided: {case.date_decided}")
            if case.opinion_author:
                print(f"   Opinion by: {case.opinion_author}")
            print()
    else:
        print("No cases found for the specified criteria.")

if __name__ == "__main__":
    main()
