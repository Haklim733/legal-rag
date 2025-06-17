"""
test_graph.py - Unit tests for FOLIO graph functionality
"""

import os
from re import I
import pytest
from pathlib import Path
import json
from typing import List, Dict, Set, Optional, Any
import asyncio

from kg.models.folio.graph import FOLIO, FOLIOTypes, FOLIO_TYPE_IRIS
from kg.models.folio.models import OWLClass, OWLObjectProperty


@pytest.fixture
def folio_owl_path():
    """Get the path to the FOLIO.owl file."""
    # Get the directory containing the test file
    test_dir = Path(__file__).parent.resolve()
    # Navigate to the FOLIO.owl file
    owl_path = test_dir.parent.parent / "src" / "kg" / "models" / "folio" / "FOLIO.owl"
    if not owl_path.exists():
        raise FileNotFoundError(f"FOLIO.owl file not found at {owl_path}")
    return owl_path


@pytest.fixture
def folio(folio_owl_path):
    """Create a FOLIO instance from the FOLIO.owl file."""
    return FOLIO(source_type="file", file_path=str(folio_owl_path))


def test_folio_initialization(folio_owl_path):
    """Test FOLIO initialization from file."""
    folio = FOLIO(source_type="file", file_path=str(folio_owl_path))
    assert folio is not None
    assert isinstance(folio, FOLIO)


def test_folio_metadata(folio):
    """Test FOLIO metadata."""
    assert folio.title is not None
    assert folio.description is not None


def test_folio_classes(folio):
    """Test FOLIO classes."""
    assert len(folio.classes) > 0
    for cls in folio.classes:
        if cls.iri == "http://www.w3.org/2002/07/owl#Thing":
            continue
        assert isinstance(cls, OWLClass)
        assert cls.iri is not None
        assert cls.label is not None


def test_folio_object_properties(folio):
    """Test FOLIO object properties."""
    assert len(folio.object_properties) > 0
    for prop in folio.object_properties:
        assert isinstance(prop, OWLObjectProperty)
        assert prop.iri is not None
        assert prop.label is not None


def test_folio_type_iris(folio):
    """Test FOLIO type IRIs."""
    BASE_URL = "https://folio.openlegalstandard.org/"
    for folio_type in FOLIOTypes:
        assert folio_type in FOLIO_TYPE_IRIS
        iri = BASE_URL + FOLIO_TYPE_IRIS[folio_type]
        assert iri in folio.iri_to_index


def test_folio_get_by_label(folio):
    """Test getting classes by label."""
    # Test with a known label
    test_label = "Actor / Player"
    print(test_label)
    classes = folio.get_by_label(test_label)
    assert len(classes) > 0
    assert all(isinstance(cls, OWLClass) for cls in classes)
    assert all(cls.label == test_label for cls in classes)


def test_folio_get_by_alt_label(folio):
    """Test getting classes by alternative label."""
    # Test with a known alternative label
    test_alt_label = "BIZO"
    classes = folio.get_by_alt_label(test_alt_label)
    assert len(classes) > 0
    assert all(isinstance(cls, OWLClass) for cls in classes)
    assert all(test_alt_label in cls.alternative_labels for cls in classes)


def test_folio_get_properties_by_label(folio):
    """Test getting properties by label."""
    # Test with a known property label
    test_label = "hasPart"
    test_label = "Actor / Player"
    properties = folio.get_properties_by_label(test_label)
    print(properties)
    assert len(properties) > 0
    assert all(isinstance(prop, OWLObjectProperty) for prop in properties)
    assert all(prop.label == test_label for prop in properties)


def test_folio_get_subgraph(folio):
    """Test getting subgraph."""
    # Test with a known class IRI
    test_iri = FOLIO_TYPE_IRIS[FOLIOTypes.LEGAL_ENTITY]
    subgraph = folio.get_subgraph(test_iri)
    assert len(subgraph) > 0
    assert all(isinstance(cls, OWLClass) for cls in subgraph)


def test_folio_get_children(folio):
    """Test getting children."""
    # Test with a known class IRI
    test_iri = FOLIO_TYPE_IRIS[FOLIOTypes.LEGAL_ENTITY]
    children = folio.get_children(test_iri)
    assert len(children) > 0
    assert all(isinstance(cls, OWLClass) for cls in children)


def test_folio_get_parents(folio):
    """Test getting parents."""
    # Test with a known class IRI
    test_iri = FOLIO_TYPE_IRIS[FOLIOTypes.LEGAL_ENTITY]
    parents = folio.get_parents(test_iri)
    assert len(parents) > 0
    assert all(isinstance(cls, OWLClass) for cls in parents)


def test_folio_normalize_iri(folio):
    """Test IRI normalization."""
    # Test with different IRI formats
    test_iris = [
        "https://folio.openlegalstandard.org/test",
        "folio:test",
        "soli:test",
        "lmss:test",
        "http://lmss.sali.org/test",
        "test",
    ]

    for iri in test_iris:
        normalized = folio.normalize_iri(iri)
        assert normalized.startswith("https://folio.openlegalstandard.org/")


def test_folio_get_folio_branches(folio):
    """Test getting FOLIO branches."""
    branches = folio.get_folio_branches()
    assert len(branches) == len(FOLIOTypes)
    for folio_type in FOLIOTypes:
        assert folio_type in branches
        assert len(branches[folio_type]) > 0


def test_folio_get_all_properties(folio):
    """Test getting all properties."""
    all_properties = folio.get_all_properties()
    assert len(all_properties) > 0
    assert all(isinstance(prop, OWLObjectProperty) for prop in all_properties)
    # Get all object properties

    # For each property, show its domain (subject) and range (object) classes
    for prop in all_properties:
        print(f"\nProperty: {prop.label} ({prop.iri})")
        print("Domain (Subject) Classes:")
        for domain in prop.domain:
            domain_class = folio[domain]
            if domain_class:
                print(f"  - {domain_class.label}")
        print("Range (Object) Classes:")
        for range_val in prop.range:
            range_class = folio[range_val]
            if range_class:
                print(f"  - {range_class.label}")


def test_folio_find_connections(folio):
    """Test finding connections between classes."""
    # Test with known class IRIs
    subject_iri = FOLIO_TYPE_IRIS[FOLIOTypes.LEGAL_ENTITY]
    object_iri = FOLIO_TYPE_IRIS[FOLIOTypes.DOCUMENT_ARTIFACT]

    connections = folio.find_connections(subject_iri, object_class=object_iri)
    assert isinstance(connections, list)
    for connection in connections:
        assert len(connection) == 3
        assert isinstance(connection[0], OWLClass)
        assert isinstance(connection[1], OWLObjectProperty)
        assert isinstance(connection[2], OWLClass)


def test_folio_generate_iri(folio):
    """Test IRI generation."""
    iri = folio.generate_iri()
    assert iri.startswith("https://folio.openlegalstandard.org/")
    assert iri not in folio.iri_to_index


def test_folio_get_triples(folio):
    """Test getting triples."""
    # Test getting triples by subject
    test_iri = FOLIO_TYPE_IRIS[FOLIOTypes.LEGAL_ENTITY]
    subject_triples = folio.get_triples_by_subject(test_iri)
    assert len(subject_triples) > 0
    assert all(len(triple) == 3 for triple in subject_triples)

    # Test getting triples by predicateA
    for prop in folio.object_properties:
        print(f"Label: {prop.label}, IRI: {prop.iri}")
    predicate_triples = folio.get_triples_by_predicate(
        "https://folio.openlegalstandard.org/RlWWALyEkmi7ifaz2MpmoR"
    )
    predicate_triples = folio.get_triples_by_predicate("folio:concerning")
    print(predicate_triples)
    predicate_triples = folio.get_triples_by_predicate(
        "http://www.w3.org/2000/01/rdf-schema#authored"
    )
    print(predicate_triples)
    assert len(predicate_triples) > 0
    assert all(len(triple) == 3 for triple in predicate_triples)

    # Test getting triples by object
    object_triples = folio.get_triples_by_object(test_iri)
    assert len(object_triples) > 0
    assert all(len(triple) == 3 for triple in object_triples)


class FOLIOTypeExtractor:
    def __init__(self):
        # Initialize FOLIO instance
        self.folio = FOLIO(source_type="github")

        # Build type indices for quick lookup
        self.type_indices = self._build_type_indices()

    def _build_type_indices(self) -> Dict[str, Set[str]]:
        """Build indices of all FOLIO types and their variations"""
        indices = {}

        # Get all FOLIO branches
        folio_branches = self.folio.get_folio_branches()

        for branch_name, classes in folio_branches.items():
            indices[branch_name] = set()
            for cls in classes:
                # Add main label
                if cls.label:
                    indices[branch_name].add(cls.label.lower())
                # Add alternative labels
                indices[branch_name].update(
                    alt.lower() for alt in cls.alternative_labels
                )
                # Add hidden labels
                if cls.hidden_label:
                    indices[branch_name].add(cls.hidden_label.lower())

        return indices

    async def extract_types_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract FOLIO types from text using FOLIO's LLM search
        """
        # Define search sets for different FOLIO branches
        search_sets = [
            self.folio.get_player_actors(max_depth=2),
            self.folio.get_document_artifacts(max_depth=2),
            self.folio.get_events(max_depth=2),
            self.folio.get_areas_of_law(max_depth=1),
            self.folio.get_legal_entities(max_depth=2),
        ]

        # Perform parallel search
        results = {}
        async for result in self.folio.parallel_search_by_llm(
            text, search_sets=search_sets, limit=5  # Adjust limit as needed
        ):
            # Get the class and its branch
            cls = result[0]  # First element is the OWLClass
            score = result[1]  # Second element is the relevance score

            # Find which branch this class belongs to
            for branch_name, classes in self.folio.get_folio_branches().items():
                if cls in classes:
                    if branch_name not in results:
                        results[branch_name] = []
                    if cls.label and cls.label not in results[branch_name]:
                        results[branch_name].append(cls.label)
                    break

        return results

    def get_required_types(self) -> Dict[str, List[str]]:
        """Get list of required types for email validation"""
        return {
            "Actor / Player": self.folio.get_player_actors(),
            "Document / Artifact": self.folio.get_document_artifacts(),
            "Event": self.folio.get_events(),
        }

    async def validate_email_content(self, text: str) -> Dict[str, Any]:
        """
        Validate email content against required FOLIO types
        """
        # Extract types from text using LLM search
        extracted_types = await self.extract_types_from_text(text)

        # Get required types
        required_types = self.get_required_types()

        # Validate against requirements
        validation_results = {
            "is_valid": True,
            "missing_types": {},
            "found_types": extracted_types,
            "suggestions": [],
        }

        # Check for required types
        for branch, types in required_types.items():
            if branch not in extracted_types:
                validation_results["is_valid"] = False
                validation_results["missing_types"][branch] = [t.label for t in types]
                validation_results["suggestions"].append(
                    f"Email should contain at least one {branch} type"
                )

        return validation_results


# Example usage
async def main():
    extractor = FOLIOTypeExtractor()

    # Example email text
    email_text = """
    Dear John,
    
    I am writing to inform you about the upcoming court hearing scheduled for next month.
    The plaintiff, Mr. Smith, has submitted a motion requesting additional time to prepare
    the necessary documents. The defendant's legal team has agreed to this extension.
    
    Please find attached the updated court filing and the judge's order.
    
    Best regards,
    Legal Team
    """

    # Extract and validate types
    results = await extractor.validate_email_content(email_text)

    # Print results
    print("Validation Results:")
    print(f"Is Valid: {results['is_valid']}")
    print("\nFound Types:")
    for branch, types in results["found_types"].items():
        print(f"{branch}: {', '.join(types)}")
    print("\nMissing Types:")
    for branch, types in results["missing_types"].items():
        print(f"{branch}: {', '.join(types)}")
    print("\nSuggestions:")
    for suggestion in results["suggestions"]:
        print(f"- {suggestion}")


if __name__ == "__main__":
    asyncio.run(main())
