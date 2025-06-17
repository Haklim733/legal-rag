"""
test_folio.py - Unit tests for FOLIO functionality
"""

import pytest
from folio import FOLIO
from typing import List, Tuple, Dict


@pytest.fixture
def folio():
    """Create a FOLIO instance."""
    return FOLIO(source_type="github")


def test_find_all_connections(folio):
    """Test finding all connections between objects."""
    # Get a specific class (e.g., Legal Entity)
    legal_entity = folio.get_by_label("Actor / Player")[0]

    # Find all connections where Legal Entity is the subject
    subject_connections = folio.find_connections(
        subject_class=legal_entity,
        property_name=None,  # None means any property
        object_class=None,  # None means any object
    )

    # Print the connections
    print("\nConnections where Legal Entity is the subject:")
    for subject, prop, obj in subject_connections:
        print(f"{subject.label} --[{prop.label}]--> {obj.label}")

    # Find all connections where Legal Entity is the object
    object_connections = folio.find_connections(
        subject_class=legal_entity,  # None means any subject
        property_name=None,  # None means any property
        object_class=legal_entity,
    )

    print("\nConnections where Legal Entity is the object:")
    for subject, prop, obj in object_connections:
        print(f"{subject.label} --[{prop.label}]--> {obj.label}")


def test_get_all_triples(folio):
    """Test getting all triples for an object."""
    legal_entity = folio.get_by_label("Actor / Player")[0]

    subject_triples = folio.get_triples_by_subject(legal_entity.iri)

    object_triples = folio.get_triples_by_object(legal_entity.iri)

    # Print the triples
    print("\nTriples where Legal Entity is the subject:")
    for subject, pred, obj in subject_triples:
        print(f"{subject} --[{pred}]--> {obj}")

    print("\nTriples where Legal Entity is the object:")
    for subject, pred, obj in object_triples:
        print(f"{subject} --[{pred}]--> {obj}")


def test_get_all_possible_connections(folio):
    """Test getting all possible connections for an object."""
    legal_entity = folio.get_by_label("Actor / Player")[0]

    all_properties = folio.get_all_properties()

    domain_properties = [
        prop
        for prop in all_properties
        if any(domain == legal_entity.iri for domain in prop.domain)
    ]

    # Find properties where Legal Entity is in the range
    range_properties = [
        prop
        for prop in all_properties
        if any(range_val == legal_entity.iri for range_val in prop.range)
    ]

    print("\nProperties where Legal Entity can be the subject:")
    for prop in domain_properties:
        print(f"- {prop.label}")
        print(f"  Domain: {[folio[domain].label for domain in prop.domain]}")
        print(f"  Range: {[folio[range_val].label for range_val in prop.range]}")

    print("\nProperties where Legal Entity can be the object:")
    for prop in range_properties:
        print(f"- {prop.label}")
        print(f"  Domain: {[folio[domain].label for domain in prop.domain]}")
        print(f"  Range: {[folio[range_val].label for range_val in prop.range]}")


def test_find_specific_connections(folio):
    """Test finding specific connections between objects."""
    # Get two specific classes
    legal_entity = folio.get_by_label("Actor / Player")[0]
    document_artifact = folio.get_by_label("Document / Artifact")[0]

    # Find all connections between these two classes
    connections = folio.find_connections(
        subject_class=legal_entity,
        property_name=None,  # None means any property
        object_class=document_artifact,
    )

    print("\nConnections between Legal Entity and Document/Artifact:")
    for subject, prop, obj in connections:
        print(f"{subject.label} --[{prop.label}]--> {obj.label}")

    # Find connections with a specific property
    authored_property = folio.get_properties_by_label("folio:authored")[0]
    specific_connections = folio.find_connections(
        subject_class=legal_entity,
        property_name=authored_property,
        object_class=document_artifact,
    )

    print("\nConnections with property:")
    for subject, prop, obj in specific_connections:
        print(f"{subject.label} --[{prop.label}]--> {obj.label}")
