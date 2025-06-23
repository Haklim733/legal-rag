"""
test_folio.py - Unit tests for FOLIO functionality and RAG response validation
"""

import pytest
from folio import FOLIO
from typing import List, Tuple, Dict
from src.rag.models import RAGResponse, validate_rag_response
from src.rag.main import main


@pytest.fixture
def folio():
    """Create a FOLIO instance."""
    return FOLIO(source_type="github")


def test_get_by_iri(folio):
    """Test getting a class by IRI."""
    legal_entity = folio["R79UCI3zYVOfFextRdsi3qM"]
    print(legal_entity)


def test_get_property_by_iri(folio):
    """Test getting a class by IRI."""
    assert folio.get_property(
        "https://folio.openlegalstandard.org/R1us3pQhG9zkEb39dZHByB"
    )


def test_get_all_properties(folio):
    """Test getting all properties from the ontology."""
    all_properties = folio.get_all_properties()
    print(f"Total properties found: {len(all_properties)}")
    print([{prop.label: (prop.domain, prop.range)} for prop in all_properties])


def test_get_specific_property(folio):
    # Get all properties and test with the first one
    all_properties = folio.get_all_properties()
    assert len(all_properties) > 0

    # Use the first property for testing
    test_property = all_properties[0]
    print(f"Testing with property: {test_property.label}")

    # Verify the property has the expected attributes
    assert test_property.label is not None
    assert test_property.iri is not None
    print(f"Found property: {test_property.label}")


def test_find_all_subject_connections(folio):
    """Test finding all connections between objects."""
    # Get a specific class (e.g., Legal Entity)
    legal_entity = folio.get_by_label("Trial Court Forum")[0]

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


def test_find_all_object_connections(folio):
    """Test finding all connections between objects."""
    with pytest.raises(AttributeError):
        """his should raise an AttributeError because find_connections expects a subject_class"""
        legal_entity = folio.get_by_label("Actor / Player")[0]
        object_connections = folio.find_connections(
            subject_class=None,  # None means any subject
            property_name=None,  # None means any property
            object_class=legal_entity,
        )


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


def test_get_all_domain_range_connections(folio):
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


@pytest.mark.skip(reason="Skipping this test for now; takes too long")
def test_get_class_pairs_for_predicate(folio):
    """Test getting class pairs for a specific predicate."""
    # Get a specific property (e.g., folio:observed)
    property_label = "folio:observed"
    prop = folio.get_properties_by_label(property_label)[0]

    print(f"\nTesting property: {property_label}")
    print(f"Domains: {[folio[domain].label for domain in prop.domain]}")
    print(f"Ranges: {[folio[range_val].label for range_val in prop.range]}")

    # Get all domain classes including subclasses
    domain_classes = set()
    for domain in prop.domain:
        domain_class = folio[domain]
        if domain_class.label is None:
            continue
        # Add the domain class IRI
        domain_classes.add(domain_class.iri)
        # Add all its subclasses' IRIs
        for subclass in folio.get_children(domain_class.iri):
            if subclass.label is not None:
                domain_classes.add(subclass.iri)

    # Get all range classes including subclasses
    range_classes = set()
    for range_val in prop.range:
        range_class = folio[range_val]
        if range_class.label is None:
            continue
        # Add the range class IRI
        range_classes.add(range_class.iri)
        # Add all its subclasses' IRIs
        for subclass in folio.get_children(range_class.iri):
            if subclass.label is not None:
                range_classes.add(subclass.iri)

    # Create the mapping
    class_connections = {}
    for domain_iri in domain_classes:
        domain_class = folio[domain_iri]
        if domain_class.label not in class_connections:
            class_connections[domain_class.label] = {}

        if property_label not in class_connections[domain_class.label]:
            class_connections[domain_class.label][property_label] = []

        # Add only valid range classes for this property
        for range_iri in range_classes:
            range_class = folio[range_iri]
            # Verify the connection exists in the ontology
            connections = folio.find_connections(
                subject_class=domain_class, property_name=prop, object_class=range_class
            )
            if connections:  # Only add if the connection exists
                if (
                    range_class.label
                    not in class_connections[domain_class.label][property_label]
                ):
                    class_connections[domain_class.label][property_label].append(
                        range_class.label
                    )

    # Print the results
    print("\nClass connections found:")
    for subject, props in class_connections.items():
        print(f"\n{subject} can connect to:")
        for prop, ranges in props.items():
            print(f"  - {prop} can connect to: {', '.join(ranges)}")

    # Verify we have connections
    assert len(class_connections) > 0, "Should have at least one domain class"
    for subject, props in class_connections.items():
        assert (
            property_label in props
        ), f"Property {property_label} should be in connections for {subject}"
        assert (
            len(props[property_label]) > 0
        ), f"Should have at least one range class for {subject}"


def test_find_specific_connection(folio):
    """Test finding specific connections between objects."""
    # Get two specific classes
    subject = folio.get_by_label("Insurance Adjuster")[0]
    object = folio.get_by_label("Incapacity Event")[0]

    # Find all connections between these two classes
    connections = folio.find_connections(
        subject_class=subject,
        property_name=None,  # None means any property
        object_class=object,
    )
    print(connections)

    for subject, prop, obj in connections:
        print(f"{subject.label} --[{prop.label}]--> {obj.label}")


def test_triple(folio):
    print(folio.triples)
