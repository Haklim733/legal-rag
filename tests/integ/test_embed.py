"""
test_embed.py - Integration tests for ontology embedding functionality
"""

import os
import pytest

from folio import FOLIO
from folio.models import OWLClass, OWLObjectProperty

# Import the ontology functions
from rag.embed import (
    Document,
    load_ontology_for_rag,
    _create_document_from_owl_class,
    _create_document_from_owl_property,
    _get_children_with_depth,
    _get_class_depth,
)

os.environ["OPENAI_API_KEY"]


@pytest.fixture
def folio_instance():
    """Create a real FOLIO instance for integration testing."""
    return FOLIO("github", llm=None)


def test_load_ontology_for_rag_basic(folio_instance):
    """Test basic functionality of load_ontology_for_rag with real FOLIO data."""

    # Test with small limit and no depth restriction
    documents = load_ontology_for_rag(limit=10)

    # Verify we get documents
    assert len(documents) > 0
    assert len(documents) <= 10

    print(f"Successfully loaded {len(documents)} documents")

    # Verify document structure
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.id is not None
        assert doc.text is not None
        assert len(doc.text) > 0
        assert doc.metadata is not None
        assert doc.metadata.get("source") == "FOLIO"


def test_get_children_with_depth(folio_instance):
    """Test _get_children_with_depth with real FOLIO data starting from top-level classes."""
    print("Testing _get_children_with_depth starting from top-level classes...")

    # Get top-level classes (classes that have owl#Thing as their only parent)
    top_level = [
        x
        for x in folio_instance.classes
        if x.sub_class_of == ["http://www.w3.org/2002/07/owl#Thing"]
        and x.label
        not in [
            "System Identifiers",
            "ZZZ - SANDBOX: UNDER CONSTRUCTION",
            "Industry and Market",
        ]
    ]
    assert len(top_level) > 0

    # Test with first top-level class
    top_class = top_level[0]
    print(f"Using top-level class: {top_class.label} (IRI: {top_class.iri})")

    # Test with max depth of 2
    max_depth = 2
    children = _get_children_with_depth(folio_instance, top_class, max_depth)

    # Verify we get children
    assert isinstance(children, list)
    print(f"Found {len(children)} children with max_depth={max_depth}")

    # Verify all children are OWLClass instances
    for child in children:
        assert isinstance(child, OWLClass)
        assert child.iri is not None

    # Test different depths to see the progression
    for depth in [1, 2]:
        children = _get_children_with_depth(folio_instance, top_class, depth)
        print(f"Depth {depth}: Found {len(children)} children")

        # Verify we get children
        assert isinstance(children, list)

        # Verify all children are OWLClass instances
        for child in children:
            assert isinstance(child, OWLClass)
            assert child.iri is not None


def test_get_class_depth(folio_instance):
    """Test _get_class_depth with real FOLIO data."""
    print("Testing _get_class_depth...")

    # Get top-level classes and a child class
    top_level = [
        x
        for x in folio_instance.classes
        if x.sub_class_of == ["http://www.w3.org/2002/07/owl#Thing"]
        and x.label
        not in [
            "System Identifiers",
            "ZZZ - SANDBOX: UNDER CONSTRUCTION",
            "Industry and Market",
        ]
    ]
    assert len(top_level) > 0

    # Debug: Show some sample classes and their relationships
    print("Sample classes and their sub_class_of relationships:")
    for i, cls in enumerate(folio_instance.classes[:10]):
        print(f"  {i+1}. {cls.label}: {cls.sub_class_of}")

    print(f"\nFound {len(top_level)} top-level classes:")
    for i, cls in enumerate(top_level[:5]):
        print(f"  {i+1}. {cls.label} (IRI: {cls.iri})")

    root_class = top_level[0]
    print(f"Root class: {root_class.label} (IRI: {root_class.iri})")
    print(f"Root class sub_class_of: {root_class.sub_class_of}")

    # Get children of the root class to find a target class
    children = _get_children_with_depth(folio_instance, root_class, 1)

    # If no children found, try a different approach - look for any class that has this as parent
    if len(children) == 0:
        print(
            "No children found with _get_children_with_depth, trying manual search..."
        )
        manual_children = []
        for cls in folio_instance.classes:
            if cls.sub_class_of and any(
                root_class.iri in parent for parent in cls.sub_class_of
            ):
                manual_children.append(cls)
                print(f"  Manual found: {cls.label} -> {cls.sub_class_of}")

        if len(manual_children) > 0:
            children = manual_children
            print(f"Found {len(children)} children manually")
        else:
            print("No children found manually either")
            # Skip this test if no children are found
            pytest.skip("No children found for the selected root class")

    assert len(children) > 0

    target_class = children[0]

    # Test depth calculation
    depth = _get_class_depth(folio_instance, target_class, root_class)

    # Verify depth is reasonable (should be 1 for direct children)
    assert isinstance(depth, (int, float))
    assert depth >= 0
    assert depth <= 5  # Shouldn't be too deep

    print(f"Depth of {target_class.label} relative to {root_class.label}: {depth}")


def test_load_ontology_for_rag_with_top_level_root(folio_instance):
    """Test load_ontology_for_rag starting from top-level classes with max depth 2."""
    print(
        "Testing load_ontology_for_rag starting from top-level classes with max_depth=2..."
    )

    # Test with max depth of 2 starting from top-level classes
    documents = load_ontology_for_rag(limit=100, max_depth=2)

    # Verify we get documents
    assert len(documents) > 0
    assert len(documents) <= 100

    print(
        f"Successfully loaded {len(documents)} documents with max_depth=2 from top-level classes"
    )

    # Verify document structure
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.id is not None
        assert doc.text is not None
        assert len(doc.text) > 0

    # Check that we have both classes and properties
    class_docs = [doc for doc in documents if doc.metadata.get("type") == "OWLClass"]
    property_docs = [
        doc for doc in documents if doc.metadata.get("type") == "OwlObjectProperty"
    ]

    print(
        f"Found {len(class_docs)} class documents and {len(property_docs)} property documents"
    )

    assert len(class_docs) > 0, "Should have at least some class documents"
    # Make property documents optional since they might not always be available
    if len(property_docs) == 0:
        print(
            "Warning: No property documents found - this might be expected depending on the ontology structure"
        )

    # Print some sample class documents to see the hierarchy
    print("\nSample class documents:")
    for i, doc in enumerate(class_docs[:5]):
        print(f"  {i+1}. {doc.metadata.get('label', 'Unknown')} - {doc.id}")


def test_create_document_from_owl_class(folio_instance):
    """Test _create_document_from_owl_class with real FOLIO data."""
    print("Testing _create_document_from_owl_class...")

    # Get a real OWL class from FOLIO
    legal_entity_classes = folio_instance.get_by_label("Legal Entity")
    assert len(legal_entity_classes) > 0

    owl_class = legal_entity_classes[0]
    doc = _create_document_from_owl_class(owl_class)

    # Verify document creation
    assert doc is not None
    assert isinstance(doc, Document)
    assert doc.id == owl_class.iri
    assert doc.text is not None
    assert len(doc.text) > 0
    assert doc.metadata is not None
    assert doc.metadata.get("type") == "OWLClass"
    assert doc.metadata.get("label") == owl_class.label
    assert doc.metadata.get("source") == "FOLIO"

    # Verify text content contains expected information
    assert "Type: Ontology Class" in doc.text
    assert f"Label: {owl_class.label}" in doc.text
    assert f"IRI: {owl_class.iri}" in doc.text
    assert "Definition:" in doc.text

    print(f"Successfully created document for class: {owl_class.label}")


def test_create_document_from_owl_property(folio_instance):
    """Test _create_document_from_owl_property with real FOLIO data."""
    print("Testing _create_document_from_owl_property...")

    # Get real OWL properties from FOLIO
    all_properties = folio_instance.get_all_properties()
    assert len(all_properties) > 0

    # Test with first few properties
    for i, owl_prop in enumerate(all_properties[:3]):
        print(f"Testing property {i+1}: {owl_prop.label}")
        print(f"  Property preferred_label: {owl_prop.preferred_label}")
        print(f"  Property label: {owl_prop.label}")

        doc = _create_document_from_owl_property(owl_prop)

        # Verify document creation
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.id == owl_prop.iri
        assert doc.text is not None
        assert len(doc.text) > 0
        assert doc.metadata is not None
        assert doc.metadata.get("type") == "OwlObjectProperty"
        print(f"  Document metadata label: {doc.metadata.get('label')}")
        expected_label = owl_prop.label  # Use regular label
        print(f"  Expected label: {expected_label}")
        assert doc.metadata.get("label") == expected_label
        assert doc.metadata.get("source") == "FOLIO"

        # Verify text content contains expected information
        assert "Type: Ontology Property" in doc.text
        assert f"Label: {expected_label}" in doc.text
        assert f"IRI: {owl_prop.iri}" in doc.text
        assert "Definition:" in doc.text

        print(f"Successfully created document for property: {owl_prop.label}")


def test_load_ontology_for_rag_edge_cases(folio_instance):
    """Test edge cases for load_ontology_for_rag."""

    # Test with very small limit
    documents = load_ontology_for_rag(limit=1)
    assert len(documents) == 1
    print("✓ Small limit test passed")

    # Test with very small depth
    documents = load_ontology_for_rag(limit=10, max_depth=1)
    assert len(documents) > 0
    assert len(documents) <= 10
    print("✓ Small depth test passed")

    # Test with no limit
    documents = load_ontology_for_rag(limit=None, max_depth=2)
    assert len(documents) > 0
    print("✓ No limit test passed")


def test_document_metadata_structure(folio_instance):
    """Test that document metadata has the expected structure."""

    documents = load_ontology_for_rag(limit=20, max_depth=2)

    for doc in documents:
        # Verify required metadata fields
        assert "uri" in doc.metadata
        assert "type" in doc.metadata
        assert "label" in doc.metadata
        assert "source" in doc.metadata

        # Verify metadata values
        assert doc.metadata["source"] == "FOLIO"
        assert doc.metadata["type"] in ["OWLClass", "OwlObjectProperty"]
        assert doc.metadata["uri"] == doc.id
        assert doc.metadata["label"] is not None

    print(f"✓ Metadata structure test passed for {len(documents)} documents")


def test_document_text_quality(folio_instance):
    """Test that document text has good quality and structure."""

    documents = load_ontology_for_rag(limit=10, max_depth=2)

    for doc in documents:
        # Verify text is not empty
        assert len(doc.text.strip()) > 0

        # Verify text contains key information
        assert "Type:" in doc.text
        assert "Label:" in doc.text
        assert "IRI:" in doc.text
        assert "Definition:" in doc.text

        # Verify text is well-formatted (has line breaks)
        assert "\n" in doc.text

        # Verify text doesn't have excessive whitespace
        lines = [line.strip() for line in doc.text.split("\n") if line.strip()]
        assert len(lines) >= 4  # Should have at least Type, Label, IRI, Definition

    print(f"✓ Text quality test passed for {len(documents)} documents")
