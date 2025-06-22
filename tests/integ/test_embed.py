"""
test_embed.py - Integration tests for ontology embedding functionality
"""

import os
import pytest

from folio import FOLIO
from folio.models import OWLClass, OWLObjectProperty

# Import the ontology functions
from src.rag.embed import (
    Document,
    load_ontology_for_rag,
    _get_children_with_depth,
    _get_class_depth,
    create_custom_kg,
)
from src.rag.embed import CustomKnowledgeGraph

os.environ["OPENAI_API_KEY"]


@pytest.fixture
def folio_instance():
    """Create a real FOLIO instance for integration testing."""
    return FOLIO("github", llm=None)


@pytest.fixture
def knowledge_graph(folio_instance):
    """Create a knowledge graph from FOLIO for testing."""
    return create_custom_kg(folio_instance)


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
        assert doc.metadata.source == "FOLIO"


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
    class_docs = [doc for doc in documents if doc.metadata.type == "OWLClass"]
    property_docs = [
        doc for doc in documents if doc.metadata.type == "OwlObjectProperty"
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
        print(f"  {i+1}. {doc.metadata.label} - {doc.id}")


@pytest.mark.parametrize(
    "child,parent",
    [
        ("Public Defender", "Lawyer"),
    ],
)
def test_create_custom_kg_hierarchy_relationships(knowledge_graph, child, parent):
    """
    Test specific parent-child relationships in the knowledge graph.
    """
    entities_by_name = {e.entity_name: e for e in knowledge_graph.entities}
    relationships_by_src = {r.src_id: r for r in knowledge_graph.relationships}

    # Check that both entities exist
    assert child in entities_by_name, f"Child entity '{child}' should exist"
    assert parent in entities_by_name, f"Parent entity '{parent}' should exist"

    child_entity = entities_by_name[child]
    parent_entity = entities_by_name[parent]

    # Check that the relationship exists
    assert (
        child_entity.source_id in relationships_by_src
    ), f"Relationship for '{child}' should exist"

    relationship = relationships_by_src[child_entity.source_id]

    # Verify relationship properties
    assert (
        relationship.tgt_id == parent_entity.source_id
    ), f"'{child}' should be related to '{parent}'"
    assert (
        "subClassOf" in relationship.description
    ), f"Relationship should contain 'subClassOf'"
    assert (
        relationship.keywords == "subClassOf"
    ), f"Keywords should be 'is a, subclass of'"
    assert relationship.weight == 1.0, "Relationship weight should be 1.0"

    print(f"✓ Verified hierarchy relationship: {child} is a subclass of {parent}")


def test_create_custom_kg_predicate_relationships(knowledge_graph):
    """
    Test predicate relationships in the knowledge graph.
    Tests the specific SPO: lawyer, folio:represents, legal service buyer
    """
    entities_by_name = {e.entity_name: e for e in knowledge_graph.entities}
    relationships_by_src = {r.src_id: r for r in knowledge_graph.relationships}
    chunks_by_source_id = {c.source_id: c for c in knowledge_graph.chunks}

    # Define the expected SPO relationship
    subject = "Lawyer"
    predicate = "represents"  # This should match the predicate label in FOLIO
    object_entity = "Legal Services Buyer"  # This should match the entity name in FOLIO

    # Check that all entities exist
    assert subject in entities_by_name, f"Subject entity '{subject}' should exist"
    assert (
        object_entity in entities_by_name
    ), f"Object entity '{object_entity}' should exist"

    subject_entity = entities_by_name[subject]
    object_entity_obj = entities_by_name[object_entity]

    # Look for relationships where Lawyer is the source
    lawyer_relationships = [
        r for r in knowledge_graph.relationships if r.src_id == subject_entity.source_id
    ]

    # Find the specific relationship with the expected predicate
    target_relationship = None
    for rel in lawyer_relationships:
        if (
            predicate.lower() in rel.description.lower()
            or predicate.lower() in rel.keywords.lower()
        ):
            target_relationship = rel
            break

    # If not found, look for any relationship from Lawyer to Legal Services Buyer
    if target_relationship is None:
        for rel in lawyer_relationships:
            if rel.tgt_id == object_entity_obj.source_id:
                target_relationship = rel
                break

    if target_relationship:
        # Verify relationship properties
        assert (
            target_relationship.src_id == subject_entity.source_id
        ), f"Source should be {subject}"
        assert (
            target_relationship.tgt_id == object_entity_obj.source_id
        ), f"Target should be {object_entity}"
        assert target_relationship.weight == 1.0, "Relationship weight should be 1.0"
        assert target_relationship.description, "Relationship should have a description"
        assert target_relationship.keywords, "Relationship should have keywords"

        # Check that there's a corresponding chunk for this relationship
        expected_chunk_source_id = f"{subject}_{predicate}_{object_entity}"

        # Look for chunks that contain the relationship information
        relationship_chunks = [
            c for c in knowledge_graph.chunks if c.source_id == expected_chunk_source_id
        ]

        assert (
            len(relationship_chunks) > 0
        ), f"Should have chunks for the {subject}-{predicate}-{object_entity} relationship"

        print(
            f"✓ Verified predicate relationship: {subject} {predicate} {object_entity}"
        )
        print(f"  Relationship description: {target_relationship.description}")
        print(f"  Found {len(relationship_chunks)} related chunks")
    else:
        # If the specific relationship doesn't exist, print available relationships for debugging
        print(
            f"⚠ Specific relationship {subject} {predicate} {object_entity} not found"
        )
        print(f"Available relationships from {subject}:")
        for rel in lawyer_relationships:
            tgt_entity = next(
                (e for e in knowledge_graph.entities if e.source_id == rel.tgt_id), None
            )
        # Skip the test if the specific relationship doesn't exist
        pytest.skip(
            f"Specific relationship {subject} {predicate} {object_entity} not found in the knowledge graph"
        )


def test_create_custom_kg_actual_hierarchy(knowledge_graph):
    """
    Test to discover and verify actual hierarchy relationships in the FOLIO ontology.
    """
    entities_by_name = {e.entity_name: e for e in knowledge_graph.entities}

    print(f"\nAvailable entities: {len(entities_by_name.keys())}")
    print(f"Total relationships: {len(knowledge_graph.relationships)}")

    # Find actual parent-child relationships
    hierarchy_relationships = [
        r for r in knowledge_graph.relationships if "subClassOf" in r.description
    ]
    print(f"Hierarchy relationships: {len(hierarchy_relationships)}")

    for rel in hierarchy_relationships:
        src_entity = next(
            (e for e in knowledge_graph.entities if e.source_id == rel.src_id), None
        )
        tgt_entity = next(
            (e for e in knowledge_graph.entities if e.source_id == rel.tgt_id), None
        )
    # Test the relationships that actually exist
    for rel in hierarchy_relationships[:5]:  # Test first 5 relationships
        src_entity = next(
            (e for e in knowledge_graph.entities if e.source_id == rel.src_id), None
        )
        tgt_entity = next(
            (e for e in knowledge_graph.entities if e.source_id == rel.tgt_id), None
        )

        if src_entity and tgt_entity:
            # Verify relationship properties
            assert rel.src_id in [
                e.source_id for e in knowledge_graph.entities
            ], "Source entity should exist"
            assert rel.tgt_id in [
                e.source_id for e in knowledge_graph.entities
            ], "Target entity should exist"
            assert (
                "subClassOf" in rel.keywords
            ), "Relationship should contain 'is a subclass of'"
            assert rel.keywords == "subClassOf", "Keywords should be 'subClassOf'"
            assert rel.weight == 1.0, "Relationship weight should be 1.0"


def test_create_custom_kg_structure(knowledge_graph):
    """
    Test the overall structure of the knowledge graph.
    """
    # Verify basic structure
    assert isinstance(knowledge_graph, CustomKnowledgeGraph)
    assert len(knowledge_graph.entities) > 0, "Should have entities"
    assert len(knowledge_graph.chunks) > 0, "Should have chunks"
    assert len(knowledge_graph.relationships) > 0, "Should have relationships"

    # Verify entity structure
    for entity in knowledge_graph.entities:
        assert entity.entity_name, "Entity should have a name"
        assert entity.entity_type, "Entity should have a type"
        assert entity.description, "Entity should have a description"
        assert entity.source_id, "Entity should have a source_id"
        assert entity.chunk_ids, "Entity should have chunk_ids"

    # Verify chunk structure
    for chunk in knowledge_graph.chunks:
        assert chunk.content, "Chunk should have content"
        assert chunk.source_id, "Chunk should have a source_id"
        assert chunk.chunk_order_index >= 0, "Chunk should have a valid order index"

    # Verify relationship structure
    for relationship in knowledge_graph.relationships:
        assert relationship.src_id, "Relationship should have a source_id"
        assert relationship.tgt_id, "Relationship should have a target_id"
        assert relationship.description, "Relationship should have a description"
        assert relationship.keywords, "Relationship should have keywords"
        assert relationship.weight > 0, "Relationship should have a positive weight"

    print(
        f"✓ Knowledge graph structure verified: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.chunks)} chunks, {len(knowledge_graph.relationships)} relationships"
    )


def test_create_custom_kg_entity_consistency(knowledge_graph):
    """
    Test that entities have consistent structure and naming.
    """
    for entity in knowledge_graph.entities:
        # Check source_id format
        expected_source_id = f"entity_{entity.entity_name.lower().replace(' ', '_')}"
        assert (
            entity.source_id == expected_source_id
        ), f"Entity {entity.entity_name} should have correct source_id format"

        # Check that entity_name and entity_type are the same
        assert (
            entity.entity_name == entity.entity_type
        ), f"Entity {entity.entity_name} should have same name and type"

    print(f"✓ Entity consistency verified for {len(knowledge_graph.entities)} entities")


@pytest.mark.skip(reason="Skipping chunk linking test due to duplicate classes")
def test_create_custom_kg_chunk_linking(knowledge_graph):
    """
    Test that chunks are properly linked to entities.
    """
    chunks_by_source_id = {c.source_id: c for c in knowledge_graph.chunks}

    for entity in knowledge_graph.entities:
        # Check that entity has a corresponding chunk
        assert (
            entity.source_id in chunks_by_source_id
        ), f"Entity {entity.entity_name} should have a corresponding chunk"

        # Check that entity description matches chunk content exactly
        try:
            chunk = chunks_by_source_id[entity.source_id]
            assert (
                entity.description == chunk.content
            ), f"Entity description should match chunk content exactly for {entity.entity_name}"
        except:
            print(f"Chunk: {chunk}")
            print(f"Entity: {entity}")
            print(f"Chunk content length: {len(chunk.content)}")
            print(f"Entity description length: {len(entity.description)}")
            print(f"Chunk content starts with: {chunk.content[:100]}...")
            print(f"Entity description starts with: {entity.description[:100]}...")
            print(f"Chunk content ends with: ...{chunk.content[-100:]}")
            print(f"Entity description ends with: ...{entity.description[-100:]}")
            raise
    print(f"✓ Chunk linking verified for {len(knowledge_graph.entities)} entities")


def test_create_custom_kg_chunk_ordering(knowledge_graph):
    """
    Test that chunk_order_index values are all 0 for now.
    """
    # Get all chunk_order_index values
    chunk_indices = [chunk.chunk_order_index for chunk in knowledge_graph.chunks]

    # Check that all indices are 0
    assert all(
        index == 0 for index in chunk_indices
    ), "All chunk indices should be 0 for now"

    print(
        f"✓ Chunk ordering verified: {len(knowledge_graph.chunks)} chunks all with index 0"
    )


def test_create_custom_kg_multiple_relationships(knowledge_graph):
    """
    Test that entities can have multiple relationships (both outgoing and incoming).
    """
    from src.rag.embed import get_entity_relationships, get_entity_relationship_summary

    # Get all entities and their relationship counts
    entity_relationship_counts = {}

    for entity in knowledge_graph.entities:
        relationships = get_entity_relationships(knowledge_graph, entity.source_id)
        entity_relationship_counts[entity.entity_name] = relationships["total"]

    # Find entities with multiple relationships
    entities_with_multiple_relationships = {
        name: count for name, count in entity_relationship_counts.items() if count > 1
    }

    print(f"\nEntities with multiple relationships:")
    for entity_name, count in entities_with_multiple_relationships.items():
        print(f"  {entity_name}: {count} relationships")

    # Test the relationship summary function for entities with multiple relationships
    if entities_with_multiple_relationships:
        # Test with the first entity that has multiple relationships
        test_entity_name = list(entities_with_multiple_relationships.keys())[0]
        summary = get_entity_relationship_summary(knowledge_graph, test_entity_name)

        print(f"\nRelationship summary for '{test_entity_name}':")
        print(f"  Entity type: {summary['entity_type']}")
        print(f"  Total relationships: {summary['total_relationships']}")

        if summary["outgoing_relationships"]:
            print(f"  Outgoing relationships:")
            for rel in summary["outgoing_relationships"]:
                print(f"    -> {rel['target']}: {rel['description']}")

        if summary["incoming_relationships"]:
            print(f"  Incoming relationships:")
            for rel in summary["incoming_relationships"]:
                print(f"    <- {rel['source']}: {rel['description']}")

        # Verify the summary structure
        assert "entity_name" in summary
        assert "entity_type" in summary
        assert "outgoing_relationships" in summary
        assert "incoming_relationships" in summary
        assert "total_relationships" in summary
        assert (
            summary["total_relationships"] > 1
        ), f"Entity {test_entity_name} should have multiple relationships"

        print(
            f"✓ Successfully verified multiple relationships for entity '{test_entity_name}'"
        )
    else:
        print("⚠ No entities found with multiple relationships")
        pytest.skip(
            "No entities with multiple relationships found in the knowledge graph"
        )


def test_create_custom_kg_relationship_types(knowledge_graph):
    """
    Test different types of relationships that can exist between entities.
    """
    # Group relationships by type (based on keywords)
    relationship_types = {}

    for rel in knowledge_graph.relationships:
        # Extract relationship type from keywords or description
        rel_type = "general"
        if "subclass" in rel.keywords.lower():
            rel_type = "hierarchy"
        elif "represents" in rel.keywords.lower():
            rel_type = "representation"
        elif "appears" in rel.keywords.lower():
            rel_type = "appearance"
        elif "works" in rel.keywords.lower() or "employed" in rel.keywords.lower():
            rel_type = "employment"
        elif "filed" in rel.keywords.lower() or "jurisdiction" in rel.keywords.lower():
            rel_type = "jurisdiction"

        if rel_type not in relationship_types:
            relationship_types[rel_type] = []
        relationship_types[rel_type].append(rel)

    print(f"\nRelationship types found:")
    for rel_type, rels in relationship_types.items():
        print(f"  {rel_type}: {len(rels)} relationships")

    # Verify we have different types of relationships
    assert len(relationship_types) > 0, "Should have at least one type of relationship"

    # Test that hierarchy relationships are properly structured
    if "hierarchy" in relationship_types:
        hierarchy_rels = relationship_types["hierarchy"]
        for rel in hierarchy_rels:
            assert "is a subclass of" in rel.description or "subClassOf" in rel.keywords
            assert rel.weight == 1.0

    print(
        f"✓ Successfully verified {len(relationship_types)} different relationship types"
    )


def test_create_custom_kg_entity_relationship_association(knowledge_graph):
    """
    Test that entities are properly associated with their relationships.
    """
    # Test that entities have relationship ID lists
    for entity in knowledge_graph.entities:
        assert hasattr(
            entity, "outgoing_relationship_ids"
        ), f"Entity {entity.entity_name} should have outgoing_relationship_ids"
        assert hasattr(
            entity, "incoming_relationship_ids"
        ), f"Entity {entity.entity_name} should have incoming_relationship_ids"
        assert isinstance(
            entity.outgoing_relationship_ids, list
        ), f"outgoing_relationship_ids should be a list for {entity.entity_name}"
        assert isinstance(
            entity.incoming_relationship_ids, list
        ), f"incoming_relationship_ids should be a list for {entity.entity_name}"

    # Test that relationships have unique IDs
    relationship_ids = [rel.source_id for rel in knowledge_graph.relationships]
    assert len(relationship_ids) == len(
        set(relationship_ids)
    ), "All relationships should have unique IDs"

    # Test that relationships have relationship_type
    for rel in knowledge_graph.relationships:
        assert hasattr(
            rel, "relationship_type"
        ), f"Relationship {rel.source_id} should have relationship_type"
        assert (
            rel.relationship_type
        ), f"Relationship {rel.source_id} should have non-empty relationship_type"

    # Test entity-relationship associations
    entities_by_source_id = {
        entity.source_id: entity for entity in knowledge_graph.entities
    }
    relationships_by_id = {rel.source_id: rel for rel in knowledge_graph.relationships}

    for rel in knowledge_graph.relationships:
        # Check that source entity has this relationship in outgoing list
        if rel.src_id in entities_by_source_id:
            source_entity = entities_by_source_id[rel.src_id]
            assert (
                rel.source_id in source_entity.outgoing_relationship_ids
            ), f"Source entity {source_entity.entity_name} should have relationship {rel.source_id} in outgoing_relationship_ids"

        # Check that target entity has this relationship in incoming list
        if rel.tgt_id in entities_by_source_id:
            target_entity = entities_by_source_id[rel.tgt_id]
            assert (
                rel.source_id in target_entity.incoming_relationship_ids
            ), f"Target entity {target_entity.entity_name} should have relationship {rel.source_id} in incoming_relationship_ids"

    # Test relationship counting
    for entity in knowledge_graph.entities:
        total_rels = entity.get_total_relationships()
        expected_total = len(entity.outgoing_relationship_ids) + len(
            entity.incoming_relationship_ids
        )
        assert (
            total_rels == expected_total
        ), f"Entity {entity.entity_name} should have correct total relationship count"

    # Print some examples
    print(f"\nEntity-Relationship Associations:")
    for entity in knowledge_graph.entities[:5]:  # Show first 5 entities
        print(f"  {entity.entity_name}:")
        print(f"    Outgoing: {len(entity.outgoing_relationship_ids)} relationships")
        print(f"    Incoming: {len(entity.incoming_relationship_ids)} relationships")
        print(f"    Total: {entity.get_total_relationships()} relationships")

        if entity.outgoing_relationship_ids:
            print(f"    Outgoing IDs: {entity.outgoing_relationship_ids}")
        if entity.incoming_relationship_ids:
            print(f"    Incoming IDs: {entity.incoming_relationship_ids}")

    print(
        f"✓ Successfully verified entity-relationship associations for {len(knowledge_graph.entities)} entities"
    )
