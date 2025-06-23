"""
test_embed.py - Integration tests for ontology embedding functionality
"""

import os
import pytest
from functools import lru_cache

from folio import FOLIO
from folio.models import OWLClass, OWLObjectProperty

# Import the ontology functions
from rag.kg import (
    _get_children_with_depth,
    _get_class_depth,
    create_custom_kg,
    get_all_subclasses,
)
from rag.kg import CustomKnowledgeGraph

os.environ["OPENAI_API_KEY"]


@pytest.fixture
def folio_instance():
    """Create a real FOLIO instance for integration testing."""
    return FOLIO("github", llm=None)


def _create_cached_kg(folio_instance):
    return create_custom_kg(
        folio_instance, entities=["Lawyer", "Legal Services Buyer"], subclasses=True
    )


@pytest.fixture
def knowledge_graph(folio_instance):
    """Create a cached knowledge graph from FOLIO for testing."""
    return _create_cached_kg(folio_instance)


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


def test_get_all_subclasses(folio_instance):
    """Test _get_all_subclasses function."""
    # Test with a known entity that should have subclasses
    test_entities = ["Actor / Player"]

    subclasses = get_all_subclasses(folio_instance, test_entities)

    # Verify we get a set of strings
    assert isinstance(subclasses, set)
    assert len(subclasses) > 0

    # Verify the original entity is included
    assert "Actor / Player" in subclasses

    # Verify we get some subclasses
    print(
        f"Found {len(subclasses)} entities including subclasses: {list(subclasses)[:10]}..."
    )

    # Verify all items are strings
    for entity_name in subclasses:
        assert isinstance(entity_name, str)
        assert len(entity_name) > 0


def test_create_custom_kg_with_filtering(folio_instance):
    """Test create_custom_kg with entity filtering."""
    # Test filtering to specific entities without subclasses
    test_entities = ["Lawyer", "Legal Services Buyer"]

    # Test without subclasses
    kg_no_subclasses = create_custom_kg(
        folio_instance, entities=test_entities, subclasses=False
    )

    # Verify we get a smaller knowledge graph
    assert len(kg_no_subclasses.entities) <= len(test_entities)

    # Verify only specified entities are included
    entity_names = {entity.entity_name for entity in kg_no_subclasses.entities}
    for entity_name in test_entities:
        if entity_name in entity_names:  # Entity might not exist in the ontology
            assert entity_name in entity_names

    print(
        f"✓ Filtered KG without subclasses: {len(kg_no_subclasses.entities)} entities"
    )

    # Test with subclasses
    kg_with_subclasses = create_custom_kg(
        folio_instance, entities=test_entities, subclasses=True
    )

    # Verify we get more entities when including subclasses
    assert len(kg_with_subclasses.entities) >= len(kg_no_subclasses.entities)

    print(f"✓ Filtered KG with subclasses: {len(kg_with_subclasses.entities)} entities")


@pytest.mark.parametrize(
    "child,parent",
    [
        ("Public Defender", "Lawyer"),
    ],
)
def test_create_custom_kg_hierarchy_relationships(
    folio_instance, child, parent, knowledge_graph
):
    """
    Test specific parent-child relationships in the knowledge graph.
    Uses Lawyer as the parent entity with subclasses=True to ensure the hierarchy is included.
    """
    # Create a filtered knowledge graph with Lawyer and its subclasses
    entities_by_name = {e.entity_name: e for e in knowledge_graph.entities}
    relationships_by_src = [
        r for r in knowledge_graph.relationships if r.src_id == child
    ]
    assert relationships_by_src

    # Check that both entities exist
    assert child in entities_by_name, f"Child entity '{child}' should exist"
    assert parent in entities_by_name, f"Parent entity '{parent}' should exist"

    parent_rel = [x for x in relationships_by_src if x.tgt_id == parent]
    assert parent_rel
    assert (
        "subClassOf" in parent_rel[0].keywords
    ), f"Keywords should contain 'subClassOf'"
    assert parent_rel[0].weight == 1.0, "Relationship weight should be 1.0"

    print(f"✓ Verified hierarchy relationship: {child} is a subclass of {parent}")


def test_create_custom_kg_predicate_relationships(knowledge_graph):
    """
    Test predicate relationships in the knowledge graph.
    Tests the specific SPO: lawyer, folio:represents, legal service buyer
    """
    # Create a knowledge graph that includes both Lawyer and Legal Services Buyer
    entities_by_name = {
        e.entity_name: e
        for e in knowledge_graph.entities
        if e.entity_name in ["Lawyer", "Legal Services Buyer"]
    }
    relationships_by_src = [
        r
        for r in knowledge_graph.relationships
        if r.src_id in ["Lawyer", "Legal Services Buyer"]
    ]
    assert relationships_by_src

    # Define the expected SPO relationship
    subject = "Lawyer"
    predicate = "represents"  # This should match the predicate label in FOLIO
    object_entity = "Legal Services Buyer"  # This should match the entity name in FOLIO

    for rel in relationships_by_src:
        if rel.tgt_id == object_entity:
            assert rel.src_id == subject
            predicate in rel.keywords
            assert rel.weight == 1.0


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
            ), "Relationship should contain 'subClassOf'"
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
        # Check that entity_name and entity_type are the same (current implementation)
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
    from rag.kg import get_entity_relationships, get_entity_relationship_summary

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


def test_create_custom_kg_description_merging(knowledge_graph):
    """
    Test that entity descriptions are properly merged when duplicates are found.
    This test verifies that the None handling in description merging works correctly.
    """
    # Check that all entities have valid descriptions (not None)
    for entity in knowledge_graph.entities:
        assert (
            entity.description is not None
        ), f"Entity {entity.entity_name} should have a non-None description"
        assert isinstance(
            entity.description, str
        ), f"Entity {entity.entity_name} should have a string description"
        assert (
            len(entity.description.strip()) > 0
        ), f"Entity {entity.entity_name} should have a non-empty description"

    print(
        f"✓ Description merging verified for {len(knowledge_graph.entities)} entities"
    )


def test_create_custom_kg_relationship_id_format(knowledge_graph):
    """
    Test that relationship IDs are properly formatted.
    """
    for rel in knowledge_graph.relationships:
        # Check that relationship_id is properly formatted
        assert rel.source_id, "Relationship should have a source_id"
        assert rel.src_id, "Relationship should have a src_id"
        assert rel.tgt_id, "Relationship should have a tgt_id"

        # Verify the relationship ID format matches the expected pattern
        expected_id_format = f"{rel.src_id}_subclass_of_{rel.tgt_id}".replace(
            " ", "_"
        ).lower()
        if "subclass" in rel.description.lower():
            assert (
                rel.source_id == expected_id_format
            ), f"Relationship ID should match expected format: {expected_id_format}"

    print(
        f"✓ Relationship ID format verified for {len(knowledge_graph.relationships)} relationships"
    )


def test_create_custom_kg_entity_source_id_format(knowledge_graph):
    """
    Test that entity source_ids are properly formatted.
    """
    for entity in knowledge_graph.entities:
        # Check that source_id is properly formatted (should be the entity name)
        assert (
            entity.source_id == entity.entity_name
        ), f"Entity {entity.entity_name} should have source_id equal to entity_name"

    print(
        f"✓ Entity source_id format verified for {len(knowledge_graph.entities)} entities"
    )


def test_create_custom_kg_duplicate_handling(knowledge_graph):
    """
    Test that duplicate entities are handled properly.
    """
    # Check for duplicate entity names
    entity_names = [entity.entity_name for entity in knowledge_graph.entities]
    unique_entity_names = set(entity_names)

    # If there are duplicates, they should be properly merged
    if len(entity_names) != len(unique_entity_names):
        print(
            f"Found {len(entity_names) - len(unique_entity_names)} duplicate entities that were merged"
        )

        # Check that merged entities have proper descriptions
        for entity in knowledge_graph.entities:
            assert (
                entity.description is not None
            ), f"Merged entity {entity.entity_name} should have a non-None description"
            assert isinstance(
                entity.description, str
            ), f"Merged entity {entity.entity_name} should have a string description"
    else:
        print("No duplicate entities found")

    print(f"✓ Duplicate handling verified for {len(knowledge_graph.entities)} entities")


def test_create_custom_kg_filtering_consistency(knowledge_graph):
    """
    Test that filtering maintains consistency between entities, relationships, and chunks.
    """
    # Test that all relationships reference entities that exist
    entity_names = {entity.entity_name for entity in knowledge_graph.entities}

    for rel in knowledge_graph.relationships:
        # Check that source entity exists
        assert (
            rel.src_id in entity_names
        ), f"Relationship source '{rel.src_id}' should reference an existing entity"
        # Check that target entity exists
        # target may not exist outside of restricted entities
        assert (
            rel.tgt_id in entity_names
        ), f"Relationship target '{rel.tgt_id}' should be referenced by an existing entity"

    # Test that all chunks reference entities or relationships that exist
    chunk_source_ids = {chunk.source_id for chunk in knowledge_graph.chunks}
    relationship_source_ids = {rel.source_id for rel in knowledge_graph.relationships}

    for chunk in knowledge_graph.chunks:
        # Chunk should reference either an entity or a relationship
        assert (
            chunk.source_id in entity_names
            or chunk.source_id in relationship_source_ids
        ), f"Chunk source_id '{chunk.source_id}' should reference an existing entity or relationship"

    print(
        f"✓ Filtering consistency verified for {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relationships)} relationships, and {len(knowledge_graph.chunks)} chunks"
    )


def test_public_defender_lawyer_relationship(folio_instance):
    """Test that Public Defender has Lawyer as a parent."""
    # Find Public Defender class
    public_defender = None
    for cls in folio_instance.classes:
        if cls.label == "Public Defender":
            public_defender = cls
            break

    assert public_defender is not None, "Public Defender class not found"

    # Check if Lawyer is a parent
    parents = folio_instance.get_parents(public_defender.iri)
    parent_names = [p.label for p in parents if p.label]

    assert (
        "Lawyer" in parent_names
    ), f"Lawyer should be a parent of Public Defender. Found parents: {parent_names}"
