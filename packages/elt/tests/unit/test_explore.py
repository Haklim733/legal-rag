"""
test_explore.py - Tests for FOLIOExplorer class
"""

import pytest
from pathlib import Path
import json

from folio import FOLIO_TYPE_IRIS, OWLClass, OWLObjectProperty

from kg._folio.models.explorer import (
    ClassConnection,
    ClassConnections,
    ClassConnectionsStructure,
)
from kg._folio.explore import (
    FOLIOExplorer,
    OntologyMetadata,
    OntologyStructure,
    OntologySearchRequest,
    OntologySearchContext,
    OntologyClass,
    OntologyProperty,
    GraphNode,
    GraphEdge,
    OntologyGraph,
    TripleStructure,
    Triple,
    CompleteHierarchy,
    TypeHierarchy,
)


@pytest.fixture
def explorer():
    """Fixture to provide a FOLIOExplorer instance."""
    return FOLIOExplorer()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to provide a temporary output directory."""
    return tmp_path


def test_singleton_pattern():
    """Test that FOLIOExplorer maintains singleton pattern."""
    explorer1 = FOLIOExplorer()
    explorer2 = FOLIOExplorer()
    assert explorer1 is explorer2
    assert explorer1._folio is explorer2._folio


def test_initialization(explorer):
    """Test FOLIOExplorer initialization."""
    assert explorer._folio is not None
    assert explorer._init_time is not None
    assert explorer._init_time > 0


def test_get_folio_types(explorer):
    """Test getting FOLIO types."""
    types = explorer.get_folio_types()
    assert isinstance(types, list)
    assert len(types) > 0
    # Check that each type has required attributes
    for type_class in types:
        assert hasattr(type_class, "label")
        assert hasattr(type_class, "iri")


def test_get_class_hierarchy(explorer):
    """Test getting class hierarchy."""
    hierarchy = explorer.get_class_hierarchy()
    assert isinstance(hierarchy, ClassHierarchyStructure)
    assert len(hierarchy.hierarchies) > 0
    assert len(hierarchy.root_classes) > 0

    # Check structure of hierarchy
    for hierarchy_entry in hierarchy.hierarchies:
        assert hierarchy_entry.class_name is not None
        assert isinstance(hierarchy_entry.class_name, str)
        assert isinstance(hierarchy_entry.parent_labels, list)
        assert isinstance(hierarchy_entry.child_labels, list)
        assert isinstance(hierarchy_entry.level, int)

        # Check each parent
        for parent in hierarchy_entry.parent_labels:
            assert parent is not None
            assert isinstance(parent, str)


def test_get_properties(explorer):
    """Test getting properties."""
    properties = explorer.get_properties()
    assert isinstance(properties, list)
    assert len(properties) > 0
    # Check that each property has required attributes
    for prop in properties:
        assert hasattr(prop, "label")
        assert hasattr(prop, "iri")
        assert hasattr(prop, "domain")
        assert hasattr(prop, "range")


def test_get_valid_predicates(explorer):
    """Test getting valid predicates for a class."""
    # Get a class IRI from the first class
    class_iri = explorer._folio.classes[0].iri
    domain_props, range_props = explorer.get_valid_predicates(class_iri)

    assert isinstance(domain_props, list)
    assert isinstance(range_props, list)
    # Check that each property has required attributes
    for prop in domain_props + range_props:
        assert hasattr(prop, "label")
        assert hasattr(prop, "iri")
        assert hasattr(prop, "domain")
        assert hasattr(prop, "range")
        print(prop.label, prop.domain, prop.range)


def test_get_class_connections(explorer):
    """Test getting class connections."""
    connections = explorer.get_class_connections()
    assert isinstance(connections, ClassConnectionsStructure)
    assert connections.total_connections > 0
    assert connections.classes_with_connections > 0

    # Check structure of connections
    for class_label, class_conns in connections.connections.items():
        assert class_label is not None
        assert isinstance(class_label, str)
        assert isinstance(class_conns, ClassConnections)

        # Check subject connections
        for conn in class_conns.as_subject:
            assert isinstance(conn, ClassConnection)
            assert conn.property is not None
            assert conn.property_iri is not None
            assert isinstance(conn.valid_objects, list)

        # Check object connections
        for conn in class_conns.as_object:
            assert isinstance(conn, ClassConnection)
            assert conn.property is not None
            assert conn.property_iri is not None
            assert isinstance(conn.valid_subjects, list)


def test_get_results_structure(explorer):
    """Test getting results structure."""
    results = explorer.get_results_structure()
    assert isinstance(results, OntologyStructure)

    # Check taxonomy structure
    assert isinstance(results.taxonomy.classes, list)
    assert isinstance(results.taxonomy.properties, list)
    assert len(results.taxonomy.classes) > 0
    assert len(results.taxonomy.properties) > 0

    # Check classes structure
    for cls in results.taxonomy.classes:
        assert isinstance(cls, OntologyClass)
        assert cls.label is not None
        assert cls.iri is not None
        assert isinstance(cls.parents, list)
        assert isinstance(cls.children, list)
        assert isinstance(cls.alternative_labels, list)

    # Check properties structure
    for prop in results.taxonomy.properties:
        assert isinstance(prop, OntologyProperty)
        assert prop.label is not None
        assert prop.iri is not None
        assert isinstance(prop.domain, list)
        assert isinstance(prop.range, list)
        assert isinstance(prop.alternative_labels, list)

    # Check graph structure
    assert isinstance(results.graph, OntologyGraph)
    assert isinstance(results.graph.nodes, list)
    assert isinstance(results.graph.edges, list)

    for node in results.graph.nodes:
        assert isinstance(node, GraphNode)
        assert node.id is not None
        assert node.label is not None
        assert node.type in ["class", "property"]

    for edge in results.graph.edges:
        assert isinstance(edge, GraphEdge)
        assert edge.source is not None
        assert edge.target is not None
        assert edge.label is not None
        assert edge.type is not None


@pytest.mark.skip(reason="Skipping save_results test")
def test_save_results(explorer, temp_output_dir):
    """Test saving results to files."""
    results = explorer.get_results_structure()
    saved_paths = explorer.save_results(results, temp_output_dir)

    # Check that all expected files were created
    assert "full_results" in saved_paths
    assert "graph" in saved_paths
    assert "dot" in saved_paths

    # Check that files exist
    for path in saved_paths.values():
        assert path.exists()
        assert path.is_file()

    # Check JSON files can be loaded
    with open(saved_paths["full_results"], "r") as f:
        loaded_results = json.load(f)
        assert isinstance(loaded_results, dict)
        assert "taxonomy" in loaded_results
        assert "graph" in loaded_results
        assert "connections" in loaded_results


@pytest.mark.skip(reason="Skipping generate_dot_file test")
def test_generate_dot_file(explorer):
    """Test generating DOT file content."""
    results = explorer.get_results_structure()
    dot_content = explorer.generate_dot_file(results["graph"])

    assert isinstance(dot_content, str)
    assert "digraph FOLIO" in dot_content
    assert "subgraph cluster_classes" in dot_content
    assert "subgraph cluster_properties" in dot_content


def test_error_handling(explorer):
    """Test error handling for invalid inputs."""
    # Test with invalid class IRI
    with pytest.raises(ValueError, match="Class IRI not found in ontology"):
        explorer.get_valid_predicates("invalid_iri")

    # Test with empty IRI
    with pytest.raises(ValueError, match="Invalid class IRI"):
        explorer.get_valid_predicates("")

    # Test with non-string IRI
    with pytest.raises(ValueError, match="Invalid class IRI"):
        explorer.get_valid_predicates(123)  # type: ignore

    # Test with invalid output directory
    with pytest.raises(OSError):
        explorer.save_results({}, Path("/invalid/path"))


def test_identify_classes_without_labels(explorer):
    """Test to identify classes without labels in FOLIO."""
    classes_without_labels = []
    for cls in explorer._folio.classes:
        if cls.label is None:
            classes_without_labels.append(
                {"iri": cls.iri, "type": cls.type if hasattr(cls, "type") else None}
            )

    # Print the results
    if classes_without_labels:
        print("\nClasses without labels:")
        for cls in classes_without_labels:
            print(f"IRI: {cls['iri']}")
            print(f"Type: {cls['type']}")
            print("-" * 50)
    else:
        print("\nNo classes found without labels")

    # Assert that we found some classes without labels
    assert len(classes_without_labels) > 0, "Expected some classes without labels"


def test_search_ontology(explorer):
    """Test searching ontology with LLM-based semantic matching."""
    # Test with a sample query
    query = "Facilities Support Services"
    results = explorer.search_ontology(query)

    # Check basic structure
    assert isinstance(results, OntologySearchRequest)
    assert results.query == query
    assert isinstance(results.ontology_structure, OntologySearchContext)
    assert isinstance(results.metadata, OntologyMetadata)

    # Check ontology structure
    structure = results.ontology_structure
    assert isinstance(structure.classes, list)
    assert isinstance(structure.properties, list)
    assert isinstance(structure.connections, ClassConnectionsStructure)
    assert isinstance(structure.graph, OntologyGraph)

    # Verify content
    assert len(structure.classes) > 0
    assert len(structure.properties) > 0
    assert structure.connections.total_connections > 0
    assert len(structure.graph.nodes) > 0
    assert len(structure.graph.edges) > 0

    # Check metadata
    assert results.metadata.version is not None
    assert results.metadata.timestamp is not None


def test_create_domain_range_connections(explorer):
    """Test creating RDF triples for connection checking using property definitions."""
    # Get all triples from the ontology
    triple_structure = explorer.create_domain_range_connections()
    print(f"\nTotal triples found: {len(triple_structure.triples)}")

    # Verify the structure
    assert isinstance(triple_structure, TripleStructure)
    assert isinstance(triple_structure.triples, list)
    # Verify each triple
    for triple in triple_structure.triples:
        # Check triple structure
        assert isinstance(triple, Triple)
        assert isinstance(triple.subject, OWLClass)
        assert isinstance(triple.predicate, OWLObjectProperty)
        assert isinstance(triple.object, OWLClass)

        # Verify the triple components exist in the ontology
        assert explorer._folio[triple.subject.iri] is not None
        assert explorer._folio.get_property(triple.predicate.iri) is not None
        assert explorer._folio[triple.object.iri] is not None
        print(
            f"Triple: {triple.subject.label} --[{triple.predicate.label}]--> {triple.object.label}"
        )


def test_get_all_connections(explorer):
    """Test getting class connections from the ontology."""
    try:
        # Get class connections
        connections = explorer.get_all_connections()

        # Basic structure checks
        assert isinstance(connections, ClassConnectionsStructure)
        assert hasattr(connections, "connections")
        assert hasattr(connections, "total_connections")
        assert hasattr(connections, "classes_with_connections")

        # Verify we have connections
        assert connections.total_connections > 0, "Should have at least one connection"
        assert (
            connections.classes_with_connections > 0
        ), "Should have at least one class with connections"

        # Check each class's connections
        for class_label, class_conns in connections.connections.items():
            # Verify class label
            assert isinstance(class_label, str)
            assert class_label, "Class label should not be empty"

            # Verify connections structure
            assert isinstance(class_conns, ClassConnections)
            assert hasattr(class_conns, "as_subject")
            assert hasattr(class_conns, "as_object")

            # Check subject connections
            for conn in class_conns.as_subject:
                assert isinstance(conn, ClassConnection)
                assert conn.property, "Property should not be empty"
                assert conn.property_iri, "Property IRI should not be empty"
                assert conn.target_class, "Target class should not be empty"

                # Verify the property exists in FOLIO
                prop = explorer._folio.get_property(conn.property_iri)
                assert (
                    prop is not None
                ), f"Property {conn.property} should exist in FOLIO"

                # Verify the target class exists
                target = explorer._folio.get_by_label(conn.target_class)
                assert (
                    target is not None
                ), f"Target class {conn.target_class} should exist in FOLIO"

            # Check object connections
            for conn in class_conns.as_object:
                assert isinstance(conn, ClassConnection)
                assert conn.property, "Property should not be empty"
                assert conn.property_iri, "Property IRI should not be empty"
                assert conn.target_class, "Target class should not be empty"

                # Verify the property exists in FOLIO
                prop = explorer._folio.get_property(conn.property_iri)
                assert (
                    prop is not None
                ), f"Property {conn.property} should exist in FOLIO"

                # Verify the target class exists
                target = explorer._folio.get_class_by_label(conn.target_class)
                assert (
                    target is not None
                ), f"Target class {conn.target_class} should exist in FOLIO"

        # Print summary for debugging
        print("\nConnection Summary:")
        print(f"Total connections: {connections.total_connections}")
        print(f"Classes with connections: {connections.classes_with_connections}")

        # Print detailed connections for debugging
        print("\nDetailed Connections:")
        for class_label, class_conns in connections.connections.items():
            print(f"\n{class_label}:")
            if class_conns.as_subject:
                print("  As subject:")
                for conn in class_conns.as_subject:
                    print(f"    - {conn.property} -> {conn.target_class}")
            if class_conns.as_object:
                print("  As object:")
                for conn in class_conns.as_object:
                    print(f"    - {conn.target_class} -> {conn.property}")

    except Exception as e:
        # Print full error details
        print("\nFull Error Details:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        if hasattr(e, "errors"):
            print("\nValidation Errors:")
            for error in e.errors():
                print(f"Location: {error['loc']}")
                print(f"Message: {error['msg']}")
                print(f"Type: {error['type']}")
                print(f"Input: {error.get('input', 'No input')}")
                print("---")
        raise  # Re-raise the exception after printing details


def get_all_connections(self) -> ClassConnectionsStructure:
    """Get all class connections in the ontology, validating actual connections."""
    # Get initial triples based on domain/range definitions
    triple_structure = self.create_domain_range_connections()

    # Initialize connections dictionary
    connections = {}

    # Pre-compute subclass mappings for faster lookups
    subclass_cache = {}
    for triple in triple_structure.triples:
        subject_iri = triple.subject.iri
        object_iri = triple.object.iri

        # Cache subject subclasses - include all subclasses recursively
        if subject_iri not in subclass_cache:
            subject_class = self._folio[subject_iri]
            all_subclasses = [subject_class]
            # Get all subclasses recursively
            for subclass in self._folio.get_children(subject_iri):
                all_subclasses.append(subclass)
                # Get subclasses of subclasses
                all_subclasses.extend(self._folio.get_children(subclass.iri))
            subclass_cache[subject_iri] = all_subclasses
            logger.debug(
                f"Found {len(all_subclasses)} subclasses for {subject_class.label}"
            )

        # Cache object subclasses - include all subclasses recursively
        if object_iri not in subclass_cache:
            object_class = self._folio[object_iri]
            all_subclasses = [object_class]
            # Get all subclasses recursively
            for subclass in self._folio.get_children(object_iri):
                all_subclasses.append(subclass)
                # Get subclasses of subclasses
                all_subclasses.extend(self._folio.get_children(subclass.iri))
            subclass_cache[object_iri] = all_subclasses
            logger.debug(
                f"Found {len(all_subclasses)} subclasses for {object_class.label}"
            )

    # Process triples in batches by subject
    subject_triples = {}
    for triple in triple_structure.triples:
        subject_iri = triple.subject.iri
        if subject_iri not in subject_triples:
            subject_triples[subject_iri] = []
        subject_triples[subject_iri].append(triple)

    # Process each subject's triples
    for subject_iri, triples in subject_triples.items():
        subject_class = self._folio[subject_iri]
        if subject_class.label is None:
            continue

        # Initialize connections for this subject
        if subject_class.label not in connections:
            connections[subject_class.label] = ClassConnections(
                as_subject=[], as_object=[]
            )

        # Get all subclasses for this subject
        subject_subclasses = subclass_cache[subject_iri]

        # Process each triple for this subject
        for triple in triples:
            prop = triple.predicate
            object_subclasses = subclass_cache[triple.object.iri]

            # Validate connections for each subject subclass with each object subclass
            for subject_subclass in subject_subclasses:
                if subject_subclass.label is None:
                    continue

                # Find all valid connections for this subject subclass and property
                found_connections = self._folio.find_connections(
                    subject_class=subject_subclass, property_name=prop.label
                )

                # Create a set of valid object IRIs for faster lookup
                valid_object_iris = {conn[2].iri for conn in found_connections}

                # Add connections for valid object subclasses
                for object_subclass in object_subclasses:
                    if object_subclass.label is None:
                        continue

                    if object_subclass.iri in valid_object_iris:
                        connection = ClassConnection(
                            property=prop.label,
                            property_iri=prop.iri,
                            target_class=object_subclass.label,
                        )
                        # Add to connections if not already present
                        if (
                            connection
                            not in connections[subject_class.label].as_subject
                        ):
                            connections[subject_class.label].as_subject.append(
                                connection
                            )
                            logger.debug(
                                f"Added validated connection: {subject_subclass.label} --[{prop.label}]--> {object_subclass.label}"
                            )

    # Calculate totals
    total_connections = sum(
        len(conn.as_subject) + len(conn.as_object) for conn in connections.values()
    )
    classes_with_connections = len(connections)

    return ClassConnectionsStructure(
        connections=connections,
        total_connections=total_connections,
        classes_with_connections=classes_with_connections,
    )


def test_traverse_folio_taxonomy(explorer):
    """Test traversing the FOLIO taxonomy and verifying the nested structure."""
    # Get the taxonomy structure
    taxonomy = explorer.traverse_folio_taxonomy()

    def print_subclass_hierarchy(
        subclass_iri: str, level: int = 0, visited: set = None
    ):
        """Helper function to print hierarchy starting from a specific subclass"""
        if visited is None:
            visited = set()

        # Skip if we've already visited this node
        if subclass_iri in visited:
            return

        visited.add(subclass_iri)
        subclass_class = explorer._folio[subclass_iri]
        indent = "  " * level
        print(f"{indent}└─ {subclass_class.label} ({subclass_iri})")

        # Get children from class_edges
        children = explorer._folio.class_edges.get(subclass_iri, [])
        if children:
            print(f"{indent}   Children:")
            for child_iri in children:
                if child_iri not in visited:  # Only print if not visited
                    print(
                        f"{indent}   └─ {explorer._folio[child_iri].label} ({child_iri})"
                    )
                    # Recursively print hierarchy for each child
                    print_subclass_hierarchy(child_iri, level + 2, visited)

    # First, let's find some interesting subclasses to traverse
    legal_entity_type = taxonomy.types["Legal Entity"]
    print("\nLegal Entity Subclasses:")
    for subclass in legal_entity_type.subclasses:
        subclass_class = explorer._folio[subclass.iri]
        print(f"\nStarting hierarchy from: {subclass_class.label}")
        print("=" * 80)
        print_subclass_hierarchy(subclass.iri)

    # Let's also try with Actor / Player
    actor_type = taxonomy.types["Actor / Player"]
    print("\nActor / Player Subclasses:")
    for subclass in actor_type.subclasses:
        subclass_class = explorer._folio[subclass.iri]
        print(f"\nStarting hierarchy from: {subclass_class.label}")
        print("=" * 80)
        print_subclass_hierarchy(subclass.iri)

    # Verify the structure is valid
    for type_name, type_taxonomy in taxonomy.types.items():
        # Verify type IRI
        assert (
            type_taxonomy.iri in FOLIO_TYPE_IRIS.values()
        ), f"Type IRI {type_taxonomy.iri} should be valid"

        # Verify each subclass
        for subclass in type_taxonomy.subclasses:
            # Verify subclass exists in FOLIO
            subclass_class = explorer._folio[subclass.iri]
            assert (
                subclass_class is not None
            ), f"Subclass {subclass.iri} should exist in FOLIO"

            # Verify children are valid
            for child_iri in subclass.children:
                child_class = explorer._folio[child_iri]
                assert (
                    child_class is not None
                ), f"Child {child_iri} should exist in FOLIO"

            # Verify parents are valid
            for parent_iri in subclass.parents:
                parent_class = explorer._folio[parent_iri]
                assert (
                    parent_class is not None
                ), f"Parent {parent_iri} should exist in FOLIO"


def test_build_complete_hierarchy(explorer):
    """Test building and caching the complete class hierarchy using BFS."""
    # Debug: Print all labels for Engagement Attributes
    engagement_class = explorer._folio.get_by_label("Engagement Attributes")[0]
    print("\nEngagement Attributes class details:")
    print(f"IRI: {engagement_class.iri}")
    print(f"Label: {engagement_class.label}")
    print(f"Preferred Label: {engagement_class.preferred_label}")
    print(f"Alternative Labels: {engagement_class.alternative_labels}")
    print(f"Hidden Label: {engagement_class.hidden_label}")
    print("\nAll triples for this class:")
    for triple in explorer._folio.get_triples_by_subject(engagement_class.iri):
        print(f"{triple[1]}: {triple[2]}")

    # Clear any existing cache
    explorer.clear_hierarchy_cache()

    # Build the hierarchy
    hierarchy = explorer.build_complete_hierarchy()

    # Basic structure validation
    assert isinstance(hierarchy, CompleteHierarchy)
    assert isinstance(hierarchy.hierarchies, dict)
    assert len(hierarchy.hierarchies) > 0

    # Test each type hierarchy
    for type_label, type_hierarchy in hierarchy.hierarchies.items():
        print(f"\nTesting hierarchy for {type_label}")

        # Get the root class for this type
        root_classes = explorer._folio.get_by_label(type_label)
        assert len(root_classes) > 0, f"Root class {type_label} not found in FOLIO"
        root_class = root_classes[0]

        # Verify root class is at level 0
        assert 0 in type_hierarchy.levels, f"No level 0 found for {type_label}"
        root_level_classes = type_hierarchy.levels[0]
        assert any(
            cls.iri == root_class.iri for cls in root_level_classes
        ), f"Root class {type_label} not found at level 0"

        # Verify all classes in the hierarchy exist in FOLIO
        all_classes = set()
        for level, classes in type_hierarchy.levels.items():
            for cls in classes:
                # Verify class exists in FOLIO
                folio_class = explorer._folio[cls.iri]
                assert folio_class is not None, f"Class {cls.iri} not found in FOLIO"
                all_classes.add(cls.iri)

        # Verify all subclasses from FOLIO are in the hierarchy
        def get_all_subclasses(class_iri, visited=None):
            if visited is None:
                visited = set()
            if class_iri in visited:
                return set()
            visited.add(class_iri)

            subclasses = set()
            for child_iri in explorer._folio.class_edges.get(class_iri, []):
                subclasses.add(child_iri)
                subclasses.update(get_all_subclasses(child_iri, visited))
            return subclasses

        # Get all subclasses from FOLIO
        folio_subclasses = get_all_subclasses(root_class.iri)

        # Verify all FOLIO subclasses are in our hierarchy
        missing_classes = folio_subclasses - all_classes
        assert (
            not missing_classes
        ), f"Missing classes in hierarchy for {type_label}: {missing_classes}"

        # Build a map of class to its level for easier lookup
        class_to_level = {}
        for level, classes in type_hierarchy.levels.items():
            for cls in classes:
                class_to_level[cls.iri] = level

        # Verify hierarchy levels are correct
        for level, classes in type_hierarchy.levels.items():
            for cls in classes:
                if cls.iri == root_class.iri:
                    continue  # Skip root class

                # Get all parents from FOLIO
                parents = set()
                for parent_iri, children in explorer._folio.class_edges.items():
                    if cls.iri in children:
                        parents.add(parent_iri)

                # Skip if parent is owl:Thing
                parents = {
                    p for p in parents if p != "http://www.w3.org/2002/07/owl#Thing"
                }

                if parents:
                    # Find parent levels
                    parent_levels = set()
                    for parent_iri in parents:
                        if parent_iri in class_to_level:
                            parent_levels.add(class_to_level[parent_iri])
                        else:
                            print(
                                f"Warning: Parent {parent_iri} not found in hierarchy"
                            )

                    if parent_levels:
                        print(f"\nClass: {cls.iri}")
                        print(f"Level: {level}")
                        print(f"Parents: {parents}")
                        print(f"Parent levels: {parent_levels}")

                        # Verify this class is at a higher level than all its parents
                        assert all(
                            pl < level for pl in parent_levels
                        ), f"Class {cls.iri} at level {level} should be at a higher level than its parents at {parent_levels}"


def test_print_type_hierarchy(explorer):
    """Test printing the type hierarchy."""
    explorer.print_hierarchy_from_label("Proceeding Status")
    hierarchy = explorer.get_hierarchy_from_label("Proceeding Status")
    assert hierarchy is not None
    assert isinstance(hierarchy, dict)
    assert len(hierarchy) > 0
    assert hierarchy[0] is not None
    assert hierarchy[0][0].label == "Proceeding Status"
    iri = explorer._folio.get_by_label("Proceeding Status")[0].iri
    assert hierarchy[0][0].iri == iri
    for _, classes in hierarchy.items():
        for obj in classes:
            assert isinstance(obj, OWLClass)


def test_get_hierarchy_from_label(explorer):
    """Test getting hierarchy from label."""
    # Test with a known type
    hierarchy = explorer.get_hierarchy_from_label("Legal Entity")

    # Verify hierarchy structure
    assert isinstance(hierarchy, dict)
    assert len(hierarchy) > 0  # Should have at least one level

    # Verify each level contains OWLClass objects
    for level, classes in hierarchy.items():
        assert isinstance(level, int)
        assert isinstance(classes, list)
        assert all(isinstance(cls, OWLClass) for cls in classes)

    # Verify root class is present
    root_classes = hierarchy[0]  # Level 0 should contain root classes
    assert len(root_classes) > 0
    assert all(isinstance(cls, OWLClass) for cls in root_classes)

    # Verify subclasses exist
    assert any(len(classes) > 0 for level, classes in hierarchy.items() if level > 0)

    # Test with invalid type
    with pytest.raises(ValueError):
        explorer.get_hierarchy_from_label("Invalid Type")


def test_get_hierarchy_from_iri(explorer):
    """Test getting hierarchy from an IRI."""
    # First get a known IRI
    matches = explorer._folio.get_by_label("Proceeding Status")
    assert len(matches) > 0, "Should find Proceeding Status"
    test_iri = matches[0].iri

    # Get hierarchy from IRI
    hierarchy = explorer.get_hierarchy_from_iri(test_iri)

    # Verify the hierarchy structure
    assert isinstance(hierarchy, dict), "Hierarchy should be a dictionary"
    assert len(hierarchy) > 0, "Hierarchy should not be empty"

    # Verify each level contains a list of OWLClass objects
    for level, classes in hierarchy.items():
        assert isinstance(level, int), "Level should be an integer"
        assert isinstance(classes, list), "Classes should be a list"
        assert all(
            isinstance(cls, OWLClass) for cls in classes
        ), "All items should be OWLClass instances"

    # Verify the root class
    root_classes = hierarchy[0]
    assert len(root_classes) == 1, "Should have exactly one root class"
    root_class = root_classes[0]
    assert root_class.iri == test_iri, "Root class should match the input IRI"

    # Verify there are subclasses
    assert any(
        len(classes) > 0 for level, classes in hierarchy.items() if level > 0
    ), "Should have subclasses"
