"""
test_explore.py - Tests for FOLIOExplorer class
"""

import pytest
from pathlib import Path
from typing import Dict, List
import json
import os
from datetime import datetime
from tqdm import tqdm

from kg.models._folio.explore import FOLIOExplorer


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
    assert isinstance(hierarchy, dict), "Hierarchy should be a dictionary"
    assert len(hierarchy) > 0, "Hierarchy should not be empty"

    # Check structure of hierarchy
    for class_label, parents in hierarchy.items():
        assert class_label is not None, "Class label should not be None"
        assert isinstance(
            class_label, str
        ), f"Class label should be string, got {type(class_label)}"
        assert isinstance(
            parents, list
        ), f"Parents should be a list, got {type(parents)}"

        # Check each parent
        for parent in parents:
            assert parent is not None, "Parent label should not be None"
            assert isinstance(
                parent, str
            ), f"Parent label should be string, got {type(parent)}"


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


def test_get_class_connections(explorer):
    """Test getting class connections."""
    connections = explorer.get_class_connections()
    assert isinstance(connections, dict)
    assert len(connections) > 0

    # Check structure of connections
    for class_label, class_conns in connections.items():
        assert class_label is not None, "Class label should not be None"
        assert isinstance(
            class_label, str
        ), f"Class label should be string, got {type(class_label)}"
        assert isinstance(
            class_conns, dict
        ), f"Connections should be dict, got {type(class_conns)}"
        assert "as_subject" in class_conns, "Connections should have 'as_subject' key"
        assert "as_object" in class_conns, "Connections should have 'as_object' key"
        assert isinstance(
            class_conns["as_subject"], list
        ), "as_subject should be a list"
        assert isinstance(class_conns["as_object"], list), "as_object should be a list"


def test_get_results_structure(explorer):
    """Test getting results structure."""
    results = explorer.get_results_structure()
    assert isinstance(results, dict)

    # Check required keys
    assert "connections" in results
    assert "graph" in results
    assert "taxonomy" in results

    # Check taxonomy structure
    taxonomy = results["taxonomy"]
    assert "classes" in taxonomy
    assert "properties" in taxonomy

    # Check classes structure
    for cls in taxonomy["classes"]:
        assert "label" in cls
        assert "iri" in cls
        assert "parents" in cls
        assert "children" in cls
        assert isinstance(cls["parents"], list)
        assert isinstance(cls["children"], list)

    # Check properties structure
    for prop in taxonomy["properties"]:
        assert "label" in prop
        assert "iri" in prop
        assert "domain" in prop
        assert "range" in prop
        assert isinstance(prop["domain"], list)
        assert isinstance(prop["range"], list)

    # Check graph structure
    graph = results["graph"]
    assert "nodes" in graph
    assert "edges" in graph
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)

    # Check connections structure
    connections = results["connections"]
    assert isinstance(connections, dict)
    for class_conns in connections.values():
        assert "as_subject" in class_conns
        assert "as_object" in class_conns
        assert isinstance(class_conns["as_subject"], list)
        assert isinstance(class_conns["as_object"], list)


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
        assert "classes" in loaded_results
        assert "properties" in loaded_results
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
    assert isinstance(results, dict)
    assert "query" in results
    assert "ontology_structure" in results
    assert "metadata" in results

    # Check ontology structure
    structure = results["ontology_structure"]
    assert "classes" in structure
    assert "properties" in structure
    assert "connections" in structure
    assert "graph" in structure

    # Verify content
    assert len(structure["classes"]) > 0
    assert len(structure["properties"]) > 0
    assert isinstance(structure["connections"], dict)
    assert "nodes" in structure["graph"]
    assert "edges" in structure["graph"]

    # Check metadata
    assert "version" in results["metadata"]
    assert "timestamp" in results["metadata"]


if __name__ == "__main__":
    pytest.main([__file__])
