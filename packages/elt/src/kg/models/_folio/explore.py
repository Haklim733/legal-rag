"""
explore_folio.py - Script to explore FOLIO ontology structure
"""

from folio import FOLIO
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from pprint import pprint
from datetime import datetime
import time
from tqdm import tqdm


class FOLIOExplorer:
    _instance = None
    _folio: Optional[FOLIO] = None
    _init_time: Optional[float] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FOLIOExplorer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._folio is None:
            self._initialize_folio()

    def _initialize_folio(self) -> None:
        """Initialize FOLIO with caching."""
        start_time = time.time()
        self._folio = FOLIO("github", use_cache=True)
        self._init_time = time.time() - start_time

    @property
    def folio(self) -> FOLIO:
        """Get the FOLIO instance."""
        return self._folio

    @property
    def init_time(self) -> float:
        """Get the initialization time."""
        return self._init_time

    def get_folio_types(self) -> List:
        """Get all FOLIO types."""
        return self._folio.get_folio_types()

    def get_class_hierarchy(self) -> Dict[str, List[str]]:
        """Get the class hierarchy."""
        hierarchy = {}
        for cls in tqdm(self._folio.classes, desc="Building class hierarchy"):
            # Skip owl:Thing as it's the root class and doesn't need to be in our hierarchy
            if cls.iri == "http://www.w3.org/2002/07/owl#Thing":
                continue

            if cls.label is None:  # Skip classes with no label
                continue

            # Get parents and filter out None values and None labels
            parents = self._folio.get_parents(cls.iri)
            parent_labels = []
            for parent in parents:
                # Skip owl:Thing in parent relationships
                if (
                    parent is not None
                    and parent.iri != "http://www.w3.org/2002/07/owl#Thing"
                ):
                    if parent.label is not None:
                        parent_labels.append(parent.label)

            # Add to hierarchy regardless of whether we have parents
            hierarchy[cls.label] = parent_labels

        return hierarchy

    def get_properties(self) -> List:
        """Get all predicates (object properties)."""
        return self._folio.get_all_properties()

    def get_valid_predicates(self, class_iri: str) -> Tuple[List, List]:
        """Get valid predicates where a class can be subject or object."""
        if not class_iri or not isinstance(class_iri, str):
            raise ValueError("Invalid class IRI: must be a non-empty string")

        # Check if the class exists in the ontology
        if class_iri not in self._folio:
            raise ValueError(f"Class IRI not found in ontology: {class_iri}")

        all_properties = self._folio.get_all_properties()

        domain_properties = [
            prop
            for prop in all_properties
            if any(domain == class_iri for domain in prop.domain)
        ]

        range_properties = [
            prop
            for prop in all_properties
            if any(range_val == class_iri for range_val in prop.range)
        ]

        return domain_properties, range_properties

    def get_class_connections(self) -> Dict:
        """Get all possible class connections."""
        class_connections = {}
        for cls in tqdm(self._folio.classes, desc="Processing class connections"):
            if cls.label is None:  # Skip classes with no label
                continue
            connections = self._get_class_connections_for_class(cls.iri)
            class_connections[cls.label] = connections
        return class_connections

    def _get_class_connections_for_class(self, class_iri: str) -> Dict:
        """Get connections for a specific class."""
        domain_props, range_props = self.get_valid_predicates(class_iri)

        connections = {"as_subject": [], "as_object": []}

        # For each domain property, find valid range classes
        for prop in domain_props:
            valid_objects = []
            for range_val in prop.range:
                if self._folio[range_val] is not None:
                    valid_objects.append(
                        {"class": self._folio[range_val].label, "iri": range_val}
                    )

            if valid_objects:
                connections["as_subject"].append(
                    {
                        "property": prop.label,
                        "property_iri": prop.iri,
                        "valid_objects": valid_objects,
                    }
                )

        # For each range property, find valid domain classes
        for prop in range_props:
            valid_subjects = []
            for domain in prop.domain:
                if self._folio[domain] is not None:
                    valid_subjects.append(
                        {"class": self._folio[domain].label, "iri": domain}
                    )

            if valid_subjects:
                connections["as_object"].append(
                    {
                        "property": prop.label,
                        "property_iri": prop.iri,
                        "valid_subjects": valid_subjects,
                    }
                )

        return connections

    def get_results_structure(self) -> Dict:
        """Create the structured results dictionary."""
        properties = self.get_properties()
        class_connections = self.get_class_connections()

        return {
            "metadata": {
                "version": "1.0",
                "description": "FOLIO Ontology Exploration Results",
                "timestamp": str(datetime.now()),
            },
            "taxonomy": {
                "classes": [
                    {
                        "label": cls.label,
                        "iri": cls.iri,
                        "parents": [
                            parent.label
                            for parent in self._folio.get_parents(cls.iri)
                            if parent is not None
                        ],
                        "children": [
                            child.label
                            for child in self._folio.get_children(cls.iri)
                            if child is not None
                        ],
                        "type": cls.type if hasattr(cls, "type") else None,
                    }
                    for cls in self._folio.classes
                ],
                "properties": [
                    {
                        "label": prop.label,
                        "iri": prop.iri,
                        "domain": [
                            self._folio[domain].label
                            for domain in prop.domain
                            if self._folio[domain] is not None
                        ],
                        "range": [
                            self._folio[range_val].label
                            for range_val in prop.range
                            if self._folio[range_val] is not None
                        ],
                        "sub_property_of": [
                            self._folio[parent].label
                            for parent in prop.sub_property_of
                            if self._folio[parent] is not None
                        ],
                        "inverse_of": (
                            prop.inverse_of if hasattr(prop, "inverse_of") else None
                        ),
                    }
                    for prop in properties
                ],
            },
            "graph": {
                "nodes": [
                    {"id": cls.iri, "label": cls.label, "type": "class"}
                    for cls in self._folio.classes
                ]
                + [
                    {"id": prop.iri, "label": prop.label, "type": "property"}
                    for prop in properties
                ],
                "edges": [
                    {
                        "source": cls.iri,
                        "target": parent.iri,
                        "label": "subClassOf",
                        "type": "taxonomy",
                    }
                    for cls in self._folio.classes
                    for parent in self._folio.get_parents(cls.iri)
                    if parent is not None
                ],
            },
            "connections": class_connections,
        }

    def search_ontology(self, text: str) -> Dict:
        """
        Search the FOLIO ontology structure using LLM-based semantic matching.

        Args:
            text (str): The text to search for in the ontology

        Returns:
            Dict containing matched concepts and their relationships
        """
        # Get the full ontology structure
        structure = self.get_results_structure()

        # Prepare context for LLM
        context = {
            "classes": structure["taxonomy"]["classes"],
            "properties": structure["taxonomy"]["properties"],
            "connections": structure["connections"],
            "graph": structure["graph"],
        }

        # Return the context for LLM processing
        return {
            "query": text,
            "ontology_structure": context,
            "metadata": structure["metadata"],
        }

    def save_results(
        self, results: Dict, output_dir: Path = Path(".")
    ) -> Dict[str, Path]:
        """Save all results to files and return paths to saved files."""
        saved_paths = {}

        # Save full results
        output_path = output_dir / "folio_exploration_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        saved_paths["full_results"] = output_path

        # Save graph-specific format
        graph_output_path = output_dir / "folio_graph.json"
        with open(graph_output_path, "w") as f:
            json.dump(results["graph"], f, indent=2)
        saved_paths["graph"] = graph_output_path

        # Generate and save DOT file
        dot_content = self.generate_dot_file(results["graph"])
        dot_output_path = output_dir / "folio_graph.dot"
        with open(dot_output_path, "w") as f:
            f.write(dot_content)
        saved_paths["dot"] = dot_output_path

        return saved_paths

    def generate_dot_file(self, graph_data: Dict) -> str:
        """Generate a DOT file content from the graph data."""
        dot_content = [
            "digraph FOLIO {",
            "    // Graph settings",
            "    graph [",
            "        rankdir = TB,",
            "        splines = true,",
            "        nodesep = 0.8,",
            "        ranksep = 0.5,",
            '        fontname = "Arial"',
            "    ];",
            "",
            "    // Node settings",
            "    node [",
            "        shape = box,",
            "        style = filled,",
            "        fillcolor = lightblue,",
            '        fontname = "Arial"',
            "    ];",
            "",
            "    // Edge settings",
            "    edge [",
            '        fontname = "Arial"',
            "    ];",
            "",
            "    // Subgraph for classes",
            "    subgraph cluster_classes {",
            '        label = "Classes";',
            "        style = filled;",
            "        color = lightgrey;",
            "        node [style = filled, fillcolor = lightblue];",
        ]

        # Add class nodes
        for node in graph_data["nodes"]:
            if node["type"] == "class":
                dot_content.append(
                    f'        "{node["label"]}" [label="{node["label"]}"];'
                )

        dot_content.append("    }")

        # Add property nodes
        dot_content.extend(
            [
                "",
                "    // Subgraph for properties",
                "    subgraph cluster_properties {",
                '        label = "Properties";',
                "        style = filled;",
                "        color = lightgrey;",
                "        node [style = filled, fillcolor = lightgreen];",
            ]
        )

        for node in graph_data["nodes"]:
            if node["type"] == "property":
                dot_content.append(
                    f'        "{node["label"]}" [label="{node["label"]}"];'
                )

        dot_content.append("    }")

        # Add edges
        dot_content.extend(
            [
                "",
                "    // Edges",
            ]
        )

        for edge in graph_data["edges"]:
            source = next(
                node["label"]
                for node in graph_data["nodes"]
                if node["id"] == edge["source"]
            )
            target = next(
                node["label"]
                for node in graph_data["nodes"]
                if node["id"] == edge["target"]
            )
            dot_content.append(
                f'    "{source}" -> "{target}" [label="{edge["label"]}"];'
            )

        dot_content.append("}")

        return "\n".join(dot_content)


if __name__ == "__main__":
    explorer = FOLIOExplorer()
    results = explorer.explore_ontology()

    # The same instance will be reused
    explorer2 = FOLIOExplorer()
    assert explorer is explorer2  # True
