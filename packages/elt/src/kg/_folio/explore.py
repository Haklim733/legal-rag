"""
explore_folio.py - Script to explore FOLIO ontology structure
"""

from functools import lru_cache
import hashlib
from folio import FOLIO, FOLIOTypes, FOLIO_TYPE_IRIS, OWLClass
from typing import Dict, List, Tuple, Optional, Set, Deque
from collections import deque, defaultdict
from pathlib import Path
import json
import time
import logging
from pydantic import BaseModel, Field

from typing import Dict, List, Optional
from openai import OpenAI
from kg._folio.models.explorer import (
    CompleteHierarchy,
    TypeHierarchy,
    OntologySearchRequest,
    OntologySearchResult,
    OntologySearchContext,
    OntologyStructure,
    OntologyMetadata,
    OntologyTaxonomy,
    OntologyGraph,
    OntologyClass,
    OntologyProperty,
    GraphNode,
    GraphEdge,
    ClassConnections,
    ClassConnection,
    ClassConnectionsStructure,
    SubclassInfo,
    TypeTaxonomy,
    TaxonomyStructure,
    Triple,
    TripleStructure,
)

logger = logging.getLogger(__name__)


class FOLIOExplorer:
    _instance = None
    _folio: Optional[FOLIO] = None
    _init_time: Optional[float] = None
    _all_properties: Optional[List[OntologyProperty]] = None
    _complete_hierarchy: Optional[CompleteHierarchy] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FOLIOExplorer, cls).__new__(cls)
        return cls._instance

    def __init__(self, folio: Optional[FOLIO] = None):
        self._complete_hierarchy = None
        self._initialize_folio(folio)

    def _initialize_folio(self, folio: Optional[FOLIO] = None) -> None:
        """Initialize FOLIO with caching."""
        start_time = time.time()
        self._folio = folio or FOLIO("github", use_cache=True)
        self._init_time = time.time() - start_time
        self._all_properties = self._folio.get_all_properties()
        self._complete_hierarchy = self.build_complete_hierarchy()

    @lru_cache(maxsize=1)
    def build_complete_hierarchy(self) -> CompleteHierarchy:
        """
        Build the complete class hierarchy for all FOLIO types using BFS and topological sorting.
        This method is cached to avoid rebuilding the hierarchy on every call.

        Returns:
            CompleteHierarchy: Pydantic model containing the complete hierarchy structure
        """
        hierarchies = {}

        # Build hierarchy for each FOLIO type
        for folio_type in FOLIOTypes:
            try:
                # Get the IRI for this type and normalize it
                type_iri = self._folio.normalize_iri(FOLIO_TYPE_IRIS[folio_type])
                # Get the hierarchy using normalized IRI
                levels = self.get_hierarchy_from_iri(type_iri)

                # Create TypeHierarchy instance
                type_hierarchy = TypeHierarchy(
                    type_label=folio_type.value, levels=levels
                )
                hierarchies[folio_type.value] = type_hierarchy

            except ValueError as e:
                logging.warning(
                    f"Could not build hierarchy for {folio_type.value}: {str(e)}"
                )
                continue

        # Create and return CompleteHierarchy instance
        return CompleteHierarchy(hierarchies=hierarchies)

    def clear_hierarchy_cache(self, rebuild: bool = False) -> None:
        """
        Clear the cached complete hierarchy and optionally rebuild it.

        Args:
            rebuild (bool): If True, rebuild the hierarchy after clearing the cache.
                          Defaults to False.
        """
        self.__complete_hierarchy = None
        if rebuild:
            self._complete_hierarchy = self.build_complete_hierarchy()

    def _get_cache_key(self, triple_structure: TripleStructure) -> Tuple[str, float]:
        """Generate a cache key based on the triple structure and current time"""
        # Create a hash of the relevant triple data
        triple_data = [
            (t.subject.iri, t.predicate.iri, t.object.iri)
            for t in triple_structure.triples
        ]
        # Include timestamp in the key for expiration
        current_time = time.time()
        return (hashlib.md5(json.dumps(triple_data).encode()).hexdigest(), current_time)

    @lru_cache(maxsize=32)
    def _cached_get_all_connections(
        self, cache_key: Tuple[str, float]
    ) -> Optional[ClassConnectionsStructure]:
        """Cached version of get_all_connections with expiration"""
        # Check if cache has expired
        if time.time() - cache_key[1] > self._cache_ttl:
            return None
        return self._get_all_connections_impl()

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

    def get_properties(self) -> List:
        """Get all predicates (object properties)."""
        return self.all_properties

    def get_valid_predicates(self, class_iri: str) -> Tuple[List, List]:
        """Get valid predicates where a class can be subject or object."""
        if not class_iri or not isinstance(class_iri, str):
            raise ValueError("Invalid class IRI: must be a non-empty string")

        # Check if the class exists in the ontology
        if class_iri not in self._folio:
            raise ValueError(f"Class IRI not found in ontology: {class_iri}")

        domain_properties = [
            prop
            for prop in self._all_properties
            if any(domain == class_iri for domain in prop.domain)
        ]

        range_properties = [
            prop
            for prop in self._all_properties
            if any(range_val == class_iri for range_val in prop.range)
        ]

        return domain_properties, range_properties

    def create_domain_range_connections(self) -> TripleStructure:
        """
        Create a mapping of classes to their properties and possible range classes based on domain/range definitions.
        This is a fast initial mapping that doesn't validate actual connections.

        Returns:
            TripleStructure: A Pydantic model containing a list of RDF triples
        """
        # Get all properties from the ontology
        all_properties = self._all_properties
        logger.debug(f"Total properties found: {len(all_properties)}")

        # Create a list to store the triples
        triples = []

        # For each property, get its domain and range classes
        for prop in all_properties:
            if not prop.label:  # Skip properties without labels
                continue

            logger.debug(f"\nProcessing property: {prop.label}")

            # Get all domain classes (subjects) - using set for faster lookups
            domain_iris = set()
            for domain_iri in prop.domain:
                # domain_class = self._folio[domain_iri]
                domain_iris.add(domain_iri)

            if not domain_iris:
                logger.debug(f"No domain classes found for property: {prop.label}")
                continue

            # Get all range classes (objects) - using set for faster lookups
            range_iris = set()
            for range_iri in prop.range:
                range_iris.add(range_iri)

            if not range_iris:
                logger.debug(f"No range classes found for property: {prop.label}")
                continue

            # Get all valid connections for this domain class and property
            for domain_iri in domain_iris:
                connections = self._folio.find_connections(
                    subject_class=domain_iri,
                    property_name=prop.label,
                )

                #  Create triples only for valid connections
                for subject, predicate, obj in connections:
                    logger.debug(predicate)
                    logger.debug(type(predicate))
                    triple = Triple(subject=subject, predicate=predicate, object=obj)
                    triples.append(triple)
                    logger.debug(
                        f"Added valid triple: {subject.label} --[{predicate.label}]--> {obj.label}"
                    )

        return TripleStructure(triples=triples)

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

            # Cache subject subclasses
            if subject_iri not in subclass_cache:
                subclass_cache[subject_iri] = [triple.subject] + list(
                    self._folio.get_children(subject_iri)
                )

            # Cache object subclasses
            if object_iri not in subclass_cache:
                subclass_cache[object_iri] = [triple.object] + list(
                    self._folio.get_children(object_iri)
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

    def get_results_structure(self) -> OntologyStructure:
        """Create the structured results using Pydantic models."""
        properties = self.get_properties()
        class_connections = self.get_class_connections()

        # Create classes
        classes = [
            OntologyClass(
                label=cls.label,
                iri=cls.iri,
                preferred_label=cls.preferred_label,
                alternative_labels=cls.alternative_labels,
                hidden_label=cls.hidden_label,
                translations=cls.translations,
                definition=cls.definition,
                parents=[
                    parent.label
                    for parent in self._folio.get_parents(cls.iri)
                    if parent is not None and parent.label is not None
                ],
                children=[
                    child.label
                    for child in self._folio.get_children(cls.iri)
                    if child is not None and child.label is not None
                ],
                type=cls.type if hasattr(cls, "type") else None,
            )
            for cls in self._folio.classes
            if cls.label is not None
        ]

        # Create properties
        properties_list = [
            OntologyProperty(
                label=prop.label,
                iri=prop.iri,
                preferred_label=prop.preferred_label,
                alternative_labels=prop.alternative_labels,
                definition=prop.definition,
                domain=[
                    self._folio[domain].label
                    for domain in prop.domain
                    if self._folio[domain] is not None
                    and self._folio[domain].label is not None
                ],
                range=[
                    self._folio[range_val].label
                    for range_val in prop.range
                    if self._folio[range_val] is not None
                    and self._folio[range_val].label is not None
                ],
                sub_property_of=[
                    self._folio[parent].label
                    for parent in prop.sub_property_of
                    if self._folio[parent] is not None
                    and self._folio[parent].label is not None
                ],
                inverse_of=prop.inverse_of if hasattr(prop, "inverse_of") else None,
            )
            for prop in properties
            if prop.label is not None  # Skip properties with no label
        ]

        # Create graph nodes
        nodes = [
            GraphNode(
                id=cls.iri,
                label=cls.label,
                preferred_label=cls.preferred_label,
                alternative_labels=cls.alternative_labels,
                type="class",
            )
            for cls in self._folio.classes
        ] + [
            GraphNode(
                id=prop.iri,
                label=prop.label,
                preferred_label=prop.preferred_label,
                alternative_labels=prop.alternative_labels,
                type="property",
            )
            for prop in properties
        ]

        # Create graph edges
        edges = [
            GraphEdge(
                source=cls.iri,
                target=parent.iri,
                label="subClassOf",
                type="taxonomy",
            )
            for cls in self._folio.classes
            for parent in self._folio.get_parents(cls.iri)
            if parent is not None
        ]

        # Create the complete structure
        return OntologyStructure(
            metadata=OntologyMetadata(),
            taxonomy=OntologyTaxonomy(classes=classes, properties=properties_list),
            graph=OntologyGraph(nodes=nodes, edges=edges),
            connections=class_connections,
        )

    def search_ontology(self, text: str) -> OntologySearchRequest:
        """
        Search the FOLIO ontology structure using LLM-based semantic matching.

        Args:
            text (str): The text to search for in the ontology

        Returns:
            OntologySearchRequest containing the search context and metadata
        """
        # Get the full ontology structure
        structure = self.get_results_structure()

        # Create search context
        search_context = OntologySearchContext(
            query=text,
            classes=structure.taxonomy.classes,
            properties=structure.taxonomy.properties,
            connections=structure.connections,
            graph=structure.graph,
        )

        # Create and return the complete search request
        return OntologySearchRequest(
            query=text, ontology_structure=search_context, metadata=structure.metadata
        )

    def save_results(
        self, results: OntologyStructure, output_dir: Path = Path(".")
    ) -> Dict[str, Path]:
        """Save all results to files and return paths to saved files."""
        saved_paths = {}

        # Save full results
        output_path = output_dir / "folio_exploration_results.json"
        with open(output_path, "w") as f:
            json.dump(results.model_dump(), f, indent=2)
        saved_paths["full_results"] = output_path

        # Save graph-specific format
        graph_output_path = output_dir / "folio_graph.json"
        with open(graph_output_path, "w") as f:
            json.dump(results.graph.model_dump(), f, indent=2)
        saved_paths["graph"] = graph_output_path

        # Generate and save DOT file
        dot_content = self.generate_dot_file(results.graph)
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

    def search_ontology_in_text(
        self, text: str, client: OpenAI
    ) -> OntologySearchResult:
        """
        Search for FOLIO ontology concepts in email text using LLM.

        Args:
            explorer: FOLIOExplorer instance
            text: The text to search through
            client: OpenAI client with instructor patch

        Returns:
            OntologySearchResult containing matched concepts and their relationships
        """
        # Get the complete ontology structure
        ontology_structure = explorer.get_results_structure()

        # Create the prompt for the LLM
        prompt = f"""
        Analyze this email text and identify relevant FOLIO ontology concepts.
        Consider both main labels and alternative labels when matching concepts.
        
        Text:
        {text}
        
        FOLIO Ontology Structure:
        {ontology_structure}
        
        For each concept found in the email:
        1. Identify if it matches a FOLIO class or property (including alternative labels)
        2. Provide the exact text that matched
        3. Include the surrounding context
        4. Assign a relevance score (0-10)
        5. Explain why it's relevant
        6. Note any relationships between matched concepts
        
        Return the results in a structured format.
        """
        # Use instructor to get structured response
        response = client.chat.completions.create(
            model="gpt-4",
            response_model=OntologySearchResult,
            messages=[{"role": "user", "content": prompt}],
        )

        return response

    def traverse_folio_taxonomy(self) -> TaxonomyStructure:
        """
        Traverse the FOLIO taxonomy starting from the top-level FOLIOTypes.
        Returns a complete taxonomy structure with all subclasses and their relationships.

        Returns:
            TaxonomyStructure: Pydantic model containing the taxonomy structure
        """
        taxonomy = {}

        # Iterate through each top-level FOLIO type
        for folio_type in FOLIOTypes:
            # Get the IRI for this type
            type_iri = FOLIO_TYPE_IRIS[folio_type]

            # Get all subclasses recursively
            def get_all_subclasses(
                class_iri: str, visited: set = None
            ) -> List[SubclassInfo]:
                if visited is None:
                    visited = set()

                if class_iri in visited:
                    return []

                visited.add(class_iri)
                subclasses = []

                # Get direct children
                children = self._folio.get_children(class_iri)
                for child in children:
                    if child.iri not in visited:
                        # Get parents and children for this subclass
                        parents = [p.iri for p in self._folio.get_parents(child.iri)]
                        child_children = [
                            c.iri for c in self._folio.get_children(child.iri)
                        ]

                        subclass_info = SubclassInfo(
                            iri=child.iri, children=child_children, parents=parents
                        )
                        subclasses.append(subclass_info)

                        # Recursively get subclasses of this child
                        subclasses.extend(get_all_subclasses(child.iri, visited))

                return subclasses

            # Build the taxonomy structure for this type
            type_taxonomy = TypeTaxonomy(
                iri=type_iri, subclasses=get_all_subclasses(type_iri)
            )

            taxonomy[folio_type.value] = type_taxonomy

        return TaxonomyStructure(types=taxonomy)

    def get_subclasses(self, class_iri: str) -> List[str]:
        """
        Get all subclasses of a class using the triples.

        Args:
            class_iri: The IRI of the class to get subclasses for

        Returns:
            List[str]: List of all subclass IRIs
        """
        return [
            subject
            for subject, predicate, obj in self._folio.triples
            if predicate == "rdfs:subClassOf" and obj == class_iri
        ]

    def get_all_connections_for_class(self, class_label: str) -> Dict[str, List[str]]:
        """
        Get all connections for a class using the existing FOLIOExplorer structure.

        Args:
            class_label: The label of the class to get connections for

        Returns:
            Dict[str, List[str]]: Dictionary mapping property labels to lists of connected class labels
        """
        # Get the class connections
        connections = self.get_all_connections()

        if class_label not in connections.connections:
            return {}

        class_conns = connections.connections[class_label]

        # Format the connections
        result = {
            "as_subject": {
                conn.property: conn.target_class for conn in class_conns.as_subject
            },
            "as_object": {
                conn.property: conn.target_class for conn in class_conns.as_object
            },
        }

        return result

    def traverse_from_type(self, type_label: str) -> Dict[int, Set[str]]:
        """
        Traverse the hierarchy starting from a specific type (e.g., 'Legal Entity').

        Args:
            type_label: The label of the type to start traversal from

        Returns:
            Dict[int, Set[str]]: Dictionary mapping levels to sets of class IRIs
        """
        # Find the type's IRI
        type_matches = self._folio.get_by_label(type_label)
        if not type_matches:
            raise ValueError(f"Type '{type_label}' not found in FOLIO")
        type_iri = type_matches[0].iri

        # Get the hierarchy for this type
        if type_iri not in self._complete_hierarchy:
            raise ValueError(f"No hierarchy found for type '{type_label}'")

        return self._complete_hierarchy[type_iri]

    def traverse_from_level(self, level: int, type_label: str) -> Dict[int, Set[str]]:
        """
        Traverse the hierarchy starting from a specific level of a type.

        Args:
            level: The level to start traversal from
            type_label: The label of the type to traverse

        Returns:
            Dict[int, Set[str]]: Dictionary mapping levels to sets of class IRIs
        """
        # Get the full hierarchy for the type
        hierarchy = self.traverse_from_type(type_label)

        # Filter to only include levels >= the specified level
        return {l: classes for l, classes in hierarchy.items() if l >= level}

    def get_subclasses_at_level(self, type_label: str, level: int) -> Set[str]:
        """
        Get all subclasses at a specific level for a type.

        Args:
            type_label: The label of the type to get subclasses for
            level: The level to get subclasses from

        Returns:
            Set[str]: Set of class IRIs at the specified level
        """
        hierarchy = self.traverse_from_type(type_label)
        return hierarchy.get(level, set())

    def get_subtree_from_node(self, node_iri: str) -> Dict[int, Set[str]]:
        """
        Get the subtree starting from a specific node.

        Args:
            node_iri: The IRI of the node to start from

        Returns:
            Dict[int, Set[str]]: Dictionary mapping levels to sets of class IRIs
        """
        # Find which type this node belongs to
        for type_iri, hierarchy in self._complete_hierarchy.items():
            # Check all levels for the node
            for level, nodes in hierarchy.items():
                if node_iri in nodes:
                    # Found the node, now get its subtree
                    subtree = defaultdict(set)
                    current_level = 0

                    # Add the node as root of subtree
                    subtree[current_level].add(node_iri)

                    # Get all subclasses at deeper levels
                    for deeper_level, nodes in hierarchy.items():
                        if deeper_level > level:
                            subtree[deeper_level - level] = nodes

                    return dict(subtree)

        raise ValueError(f"Node {node_iri} not found in any hierarchy")

    def get_all_subclasses_for_type(self, type_label: str) -> Dict[int, List[OWLClass]]:
        """
        Get all subclasses for a specific type, organized by their levels in the hierarchy.

        Args:
            type_label (str): The label of the type to get subclasses for (e.g., 'Legal Entity', 'Actor / Player')

        Returns:
            Dict[int, List[OWLClass]]: Dictionary mapping levels to lists of OWLClass objects
        """
        # Find the type's IRI
        type_matches = self._folio.get_by_label(type_label)
        if not type_matches:
            raise ValueError(f"Type '{type_label}' not found in FOLIO")
        type_iri = type_matches[0].iri

        # Get the hierarchy for this type
        if type_iri not in self._complete_hierarchy:
            raise ValueError(f"No hierarchy found for type '{type_label}'")

        # Convert the hierarchy from IRIs to OWLClass objects
        hierarchy = {}
        for level, iris in self._complete_hierarchy[type_iri].items():
            hierarchy[level] = [self._folio[iri] for iri in iris]

        return hierarchy

    def print_type_hierarchy(self, type_label: str) -> None:
        """
        Print the complete hierarchy for a specific type, showing all subclasses at each level.

        Args:
            type_label (str): The label of the type to print hierarchy for
        """
        hierarchy = self.get_all_subclasses_for_type(type_label)

        print(f"\nHierarchy for {type_label}:")
        print("=" * 80)

        for level, classes in sorted(hierarchy.items()):
            print(f"\nLevel {level}:")
            for cls in classes:
                print(f"  - {cls.label} ({cls.iri})")

    def print_subclass_hierarchy(self, parent_type: str, subclass_label: str) -> None:
        """
        Print the hierarchy starting from a specific subclass under a parent type.

        Args:
            parent_type (str): The parent type (e.g., 'Status')
            subclass_label (str): The label of the subclass to start from (e.g., 'Proceeding Status')
        """
        # First get the complete hierarchy for the parent type
        parent_hierarchy = self.get_all_subclasses_for_type(parent_type)

        # Find the subclass in the hierarchy
        subclass_found = False
        subclass_level = None
        subclass_iri = None

        for level, classes in parent_hierarchy.items():
            for cls in classes:
                if cls.label == subclass_label:
                    subclass_found = True
                    subclass_level = level
                    subclass_iri = cls.iri
                    break
            if subclass_found:
                break

        if not subclass_found:
            raise ValueError(
                f"Subclass '{subclass_label}' not found under '{parent_type}'"
            )

        print(f"\nHierarchy starting from {subclass_label} under {parent_type}:")
        print("=" * 80)

        # Print the subclass and all its descendants
        for level, classes in sorted(parent_hierarchy.items()):
            if level >= subclass_level:  # Only print from the subclass level onwards
                print(
                    f"\nLevel {level - subclass_level}:"
                )  # Adjust level numbers to start from 0
                for cls in classes:
                    # Check if this class is a descendant of our subclass
                    if level == subclass_level or self._is_descendant(
                        cls.iri, subclass_iri
                    ):
                        print(f"  - {cls.label} ({cls.iri})")

    def _is_descendant(self, class_iri: str, ancestor_iri: str) -> bool:
        """
        Check if a class is a descendant of an ancestor class.

        Args:
            class_iri (str): The IRI of the class to check
            ancestor_iri (str): The IRI of the potential ancestor

        Returns:
            bool: True if class_iri is a descendant of ancestor_iri
        """
        # Get all parents of the class
        parents = self._folio.get_parents(class_iri)

        # Check if ancestor is among the parents
        if ancestor_iri in [p.iri for p in parents]:
            return True

        # Recursively check parents
        for parent in parents:
            if self._is_descendant(parent.iri, ancestor_iri):
                return True

        return False

    def get_subclasses_at_depth(self, type_label: str, depth: int) -> List[OWLClass]:
        """
        Get all subclasses that are exactly 'depth' levels from the root type.

        Args:
            type_label (str): The root type (e.g., 'Status')
            depth (int): The depth level to get subclasses from (1-based, where 1 is direct children)

        Returns:
            List[OWLClass]: List of subclasses at the specified depth
        """
        # Get the complete hierarchy for the type
        hierarchy = self.get_all_subclasses_for_type(type_label)

        # Find the root level (should be 0)
        root_level = min(hierarchy.keys())

        # Calculate the target level
        target_level = root_level + depth

        # Get subclasses at the target level
        return hierarchy.get(target_level, [])

    def print_subclasses_at_depth(self, type_label: str, depth: int) -> None:
        """
        Print all subclasses that are exactly 'depth' levels from the root type.

        Args:
            type_label (str): The root type (e.g., 'Status')
            depth (int): The depth level to get subclasses from (1-based, where 1 is direct children)
        """
        subclasses = self.get_subclasses_at_depth(type_label, depth)

        print(f"\nSubclasses at depth {depth} from {type_label}:")
        print("=" * 80)

        if not subclasses:
            print(f"No subclasses found at depth {depth}")
            return

        for cls in subclasses:
            print(f"  - {cls.label} ({cls.iri})")

            # Print the path from root to this subclass
            path = self._get_path_to_root(cls.iri)
            path_str = " -> ".join(reversed([self._folio[iri].label for iri in path]))
            print(f"    Path: {path_str}")

    def _get_path_to_root(self, class_iri: str) -> List[str]:
        """
        Get the path from a class to the root of its hierarchy.

        Args:
            class_iri (str): The IRI of the class to get the path for

        Returns:
            List[str]: List of IRIs representing the path from root to the class
        """
        path = [class_iri]
        current = class_iri

        while True:
            parents = self._folio.get_parents(current)
            if not parents:
                break
            # Take the first parent (assuming single inheritance)
            current = parents[0].iri
            path.append(current)

        return path

    def get_hierarchy_from_iri(self, iri: str) -> Dict[int, List[OWLClass]]:
        """
        Get the complete subclass hierarchy starting from a specific IRI.

        Args:
            iri (str): The IRI to start the hierarchy from

        Returns:
            Dict[int, List[OWLClass]]: Dictionary mapping levels to lists of OWLClass objects
        """
        # Verify the IRI exists
        if iri not in self._folio.iri_to_index:
            raise ValueError(f"IRI '{iri}' not found in FOLIO")

        # First pass: Build parent-child relationships
        parent_map = defaultdict(set)
        child_map = defaultdict(set)
        all_nodes = set()

        # Build the complete graph
        for parent_iri, children in self._folio.class_edges.items():
            if parent_iri != "http://www.w3.org/2002/07/owl#Thing":
                all_nodes.add(parent_iri)
                for child_iri in children:
                    parent_map[child_iri].add(parent_iri)
                    child_map[parent_iri].add(child_iri)
                    all_nodes.add(child_iri)

        # Calculate levels for all nodes
        initial_levels = {}

        def calculate_level(node_iri, visited=None):
            if visited is None:
                visited = set()
            if node_iri in visited:
                return initial_levels.get(node_iri, 0)
            visited.add(node_iri)

            parents = parent_map[node_iri]
            if not parents:
                initial_levels[node_iri] = 0
                return 0

            # Calculate levels for all parents first
            parent_levels = []
            for parent in parents:
                if parent not in initial_levels:
                    calculate_level(parent, visited)
                parent_levels.append(initial_levels[parent])

            initial_levels[node_iri] = max(parent_levels) + 1
            return initial_levels[node_iri]

        # Calculate levels for all nodes in the graph
        for node in all_nodes:
            if node not in initial_levels:
                calculate_level(node)

        # Initialize the hierarchy
        hierarchy = defaultdict(list)
        visited = set()
        queue = deque([(iri, 0)])  # (node_iri, level)
        in_queue = {iri}

        while queue:
            current_iri, level = queue.popleft()
            if current_iri in in_queue:
                in_queue.remove(current_iri)

            if current_iri in visited:
                continue

            visited.add(current_iri)
            current_class = self._folio[current_iri]

            # Ensure the level is at least as high as all parents
            parent_levels = {initial_levels[p] for p in parent_map[current_iri]}
            if parent_levels:
                level = max(level, max(parent_levels) + 1)

            hierarchy[level].append(current_class)

            # Get all children
            children = self._folio.get_children(current_iri)

            # Sort children by their initial level and parent count
            children_with_metadata = []
            for child in children:
                if child.iri not in visited and child.iri not in in_queue:
                    initial_level = initial_levels.get(child.iri, 0)
                    parent_count = len(parent_map[child.iri])
                    # Combine initial level and parent count for sorting
                    score = initial_level * 2 + parent_count
                    children_with_metadata.append((child, score))

            # Sort by score in descending order
            children_with_metadata.sort(key=lambda x: x[1], reverse=True)

            # Add children to the next level
            for child, _ in children_with_metadata:
                if child.iri not in visited and child.iri not in in_queue:
                    queue.append((child.iri, level + 1))
                    in_queue.add(child.iri)

        return dict(hierarchy)

    def get_hierarchy_from_label(self, label: str) -> Dict[int, List[OWLClass]]:
        """
        Get the complete subclass hierarchy starting from a specific label.
        Uses the cached BFS results from build_complete_hierarchy for efficiency.

        Args:
            label (str): The label to start the hierarchy from

        Returns:
            Dict[int, List[OWLClass]]: Dictionary mapping levels to lists of OWLClass objects
        """
        # Find the class with this label
        matches = self._folio.get_by_label(label)
        if not matches:
            raise ValueError(f"Label '{label}' not found in FOLIO")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple classes found with label '{label}'. Please use get_hierarchy_from_iri() with specific IRI."
            )

        # Get the hierarchy directly using the IRI
        return self.get_hierarchy_from_iri(matches[0].iri)

    def print_hierarchy_from_iri(self, iri: str) -> None:
        """
        Print the complete subclass hierarchy starting from a specific IRI.

        Args:
            iri (str): The IRI to start the hierarchy from
        """
        hierarchy = self.get_hierarchy_from_iri(iri)
        root_class = self._folio[iri]

        print(f"\nHierarchy starting from {root_class.label} ({iri}):")
        print("=" * 80)

        for level, classes in sorted(hierarchy.items()):
            print(f"\nLevel {level}:")
            for cls in classes:
                print(f"  - {cls.label} ({cls.iri})")

    def print_hierarchy_from_label(self, label: str, indent: str = "  ") -> None:
        """
        Print the complete subclass hierarchy starting from a specific label in a readable format.

        Args:
            label (str): The label to start the hierarchy from
            indent (str): The indentation string to use for each level (default: "  ")

        Example output:
        Status
          Proceeding Status
            Active
            Closed
            Consolidated
            Inactive
            Filing Fee Waived
        """
        try:
            hierarchy = self.get_hierarchy_from_label(label)

            # Sort levels to ensure consistent output
            for level in sorted(hierarchy.keys()):
                classes = hierarchy[level]
                # Print each class at this level
                for cls in classes:
                    print(f"{indent * level}{cls.label}")
                    if cls.preferred_label and cls.preferred_label != cls.label:
                        print(
                            f"{indent * (level + 1)}Preferred Label: {cls.preferred_label}"
                        )
                    if cls.alternative_labels:
                        print(
                            f"{indent * (level + 1)}Alternative Labels: {', '.join(cls.alternative_labels)}"
                        )
                    if cls.definition:
                        print(f"{indent * (level + 1)}Definition: {cls.definition}")
                    print()  # Add blank line between classes

        except ValueError as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    explorer = FOLIOExplorer()
    results = explorer.explore_ontology()

    # The same instance will be reused
    explorer2 = FOLIOExplorer()
    assert explorer is explorer2  # True

    # Example usage:
    # 1. Traverse from Legal Entity
    legal_entity_hierarchy = explorer.traverse_from_type("Legal Entity")
    print("\nLegal Entity Hierarchy:")
    for level, classes in sorted(legal_entity_hierarchy.items()):
        print(f"\nLevel {level}:")
        for class_iri in classes:
            class_obj = explorer._folio[class_iri]
            print(f"  - {class_obj.label}")

    # 2. Traverse from level 3 of Legal Entity
    level_3_hierarchy = explorer.traverse_from_level(3, "Legal Entity")
    print("\nLegal Entity Hierarchy from Level 3:")
    for level, classes in sorted(level_3_hierarchy.items()):
        print(f"\nLevel {level}:")
        for class_iri in classes:
            class_obj = explorer._folio[class_iri]
            print(f"  - {class_obj.label}")

    # 3. Get subclasses at level 2
    level_2_subclasses = explorer.get_subclasses_at_level("Legal Entity", 2)
    print("\nSubclasses at Level 2:")
    for class_iri in level_2_subclasses:
        class_obj = explorer._folio[class_iri]
        print(f"  - {class_obj.label}")

    # 4. Get subtree from a specific node
    # First find a node at level 3
    level_3_nodes = explorer.get_subclasses_at_level("Legal Entity", 3)
    if level_3_nodes:
        node_iri = next(iter(level_3_nodes))
        subtree = explorer.get_subtree_from_node(node_iri)
        print("\nSubtree from node:")
        for level, classes in sorted(subtree.items()):
            print(f"\nLevel {level}:")
            for class_iri in classes:
                class_obj = explorer._folio[class_iri]
                print(f"  - {class_obj.label}")

    # Get all subclasses for Legal Entity
    legal_entity_hierarchy = explorer.get_all_subclasses_for_type("Legal Entity")

    # Or print the hierarchy in a readable format
    explorer.print_type_hierarchy("Legal Entity")

    # You can also get subclasses for Actor / Player
    actor_hierarchy = explorer.get_all_subclasses_for_type("Actor / Player")
    explorer.print_type_hierarchy("Actor / Player")
