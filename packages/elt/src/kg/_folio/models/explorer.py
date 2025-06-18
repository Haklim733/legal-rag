from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import time
from folio import OWLClass, OWLObjectProperty


class TypeHierarchy(BaseModel):
    """Model for the hierarchy of a single FOLIO type."""

    type_label: str = Field(..., description="The label of the FOLIO type")
    levels: Dict[int, List[OWLClass]] = Field(
        ..., description="Dictionary mapping levels to lists of OWLClass objects"
    )


class CompleteHierarchy(BaseModel):
    """Model for the complete FOLIO hierarchy."""

    hierarchies: Dict[str, TypeHierarchy] = Field(
        ..., description="Dictionary mapping type labels to their hierarchies"
    )
    metadata: Dict[str, str] = Field(
        default_factory=lambda: {
            "version": "1.0",
            "description": "Complete FOLIO class hierarchy",
            "timestamp": str(time.time()),
        },
        description="Metadata about the hierarchy",
    )

    @property
    def type_labels(self) -> List[str]:
        """Get list of all type labels in the hierarchy."""
        return list(self.hierarchies.keys())

    def get_hierarchy_for_type(self, type_label: str) -> Optional[TypeHierarchy]:
        """Get the hierarchy for a specific type."""
        return self.hierarchies.get(type_label)

    def get_classes_at_level(
        self, type_label: str, level: int
    ) -> Optional[List[OWLClass]]:
        """Get all classes at a specific level for a type."""
        hierarchy = self.get_hierarchy_for_type(type_label)
        if hierarchy:
            return hierarchy.levels.get(level)
        return None


class OntologyConcept(BaseModel):
    """Model for a matched ontology concept."""

    label: str = Field(..., description="The main label of the concept")
    iri: str = Field(..., description="The IRI of the concept")
    preferred_label: Optional[str] = Field(
        None, description="The preferred label if different from main label"
    )
    alternative_labels: List[str] = Field(
        default_factory=list, description="List of alternative labels"
    )
    relevance_score: float = Field(
        ..., ge=0, le=10, description="Relevance score from 0-10"
    )
    explanation: str = Field(
        ..., description="Explanation of why this concept is relevant"
    )
    matched_text: str = Field(
        ..., description="The specific text from the email that matched this concept"
    )
    context: str = Field(..., description="The surrounding context of the matched text")


class OntologySearchResult(BaseModel):
    """Model for the complete search results."""

    matched_concepts: List[OntologyConcept] = Field(
        ..., description="List of matched concepts"
    )
    summary: str = Field(
        ..., description="Summary of the findings and relationships between concepts"
    )


# Pydantic models for the ontology structure
class OntologyMetadata(BaseModel):
    """Model for ontology metadata."""

    version: str = Field(default="1.0", description="Version of the ontology structure")
    description: str = Field(
        default="FOLIO Ontology Exploration Results",
        description="Description of the ontology structure",
    )
    timestamp: str = Field(
        default_factory=lambda: str(datetime.now()),
        description="Timestamp of when the structure was created",
    )


class OntologyClass(BaseModel):
    label: str
    iri: str
    preferred_label: Optional[str] = None
    alternative_labels: List[str] = Field(default_factory=list)
    hidden_label: Optional[str] = None
    translations: Dict[str, str] = Field(default_factory=dict)
    definition: Optional[str] = None
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    type: Optional[str] = None


class OntologyProperty(BaseModel):
    label: str
    iri: str
    preferred_label: Optional[str] = None
    alternative_labels: List[str] = Field(default_factory=list)
    definition: Optional[str] = None
    domain: List[str] = Field(default_factory=list)
    range: List[str] = Field(default_factory=list)
    sub_property_of: List[str] = Field(default_factory=list)
    inverse_of: Optional[str] = None


class OntologyTaxonomy(BaseModel):
    classes: List[OntologyClass]
    properties: List[OntologyProperty]


class GraphNode(BaseModel):
    id: str
    label: str
    preferred_label: Optional[str] = None
    alternative_labels: List[str] = Field(default_factory=list)
    type: str


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    type: str


class OntologyGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ClassConnection(BaseModel):
    """Model for a connection between classes."""

    property: str = Field(..., description="The property label")
    property_iri: str = Field(..., description="The property IRI")
    target_class: str = Field(..., description="The target class label")


class ClassConnections(BaseModel):
    as_subject: List[ClassConnection] = Field(default_factory=list)
    as_object: List[ClassConnection] = Field(default_factory=list)


class ClassConnectionsStructure(BaseModel):
    """Model for the complete class connections structure."""

    connections: Dict[str, ClassConnections] = Field(
        ..., description="Dictionary mapping class labels to their connections"
    )
    total_connections: int = Field(
        ..., description="Total number of connections across all classes"
    )
    classes_with_connections: int = Field(
        ..., description="Number of classes that have at least one connection"
    )


class HierarchyLevel(BaseModel):
    """Model for a single level in the hierarchy."""

    level: int = Field(..., description="The level number in the hierarchy")
    classes: List[OWLClass] = Field(
        ..., description="List of OWLClass objects at this level"
    )


class TypeHierarchy(BaseModel):
    """Model for the hierarchy of a single FOLIO type."""

    type_label: str = Field(..., description="The label of the FOLIO type")
    levels: Dict[int, List[OWLClass]] = Field(
        ..., description="Dictionary mapping levels to lists of OWLClass objects"
    )


class CompleteHierarchy(BaseModel):
    """Model for the complete FOLIO hierarchy."""

    hierarchies: Dict[str, TypeHierarchy] = Field(
        ..., description="Dictionary mapping type labels to their hierarchies"
    )
    metadata: Dict[str, str] = Field(
        default_factory=lambda: {
            "version": "1.0",
            "description": "Complete FOLIO class hierarchy",
            "timestamp": str(time.time()),
        },
        description="Metadata about the hierarchy",
    )

    @property
    def type_labels(self) -> List[str]:
        """Get list of all type labels in the hierarchy."""
        return list(self.hierarchies.keys())

    def get_hierarchy_for_type(self, type_label: str) -> Optional[TypeHierarchy]:
        """Get the hierarchy for a specific type."""
        return self.hierarchies.get(type_label)

    def get_classes_at_level(
        self, type_label: str, level: int
    ) -> Optional[List[OWLClass]]:
        """Get all classes at a specific level for a type."""
        hierarchy = self.get_hierarchy_for_type(type_label)
        if hierarchy:
            return hierarchy.levels.get(level)
        return None


class OntologyStructure(BaseModel):
    metadata: OntologyMetadata
    taxonomy: OntologyTaxonomy
    graph: OntologyGraph
    connections: Dict[str, ClassConnections]


class ClassHierarchy(BaseModel):
    """Model for class hierarchy relationships."""

    class_name: str = Field(..., description="The name of the class")
    parent_labels: List[str] = Field(
        default_factory=list, description="List of parent class labels"
    )
    child_labels: List[str] = Field(
        default_factory=list, description="List of child class labels"
    )
    level: int = Field(default=0, description="Hierarchy level (0 for root classes)")


class ClassHierarchyStructure(BaseModel):
    """Model for the complete class hierarchy structure."""

    hierarchies: List[ClassHierarchy] = Field(
        ..., description="List of class hierarchies"
    )
    root_classes: List[str] = Field(
        ..., description="List of root class labels (classes with no parents)"
    )


class OntologySearchContext(BaseModel):
    """Model for the search context provided to the LLM."""

    query: str = Field(..., description="The search query text")
    classes: List[OntologyClass] = Field(..., description="List of ontology classes")
    properties: List[OntologyProperty] = Field(
        ..., description="List of ontology properties"
    )
    connections: ClassConnectionsStructure = Field(
        ..., description="Class connections structure"
    )
    graph: OntologyGraph = Field(..., description="Ontology graph structure")


class OntologySearchRequest(BaseModel):
    """Model for the complete search request."""

    query: str = Field(..., description="The search query text")
    ontology_structure: OntologySearchContext = Field(
        ..., description="The search context"
    )
    metadata: OntologyMetadata = Field(..., description="Metadata about the search")


class SubclassInfo(BaseModel):
    """Model for subclass information"""

    iri: str
    children: List[str]  # List of child IRIs
    parents: List[str]  # List of parent IRIs


class TypeTaxonomy(BaseModel):
    """Model for type taxonomy structure"""

    iri: str
    subclasses: List[SubclassInfo]


class TaxonomyStructure(BaseModel):
    """Model for the complete taxonomy structure"""

    types: Dict[str, TypeTaxonomy]

    def __getitem__(self, key: str) -> TypeTaxonomy:
        return self.types[key]

    def items(self):
        return self.types.items()


class Triple(BaseModel):
    """Model for an RDF triple (subject-predicate-object)."""

    subject: OWLClass
    predicate: OWLObjectProperty
    object: OWLClass


class TripleStructure(BaseModel):
    """Structure containing all RDF triples."""

    triples: List[Triple] = Field(default_factory=list)
