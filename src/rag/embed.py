import logging
import requests
import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from folio import FOLIO
from folio.models import OWLClass, OWLObjectProperty
from .models import (
    CustomKnowledgeGraph,
    Chunk,
    Entity,
    Relationship,
)

logger = logging.getLogger(__name__)


class ClassMetadata(BaseModel):
    """Metadata for OWL Class documents."""

    uri: str
    type: str = "OWLClass"
    label: str
    source: str = "FOLIO"
    alternative_labels: Optional[List[str]] = None
    sub_class_of: Optional[List[str]] = None


class PropertyMetadata(BaseModel):
    """Metadata for OWL Object Property documents."""

    uri: str
    type: str = "OwlObjectProperty"
    label: str
    source: str = "FOLIO"
    alternative_labels: Optional[List[str]] = None
    domain: Optional[List[str]] = None
    range: Optional[List[str]] = None


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: Optional[ClassMetadata | PropertyMetadata] = None

    class Config:
        pass


def wait_for_lightrag_service(base_api_url, timeout=180, health_endpoint="/health"):
    start_time = time.time()
    # Construct health URL: ensure it doesn't duplicate slashes if base_api_url ends with one
    health_url = base_api_url.rstrip("/") + health_endpoint
    logger.info(f"Waiting for LightRAG service at {health_url}...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                logger.info("LightRAG service is healthy.")
                return True
            else:
                logger.info(
                    f"LightRAG service responded with {response.status_code}. Retrying..."
                )
        except requests.ConnectionError:
            logger.error(
                "LightRAG service not yet available (ConnectionError). Retrying..."
            )
        except requests.Timeout:
            logger.error("LightRAG service health check timed out. Retrying...")
        except requests.RequestException as e:
            logger.error(
                f"An unexpected error occurred while checking LightRAG health: {e}. Retrying..."
            )
        time.sleep(10)  # Increased sleep time
    logger.error(f"Timeout waiting for LightRAG service at {health_url}.")
    return False


# Helper function to create a Document from an OWLClass
def _create_document_from_owl_class(owl_class: OWLClass) -> Document | None:
    if not owl_class.iri:
        print(f"Skipping class with no IRI: {owl_class.label}")
        return None

    doc_id = owl_class.iri
    label = owl_class.preferred_label or owl_class.label or "Unnamed Class"
    definition = (
        owl_class.definition
        or owl_class.description
        or owl_class.comment
        or "No definition available."
    )

    text_content_parts = [
        f"Type: Ontology Class",
        f"Label: {label}",
        f"IRI: {doc_id}",
        f"Definition: {definition}",
    ]
    if owl_class.alternative_labels:
        # Filter out None values before joining
        valid_alt_labels = [
            label for label in owl_class.alternative_labels if label is not None
        ]
        if valid_alt_labels:
            text_content_parts.append(
                f"Alternate Labels: {', '.join(valid_alt_labels)}"
            )
    if owl_class.examples:
        # Filter out None values before joining
        valid_examples = [ex for ex in owl_class.examples if ex is not None]
        if valid_examples:
            text_content_parts.append(f"Examples: {'; '.join(valid_examples)}")
    if owl_class.notes:
        # Filter out None values before joining
        valid_notes = [note for note in owl_class.notes if note is not None]
        if valid_notes:
            text_content_parts.append(f"Notes: {'; '.join(valid_notes)}")
    if owl_class.sub_class_of:
        # Filter out None values before joining
        valid_sub_classes = [sub for sub in owl_class.sub_class_of if sub is not None]
        if valid_sub_classes:
            text_content_parts.append(f"Subclass Of: {', '.join(valid_sub_classes)}")

    text_content = "\n".join(text_content_parts)

    # Create metadata using Pydantic model
    metadata = ClassMetadata(
        uri=doc_id,
        label=label,
        alternative_labels=owl_class.alternative_labels,
        sub_class_of=owl_class.sub_class_of,
    )

    return Document(id=doc_id, text=text_content, metadata=metadata)


# Helper function to create a Document from an OWLObjectProperty
def _create_document_from_owl_property(owl_prop: OWLObjectProperty) -> Document | None:
    if not owl_prop.iri:
        print(f"Skipping property with no IRI: {owl_prop.label}")
        return None

    doc_id = owl_prop.iri
    label = owl_prop.preferred_label or owl_prop.label or "Unnamed Property"
    definition = owl_prop.definition or "No definition available."

    text_content_parts = [
        f"Type: Ontology Property",
        f"Label: {label}",
        f"IRI: {doc_id}",
        f"Definition: {definition}",
    ]
    if owl_prop.alternative_labels:
        # Filter out None values before joining
        valid_alt_labels = [
            label for label in owl_prop.alternative_labels if label is not None
        ]
        if valid_alt_labels:
            text_content_parts.append(
                f"Alternate Labels: {', '.join(valid_alt_labels)}"
            )
    if owl_prop.domain:
        # Filter out None values before joining
        valid_domain = [domain for domain in owl_prop.domain if domain is not None]
        if valid_domain:
            text_content_parts.append(f"Domain: {', '.join(valid_domain)}")
    if owl_prop.range:
        # Filter out None values before joining
        valid_range = [
            range_val for range_val in owl_prop.range if range_val is not None
        ]
        if valid_range:
            text_content_parts.append(f"Range: {', '.join(valid_range)}")
    if owl_prop.examples:
        # Filter out None values before joining
        valid_examples = [ex for ex in owl_prop.examples if ex is not None]
        if valid_examples:
            text_content_parts.append(f"Examples: {'; '.join(valid_examples)}")
    if owl_prop.definition:
        # Filter out None values before joining
        valid_definition = [
            definition for definition in owl_prop.definition if definition is not None
        ]
        if valid_definition:
            text_content_parts.append(f"Definition: {'; '.join(valid_definition)}")

    text_content = "\n".join(text_content_parts)

    # Create metadata using Pydantic model
    metadata = PropertyMetadata(
        uri=doc_id,
        label=label,
        alternative_labels=owl_prop.alternative_labels,
        domain=owl_prop.domain,
        range=owl_prop.range,
    )

    return Document(id=doc_id, text=text_content, metadata=metadata)


def load_ontology_for_rag(limit: int = None, max_depth: int = 2):
    """Loads ontology data, processes it into Document objects, and returns them."""
    logger.info(f"Loading FOLIO ontology with limit={limit}, max_depth={max_depth}")
    folio_instance = FOLIO()
    documents = []

    # Process classes
    logger.info(f"Processing {len(folio_instance.classes)} classes...")
    for owl_class in folio_instance.classes:
        # Apply depth filtering if max_depth is set
        # This simple check assumes root classes have no sub_class_of or specific roots are known.
        # For more complex ontologies, a proper root finding or explicit root list might be needed.
        # if max_depth is not None and _get_class_depth(folio_instance, owl_class, ???) > max_depth:
        # continue
        doc = _create_document_from_owl_class(owl_class)
        if doc:
            documents.append(doc)

    # Process properties - use get_all_properties() instead of properties attribute
    all_properties = folio_instance.get_all_properties()
    logger.info(f"Processing {len(all_properties)} properties...")
    for owl_prop in all_properties:
        doc = _create_document_from_owl_property(owl_prop)
        if doc:
            documents.append(doc)

    if limit:
        logger.info(f"Limiting documents to {limit}")
        documents = documents[:limit]

    logger.info(f"Loaded {len(documents)} documents from FOLIO ontology.")
    return documents


def _get_children_with_depth(folio_instance, parent_class, max_depth, current_depth=1):
    """Get all children of a class up to a specified depth using manual traversal."""
    if current_depth >= max_depth:
        return []

    children = []
    logger.info(
        f"Looking for children of {parent_class.label} (IRI: {parent_class.iri}) at depth {current_depth}"
    )

    # Use manual traversal since get_children might not be available or working
    for owl_class in folio_instance.classes:
        if owl_class.sub_class_of and any(
            parent_class.iri in parent for parent in owl_class.sub_class_of
        ):
            children.append(owl_class)
            # Recursively get children of this class
            children.extend(
                _get_children_with_depth(
                    folio_instance, owl_class, max_depth, current_depth + 1
                )
            )

    logger.info(f"Total children found for {parent_class.label}: {len(children)}")
    return children


def _get_class_depth(folio_instance, target_class, root_class, current_depth=0):
    if target_class == root_class:
        return current_depth

    # Use the sub_class_of attribute instead of triples_for_class
    if target_class.sub_class_of:
        for parent_iri in target_class.sub_class_of:
            # Find the parent class by IRI
            parent_class = None
            for cls in folio_instance.classes:
                if cls.iri == parent_iri:
                    parent_class = cls
                    break

            if parent_class:
                depth = _get_class_depth(
                    folio_instance, parent_class, root_class, current_depth + 1
                )
                if depth is not None:
                    return depth

    return None


def create_entities(
    folio_instance: FOLIO,
) -> tuple[list[Chunk], list[Entity], dict[str, str], list[Relationship]]:
    """
    Creates entities and chunks from FOLIO ontology classes.
    Returns chunks, entities, a mapping of IRI to source_id.
    """
    chunks = []
    entities = []
    added_entities = {}  # Maps IRI to source_id
    hierarchy_relationships = (
        []
    )  # List of Relationship objects for subclass relationships

    logger.info(f"Processing {len(folio_instance.classes)} classes for entities...")

    for i, owl_class in enumerate(folio_instance.classes):
        if not owl_class.iri:
            logger.warning(f"Skipping class with no IRI at index {i}")
            continue

        # Skip deprecated classes
        if hasattr(owl_class, "deprecated") and owl_class.deprecated:
            logger.debug(
                f"Skipping deprecated class: {owl_class.label or owl_class.iri}"
            )
            continue

        # Validate required fields before creating entities
        class_name = owl_class.label
        if not class_name:
            logger.warning(
                f"Skipping class with no label and no valid IRI: {owl_class.iri}"
            )
            continue

        class_iri = owl_class.iri

        # Create chunk for the class - use the same definition logic for both entity and chunk
        definition = (
            owl_class.definition
            or owl_class.description
            or owl_class.comment
            or "No definition available."
        )

        # Enhanced definition handling for reference-style definitions
        if definition and definition.startswith("See industry description for"):
            # For reference-style definitions, create a more comprehensive description
            enhanced_definition = f"{class_name}: {definition}"
        elif (
            definition and len(definition.strip()) < 50 and "see" in definition.lower()
        ):
            # Handle other short reference-style definitions
            enhanced_definition = f"{class_name}: {definition}"
        else:
            enhanced_definition = definition

        # Build comprehensive description including labels
        description_parts = [enhanced_definition]

        # Add preferred label if it exists and is different from the main label
        if hasattr(owl_class, "preferred_label") and owl_class.preferred_label:
            if owl_class.preferred_label != class_name:
                description_parts.append(
                    f"Preferred label: {owl_class.preferred_label}"
                )

        # Add alternative labels if they exist
        if hasattr(owl_class, "alternative_labels") and owl_class.alternative_labels:
            # Filter out None values and duplicates
            valid_alt_labels = [
                label
                for label in owl_class.alternative_labels
                if label and label != class_name
            ]
            if valid_alt_labels:
                alt_labels_str = ", ".join(valid_alt_labels)
                description_parts.append(f"Alternative labels: {alt_labels_str}")

        full_description = ". ".join(description_parts)

        # Create chunk content
        chunk_text = full_description
        if owl_class.examples:
            # Filter out None values from examples
            valid_examples = [ex for ex in owl_class.examples if ex is not None]
            if valid_examples:
                chunk_text += f"\n(Examples): {'; '.join(valid_examples)}"

        # Add notes if definition is minimal and notes are available
        if owl_class.notes and (
            len(enhanced_definition.strip()) < 100
            or "see" in enhanced_definition.lower()
        ):
            valid_notes = [note for note in owl_class.notes if note is not None]
            if valid_notes:
                chunk_text += f"\n(Notes): {'; '.join(valid_notes)}"

        try:
            chunk = Chunk(
                content=chunk_text,
                source_id=class_name,
                chunk_order_index=0,
            )
            chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Failed to create chunk for {class_name}: {e}")
            continue

        try:
            entity = Entity(
                entity_name=class_name,
                entity_type=class_name,
                description=chunk_text,
                source_id=class_name,
                chunk_ids=[class_name],
            )
            entities.append(entity)
            added_entities[class_iri] = class_name
        except Exception as e:
            logger.warning(f"Failed to create entity for {class_name}: {e}")
            continue

        # Build hierarchy relationships while iterating through entities
        try:
            parents = folio_instance.get_parents(owl_class.iri)
            if parents:
                for parent_class in parents:
                    # Skip if parent is deprecated
                    if hasattr(parent_class, "deprecated") and parent_class.deprecated:
                        logger.debug(
                            f"Skipping deprecated parent: {parent_class.label or parent_class.iri}"
                        )
                        continue

                    parent_iri = parent_class.iri
                    parent_name = parent_class.label or parent_iri.split("/")[-1]

                    if parent_iri and parent_iri != class_iri:  # Avoid self-references
                        # Create the relationship object with unique ID
                        relationship_id = Relationship.create_relationship_id(
                            class_name,
                            "subclass_of",
                            parent_name,
                        )

                        relationship = Relationship(
                            relationship_id=relationship_id,
                            src_id=class_name,
                            tgt_id=parent_name,
                            description=f"{class_name} is a subclass of {parent_name}.",
                            keywords="subClassOf",
                            weight=1.0,
                            source_id=relationship_id,
                        )
                        hierarchy_relationships.append(relationship)

                        # Associate the relationship with the source entity
                        entity.add_outgoing_relationship(relationship_id)

                        logger.debug(
                            f"Added hierarchy relationship: {class_name} -> {parent_name} (ID: {relationship_id})"
                        )
                        break  # Only create relationship to first valid parent
        except Exception as e:
            logger.warning(f"Could not determine parents for {class_name}: {e}")

    logger.info(
        f"Successfully created {len(chunks)} chunks and {len(entities)} entities, and {len(hierarchy_relationships)} hierarchy relationships"
    )
    return chunks, entities, added_entities, hierarchy_relationships


def create_predicates(
    folio_instance: FOLIO, added_entities: dict[str, str]
) -> tuple[list[Relationship], list[Chunk]]:
    """
    Creates relationships and chunks from FOLIO triples.
    Requires the added_entities mapping from create_entities.
    Returns relationships and chunks.
    """
    relationships = []
    chunks = []

    logger.info(
        f"Processing {len(folio_instance.triples)} triples for relationships and chunks..."
    )

    for subject_iri, predicate_iri, object_iri in [
        x
        for x in folio_instance.triples
        if (x[1].startswith("folio") or x[1].startswith("oasis"))
        and x[1] not in ["folio:operators"]
    ]:
        try:
            subject_info = folio_instance[subject_iri]
            object_info = folio_instance[object_iri]

            # Skip if either entity is deprecated
            if object_info.deprecated or subject_info.deprecated:
                continue

            subject_name = subject_info.label
            object_name = object_info.label

            predicate_label = None
            predicate_definition = None

            try:
                predicate_info = folio_instance.get_property_by_label(predicate_iri)
                if predicate_info:
                    predicate_label = (
                        getattr(predicate_info, "label", None)
                        or predicate_iri.split("/")[-1]
                    )
                    predicate_definition = (
                        getattr(predicate_info, "definition", None)
                        or getattr(predicate_info, "description", None)
                        or getattr(predicate_info, "comment", None)
                        or "No definition available"
                    )
            except Exception as e:
                logger.debug(f"Could not get predicate info for {predicate_iri}: {e}")
                predicate_label = predicate_iri.split("/")[-1]
                predicate_definition = "No definition available"

            # Create relationship
            src_id = subject_name
            tgt_id = object_name

            relationship_id = Relationship.create_relationship_id(
                src_id, predicate_label, tgt_id
            )

            relationship = Relationship(
                src_id=src_id,
                tgt_id=tgt_id,
                description=predicate_definition,
                keywords=f"{predicate_label},{subject_name}, {object_name}",
                weight=1.0,
                source_id=relationship_id,
            )
            relationships.append(relationship)

            # Create chunk content
            chunk_text = predicate_definition
            try:
                chunk = Chunk(
                    content=chunk_text,
                    source_id=relationship_id,
                    chunk_order_index=0,
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to create chunk for {predicate_label}: {e}")
                continue

        except Exception as e:
            logger.warning(
                f"Failed to create relationship for triple {subject_iri} {predicate_iri} {object_iri}: {e}"
            )
            continue

    logger.info(
        f"Successfully created {len(relationships)} relationships and {len(chunks)} chunks from triples"
    )
    return relationships, chunks


def get_entity_relationships(
    knowledge_graph: CustomKnowledgeGraph, entity_source_id: str
) -> dict[str, list[Relationship]]:
    """
    Get all relationships for a specific entity.

    Args:
        knowledge_graph: The knowledge graph containing entities and relationships
        entity_source_id: The source_id of the entity to find relationships for

    Returns:
        Dictionary with 'outgoing' and 'incoming' relationships for the entity
    """
    outgoing = [
        rel for rel in knowledge_graph.relationships if rel.src_id == entity_source_id
    ]
    incoming = [
        rel for rel in knowledge_graph.relationships if rel.tgt_id == entity_source_id
    ]

    return {
        "outgoing": outgoing,
        "incoming": incoming,
        "total": len(outgoing) + len(incoming),
    }


def get_entity_relationship_summary(
    knowledge_graph: CustomKnowledgeGraph, entity_name: str
) -> dict:
    """
    Get a summary of all relationships for an entity by name.

    Args:
        knowledge_graph: The knowledge graph containing entities and relationships
        entity_name: The name of the entity to find relationships for

    Returns:
        Dictionary with relationship summary for the entity
    """
    # Find the entity by name
    entity = None
    for e in knowledge_graph.entities:
        if e.entity_name == entity_name:
            entity = e
            break

    if not entity:
        return {"error": f"Entity '{entity_name}' not found"}

    # Get relationships
    relationships = get_entity_relationships(knowledge_graph, entity.source_id)

    # Get target entity names for outgoing relationships
    outgoing_targets = []
    for rel in relationships["outgoing"]:
        target_entity = next(
            (e for e in knowledge_graph.entities if e.source_id == rel.tgt_id), None
        )
        if target_entity:
            outgoing_targets.append(
                {
                    "target": target_entity.entity_name,
                    "description": rel.description,
                    "keywords": rel.keywords,
                }
            )

    # Get source entity names for incoming relationships
    incoming_sources = []
    for rel in relationships["incoming"]:
        source_entity = next(
            (e for e in knowledge_graph.entities if e.source_id == rel.src_id), None
        )
        if source_entity:
            incoming_sources.append(
                {
                    "source": source_entity.entity_name,
                    "description": rel.description,
                    "keywords": rel.keywords,
                }
            )

    return {
        "entity_name": entity_name,
        "entity_type": entity.entity_type,
        "outgoing_relationships": outgoing_targets,
        "incoming_relationships": incoming_sources,
        "total_relationships": relationships["total"],
    }


def create_custom_kg(folio_instance: FOLIO) -> CustomKnowledgeGraph:
    """
    Creates a custom knowledge graph from FOLIO ontology data.
    This function orchestrates the creation of entities and relationships.
    """
    # First, create entities and chunks, along with hierarchy relationships
    entity_chunks, entities, added_entities, hierarchy_relationships = create_entities(
        folio_instance
    )

    # Then, create relationships and chunks from triples
    triple_relationships, triple_chunks = create_predicates(
        folio_instance, added_entities
    )

    # Combine all relationships
    relationships = triple_relationships + hierarchy_relationships

    all_chunks = entity_chunks + triple_chunks

    logger.info(
        f"Final knowledge graph: {len(all_chunks)} chunks, {len(entities)} entities, {len(relationships)} relationships"
    )

    return CustomKnowledgeGraph(
        chunks=all_chunks, entities=entities, relationships=relationships
    )


def main():
    # Determine the base URL for LightRAG API (without /api/v1/load)
    # This is so we can call /health on the same host and port
    lightrag_base_url = "LIGHTRAG_API_URL"
    if "/api/v1/load" in lightrag_base_url:
        lightrag_base_url = lightrag_base_url.split("/api/v1/load")[0]
    elif "/load" in lightrag_base_url:  # More general split if only /load is present
        lightrag_base_url = lightrag_base_url.split("/load")[0]

    if not wait_for_lightrag_service(lightrag_base_url):
        logger.error("Exiting: LightRAG service did not become healthy.")
        return

    logger.info("Fetching FOLIO ontology terms for RAG...")
    # When calling from main() in this script, we might not want a limit, or a specific one for testing.
    # For now, let's assume no limit if main() is run directly, or pass a specific one.
    docs_to_load = load_ontology_for_rag(
        limit=None, max_depth=6
    )  # Or set a specific limit for standalone run e.g. limit=100

    if not docs_to_load:
        logger.error("No documents to load.")
        return

    payload_docs = [doc.to_dict() for doc in docs_to_load]

    # Send in batches if the total number of documents is very large
    batch_size = 1000  # Adjust as needed
    for i in range(0, len(payload_docs), batch_size):
        batch = payload_docs[i : i + batch_size]
        logger.info(
            f"Sending batch {i // batch_size + 1} with {len(batch)} documents..."
        )
        try:
            response = requests.post(
                lightrag_base_url, json={"documents": batch}, timeout=300
            )
            response.raise_for_status()
            logger.info(
                f"Successfully sent batch. Response: {response.json().get('message', 'OK')}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending batch to LightRAG: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            logger.error("Stopping further batches due to error.")
            break  # Stop sending further batches if one fails

    # Create custom KG from FOLIO instance and documents
    custom_kg_data = create_custom_kg(FOLIO())
    logger.info("Custom KG data created.")


if __name__ == "__main__":
    main()
