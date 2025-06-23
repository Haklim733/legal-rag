import logging
import requests
import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Configure logging to handle ascii_colors errors gracefully
logging.getLogger("ascii_colors").setLevel(logging.ERROR)

from folio import FOLIO
from folio.models import OWLClass
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


def _get_children_with_depth(folio_instance, parent_class, max_depth, current_depth=1):
    """Get all children of a class up to a specified depth using manual traversal."""
    if current_depth >= max_depth:
        return []

    children = []

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


def get_labels(owl_class: OWLClass) -> str:
    labels = []
    if owl_class.label:
        labels.append(owl_class.label)
    if owl_class.preferred_label:
        labels.append(owl_class.preferred_label)
    if owl_class.alternative_labels:
        labels.extend(owl_class.alternative_labels)
    return ", ".join(labels)


def get_definition(owl_class: OWLClass) -> str:
    full_definition = ""
    if hasattr(owl_class, "definition"):
        full_definition += f" {owl_class.definition}"
    if hasattr(owl_class, "description"):
        full_definition += f" {owl_class.description}"
    if hasattr(owl_class, "comment"):
        full_definition += f" {owl_class.comment}"
    if hasattr(owl_class, "notes"):
        full_definition += f" {owl_class.notes}"
    if hasattr(owl_class, "examples"):
        full_definition += (
            f" (Examples): {'; '.join([ x for x in owl_class.examples if x])}"
        )
    labels = get_labels(owl_class)
    full_definition += f" Labels: {labels}"
    return full_definition


def create_relationship(
    folio_instance: FOLIO,
    subject_iri: str,
    object_iri: str,
    predicate: str,
    bypass_definition: bool = False,
) -> Optional[Relationship]:
    try:
        subject_info = folio_instance[subject_iri]
        object_info = folio_instance[object_iri]
        subject_name = subject_info.label
        object_name = object_info.label

        if not subject_name or not object_name:
            logger.warning(
                f"Skipping relationship with no name: {subject_name} or {object_name}"
            )
            return None

        if subject_info.deprecated or object_info.deprecated:
            logger.warning(
                f"Skipping deprecated subject or object: {subject_name} or {object_name}"
            )
            return None

        definition = f"{subject_name} {predicate} {object_name}"
        labels = predicate.split(":")[-1]

        if not bypass_definition:
            predicate_info = folio_instance.get_properties_by_label(predicate)[0]
            definition = get_definition(predicate_info)
            labels = get_labels(predicate_info)
        else:
            # For hierarchy relationships, create a more natural description
            if predicate == "subclass_of":
                definition = f"{subject_name} is a subclass of {object_name}"
                labels = "subClassOf"

        # Create relationship
        src_id = subject_name
        tgt_id = object_name

        relationship_id = Relationship.create_relationship_id(src_id, predicate, tgt_id)

        relationship = Relationship(
            src_id=src_id,
            tgt_id=tgt_id,
            description=definition,
            keywords=f"{predicate}, {subject_name}, {object_name}, {labels}",
            weight=1.0,
            source_id=relationship_id,
        )
    except Exception as e:
        logger.error(
            f"Failed to create relationship for {subject_name} {predicate} {object_name}: {e}"
        )
        raise
    return relationship


def create_hierarchy_relationships(
    folio_instance: FOLIO,
    owl_class: OWLClass,
    entities: list[str] = None,
) -> tuple[dict[str, Relationship], list[Chunk]]:
    """
    Creates hierarchy relationships from FOLIO ontology classes.
    Only creates relationships where both source and target are in the entities list.
    """
    relationships = {}
    chunks = []
    parent_iris = owl_class.sub_class_of
    if not parent_iris:
        return {}, []

    for parent_iri in parent_iris:
        # Get parent info to check if it's in the filtered entities
        try:
            parent_info = folio_instance[parent_iri]
            parent_name = parent_info.label

            # Skip if we can't get valid names
            if not parent_name:
                continue

            # Filter by entities if specified - both source and target must be in entities
            if entities is not None:
                if owl_class.label not in entities or parent_name not in entities:
                    continue

        except Exception as e:
            logger.warning(f"Could not get parent info for {parent_iri}: {e}")
            continue

        relationship = create_relationship(
            folio_instance, owl_class.iri, parent_iri, "subclass_of", True
        )
        if relationship:
            if relationship.source_id in relationships:
                logger.warning(
                    f"Found duplicate relationship: {relationship.source_id}"
                )
            relationships[relationship.source_id] = relationship

            chunk = Chunk(
                content=relationship.description,
                source_id=relationship.source_id,
                chunk_order_index=0,
            )
            chunks.append(chunk)

    return relationships, chunks


def create_entities(
    folio_instance: FOLIO,
    entities: list[str] = None,
) -> dict[str, list[Chunk] | dict[str, Entity] | dict[str, Relationship]]:
    """
    Creates entities and chunks from FOLIO ontology classes.
    Returns chunks, entities, a mapping of IRI to source_id.
    """
    chunks = []
    entities_dict = {}
    relationships = {}

    logger.info(f"Processing {len(folio_instance.classes)} classes for entities...")

    for i, owl_class in enumerate(folio_instance.classes):
        if not owl_class.iri:
            logger.warning(f"Skipping class with no IRI at index {i}")
            continue

        # Skip deprecated classes - handle None label safely
        if (
            hasattr(owl_class, "deprecated")
            and owl_class.deprecated
            or (owl_class.label and "deprecated" in owl_class.label.lower())
        ):
            logger.debug(
                f"Skipping deprecated class: {owl_class.label or owl_class.iri}"
            )
            continue

        # Validate required fields before creating entities
        class_name = owl_class.label
        class_iri = owl_class.iri
        if not class_name:
            logger.warning(
                f"Skipping class with no label and no valid IRI: {class_iri}"
            )
            continue

        # Only process if in target_entities
        if entities is not None and class_name not in entities:
            logger.debug(f"Skipping {class_name} - not in target entities")
            continue
        else:
            if entities is not None:
                logger.debug(f"Including {class_name} - found in target entities")

        description = get_definition(owl_class)

        try:
            chunk = Chunk(
                content=description,
                source_id=class_name,
                chunk_order_index=0,
            )
            chunks.append(chunk)
        except Exception as e:
            logger.error(f"Failed to create chunk for {class_name}: {e}")
            continue

        try:
            entity = Entity(
                entity_name=class_name,
                entity_type=class_name,
                description=description,
                source_id=class_name,
                chunk_ids=[class_name],
            )
            if class_name in entities_dict:
                logger.warning(f"found duplicate entity {class_name}")
                exist_description = entities_dict[class_name].description
                # Handle None values when merging descriptions
                current_desc = entity.description or ""
                exist_desc = exist_description or ""
                entity.description = f"{current_desc}\n{exist_desc}".strip()
            else:
                entities_dict[class_name] = entity
        except Exception as e:
            logger.error(f"Failed to create entity for {class_name}: {e}")
            continue

        heirarchical_relationships, heirarchical_chunks = (
            create_hierarchy_relationships(folio_instance, owl_class, entities) or {}
        )

        relationships = relationships | heirarchical_relationships
        chunks = chunks + heirarchical_chunks

    logger.info(
        f"Successfully created {len(chunks)} chunks and {entities_dict.__len__()} entities, and {relationships.__len__()} hierarchy relationships"
    )
    return chunks, entities_dict, relationships


def create_predicates(
    folio_instance: FOLIO,
    entities: list[str] = None,
) -> tuple[dict[str, Relationship], list[Chunk]]:
    """
    Creates relationships and chunks from FOLIO triples.
    Requires the relationships and chunks from create_entities.
    Returns relationships and chunks.
    """
    relationships = {}
    chunks = []

    for subject_iri, predicate, object_iri in folio_instance.triples:
        if predicate == "folio:operators":
            continue
        if not predicate.startswith("folio") and not predicate.startswith("oasis"):
            continue

        try:
            subject_info = folio_instance[subject_iri]
            object_info = folio_instance[object_iri]
            subject_name = subject_info.label
            object_name = object_info.label

            # Skip if we can't get valid names
            if not subject_name or not object_name:
                continue

            # Filter by target entities if specified - BOTH subject and object must be in entities
            if entities is not None:
                if subject_name not in entities or object_name not in entities:
                    continue

            relationship = create_relationship(
                folio_instance, subject_iri, object_iri, predicate
            )
            if relationship.source_id in relationships:
                logger.warning(f"found duplicate relationship {relationship.source_id}")
                continue
            relationships[relationship.source_id] = relationship

            # Create chunk content
            try:
                chunk = Chunk(
                    content=relationship.description,
                    source_id=relationship.source_id,
                    chunk_order_index=0,
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to create chunk for {predicate}: {e}")
                continue

        except Exception as e:
            logger.warning(
                f"Failed to create relationship for triple {subject_iri} {predicate} {object_iri}: {e}"
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


def get_all_subclasses(folio_instance: FOLIO, entity_names: list[str]) -> set[str]:
    """
    Get all subclasses of the specified entities using FOLIO's built-in methods.
    Also includes parents of subclasses to ensure all hierarchy relationships are valid.

    Args:
        folio_instance: The FOLIO instance
        entity_names: List of entity names to find subclasses for

    Returns:
        Set of all subclass entity names (including the original entities and their parents)
    """
    all_entities = set()
    for name in entity_names:
        try:
            # Try to find the entity by label
            entities_found = folio_instance.get_by_label(name)
            if not entities_found:
                logger.warning(f"Entity '{name}' not found in FOLIO ontology")
                continue

            iri = entities_found[0].iri
            logger.info(f"Found entity '{name}' with IRI: {iri}")

            # Get children
            children = folio_instance.get_children(iri)
            logger.info(f"Found {len(children)} children for '{name}'")

            all_entities.add(name)
            child_labels = [x.label for x in children if x.label]
            all_entities.update(child_labels)
            logger.info(f"Added children for '{name}': {child_labels}")

        except Exception as e:
            logger.error(f"Error getting subclasses for '{name}': {e}")
            # Still add the original entity name even if we can't find subclasses
            all_entities.add(name)

    return all_entities


def create_custom_kg(
    folio_instance: FOLIO, entities: list[str] = None, subclasses: bool = True
) -> CustomKnowledgeGraph:
    """
    Creates a custom knowledge graph from FOLIO ontology data.
    This function orchestrates the creation of entities and relationships.

    Args:
        folio_instance: The FOLIO instance to create the knowledge graph from
        entities: Optional list of entity names to filter by
        subclasses: If True and entities are specified, include all subclasses of the specified entities

    Returns:
        CustomKnowledgeGraph with filtered entities and relationships
    """
    # First, create entities and chunks, along with hierarchy relationships
    if entities is not None:
        if subclasses:
            target_entities = get_all_subclasses(folio_instance, entities)
        else:
            target_entities = set(entities)

    entity_chunks, entities_dict, relationships = create_entities(
        folio_instance, target_entities
    )

    # Then, create relationships and chunks from triples
    triple_relationships, triple_chunks = create_predicates(
        folio_instance, target_entities
    )

    # Combine all relationships
    all_relationships = relationships | triple_relationships
    all_chunks = entity_chunks + triple_chunks

    # Convert entities dict to list
    entities_list = list(entities_dict.values())

    # Convert relationships dict to list
    relationships_list = list(all_relationships.values())

    logger.info(
        f"Final knowledge graph: {len(all_chunks)} chunks, {len(entities_list)} entities, {len(relationships_list)} relationships"
    )

    return CustomKnowledgeGraph(
        chunks=all_chunks, entities=entities_list, relationships=relationships_list
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
