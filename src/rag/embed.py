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
        text_content_parts.append(
            f"Alternate Labels: {', '.join(owl_class.alternative_labels)}"
        )
    if owl_class.examples:
        text_content_parts.append(f"Examples: {'; '.join(owl_class.examples)}")
    if owl_class.notes:
        text_content_parts.append(f"Notes: {'; '.join(owl_class.notes)}")
    if owl_class.sub_class_of:
        text_content_parts.append(f"Subclass Of: {', '.join(owl_class.sub_class_of)}")

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
    definition = (
        owl_prop.definition
        or owl_prop.description
        or owl_prop.comment
        or "No definition available."
    )

    text_content_parts = [
        f"Type: Ontology Property",
        f"Label: {label}",
        f"IRI: {doc_id}",
        f"Definition: {definition}",
    ]
    if owl_prop.alternative_labels:
        text_content_parts.append(
            f"Alternate Labels: {', '.join(owl_prop.alternative_labels)}"
        )
    if owl_prop.domain:
        text_content_parts.append(f"Domain: {', '.join(owl_prop.domain)}")
    if owl_prop.range:
        text_content_parts.append(f"Range: {', '.join(owl_prop.range)}")
    if owl_prop.examples:
        text_content_parts.append(f"Examples: {'; '.join(owl_prop.examples)}")
    if owl_prop.notes:
        text_content_parts.append(f"Notes: {'; '.join(owl_prop.notes)}")

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
    """Loads ontology data, processes it into Document objects, and returns them along with the FOLIO instance."""
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

    # Process properties
    logger.info(f"Processing {len(folio_instance.properties)} properties...")
    for owl_prop in folio_instance.properties:
        doc = _create_document_from_owl_property(owl_prop)
        if doc:
            documents.append(doc)

    if limit:
        logger.info(f"Limiting documents to {limit}")
        documents = documents[:limit]

    logger.info(f"Loaded {len(documents)} documents from FOLIO ontology.")
    return documents, folio_instance


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
    """Get the depth of a class relative to a root class."""
    if target_class.iri == root_class.iri:
        return current_depth

    if not target_class.sub_class_of:
        return float("inf")  # Not reachable from root

    min_depth = float("inf")
    for parent_iri in target_class.sub_class_of:
        # Find the parent class
        for parent_class in folio_instance.classes:
            if parent_class.iri == parent_iri:
                depth = _get_class_depth(
                    folio_instance, parent_class, root_class, current_depth + 1
                )
                min_depth = min(min_depth, depth)
                break

    return min_depth


def create_custom_kg(folio_instance: FOLIO) -> CustomKnowledgeGraph:
    """
    Creates a custom knowledge graph structure from FOLIO triples.
    Uses folio[iri] to retrieve definitions and labels for subject and object,
    and folio.get_property_by_label to get definition and labels of predicates.
    """
    chunks = []
    entities = []
    relationships = []

    # Track entities we've already added to avoid duplicates
    added_entities = {}  # entity_name -> Entity object
    added_chunks = set()
    chunk_index = 0  # Track chunk index

    logger.info(f"Processing {len(folio_instance.triples)} triples from FOLIO...")

    for subject_iri, predicate_iri, object_iri in folio_instance.triples:
        try:
            # Get subject information using folio[iri]
            if not subject_iri:
                logger.debug(f"Skipping triple with no subject IRI: {subject_iri}")
                continue

            subject_info = folio_instance[subject_iri]
            subj_iri_clean = subject_iri.split("/")[-1]
            subject_label = (
                getattr(subject_info, "preferred_label", None)
                or getattr(subject_info, "label", None)
                or subj_iri_clean
            )
            subject_definition = (
                getattr(subject_info, "definition", None)
                or getattr(subject_info, "description", None)
                or getattr(subject_info, "comment", None)
                or "No definition available"
            )

            # Get object information using folio[iri]
            if not object_iri:
                logger.debug(f"Skipping triple with no object IRI: {object_iri}")
                continue

            object_info = folio_instance[object_iri]
            obj_iri_clean = object_iri.split("/")[-1]
            object_label = (
                getattr(object_info, "preferred_label", None)
                or getattr(object_info, "label", None)
                or obj_iri_clean
            )
            object_definition = (
                getattr(object_info, "definition", None)
                or getattr(object_info, "description", None)
                or getattr(object_info, "comment", None)
                or "No definition available"
            )

            # Get predicate information using folio.get_property_by_label
            predicate_label = None
            predicate_definition = None

            if predicate_iri:
                try:
                    predicate_info = folio_instance.get_property_by_label(predicate_iri)
                    if predicate_info:
                        predicate_label = (
                            getattr(predicate_info, "preferred_label", None)
                            or getattr(predicate_info, "label", None)
                            or predicate_iri.split("/")[-1]
                        )
                        predicate_definition = (
                            getattr(predicate_info, "definition", None)
                            or getattr(predicate_info, "description", None)
                            or getattr(predicate_info, "comment", None)
                            or "No definition available"
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not get predicate info for {predicate_iri}: {e}"
                    )
                    predicate_label = predicate_iri.split("/")[-1]
                    predicate_definition = "No definition available"

            # Create chunk content describing the relationship
            chunk_content = f"{subject_label} {predicate_label or 'relates to'} {object_label}. {subject_definition} {predicate_definition or ''} {object_definition}"
            chunk_id = f"triple_{subj_iri_clean}_{predicate_iri.split('/')[-1] if predicate_iri else 'unknown'}_{obj_iri_clean}"

            # Add chunk if not already added
            if chunk_id not in added_chunks:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        source_id=chunk_id,
                        chunk_order_index=chunk_index,
                    )
                )
                added_chunks.add(chunk_id)
                chunk_index += 1

            # Add or update subject entity
            if subj_iri_clean not in added_entities:
                added_entities[subj_iri_clean] = Entity(
                    entity_name=subj_iri_clean,  # Use clean name as entity_name
                    entity_type="OWLClass",
                    description=subject_definition,
                    source_id=chunk_id,  # Use chunk_id as source_id for LightRAG
                    chunk_ids=[chunk_id],
                )
            else:
                # Entity already exists, add this chunk to its list
                if chunk_id not in added_entities[subj_iri_clean].chunk_ids:
                    added_entities[subj_iri_clean].chunk_ids.append(chunk_id)
                # Update source_id to the first chunk this entity appears in
                if not added_entities[subj_iri_clean].source_id:
                    added_entities[subj_iri_clean].source_id = chunk_id

            # Add or update object entity
            if obj_iri_clean not in added_entities:
                added_entities[obj_iri_clean] = Entity(
                    entity_name=obj_iri_clean,  # Use clean name as entity_name
                    entity_type="OWLClass",
                    description=object_definition,
                    source_id=chunk_id,  # Use chunk_id as source_id for LightRAG
                    chunk_ids=[chunk_id],
                )
            else:
                # Entity already exists, add this chunk to its list
                if chunk_id not in added_entities[obj_iri_clean].chunk_ids:
                    added_entities[obj_iri_clean].chunk_ids.append(chunk_id)
                # Update source_id to the first chunk this entity appears in
                if not added_entities[obj_iri_clean].source_id:
                    added_entities[obj_iri_clean].source_id = chunk_id

            # Add relationship
            relationships.append(
                Relationship(
                    src_id=subj_iri_clean,  # Use clean name for src_id
                    tgt_id=obj_iri_clean,  # Use clean name for tgt_id
                    description=f"{subject_label} {predicate_label or 'relates to'} {object_label}",
                    keywords=f"{predicate_label or 'relationship'} {subject_label} {object_label}",
                    weight=1.0,
                    source_id=chunk_id,  # Use chunk_id as source_id for LightRAG
                )
            )

        except Exception as e:
            logger.warning(
                f"Error processing triple ({subject_iri}, {predicate_iri}, {object_iri}): {e}"
            )
            continue

    # Convert the entities dictionary to a list
    entities = list(added_entities.values())

    logger.info(
        f"Created custom KG with {len(chunks)} chunks, {len(entities)} entities, and {len(relationships)} relationships"
    )

    return CustomKnowledgeGraph(
        chunks=chunks, entities=entities, relationships=relationships
    )


def create_entities_and_relations_from_folio(folio_instance: FOLIO, rag):
    """
    Creates entities and relationships from FOLIO triples using LightRAG's built-in functions.
    Uses rag.create_entity() and rag.create_relation().
    """
    # Track entities we've already created to avoid duplicates
    created_entities = set()

    logger.info(f"Processing {len(folio_instance.triples)} triples from FOLIO...")

    for subject_iri, predicate_iri, object_iri in folio_instance.triples:
        try:
            # Get subject information using folio[iri]
            if not subject_iri:
                logger.debug(f"Skipping triple with no subject IRI: {subject_iri}")
                continue

            subject_info = folio_instance[subject_iri]
            subj_iri_clean = subject_iri.split("/")[-1]
            # Use default label instead of preferred_label
            subject_label = getattr(subject_info, "label", None) or subj_iri_clean
            subject_definition = (
                getattr(subject_info, "definition", None)
                or getattr(subject_info, "description", None)
                or getattr(subject_info, "comment", None)
                or "No definition available"
            )

            # Add preferred label and alternative labels to definition
            subject_definition_parts = [subject_definition]
            if (
                hasattr(subject_info, "preferred_label")
                and subject_info.preferred_label
            ):
                subject_definition_parts.append(
                    f"Preferred label: {subject_info.preferred_label}"
                )
            if (
                hasattr(subject_info, "alternative_labels")
                and subject_info.alternative_labels
            ):
                alt_labels_str = ", ".join(subject_info.alternative_labels)
                subject_definition_parts.append(f"Alternative labels: {alt_labels_str}")
            subject_definition = ". ".join(subject_definition_parts)

            # Get object information using folio[iri]
            if not object_iri:
                logger.debug(f"Skipping triple with no object IRI: {object_iri}")
                continue

            object_info = folio_instance[object_iri]
            obj_iri_clean = object_iri.split("/")[-1]
            # Use default label instead of preferred_label
            object_label = getattr(object_info, "label", None) or obj_iri_clean
            object_definition = (
                getattr(object_info, "definition", None)
                or getattr(object_info, "description", None)
                or getattr(object_info, "comment", None)
                or "No definition available"
            )

            # Add preferred label and alternative labels to definition
            object_definition_parts = [object_definition]
            if hasattr(object_info, "preferred_label") and object_info.preferred_label:
                object_definition_parts.append(
                    f"Preferred label: {object_info.preferred_label}"
                )
            if (
                hasattr(object_info, "alternative_labels")
                and object_info.alternative_labels
            ):
                alt_labels_str = ", ".join(object_info.alternative_labels)
                object_definition_parts.append(f"Alternative labels: {alt_labels_str}")
            object_definition = ". ".join(object_definition_parts)

            # Get predicate information using folio.get_property_by_label
            predicate_label = None
            predicate_definition = None

            if predicate_iri:
                try:
                    predicate_info = folio_instance.get_property_by_label(predicate_iri)
                    if predicate_info:
                        # Use default label instead of preferred_label
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

                        # Add preferred label and alternative labels to definition
                        predicate_definition_parts = [predicate_definition]
                        if (
                            hasattr(predicate_info, "preferred_label")
                            and predicate_info.preferred_label
                        ):
                            predicate_definition_parts.append(
                                f"Preferred label: {predicate_info.preferred_label}"
                            )
                        if (
                            hasattr(predicate_info, "alternative_labels")
                            and predicate_info.alternative_labels
                        ):
                            alt_labels_str = ", ".join(
                                predicate_info.alternative_labels
                            )
                            predicate_definition_parts.append(
                                f"Alternative labels: {alt_labels_str}"
                            )
                        predicate_definition = ". ".join(predicate_definition_parts)
                except Exception as e:
                    logger.debug(
                        f"Could not get predicate info for {predicate_iri}: {e}"
                    )
                    predicate_label = predicate_iri.split("/")[-1]
                    predicate_definition = "No definition available"

            # Create subject entity if not already created
            if subj_iri_clean not in created_entities:
                entity = rag.create_entity(
                    subj_iri_clean,
                    {
                        "description": subject_definition,
                        "entity_type": "OWLClass",
                        "iri": subject_iri,
                        "label": subject_label,
                    },
                )
                created_entities.add(subj_iri_clean)
                logger.debug(f"Created entity: {subj_iri_clean}")

            # Create object entity if not already created
            if obj_iri_clean not in created_entities:
                entity = rag.create_entity(
                    obj_iri_clean,
                    {
                        "description": object_definition,
                        "entity_type": "OWLClass",
                        "iri": object_iri,
                        "label": object_label,
                    },
                )
                created_entities.add(obj_iri_clean)
                logger.debug(f"Created entity: {obj_iri_clean}")

            # Create relationship between entities
            relation_description = (
                f"{subject_label} {predicate_label or 'relates to'} {object_label}"
            )
            relation_keywords = (
                f"{predicate_label or 'relationship'} {subject_label} {object_label}"
            )

            rag.create_relation(
                subj_iri_clean,
                obj_iri_clean,
                {
                    "description": relation_description,
                    "keywords": relation_keywords,
                    "weight": 1.0,
                    "predicate_iri": predicate_iri,
                    "predicate_label": predicate_label,
                },
            )

            logger.debug(f"Created relation: {subj_iri_clean} -> {obj_iri_clean}")

        except Exception as e:
            logger.warning(
                f"Error processing triple ({subject_iri}, {predicate_iri}, {object_iri}): {e}"
            )
            continue

    logger.info(
        f"Created {len(created_entities)} entities and processed {len(folio_instance.triples)} relationships"
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
    docs_to_load, folio_instance = load_ontology_for_rag(
        limit=None, max_depth=6
    )  # Or set a specific limit for standalone run e.g. limit=100

    if not docs_to_load:
        logger.error("No documents to load.")
        return

    logger.info(
        f"Sending {len(docs_to_load)} documents to LightRAG service at {lightrag_base_url}..."
    )

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
    custom_kg_data = create_custom_kg(folio_instance)
    logger.info("Custom KG data created.")


if __name__ == "__main__":
    main()
