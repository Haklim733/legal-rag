import logging
import requests
import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from folio import FOLIO
from folio.models import OWLClass, OWLObjectProperty

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
    label = owl_prop.label or "Unnamed Property"
    definition = owl_prop.definition or "No definition available."

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


def load_ontology_for_rag(limit: int = None, max_depth: int = 2) -> list[Document]:
    folio_instance = FOLIO("github", llm=None)

    documents = []

    # --- Process Classes/Concepts starting from top-level classes with max depth ---
    print(f"Processing FOLIO classes with max depth {max_depth}...")

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

    logger.info(f"Found {len(top_level)} top-level classes")

    # Process each top-level class and its children up to max_depth
    for top_class in top_level:
        if limit is not None and len(documents) >= limit:
            break

        # Add the top-level class itself
        doc = _create_document_from_owl_class(top_class)
        if doc:
            documents.append(doc)

        # Get children up to max_depth using get_children
        children = _get_children_with_depth(folio_instance, top_class, max_depth)
        for child in children:
            if limit is not None and len(documents) >= limit:
                break
            doc = _create_document_from_owl_class(child)
            if doc:
                documents.append(doc)
                depth = _get_class_depth(folio_instance, child, top_class)

        if limit is not None and len(documents) >= limit:
            break

    # --- Process Properties/Relationships (limited) ---
    if limit is None or len(documents) < limit:
        all_properties = folio_instance.get_all_properties()
        logger.info(f"Total properties available: {len(all_properties)}")

        # Take only a subset of properties for testing
        test_properties = all_properties[:20]  # Limit to first 20 properties
        logger.info(
            f"Processing {len(test_properties)} OWL object properties from FOLIO..."
        )

        for owl_prop in test_properties:
            if limit is not None and len(documents) >= limit:
                break
            doc = _create_document_from_owl_property(owl_prop)
            if doc:
                documents.append(doc)
            if limit is not None and len(documents) >= limit:
                break

    logger.info(f"Prepared a total of {len(documents)} documents for LightRAG.")
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


if __name__ == "__main__":
    main()
