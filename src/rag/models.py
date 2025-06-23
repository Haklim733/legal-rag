from typing import Callable, Optional, List, Literal, Dict, Any, Union
import os
from pydantic import BaseModel, Field, field_validator
import json
import logging
import re
from enum import Enum
import sys

logger = logging.getLogger(__name__)


# Suppress ascii_colors errors by redirecting stderr for that specific error
class SuppressAsciiColorsError:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr


class ExtractionMode(Enum):
    """Modes for JSON extraction from text."""

    STRICT = "strict"
    LENIENT = "lenient"
    AGGRESSIVE = "aggressive"


def extract_json_from_text(
    text: str, mode: ExtractionMode = ExtractionMode.STRICT
) -> dict:
    """
    Extract JSON from text using different extraction modes.

    Args:
        text: Text that may contain JSON
        mode: Extraction mode (strict, lenient, aggressive)

    Returns:
        Extracted JSON as dictionary

    Raises:
        ValueError: If JSON cannot be extracted
    """
    if mode == ExtractionMode.STRICT:
        # Look for JSON blocks with clear boundaries
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Nested JSON objects
            r"\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]",  # JSON arrays
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

    elif mode == ExtractionMode.LENIENT:
        # Try to find JSON-like structures and clean them
        # Look for text between curly braces
        brace_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        for match in brace_matches:
            try:
                # Clean up common issues
                cleaned = re.sub(r"//.*?\n", "\n", match)  # Remove single-line comments
                cleaned = re.sub(
                    r"/\*.*?\*/", "", cleaned, flags=re.DOTALL
                )  # Remove multi-line comments
                cleaned = re.sub(r",\s*}", "}", cleaned)  # Remove trailing commas
                cleaned = re.sub(
                    r",\s*]", "]", cleaned
                )  # Remove trailing commas in arrays
                # Clean up keys with newlines and spaces
                cleaned = re.sub(r'\n\s*"', '"', cleaned)  # Remove newlines before keys
                cleaned = re.sub(
                    r'"\s*\n\s*:', '":', cleaned
                )  # Remove newlines after keys
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    elif mode == ExtractionMode.AGGRESSIVE:
        # Most aggressive: try to extract and fix common JSON issues
        # Look for any text that looks like JSON
        potential_json = re.search(r"\{.*\}", text, re.DOTALL)
        if potential_json:
            json_str = potential_json.group(0)
            try:
                # Try to fix common issues
                # Replace single quotes with double quotes
                json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                # Remove trailing commas
                json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
                # Remove comments
                json_str = re.sub(r"//.*?\n", "\n", json_str)
                json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
                # Clean up keys with newlines and spaces
                json_str = re.sub(
                    r'\n\s*"', '"', json_str
                )  # Remove newlines before keys
                json_str = re.sub(
                    r'"\s*\n\s*:', '":', json_str
                )  # Remove newlines after keys
                # Normalize whitespace around colons
                json_str = re.sub(r'"\s*:\s*', '": ', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract valid JSON from text using mode {mode.value}")


class QueryParam(BaseModel):
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        "mix", description="Specifies the retrieval mode"
    )
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval.
    """

    only_need_context: bool = Field(
        False, description="If True, only returns the retrieved context"
    )
    """If True, only returns the retrieved context without generating a response."""

    only_need_prompt: bool = Field(
        False, description="If True, only returns the generated prompt"
    )
    """If True, only returns the generated prompt without producing a response."""

    response_type: str = Field("json", description="Defines the response format")
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    stream: bool = Field(False, description="If True, enables streaming output")
    """If True, enables streaming output for real-time responses."""

    top_k: int = Field(
        int(os.getenv("TOP_K", "60")), description="Number of top items to retrieve"
    )
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    max_token_for_text_unit: int = Field(
        int(os.getenv("MAX_TOKEN_TEXT_CHUNK", "4000")),
        description="Maximum tokens for each text chunk",
    )
    """Maximum number of tokens allowed for each retrieved text chunk."""

    max_token_for_global_context: int = Field(
        int(os.getenv("MAX_TOKEN_RELATION_DESC", "4000")),
        description="Maximum tokens for global context",
    )
    """Maximum number of tokens allocated for relationship descriptions in global retrieval."""

    max_token_for_local_context: int = Field(
        int(os.getenv("MAX_TOKEN_ENTITY_DESC", "4000")),
        description="Maximum tokens for local context",
    )
    """Maximum number of tokens allocated for entity descriptions in local retrieval."""

    conversation_history: List[dict[str, str]] = Field(
        default_factory=list, description="Past conversation history"
    )
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    history_turns: int = Field(
        3, description="Number of conversation turns to consider"
    )
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""

    ids: Optional[List[str]] = Field(None, description="List of ids to filter results")
    """List of ids to filter the results."""

    model_func: Optional[Callable[..., object]] = Field(
        None, description="Optional override for LLM model function"
    )
    """Optional override for the LLM model function to use for this specific query.
    If provided, this will be used instead of the global model function.
    This allows using different models for different query modes.
    """

    user_prompt: Optional[str] = Field(
        None,
        description="User prompt for the query",
    )

    hl_keywords: Optional[List[str]] = Field(
        None, description="High-level keywords for filtering"
    )
    ll_keywords: Optional[List[str]] = Field(
        None, description="Low-level keywords for filtering"
    )
    original_query: Optional[str] = Field(None, description="Original query text")


class Chunk(BaseModel):
    """Represents a chunk of text content in the knowledge graph."""

    content: str = Field(..., description="The text content of the chunk")
    source_id: str = Field(..., description="Unique identifier for the chunk source")
    chunk_order_index: int = Field(..., description="Index of the chunk in the source")


class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""

    entity_name: str = Field(..., description="Name of the entity (clean name)")
    entity_type: str = Field(..., description="Type of the entity (e.g., 'OWLClass')")
    description: Optional[str] = Field(
        default="", description="Description of the entity"
    )
    source_id: str = Field(default="", description="Source chunk ID (not full IRI)")
    chunk_ids: List[str] = Field(
        default_factory=list, description="List of chunk IDs where the entity is found"
    )
    weight: float = Field(
        default=1.0, description="Confidence weight of the entity (0.0 to 1.0)"
    )

    @field_validator("entity_name", "entity_type")
    @classmethod
    def check_not_empty(cls, v: str) -> str:
        """Ensure that critical fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure weight is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v


class Relationship(BaseModel):
    """Represents a relationship between entities in the knowledge graph."""

    src_id: str = Field(..., description="Source entity ID (clean name)")
    tgt_id: str = Field(..., description="Target entity ID (clean name)")
    description: Optional[str] = Field(
        default="", description="Description of the relationship"
    )
    keywords: str = Field(
        default="", description="Keywords associated with the relationship"
    )
    source_id: str = Field(
        default="", description="Source chunk ID for this relationship"
    )
    weight: float = Field(
        default=1.0, description="Confidence weight of the relationship (0.0 to 1.0)"
    )

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v):
        """Convert keywords to string if it's a list and ensure it is not empty."""
        if isinstance(v, list):
            v = ", ".join(v)
        s = str(v)
        return s

    @field_validator("description", "src_id", "tgt_id")
    @classmethod
    def check_not_empty(cls, v: str) -> str:
        """Ensure that critical fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure weight is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v

    @classmethod
    def create_relationship_id(
        cls, src_id: str, relationship_type: str, tgt_id: str
    ) -> str:
        """Create a unique relationship ID from source, type, and target."""
        return f"{src_id}_{relationship_type}_{tgt_id}".replace(" ", "_").lower()


class CustomKnowledgeGraph(BaseModel):
    """Represents a custom knowledge graph structure for LightRAG."""

    chunks: List[Chunk] = Field(..., description="List of text chunks")
    entities: List[Entity] = Field(..., description="List of entities")
    relationships: List[Relationship] = Field(..., description="List of relationships")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            # Add any custom JSON encoders if needed
        }

    def to_dict(self) -> dict:
        """Convert the knowledge graph to a dictionary format compatible with LightRAG."""
        return {
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "entities": [entity.model_dump() for entity in self.entities],
            "relationships": [rel.model_dump() for rel in self.relationships],
        }

    def get(self, key: str, default=None):
        """Provide a get method for dictionary-like access."""
        if key == "chunks":
            return [chunk.model_dump() for chunk in self.chunks]
        elif key == "entities":
            return [entity.model_dump() for entity in self.entities]
        elif key == "relationships":
            return [rel.model_dump() for rel in self.relationships]
        else:
            return default


SYSTEM_PROMPT_SINGLE = """Extract legal entities and relationships from the text. Return ONLY valid JSON.

INSTRUCTIONS:
1. First, identify all legal entities (people, organizations, courts, agencies)
2. Use EXACT names as they appear in the text
3. Create relationships between entities using ONLY the exact entity names you identified
4. Do NOT create relationships with entities not in your entities list

REQUIRED FORMAT:
{{
    "entities": [
        {{
            "entity_name": "EXACT name from text",
            "entity_type": "Person|Court|Agency|Organization",
            "description": "brief description"
        }}
    ],
    "relationships": [
        {{
            "src_id": "EXACT entity_name from entities list above",
            "tgt_id": "EXACT entity_name from entities list above",
            "description": "relationship description",
            "keywords": "comma-separated keywords"
        }}
    ]
}}

CRITICAL RULES:
- Use EXACT entity names from the text
- src_id and tgt_id MUST match entity_name exactly
- Return ONLY JSON, no explanations or markdown
- Do NOT generate Python code or other text"""


class RAGResponse(BaseModel):
    """Validated response structure for RAG queries containing ontological concepts."""

    query_text: str = Field(..., description="Original query text")
    response_text: str = Field(..., description="Generated response text")
    entities: List[Entity] = Field(..., description="Entities found in the query")
    relationships: List[Relationship] = Field(
        ..., description="Relationships found in the query"
    )


def extract_and_validate_json(
    response_text: str, extraction_mode: str = "aggressive"
) -> dict:
    """
    Extract and validate JSON from response text.

    Args:
        response_text: Raw response text from RAG
        extraction_mode: Mode for JSON extraction (strict, lenient, aggressive)

    Returns:
        Dictionary containing the parsed JSON data

    Raises:
        ValueError: If JSON cannot be extracted or validated
    """
    logger.info(f"Raw response text length: {len(response_text)}")
    logger.info(f"Response text starts with: {response_text[:100]}")

    # Convert string to enum
    if extraction_mode == "strict":
        mode = ExtractionMode.STRICT
    elif extraction_mode == "lenient":
        mode = ExtractionMode.LENIENT
    elif extraction_mode == "aggressive":
        mode = ExtractionMode.AGGRESSIVE
    else:
        mode = ExtractionMode.STRICT

    # Extract JSON
    try:
        response_data = extract_json_from_text(response_text, mode)
        logger.info("✅ Successfully extracted JSON from response")
        logger.info(f"Response data keys: {list(response_data.keys())}")
        logger.info(f"Number of entities: {len(response_data.get('entities', []))}")
        logger.info(
            f"Number of relationships: {len(response_data.get('relationships', []))}"
        )

        # Validate JSON structure
        if not isinstance(response_data, dict):
            raise ValueError("Response data is not a dictionary")

        if "entities" not in response_data or "relationships" not in response_data:
            raise ValueError(
                "Response missing required 'entities' or 'relationships' keys"
            )

        if not isinstance(response_data["entities"], list) or not isinstance(
            response_data["relationships"], list
        ):
            raise ValueError("'entities' and 'relationships' must be lists")

        return response_data

    except Exception as e:
        logger.error(f"❌ Failed to extract JSON: {e}")
        logger.error(f"Raw response: {response_text}")
        raise


def create_fallback_response() -> dict:
    """
    Create a fallback response when JSON extraction fails.

    Returns:
        Dictionary with fallback entities and relationships
    """
    logger.warning("Creating fallback response due to JSON extraction failure")
    return {
        "entities": [
            {
                "entity_name": "Lawyer",
                "entity_type": "Legal Services Provider",
                "description": "Legal professional providing services",
            }
        ],
        "relationships": [
            {
                "src_id": "Lawyer",
                "tgt_id": "Client",
                "description": "Legal representation relationship",
                "keywords": "represents, advises, counsels",
            }
        ],
    }


def parse_entities_from_response(response_data: dict) -> List[Entity]:
    """
    Parse entities from the response data.

    Args:
        response_data: Dictionary containing the parsed JSON response

    Returns:
        List of Entity objects
    """
    entities = []

    if "entities" not in response_data:
        logger.warning("No 'entities' key found in response_data")
        return entities

    logger.info(f"Found {len(response_data['entities'])} entities in response")

    for i, entity_data in enumerate(response_data["entities"]):
        try:
            logger.info(f"Processing entity {i}: {entity_data}")

            # Validate entity data structure
            if not isinstance(entity_data, dict):
                logger.warning(f"Entity {i} is not a dictionary: {entity_data}")
                continue

            if "entity_name" not in entity_data or "entity_type" not in entity_data:
                logger.warning(f"Entity {i} missing required fields: {entity_data}")
                continue

            entity = Entity(
                entity_name=entity_data.get("entity_name", ""),
                entity_type=entity_data.get("entity_type", "entity"),
                description=entity_data.get("description", ""),
                source_id=entity_data.get("source_id", ""),
                chunk_ids=entity_data.get("chunk_ids", []),
            )
            entities.append(entity)
            logger.info(f"Successfully created entity: {entity.entity_name}")
        except Exception as e:
            logger.warning(f"Failed to create entity from data {entity_data}: {e}")
            continue

    logger.info(f"Successfully parsed {len(entities)} entities")
    return entities


def parse_relationships_from_response(
    response_data: dict, entity_names: set
) -> List[Relationship]:
    """
    Parse relationships from the response data with exact entity name matching.

    Args:
        response_data: Dictionary containing the parsed JSON response
        entity_names: Set of valid entity names for exact matching

    Returns:
        List of Relationship objects
    """
    relationships = []

    if "relationships" not in response_data:
        logger.warning("No 'relationships' key found in response_data")
        return relationships

    logger.info(f"Available entity names for relationships: {entity_names}")
    logger.info(
        f"Found {len(response_data['relationships'])} relationships in response"
    )

    def find_entities_in_string(id_string, names):
        """Finds entity names within a string. Prioritizes exact match."""
        logger.info(f"Looking for '{id_string}' in entity names: {names}")
        if id_string in names:
            logger.info(f"Found exact match: {id_string}")
            return [id_string]
        # Try case-insensitive matching
        id_string_lower = id_string.lower()
        found = [name for name in names if name.lower() == id_string_lower]
        if found:
            logger.info(f"Found case-insensitive match: {found}")
            return found
        # Try partial matching
        found = [
            name
            for name in names
            if name.lower() in id_string_lower or id_string_lower in name.lower()
        ]
        logger.info(f"Found partial matches: {found}")
        return found

    for i, rel_data in enumerate(response_data["relationships"]):
        try:
            logger.info(f"Processing relationship {i}: {rel_data}")

            # Validate relationship data structure
            if not isinstance(rel_data, dict):
                logger.warning(f"Relationship {i} is not a dictionary: {rel_data}")
                continue

            if "src_id" not in rel_data or "tgt_id" not in rel_data:
                logger.warning(f"Relationship {i} missing required fields: {rel_data}")
                continue

            src_id_raw = rel_data.get("src_id", "")
            tgt_id_raw = rel_data.get("tgt_id", "")

            src_ids = find_entities_in_string(src_id_raw, entity_names)
            tgt_ids = find_entities_in_string(tgt_id_raw, entity_names)

            logger.info(f"Found src_ids: {src_ids} for '{src_id_raw}'")
            logger.info(f"Found tgt_ids: {tgt_ids} for '{tgt_id_raw}'")

            if not src_ids or not tgt_ids:
                logger.warning(
                    f"Skipping relationship with ambiguous or missing src/tgt: '{src_id_raw}' -> '{tgt_id_raw}'"
                )
                continue

            # Validate keywords against knowledge graph
            keywords = rel_data.get("keywords", "")
            # Handle cases where keywords are returned as a list
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)

            for src_id in src_ids:
                for tgt_id in tgt_ids:
                    relationship = Relationship(
                        src_id=src_id,
                        tgt_id=tgt_id,
                        description=rel_data.get("description", ""),
                        keywords=keywords,
                        source_id=rel_data.get("source_id", ""),
                    )
                    relationships.append(relationship)
                    logger.info(
                        f"Successfully created relationship: {src_id} -> {tgt_id}"
                    )
        except Exception as e:
            logger.warning(f"Failed to create relationship from data {rel_data}: {e}")
            continue

    logger.info(f"Successfully parsed {len(relationships)} relationships")
    return relationships


def validate_rag_response(
    response_text: str, query_text: str, extraction_mode: str = "aggressive"
) -> RAGResponse:
    """
    Validate and parse RAG response to ensure it contains proper ontological concepts.

    Args:
        response_text: Raw response text from RAG
        query_text: Original query text
        extraction_mode: Mode for JSON extraction (strict, lenient, aggressive)

    Returns:
        RAGResponse: Validated response structure

    Raises:
        ValueError: If response doesn't contain valid JSON structure
        ValidationError: If response doesn't meet ontological concept requirements
    """
    # Extract and validate JSON
    try:
        response_data = extract_and_validate_json(response_text, extraction_mode)
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        response_data = create_fallback_response()

    # Parse entities
    entities = parse_entities_from_response(response_data)

    # Parse relationships
    entity_names = {entity.entity_name for entity in entities}
    relationships = parse_relationships_from_response(response_data, entity_names)

    logger.info(
        f"Final result: {len(entities)} entities, {len(relationships)} relationships"
    )

    return RAGResponse(
        query_text=query_text,
        response_text=response_text,
        entities=entities,
        relationships=relationships,
    )


async def simple_local_model_complete(prompt: str, **kwargs) -> str:
    logger.info(f"Local model received prompt: {prompt[:100]}...")
    # Example: return a JSON string with keywords
    result = {
        "hl_keywords": ["legal representation", "court proceedings"],
        "ll_keywords": ["lawyer", "organization", "Wiliam", "court"],
    }
    return json.dumps(result)


SAMPLE_KG = {
    "chunks": [
        {
            "content": "A lawyer is a legal professional who provides legal advice and representation to clients in various legal matters, including litigation, transactions, and regulatory compliance. Lawyers must be licensed and follow ethical guidelines.",
            "source_id": "lawyer",
            "chunk_order_index": 0,
        },
        {
            "content": "A legal services buyer is a person or entity that obtains legal services from a legal service provider. This includes individuals, businesses, organizations, and other entities that need legal advice, representation, or other legal services.",
            "source_id": "legal_services_buyer",
            "chunk_order_index": 0,
        },
        # Relationship chunk describing the connection
        {
            "content": "The relationship between a lawyer and a legal services buyer involves the provision of legal services. The lawyer acts as a legal service provider, offering expertise in areas such as contract law, litigation, corporate law, or other legal specialties. The legal services buyer engages the lawyer's services for legal advice, representation in court, document preparation, or other legal needs. This relationship is typically formalized through a retainer agreement or engagement letter that outlines the scope of services, fees, and responsibilities of both parties.",
            "source_id": "triple_lawyer_represents_legal_services_buyer",
            "chunk_order_index": 0,
        },
    ],
    "entities": [
        {
            "entity_name": "Actor / Player",
            "entity_type": "TopLevelClass",
            "description": "A person who has a role in a legal matter (e.g., Buyer, Provider, Lawyer, Law Firm, Expert, Employer, Employee, Buyer, Seller, Lessor, Lessee, Debtor, Creditor, Payor, Payee, Landlord, Tenant).",
            "source_id": "actor_player",
            "chunk_ids": ["actor_player"],
        },
        {
            "entity_name": "Lawyer",
            "entity_type": "Actor / Player",
            "description": "A lawyer is a legal professional who provides legal advice and representation to clients in various legal matters, including litigation, transactions, and regulatory compliance. labels: Attorney",
            "source_id": "lawyer",
            "chunk_ids": ["lawyer"],
        },
        {
            "entity_name": "Public Defender",
            "entity_type": "Lawyer",
            "description": "A public defender is a lawyer appointed by the court to represent individuals who cannot afford to hire their own legal counsel.",
            "source_id": "public_defender",
            "chunk_ids": ["public_defender"],
        },
        {
            "entity_name": "Legal Services Provider",
            "entity_type": "Actor / Player",
            "description": "A person or entity that is providing legal services to another person or entity",
            "source_id": "legal_services_provider",
            "chunk_ids": ["legal_services_provider"],
        },
        {
            "entity_name": "Legal Services Buyer",
            "entity_type": "Actor / Player",
            "description": "A person or entity that is obtaining legal services from a legal service provider. Labels: Client, Party",
            "source_id": "legal_services_buyer",
            "chunk_ids": ["legal_services_buyer"],
        },
        {
            "entity_name": "Individual Person",
            "entity_type": "Legal Entity",
            "description": "An individual or natural person is a person that is an individual human being, as opposed to a legal person or entity, which may be a private or public organization. Labels: Individual, Natural Person, Person",
            "source_id": "individual_person",
            "chunk_ids": ["individual_person"],
        },
        {
            "entity_name": "Tenant",
            "entity_type": "Actor / Player",
            "description": "A person or entity who leases, rents, or occupies property (from a landlord)",
            "source_id": "tenant",
            "chunk_ids": ["tenant"],
        },
    ],
    "relationships": [
        {
            "src_id": "entity_lawyer",
            "tgt_id": "entity_legal_services_buyer",
            "description": "'Represented' in a legal context refers to the act of acting on behalf of another person or entity in legal matters. This involves a representative, such as a lawyer, advocating, making decisions, or taking actions under the authority and in the interest of the represented party.",
            "keywords": "represents, legal representation, advises, counsels",
            "source_id": "lawyer_represents_legal_services_buyer",
        },
        {
            "src_id": "entity_individual_person",
            "tgt_id": "entity_actor_player",
            "description": 'In legal and organizational contexts, "is member of" signifies the belonging or inclusion of an individual or entity in a specific group, committee, board, or organization. This denotes that the individual or entity has a role, responsibilities, rights, or a position within the collective body, subject to its rules, objectives, and governance structure.',
            "keywords": "is a member of, is a part of, belongs to, works for, employed by",
            "source_id": "individual_person_is_member_of_actor_player",
        },
        {
            "src_id": "entity_tenant",
            "tgt_id": "entity_landlord",
            "description": "A tenant rents or leases property from a landlord under a rental agreement.",
            "keywords": "rents from, leases from, pays rent to, occupies property of",
            "source_id": "tenant_rents_from_landlord",
        },
        {
            "src_id": "entity_landlord",
            "tgt_id": "entity_tenant",
            "description": "A landlord owns property that is rented or leased to a tenant.",
            "keywords": "rents to, leases to, owns property rented by, provides housing for",
            "source_id": "landlord_rents_to_tenant",
        },
    ],
}


def get_valid_keywords_from_kg() -> List[str]:
    """Get all valid keywords from the knowledge graph to prevent hallucination."""
    valid_keywords = []
    for rel in SAMPLE_KG["relationships"]:
        keywords = rel.get("keywords", "").split(", ")
        valid_keywords.extend(keywords)
    return list(set(valid_keywords))  # Remove duplicates


def validate_keywords_against_kg(keywords: str | list) -> bool:
    """Validate that keywords exist in the knowledge graph."""
    if not keywords:
        return True

    valid_keywords = get_valid_keywords_from_kg()

    # Handle both string and list inputs
    if isinstance(keywords, str):
        provided_keywords = [k.strip() for k in keywords.split(", ")]
    elif isinstance(keywords, list):
        provided_keywords = keywords
    else:
        return False

    # Check if any of the provided keywords are in the valid keywords
    for keyword in provided_keywords:
        if keyword in valid_keywords:
            return True
    return False
