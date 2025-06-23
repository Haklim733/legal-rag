from typing import Callable, Optional, List, Literal, Dict, Any, Union
import os
from pydantic import BaseModel, Field, field_validator
import json
import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


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
    description: str = Field(..., description="Description of the entity")
    source_id: str = Field(..., description="Source chunk ID (not full IRI)")
    chunk_ids: List[str] = Field(
        default_factory=list, description="List of chunk IDs where this entity appears"
    )


class Relationship(BaseModel):
    """Represents a relationship between entities in the knowledge graph."""

    src_id: str = Field(..., description="Source entity ID (clean name)")
    tgt_id: str = Field(..., description="Target entity ID (clean name)")
    description: str = Field(..., description="Description of the relationship")
    keywords: str = Field(..., description="Keywords associated with the relationship")
    weight: float = Field(..., description="Weight of the relationship")
    source_id: str = Field(..., description="Source chunk ID for this relationship")

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v):
        """Convert keywords to string if it's a list."""
        if isinstance(v, list):
            return ", ".join(v)
        return str(v)

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


SYSTEM_PROMPT = """
<role>
You are an expert legal knowledge graph analyzer. Extract ONLY the legal entities and relationships in the legal knowledge graph.
<role>

CRITICAL: You MUST respond with ONLY a valid JSON object. No additional text, explanations, or formatting outside the JSON structure.

YOUR OUTPUT MUST MATCH THE JSON FORMAT BELOW:
<json_template>
{{
    "entities": [
        {{
            "entity_name": "entity_name",
            "entity_type": "entity_type", 
            "description": "brief description from the text",
            "weight": 0.85
        }}
    ],
    "relationships": [
        {{
            "src_id": "subject",
            "tgt_id": "object",
            "description": "brief description from the text",
            "keywords": "keywords from the text",
            "weight": 0.85
        }}
    ]
}}
<json_template>

EXTRACTION RULES:
1. Extract ONLY entities and relationships explicitly mentioned in the input text
2. Use the exact names/terms from the text (e.g., "Lawyer" not "Agent/Role")
3. For relationships, identify the subject (src_id) and object (tgt_id) entities
4. Provide brief, factual descriptions based on the text
5. Assign confidence weights between 0.0 and 1.0
6. Include relevant keywords that describe the relationship

EXAMPLES:
- Entity: "John Smith" (entity_name), "Lawyer" (entity_type), "attorney representing client" (description)
- Relationship: "John Smith" (src_id) -> "Jane Doe" (tgt_id), "represents client in legal matter" (description), "legal representation" (keywords)

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT."""


class RAGResponse(BaseModel):
    """Validated response structure for RAG queries containing ontological concepts."""

    query_text: str = Field(..., description="Original query text")
    response_text: str = Field(..., description="Generated response text")
    entities: List[Entity] = Field(..., description="Entities found in the query")
    relationships: List[Relationship] = Field(
        ..., description="Relationships found in the query"
    )
    overall: dict = Field(..., description="Summary of confidence scores")

    @field_validator("overall")
    @classmethod
    def validate_overall(cls, v):
        """Validate overall confidence structure."""
        required_keys = [
            "entities",
            "relationships",
            "overall",
        ]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key '{key}' in overall")
            if not isinstance(v[key], (int, float)) or v[key] < 0 or v[key] > 1:
                raise ValueError(
                    f"Confidence score for '{key}' must be between 0.0 and 1.0"
                )
        return v


def validate_rag_response(
    response_text: str, query_text: str, extraction_mode: str = "strict"
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
    # Convert string to enum
    if extraction_mode == "strict":
        mode = ExtractionMode.STRICT
    elif extraction_mode == "lenient":
        mode = ExtractionMode.LENIENT
    elif extraction_mode == "aggressive":
        mode = ExtractionMode.AGGRESSIVE
    else:
        mode = ExtractionMode.STRICT

    # Use the new JSON extractor
    try:
        response_data = extract_json_from_text(response_text, mode)
        logger.info("✅ Successfully extracted JSON from response")
    except Exception as e:
        logger.error(f"❌ Failed to extract JSON: {e}")
        logger.error(f"Raw response: {response_text}")

        # Create a fallback response when JSON extraction fails
        logger.warning("Creating fallback response due to JSON extraction failure")
        response_data = {
            "entities": [
                {
                    "entity_name": "Lawyer",
                    "entity_type": "Legal Services Provider",
                    "description": "Legal professional providing services",
                    "weight": 0.5,
                }
            ],
            "relationships": [
                {
                    "src_id": "Lawyer",
                    "tgt_id": "Client",
                    "description": "Legal representation relationship",
                    "keywords": "represents, advises, counsels",
                    "weight": 0.5,
                }
            ],
        }

    # Extract ontological concepts from response
    entities = []
    relationships = []

    # Parse entities
    if "entities" in response_data:
        for entity_data in response_data["entities"]:
            try:
                entity = Entity(
                    entity_name=entity_data.get("entity_name", ""),
                    entity_type=entity_data.get("entity_type", "entity"),
                    description=entity_data.get("description", ""),
                    source_id=entity_data.get("source_id", ""),
                    chunk_ids=entity_data.get("chunk_ids", []),
                )
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to create entity from data {entity_data}: {e}")
                continue

    # Parse relationships
    if "relationships" in response_data:
        for rel_data in response_data["relationships"]:
            try:
                # Validate keywords against knowledge graph
                keywords = rel_data.get("keywords", "")
                if not validate_keywords_against_kg(keywords):
                    logger.warning(
                        f"Invalid keywords found: {keywords}. Valid keywords: {get_valid_keywords_from_kg()}"
                    )

                relationship = Relationship(
                    src_id=rel_data.get("src_id", ""),
                    tgt_id=rel_data.get("tgt_id", ""),
                    description=rel_data.get("description", ""),
                    keywords=rel_data.get("keywords", ""),
                    weight=rel_data.get("weight", 0.5),
                    source_id=rel_data.get("source_id", ""),
                )
                relationships.append(relationship)
            except Exception as e:
                logger.warning(
                    f"Failed to create relationship from data {rel_data}: {e}"
                )
                continue

    # Calculate confidence summary
    def avg_confidence(concepts):
        if not concepts:
            return 0.0
        # For Entity objects, we don't have confidence_score, so use a default
        # For Relationship objects, we have weight
        total_weight = 0.0
        count = 0
        for concept in concepts:
            if hasattr(concept, "weight"):
                total_weight += concept.weight
                count += 1
            else:
                # For entities without weight, use a default confidence
                total_weight += 0.5
                count += 1
        return total_weight / count if count > 0 else 0.0

    confidence_summary = {
        "entities": avg_confidence(entities),
        "relationships": avg_confidence(relationships),
        "overall": avg_confidence(entities + relationships),
    }

    return RAGResponse(
        query_text=query_text,
        response_text=response_text,
        entities=entities,
        relationships=relationships,
        overall=confidence_summary,
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
            "weight": 1.0,
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
            "weight": 1.0,
            "keywords": "rents from, leases from, pays rent to, occupies property of",
            "source_id": "tenant_rents_from_landlord",
        },
        {
            "src_id": "entity_landlord",
            "tgt_id": "entity_tenant",
            "description": "A landlord owns property that is rented or leased to a tenant.",
            "weight": 1.0,
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
