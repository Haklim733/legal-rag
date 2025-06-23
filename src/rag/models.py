from typing import Callable, Optional, List, Literal
import os
from pydantic import BaseModel, Field, field_validator
import json
import logging

logger = logging.getLogger(__name__)


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


SYSTEM_PROMPT = """You are an expert legal knowledge graph analyzer. Extract ONLY the core legal entities and relationships mentioned in the input text.

CRITICAL: You MUST respond with ONLY a valid JSON object. No additional text, explanations, or formatting outside the JSON structure.

{{REQUIRED JSON FORMAT}}:
{{
    "entities": [
        {{
            "entity_name": "entity_name",
            "entity_type": "entity_type",
            "description": "brief description from the text",
            "weight": 0.85,
        }}
    ],
    "relationships": [
        {{
            "src_id": "subject",
            "tgt_id": "object",
            "description": "brief description from the text",
            "keywords": "keywords from the text",
            "weight": 0.85,
        }}
    ],
}}


EXTRACTION RULES:
1. Extract ONLY entities and relationships explicitly mentioned in the input text
2. Use the exact names/terms from the text (e.g., "Lawyer" not "Agent/Role")
3. Keep descriptions brief and based on the text context
4. Do NOT add FOLIO ontology explanations or verbose descriptions
5. Do NOT include classes or properties unless explicitly mentioned in the knowledge graph
6. weight should be a measure of confidence in the match between text and the entries found in the knowledge graph
7. src_id should be the subject of the relationship and tgt_id should be the object
8. keywords should ONLY be words/phrases from the knowledge graph - DO NOT make up keywords
9. If you're unsure about a keyword, use a simple action word from the text that closely matches the knowledge graph and assign a value to weight based on how closely it matches the knowledge graph

EXAMPLES:
- Input: "A lawyer represents a client"
- Entity: "Lawyer" (not "Actor/Player")
- Entity: "Client" (not "Actor/Player") 
- Relationship: "represented"
- Description: actual text from the input


IMPORTANT RULES:
- Respond with ONLY the JSON object, no other text
- Use exact terms from the input text
- Keep descriptions concise and relevant
- Avoid FOLIO ontology jargon in descriptions
- If no concepts found, return empty arrays but maintain JSON structure
- NEVER invent keywords that don't exist in the knowledge graph"""


class RAGResponse(BaseModel):
    """Validated response structure for RAG queries containing ontological concepts."""

    query_text: str = Field(..., description="Original query text")
    response_text: str = Field(..., description="Generated response text")
    entities_found: List[Entity] = Field(..., description="Entities found in the query")
    relationships_found: List[Relationship] = Field(
        ..., description="Relationships found in the query"
    )
    confidence_summary: dict = Field(..., description="Summary of confidence scores")
    total_concepts: int = Field(
        ..., description="Total number of ontological concepts found"
    )

    @field_validator("confidence_summary")
    @classmethod
    def validate_confidence_summary(cls, v):
        """Validate confidence summary structure."""
        required_keys = [
            "entities",
            "relationships",
            "overall",
        ]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key '{key}' in confidence_summary")
            if not isinstance(v[key], (int, float)) or v[key] < 0 or v[key] > 1:
                raise ValueError(
                    f"Confidence score for '{key}' must be between 0.0 and 1.0"
                )
        return v

    @field_validator("total_concepts")
    @classmethod
    def validate_total_concepts(cls, v, values):
        """Validate that total_concepts matches the sum of all concept types."""
        if (
            "entities_found" in values.data
            and "relationships_found" in values.data
            and "classes_found" in values.data
            and "properties_found" in values.data
        ):
            calculated_total = len(values.data["entities_found"]) + len(
                values.data["relationships_found"]
            )
            if v != calculated_total:
                raise ValueError(
                    f"total_concepts ({v}) does not match sum of individual concept types ({calculated_total})"
                )
        return v


def validate_rag_response(response_text: str, query_text: str) -> RAGResponse:
    """
    Validate and parse RAG response to ensure it contains proper ontological concepts.

    Args:
        response_text: Raw response text from RAG
        query_text: Original query text

    Returns:
        RAGResponse: Validated response structure

    Raises:
        ValueError: If response doesn't contain valid JSON structure
        ValidationError: If response doesn't meet ontological concept requirements
    """
    import json
    import re

    # Try to extract JSON from response
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("Response does not contain valid JSON structure")

    try:
        response_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")

    # Extract ontological concepts from response
    entities = []
    relationships = []

    # Parse entities
    if "entities" in response_data:
        for entity_data in response_data["entities"]:
            entities.append(
                Entity(
                    entity_name=entity_data.get("entity_name", ""),
                    entity_type=entity_data.get("entity_type", "entity"),
                    description=entity_data.get("description", ""),
                    source_id=entity_data.get("source_id", ""),
                    chunk_ids=entity_data.get("chunk_ids", []),
                )
            )

    # Parse relationships
    if "relationships" in response_data:
        for rel_data in response_data["relationships"]:
            # Validate keywords against knowledge graph
            keywords = rel_data.get("keywords", "")
            if not validate_keywords_against_kg(keywords):
                logger.warning(
                    f"Invalid keywords found: {keywords}. Valid keywords: {get_valid_keywords_from_kg()}"
                )

            relationships.append(
                Relationship(
                    src_id=rel_data.get("src_id", ""),
                    tgt_id=rel_data.get("tgt_id", ""),
                    description=rel_data.get("description", ""),
                    keywords=rel_data.get("keywords", ""),
                    weight=rel_data.get("weight", 0.5),
                    source_id=rel_data.get("source_id", ""),
                )
            )

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

    total_concepts = len(entities) + len(relationships)

    return RAGResponse(
        query_text=query_text,
        response_text=response_text,
        entities_found=entities,
        relationships_found=relationships,
        confidence_summary=confidence_summary,
        total_concepts=total_concepts,
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


def validate_keywords_against_kg(keywords: str) -> bool:
    """Validate that keywords exist in the knowledge graph."""
    if not keywords:
        return True

    valid_keywords = get_valid_keywords_from_kg()
    provided_keywords = [k.strip() for k in keywords.split(", ")]

    for keyword in provided_keywords:
        if keyword not in valid_keywords:
            return False
    return True
