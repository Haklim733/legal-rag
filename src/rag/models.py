from typing import Callable, Optional, List, Literal
import os
from pydantic import BaseModel, Field
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
    # Add relationship tracking fields
    outgoing_relationship_ids: List[str] = Field(
        default_factory=list,
        description="List of relationship IDs where this entity is the source",
    )
    incoming_relationship_ids: List[str] = Field(
        default_factory=list,
        description="List of relationship IDs where this entity is the target",
    )

    def add_outgoing_relationship(self, relationship_id: str):
        """Add an outgoing relationship ID to this entity."""
        if relationship_id not in self.outgoing_relationship_ids:
            self.outgoing_relationship_ids.append(relationship_id)

    def add_incoming_relationship(self, relationship_id: str):
        """Add an incoming relationship ID to this entity."""
        if relationship_id not in self.incoming_relationship_ids:
            self.incoming_relationship_ids.append(relationship_id)

    def get_total_relationships(self) -> int:
        """Get the total number of relationships for this entity."""
        return len(self.outgoing_relationship_ids) + len(self.incoming_relationship_ids)


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
        return f"rel_{src_id}_{relationship_type}_{tgt_id}".replace(" ", "_").lower()


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
            "keywords": "represents, legal representation",
            "source_id": "lawyer_represents_legal_services_buyer",
        },
        {
            "src_id": "entity_individual_person",
            "tgt_id": "entity_actor_player",
            "description": 'In legal and organizational contexts, "is member of" signifies the belonging or inclusion of an individual or entity in a specific group, committee, board, or organization. This denotes that the individual or entity has a role, responsibilities, rights, or a position within the collective body, subject to its rules, objectives, and governance structure.',
            "keywords": "is a member of, is a part of",
            "source_id": "individual_person_is_member_of_actor_player",
        },
    ],
}

SYSTEM_PROMPT = """You are a FOLIO ontology expert. Your task is to identify specific entities and relationships from the FOLIO legal knowledge graph that are relevant to the user's query.

RESPONSE FORMAT:
1. List ONLY entities that exist in the FOLIO knowledge graph
2. List ONLY relationships that exist in the FOLIO knowledge graph
3. For each item, explain its specific relevance to the query

EXAMPLE RESPONSE:
ENTITIES:
- Lawyer (Actor/Player): name of person, type of person
- Tenant (Actor/Player): name of person, type of person
- Eviction Notice (Document/Artifact): name of document, type of document

RELATIONSHIPS:
- Lawyer --represents--> Tenant: Legal representation relationship in eviction cases
- Tenant --receives--> Eviction Notice: Direct relationship showing tenant's receipt of eviction document

DO NOT:
- Make up entities that don't exist in FOLIO
- Use generic legal terms not in the knowledge graph
- Provide legal advice or interpretations beyond the ontology

ONLY reference actual ontological entities and relationships in the knowledge graph."""
