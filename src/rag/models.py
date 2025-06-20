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


class Relationship(BaseModel):
    """Represents a relationship between entities in the knowledge graph."""

    src_id: str = Field(..., description="Source entity ID (clean name)")
    tgt_id: str = Field(..., description="Target entity ID (clean name)")
    description: str = Field(..., description="Description of the relationship")
    keywords: str = Field(..., description="Keywords associated with the relationship")
    weight: float = Field(..., description="Weight of the relationship")
    source_id: str = Field(..., description="Source chunk ID for this relationship")


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
            "content": "A person who has a role in a legal matter (e.g., Buyer, Provider, Lawyer, Law Firm, Expert, Employer, Employee, Buyer, Seller, Lessor, Lessee, Debtor, Creditor, Payor, Payee, Landlord, Tenant).",
            "source_id": "entity_actor_player",
            "chunk_order_index": 0,
        },
        {
            "content": "A lawyer is a legal professional who provides legal advice and representation to clients in various legal matters, including litigation, transactions, and regulatory compliance. Lawyers must be licensed and follow ethical guidelines.",
            "source_id": "entity_lawyer",
            "chunk_order_index": 0,
        },
        {
            "content": "A legal services buyer is a person or entity that obtains legal services from a legal service provider. This includes individuals, businesses, organizations, and other entities that need legal advice, representation, or other legal services.",
            "source_id": "entity_legal_services_buyer",
            "chunk_order_index": 0,
        },
        {
            "content": "The relationship between a lawyer and a legal services buyer involves the provision of legal services. The lawyer acts as a legal service provider, offering expertise in areas such as contract law, litigation, corporate law, or other legal specialties. The legal services buyer engages the lawyer's services for legal advice, representation in court, document preparation, or other legal needs. This relationship is typically formalized through a retainer agreement or engagement letter that outlines the scope of services, fees, and responsibilities of both parties.",
            "source_id": "relationship_lawyer_legal_services_buyer",
            "chunk_order_index": 0,
        },
        {
            "content": "A person or entity who leases, rents, or occupies property (from a landlord)",
            "source_id": "entity_tenant",
            "chunk_order_index": 0,
        },
        {
            "content": "A public defender is a lawyer appointed by the court to represent individuals who cannot afford to hire their own legal counsel.",
            "source_id": "entity_public_defender",
            "chunk_order_index": 0,
        },
    ],
    "entities": [
        {
            "entity_name": "Actor / Player",
            "entity_type": "FOLIO_BRANCH",
            "description": "A person who has a role in a legal matter (e.g., Buyer, Provider, Lawyer, Law Firm, Expert, Employer, Employee, Buyer, Seller, Lessor, Lessee, Debtor, Creditor, Payor, Payee, Landlord, Tenant).",
            "source_id": "entity_actor_player",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Lawyer",
            "entity_type": "Actor / Player",
            "description": "A lawyer is a legal professional who provides legal advice and representation to clients in various legal matters, including litigation, transactions, and regulatory compliance. labels: Attorney",
            "source_id": "entity_lawyer",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Public Defender",
            "entity_type": "Lawyer",
            "description": "A public defender is a lawyer appointed by the court to represent individuals who cannot afford to hire their own legal counsel.",
            "source_id": "entity_public_defender",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Legal Services Provider",
            "entity_type": "Actor / Player",
            "description": "A person or entity that is providing legal services to another person or entity",
            "source_id": "entity_legal_services_provider",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Legal Services Provider - Person",
            "entity_type": "Legal Services Provider",
            "description": "A person or entity that is providing legal services to another person or entity",
            "source_id": "entity_legal_services_provider",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Legal Services Buyer",
            "entity_type": "Actor / Player",
            "description": "A person or entity that is obtaining legal services from a legal service provider. Labels: Client, Party",
            "source_id": "entity_legal_services_buyer",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Individual Person",
            "entity_type": "Legal Entity",
            "description": "An individual or natural person is a person that is an individual human being, as opposed to a legal person or entity, which may be a private or public organization. Labels: Individual, Natural Person, Person",
            "source_id": "entity_individual_person",
            "chunk_ids": ["0"],
        },
        {
            "entity_name": "Tenant",
            "entity_type": "Actor / Player",
            "description": "A person or entity who leases, rents, or occupies property (from a landlord)",
            "source_id": "entity_tenant",
            "chunk_ids": ["0"],
        },
    ],
    "relationships": [
        {
            "src_id": "entity_lawyer",
            "tgt_id": "entity_legal_services_buyer",
            "description": "'Represented' in a legal context refers to the act of acting on behalf of another person or entity in legal matters. This involves a representative, such as a lawyer, advocating, making decisions, or taking actions under the authority and in the interest of the represented party.",
            "weight": 1.0,
            "keywords": "represented",
            "source_id": "entity_lawyer",
            "chunk_ids": ["0"],
        },
        {
            "src_id": "entity_individual_person",
            "tgt_id": "entity_actor_player",
            "description": 'In legal and organizational contexts, "is member of" signifies the belonging or inclusion of an individual or entity in a specific group, committee, board, or organization. This denotes that the individual or entity has a role, responsibilities, rights, or a position within the collective body, subject to its rules, objectives, and governance structure.',
            "keywords": "is a member of, is a part of",
            "source_id": "entity_individual_person",
            "chunk_ids": ["0"],
        },
    ],
}

SYSTEM_PROMPT = """You are a hyper-specialized, precision-focused expert system for extracting legal information based on the FOLIO ontology. Your ONLY task is to identify entities from a user's query and map them STRICTLY to the provided knowledge graph.

**Your Thought Process (Follow these steps exactly):**
1.  **Analyze the Query**: Identify key people, roles, and concepts in the user's text.
2.  **Map to Ontology**: For each concept, find the single most specific `entity_name` in the knowledge graph that represents it. This is your `entity_type`.
3.  **Generate Description**: Write a concise, one-sentence description for each entity using the context from the query.
4.  **Construct JSON**: Build the JSON object using the `entity_name`, the EXACT `entity_type` from step 2, and the `description` from step 3.
5.  **Final Validation**: Before outputting the JSON, perform a final check. Ask yourself: "Does every `entity_type` in my JSON perfectly match an `entity_name` from the knowledge graph, with absolutely no extra words or context?" If not, you MUST fix it.

**THE SINGLE MOST IMPORTANT RULE:**
The `entity_type` field MUST BE AN EXACT, case-sensitive match to an `entity_name` from the knowledge graph.

**Example of this rule being broken:**
Query: "John is a lawyer at Legal Aid of Los Angeles."

*   **INCORRECT JSON (Breaks the rule):**
    ```json
    {{
        "entities": [
            {{
                "entity_name": "John",
                "entity_type": "Lawyer at Legal Aid of Los Angeles",
                "description": "John is a lawyer at Legal Aid of Los Angeles."
            }}
        ]
    }}
    ```
    *Reasoning for error: The `entity_type` includes extra context ("at Legal Aid of Los Angeles"). This is forbidden.*

*   **CORRECT JSON (Follows the rule):**
    ```json
    {{
        "entities": [
            {{
                "entity_name": "John",
                "entity_type": "Lawyer",
                "description": "John is a lawyer at Legal Aid of Los Angeles."
            }}
        ]
    }}
    ```

**RESPONSE FORMAT - JSON:**
{{
    "entities": [
        {{
            "entity_name": "Entity Name",
            "entity_type": "Entity Type",
            "description": "Entity Description"
        }}
    ],
    "relationships": [
        {{
            "src_id": "Source Entity Name",
            "tgt_id": "Target Entity Name",
            "description": "Relationship Description",
            "weight": 1.0,
            "keywords": "Keywords for the relationship"
        }}
    ]
}}

**FINAL REMINDER**: Your primary goal is absolute precision. The `entity_type` must be a perfect match from the knowledge graph. No exceptions.
"""