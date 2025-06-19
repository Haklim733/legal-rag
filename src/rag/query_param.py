from typing import Callable, Optional, List, Literal
import os
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)


class QueryParam(BaseModel):
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        "naive", description="Specifies the retrieval mode"
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

    response_type: str = Field(
        "Multiple Paragraphs", description="Defines the response format"
    )
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
        None, description="User-provided prompt for the query"
    )
    """User-provided prompt for the query.
    If provided, this will be used instead of the default value from prompt template.
    """

    hl_keywords: Optional[List[str]] = Field(
        None, description="High-level keywords for filtering"
    )
    ll_keywords: Optional[List[str]] = Field(
        None, description="Low-level keywords for filtering"
    )
    original_query: Optional[str] = Field(None, description="Original query text")


async def simple_local_model_complete(prompt: str, **kwargs) -> str:
    logger.info(f"Local model received prompt: {prompt[:100]}...")
    # Example: return a JSON string with keywords
    result = {
        "hl_keywords": ["legal representation", "court proceedings"],
        "ll_keywords": ["lawyer", "organization", "Wiliam", "court"],
    }
    return json.dumps(result)
