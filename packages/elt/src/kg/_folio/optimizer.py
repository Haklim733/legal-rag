import asyncio
from functools import lru_cache
import hashlib
import json
import os
from typing import Dict
from kg.models._folio.explore import FOLIOExplorer


class FOLIOCache:
    def __init__(self):
        self.concept_cache = {}
        self.relationship_cache = {}

    def get_cached_concept(self, concept_id):
        return self.concept_cache.get(concept_id)

    def cache_concept(self, concept_id, concept_data):
        self.concept_cache[concept_id] = concept_data


def extract_ontology_concepts(folio_explorer: FOLIOExplorer, email_text: str) -> Dict:
    """
    Extract FOLIO ontology concepts from email text using LLM.

    Args:
        email_text (str): The email text to analyze

    Returns:
        Dict containing matched FOLIO concepts and their relationships
    """
    # Get the complete FOLIO ontology structure
    ontology_structure = folio_explorer.get_results_structure()

    # Create prompt for LLM to analyze email against ontology
    prompt = f"""
    Analyze this email text against the FOLIO ontology structure:
    
    Email:
    {email_text}
    
    FOLIO Ontology Structure:
    - Classes: {ontology_structure['taxonomy']['classes']}
    - Properties: {ontology_structure['taxonomy']['properties']}
    - Connections: {ontology_structure['connections']}
    - Graph: {ontology_structure['graph']}
    
    Identify:
    1. Which FOLIO classes are mentioned or implied in the email
    2. Which FOLIO properties/relationships exist between these classes
    3. The hierarchical relationships between identified classes
    
    Return the matches in a structured format that maps to the FOLIO ontology.
    """

    # Here you would:
    # 1. Send prompt to LLM
    # 2. Get LLM response
    # 3. Return structured matches

    return {
        "email_text": email_text,
        "matched_concepts": {
            "classes": [],  # LLM identified FOLIO classes
            "properties": [],  # LLM identified FOLIO properties
            "hierarchy": {},  # LLM identified hierarchical relationships
        },
    }


def optimize_folio_data(folio_data):
    """Optimize FOLIO data for LLM processing"""
    optimized = {"entities": {}, "relationships": {}, "type_hierarchy": {}}

    # Group entities by type
    for entity in folio_data["entities"]:
        entity_type = entity.get("type", "unknown")
        if entity_type not in optimized["entities"]:
            optimized["entities"][entity_type] = []
        optimized["entities"][entity_type].append(
            {
                "id": entity["id"],
                "label": entity["label"],
                "definition": entity.get("definition", ""),
                "alt_labels": entity.get("alt_labels", []),
            }
        )

    # Group relationships by type
    for rel in folio_data["relationships"]:
        rel_type = rel.get("type", "unknown")
        if rel_type not in optimized["relationships"]:
            optimized["relationships"][rel_type] = []
        optimized["relationships"][rel_type].append(
            {
                "subject": rel["subject"],
                "predicate": rel["predicate"],
                "object": rel["object"],
            }
        )

    return optimized


def chunk_folio_data(optimized_data, chunk_size=1000):
    """Split FOLIO data into manageable chunks"""
    chunks = []
    current_chunk = {"entities": {}, "relationships": {}, "type_hierarchy": {}}

    # Chunk entities by type
    for entity_type, entities in optimized_data["entities"].items():
        for i in range(0, len(entities), chunk_size):
            chunk = entities[i : i + chunk_size]
            if entity_type not in current_chunk["entities"]:
                current_chunk["entities"][entity_type] = []
            current_chunk["entities"][entity_type].extend(chunk)

            if len(str(current_chunk)) > 100000:  # Approximate token limit
                chunks.append(current_chunk)
                current_chunk = {
                    "entities": {},
                    "relationships": {},
                    "type_hierarchy": {},
                }

    # Add remaining data
    if current_chunk["entities"]:
        chunks.append(current_chunk)

    return chunks


@lru_cache(maxsize=1000)
def get_concept_hash(concept_data):
    """Create a hash for caching concept lookups"""
    return hashlib.md5(json.dumps(concept_data, sort_keys=True).encode()).hexdigest()


def find_concept_in_data(concept_id, folio_data):
    """Find a concept in the original FOLIO data"""
    # Search in entities
    for entity in folio_data["entities"]:
        if entity["id"] == concept_id:
            return entity

    # Search in relationships
    for rel in folio_data["relationships"]:
        if rel["subject"] == concept_id or rel["object"] == concept_id:
            return rel

    return None


async def optimized_folio_search(query, folio_data, llm):
    """Optimized FOLIO search with caching and chunking"""
    # Initialize cache
    cache = FOLIOCache()

    # Optimize and chunk the data
    optimized_data = optimize_folio_data(folio_data)
    chunks = chunk_folio_data(optimized_data)

    # Search through chunks
    results = await search_folio_chunks(chunks, query, llm)

    # Post-process results
    processed_results = []
    for result in results:
        concept_id = result["concept_id"]

        # Check cache first
        cached_concept = cache.get_cached_concept(concept_id)
        if cached_concept:
            processed_results.append(
                {
                    "concept": cached_concept,
                    "relevance_score": result["relevance_score"],
                    "context": result["context"],
                }
            )
            continue

        # If not in cache, find in original data
        concept = find_concept_in_data(concept_id, folio_data)
        if concept:
            cache.cache_concept(concept_id, concept)
            processed_results.append(
                {
                    "concept": concept,
                    "relevance_score": result["relevance_score"],
                    "context": result["context"],
                }
            )

    return processed_results


async def search_folio_chunks(folio_chunks, query, llm):
    """Search through FOLIO chunks efficiently"""
    results = []

    for chunk in folio_chunks:
        # Create a focused prompt for the chunk
        prompt = f"""
        Analyze the following text for relevant FOLIO concepts:
        
        Document:
        {query}
        
        Available FOLIO concepts in this chunk:
        {json.dumps(chunk, indent=2)}
        
        Return a JSON array of matches with:
        - concept_id: The FOLIO concept ID
        - relevance_score: 1-10
        - context: Where it appears in the text
        - concept_type: The type of concept (entity/relationship)
        """

        try:
            response = await llm.json_async(prompt)
            if response and response.data:
                results.extend(response.data)
        except Exception as e:
            if "429" in str(e):
                await asyncio.sleep(5)  # Rate limit handling
                continue
            raise

    return results
