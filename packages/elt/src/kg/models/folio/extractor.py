import json
from typing import List, Dict, Set, Optional
from folio import FOLIO


class FOLIOTypeExtractor:
    def __init__(self):
        # Initialize FOLIO instance
        self.folio = FOLIO(source_type="github")

        # Build type indices for quick lookup
        self.type_indices = self._build_type_indices()

    def _build_type_indices(self) -> Dict[str, Set[str]]:
        """Build indices of all FOLIO types and their variations"""
        indices = {}

        # Get all FOLIO branches
        folio_branches = self.folio.get_folio_branches()

        for branch_name, classes in folio_branches.items():
            indices[branch_name] = set()
            for cls in classes:
                # Add main label
                if cls.label:
                    indices[branch_name].add(cls.label.lower())
                # Add alternative labels
                indices[branch_name].update(
                    alt.lower() for alt in cls.alternative_labels
                )
                # Add hidden labels
                if cls.hidden_label:
                    indices[branch_name].add(cls.hidden_label.lower())

        return indices

    def extract_types_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract FOLIO types from text using LLM
        """
        # Prepare prompt for LLM
        prompt = f"""
        Analyze the following text and identify all FOLIO types present.
        For each identified type, specify which FOLIO branch it belongs to.
        
        Available FOLIO branches:
        {', '.join(self.type_indices.keys())}
        
        Text to analyze:
        {text}
        
        Return the results in JSON format with the following structure:
        {{
            "branch_name": ["identified_type1", "identified_type2", ...],
            ...
        }}
        """

        # Call LLM (implementation depends on your LLM provider)
        llm_response = self._call_llm(prompt)

        # Parse and validate results
        results = self._validate_llm_results(llm_response)

        return results

    def _validate_llm_results(self, llm_response: str) -> Dict[str, List[str]]:
        """Validate and clean up LLM results"""
        validated_results = {}

        try:
            # Parse JSON response
            results = json.loads(llm_response)

            # Validate each branch and type
            for branch, types in results.items():
                if branch in self.type_indices:
                    # Filter and validate types
                    valid_types = [
                        t for t in types if t.lower() in self.type_indices[branch]
                    ]
                    if valid_types:
                        validated_results[branch] = valid_types

        except json.JSONDecodeError:
            # Handle invalid JSON
            return {}

        return validated_results

    def get_required_types(self) -> Dict[str, List[str]]:
        """Get list of required types for email validation"""
        return {
            "Actor / Player": self.folio.get_player_actors(),
            "Document / Artifact": self.folio.get_document_artifacts(),
            "Event": self.folio.get_events(),
        }

    def validate_email_content(self, text: str) -> Dict[str, Any]:
        """
        Validate email content against required FOLIO types
        """
        # Extract types from text
        extracted_types = self.extract_types_from_text(text)

        # Get required types
        required_types = self.get_required_types()

        # Validate against requirements
        validation_results = {
            "is_valid": True,
            "missing_types": {},
            "found_types": extracted_types,
            "suggestions": [],
        }

        # Check for required types
        for branch, types in required_types.items():
            if branch not in extracted_types:
                validation_results["is_valid"] = False
                validation_results["missing_types"][branch] = [t.label for t in types]
                validation_results["suggestions"].append(
                    f"Email should contain at least one {branch} type"
                )

        return validation_results
