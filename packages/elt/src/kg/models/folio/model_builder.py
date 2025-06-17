"""
model_builder.py - FOLIO model builder module

This module provides static methods to parse and create FOLIO models from various formats.
"""

from typing import Dict, List, Optional, Union, Tuple
import json
import lxml.etree
from pathlib import Path
import os

from .graph import FOLIO
from .models import OWLClass, OWLObjectProperty, NSMAP


class ModelBuilder:
    """
    Static class for building FOLIO models from various formats.
    """

    @staticmethod
    def get_default_owl_path() -> Path:
        """
        Get the default path to the FOLIO.owl file.

        Returns:
            Path: Path to the FOLIO.owl file.
        """
        # Get the directory containing this file
        current_dir = Path(__file__).parent.resolve()
        return current_dir / "FOLIO.owl"

    @staticmethod
    def parse_owl_file(file_path: Optional[Union[str, Path]] = None) -> FOLIO:
        """
        Parse the FOLIO.owl file and create a FOLIO instance.

        Args:
            file_path (Optional[Union[str, Path]]): Path to the OWL file. If None, uses the default FOLIO.owl.

        Returns:
            FOLIO: A FOLIO instance containing the parsed ontology.
        """
        if file_path is None:
            file_path = ModelBuilder.get_default_owl_path()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FOLIO.owl file not found at {file_path}")

        # Create FOLIO instance with file source
        folio = FOLIO(source_type="file", file_path=str(file_path))

        # Parse the ontology
        folio.parse_owl(
            folio.load_owl(source_type="file", file_path=str(file_path), use_cache=True)
        )

        return folio

    @staticmethod
    def parse_owl_string(owl_content: str) -> FOLIO:
        """
        Parse OWL content from a string and create a FOLIO instance.

        Args:
            owl_content (str): OWL content as a string.

        Returns:
            FOLIO: A FOLIO instance containing the parsed ontology.
        """
        # Create a temporary file to store the OWL content
        temp_file = Path("temp.owl")
        try:
            temp_file.write_text(owl_content)
            return ModelBuilder.parse_owl_file(temp_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    @staticmethod
    def from_owl_file(file_path: Union[str, Path]) -> FOLIO:
        """
        Create a FOLIO instance from an OWL file.

        Args:
            file_path (Union[str, Path]): Path to the OWL file.

        Returns:
            FOLIO: A FOLIO instance containing the parsed ontology.
        """
        return ModelBuilder.parse_owl_file(file_path)

    @staticmethod
    def from_owl_string(owl_content: str) -> FOLIO:
        """
        Create a FOLIO instance from an OWL string.

        Args:
            owl_content (str): OWL content as a string.

        Returns:
            FOLIO: A FOLIO instance containing the parsed ontology.
        """
        return ModelBuilder.parse_owl_string(owl_content)

    @staticmethod
    def from_json_file(file_path: Union[str, Path]) -> FOLIO:
        """
        Create a FOLIO instance from a JSON file.

        Args:
            file_path (Union[str, Path]): Path to the JSON file.

        Returns:
            FOLIO: A FOLIO instance containing the parsed ontology.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        return ModelBuilder.from_json_string(json_content)

    @staticmethod
    def from_json_string(json_content: str) -> FOLIO:
        """
        Create a FOLIO instance from a JSON string.

        Args:
            json_content (str): JSON content as a string.

        Returns:
            FOLIO: A FOLIO instance containing the parsed ontology.
        """
        # Parse the JSON content
        data = json.loads(json_content)

        # Create a new FOLIO instance
        folio = FOLIO()

        # Parse classes
        for class_data in data.get("classes", []):
            owl_class = OWLClass.from_json(json.dumps(class_data))
            folio.classes.append(owl_class)
            folio.iri_to_index[owl_class.iri] = len(folio.classes) - 1

            # Update label indices
            if owl_class.label:
                if owl_class.label not in folio.label_to_index:
                    folio.label_to_index[owl_class.label] = []
                folio.label_to_index[owl_class.label].append(len(folio.classes) - 1)

            # Update alt label indices
            for alt_label in owl_class.alternative_labels:
                if alt_label not in folio.alt_label_to_index:
                    folio.alt_label_to_index[alt_label] = []
                folio.alt_label_to_index[alt_label].append(len(folio.classes) - 1)

        # Parse object properties
        for prop_data in data.get("object_properties", []):
            owl_prop = OWLObjectProperty.from_json(json.dumps(prop_data))
            folio.object_properties.append(owl_prop)
            folio.iri_to_property_index[owl_prop.iri] = len(folio.object_properties) - 1

            # Update property label indices
            if owl_prop.label:
                if owl_prop.label not in folio.property_label_to_index:
                    folio.property_label_to_index[owl_prop.label] = []
                folio.property_label_to_index[owl_prop.label].append(
                    len(folio.object_properties) - 1
                )

        return folio

    @staticmethod
    def to_json(folio: FOLIO) -> str:
        """
        Convert a FOLIO instance to JSON string.

        Args:
            folio (FOLIO): The FOLIO instance to convert.

        Returns:
            str: JSON string representation of the FOLIO instance.
        """
        data = {
            "classes": [cls.to_jsonld() for cls in folio.classes],
            "object_properties": [prop.to_jsonld() for prop in folio.object_properties],
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def to_owl(folio: FOLIO) -> str:
        """
        Convert a FOLIO instance to OWL string.

        Args:
            folio (FOLIO): The FOLIO instance to convert.

        Returns:
            str: OWL string representation of the FOLIO instance.
        """
        # Create the root element
        root = lxml.etree.Element(f"{{{NSMAP['rdf']}}}RDF", nsmap=NSMAP)

        # Add ontology element
        ontology = lxml.etree.Element(f"{{{NSMAP['owl']}}}Ontology", nsmap=NSMAP)
        ontology.set(f"{{{NSMAP['rdf']}}}about", "https://folio.openlegalstandard.org/")
        root.append(ontology)

        # Add classes
        for cls in folio.classes:
            root.append(cls.to_owl_element())

        # Add object properties
        for prop in folio.object_properties:
            prop_element = lxml.etree.Element(
                f"{{{NSMAP['owl']}}}ObjectProperty", nsmap=NSMAP
            )
            prop_element.set(f"{{{NSMAP['rdf']}}}about", prop.iri)

            # Add label
            if prop.label:
                label = lxml.etree.Element(f"{{{NSMAP['rdfs']}}}label", nsmap=NSMAP)
                label.text = prop.label
                prop_element.append(label)

            # Add subPropertyOf
            for sub_prop in prop.sub_property_of:
                sub_prop_elem = lxml.etree.Element(
                    f"{{{NSMAP['rdfs']}}}subPropertyOf", nsmap=NSMAP
                )
                sub_prop_elem.set(f"{{{NSMAP['rdf']}}}resource", sub_prop)
                prop_element.append(sub_prop_elem)

            # Add domain
            for domain in prop.domain:
                domain_elem = lxml.etree.Element(
                    f"{{{NSMAP['rdfs']}}}domain", nsmap=NSMAP
                )
                domain_elem.set(f"{{{NSMAP['rdf']}}}resource", domain)
                prop_element.append(domain_elem)

            # Add range
            for range_val in prop.range:
                range_elem = lxml.etree.Element(
                    f"{{{NSMAP['rdfs']}}}range", nsmap=NSMAP
                )
                range_elem.set(f"{{{NSMAP['rdf']}}}resource", range_val)
                prop_element.append(range_elem)

            root.append(prop_element)

        # Convert to string
        return lxml.etree.tostring(root, pretty_print=True, encoding="unicode")

    @staticmethod
    def get_ontology_metadata(folio: FOLIO) -> Dict[str, str]:
        """
        Get the metadata from the FOLIO ontology.

        Args:
            folio (FOLIO): The FOLIO instance.

        Returns:
            Dict[str, str]: Dictionary containing ontology metadata.
        """
        return {"title": folio.title, "description": folio.description}

    @staticmethod
    def get_class_hierarchy(folio: FOLIO) -> Dict[str, List[str]]:
        """
        Get the class hierarchy from the FOLIO ontology.

        Args:
            folio (FOLIO): The FOLIO instance.

        Returns:
            Dict[str, List[str]]: Dictionary mapping class IRIs to their parent class IRIs.
        """
        return {cls.iri: cls.sub_class_of for cls in folio.classes}

    @staticmethod
    def get_property_hierarchy(folio: FOLIO) -> Dict[str, List[str]]:
        """
        Get the property hierarchy from the FOLIO ontology.

        Args:
            folio (FOLIO): The FOLIO instance.

        Returns:
            Dict[str, List[str]]: Dictionary mapping property IRIs to their parent property IRIs.
        """
        return {prop.iri: prop.sub_property_of for prop in folio.object_properties}
