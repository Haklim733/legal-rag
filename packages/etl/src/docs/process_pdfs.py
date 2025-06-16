from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Title, NarrativeText, ListItem, Table
import logging

def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract structured text and elements from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        tuple: (full_text, elements) where elements is a list of dictionaries
               containing element text and metadata
    """
    try:
        # Extract elements from PDF
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",  # Best for complex legal documents
            include_page_breaks=True,
            chunking_strategy="by_title",  # Group content by headings
            max_characters=4000,  # Maximum characters per chunk
            new_after_n_chars=3800,  # Start new chunk after this many chars
            combine_text_under_n_chars=2000,  # Combine small chunks
        )
        
        # Process elements into a structured format
        full_text = ""
        elements_data = []
        
        for element in elements:
            # Add to full text with double newlines between elements
            element_text = str(element)
            full_text += f"{element_text}\n\n"
            
            # Create element metadata
            element_data = {
                "text": element_text,
                "type": element.__class__.__name__,
                "metadata": {}
            }
            
            # Add element-specific metadata
            if hasattr(element, 'metadata'):
                element_data["metadata"]["page_number"] = getattr(element.metadata, 'page_number', None)
                element_data["metadata"]["filename"] = getattr(element.metadata, 'filename', None)
            
            elements_data.append(element_data)
        
        return full_text.strip(), elements_data
        
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
        return "", []

def process_pdf_directory(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of dictionaries containing processing results for each PDF
    """
    if not pdf_dir.is_dir():
        raise ValueError(f"{pdf_dir} is not a valid directory")
    
    results = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            text, elements = extract_text_from_pdf(pdf_path)
            results.append({
                "file": str(pdf_path),
                "text": text,
                "elements": elements,
                "success": True
            })
        except Exception as e:
            logging.error(f"Failed to process {pdf_path}: {str(e)}")
            results.append({
                "file": str(pdf_path),
                "error": str(e),
                "success": False
            })
    
    return results
