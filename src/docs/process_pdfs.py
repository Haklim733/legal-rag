from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Title, NarrativeText, ListItem, Table
import pypdf
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from .models import (
    SupremeCourtCase,
    PDFDocumentMetadata,
    ExtractionMetadata,
    CaseDocumentMetadata,
)

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Available PDF text extraction methods."""

    UNSTRUCTURED = "unstructured"
    PYPDF = "pypdf"
    UNSTRUCTURED_FAST = "unstructured_fast"
    UNSTRUCTURED_HIERARCHICAL = "unstructured_hierarchical"


@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""

    full_text: str
    elements: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    method: str
    pages_processed: int
    error_message: Optional[str] = None


def extract_text_unstructured(
    pdf_path: Path,
    max_pages: Optional[int] = None,
    strategy: str = "hi_res",
    preserve_hierarchy: bool = True,
    method_name: str = "unstructured",
) -> ExtractionResult:
    """
    Extract text using unstructured library with hierarchical structure preservation.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None for all pages)
        strategy: Extraction strategy ("hi_res", "fast", "ocr_only")
        preserve_hierarchy: Whether to preserve document hierarchy
        method_name: Name of the method being used

    Returns:
        ExtractionResult with structured text and metadata
    """
    try:
        # Extract elements from PDF
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=strategy,
            include_page_breaks=True,
            chunking_strategy="by_title" if preserve_hierarchy else "basic",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )

        # Process elements into hierarchical structure
        full_text = ""
        elements_data = []
        current_hierarchy_level = 0
        hierarchy_stack = []
        pages_processed = 0
        max_page_seen = 0

        for element in elements:
            element_text = str(element)

            # Track pages processed by checking element metadata
            if hasattr(element, "metadata") and hasattr(
                element.metadata, "page_number"
            ):
                current_page = element.metadata.page_number
                if current_page and current_page > max_page_seen:
                    max_page_seen = current_page
                    # Check if we've exceeded the page limit
                    if max_pages is not None and max_page_seen > max_pages:
                        # Stop processing elements from pages beyond the limit
                        break
                    pages_processed = max_page_seen

            # Determine hierarchy level based on element type
            if isinstance(element, Title):
                # Count # symbols to determine heading level
                heading_level = (
                    element_text.count("#") if element_text.startswith("#") else 1
                )
                current_hierarchy_level = heading_level

                # Update hierarchy stack
                while len(hierarchy_stack) >= heading_level:
                    hierarchy_stack.pop()
                hierarchy_stack.append(element_text.strip())

                # Add indentation based on hierarchy level
                indent = "  " * (heading_level - 1)
                full_text += f"{indent}{element_text}\n\n"
            else:
                # Regular text - maintain current hierarchy level
                indent = "  " * current_hierarchy_level
                full_text += f"{indent}{element_text}\n\n"

            # Create element metadata
            element_data = {
                "text": element_text,
                "type": element.__class__.__name__,
                "hierarchy_level": current_hierarchy_level,
                "hierarchy_path": " > ".join(hierarchy_stack),
                "metadata": {},
            }

            # Add element-specific metadata
            if hasattr(element, "metadata"):
                element_data["metadata"]["page_number"] = getattr(
                    element.metadata, "page_number", None
                )
                element_data["metadata"]["filename"] = getattr(
                    element.metadata, "filename", None
                )

            elements_data.append(element_data)

        return ExtractionResult(
            full_text=full_text.strip(),
            elements=elements_data,
            metadata={
                "method": method_name,
                "strategy": strategy,
                "preserve_hierarchy": preserve_hierarchy,
                "total_elements": len(elements_data),
            },
            success=True,
            method=method_name,
            pages_processed=pages_processed,
        )

    except Exception as e:
        logger.error(
            f"Error processing {pdf_path} with {method_name}: {str(e)}", exc_info=True
        )
        return ExtractionResult(
            full_text="",
            elements=[],
            metadata={},
            success=False,
            method=method_name,
            pages_processed=0,
            error_message=str(e),
        )


def extract_text_pypdf(
    pdf_path: Path, max_pages: Optional[int] = None, preserve_hierarchy: bool = True
) -> ExtractionResult:
    """
    Extract text using PyPDF library with basic structure preservation.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None for all pages)
        preserve_hierarchy: Whether to preserve basic document structure

    Returns:
        ExtractionResult with structured text and metadata
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            total_pages = len(pdf_reader.pages)
            pages_to_process = min(max_pages, total_pages) if max_pages else total_pages

            full_text = ""
            elements_data = []

            for page_num in range(pages_to_process):
                page = pdf_reader.pages[page_num]

                # Extract text from page
                page_text = page.extract_text()

                if preserve_hierarchy:
                    # Try to identify headings and structure
                    lines = page_text.split("\n")
                    structured_lines = []

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # Simple heuristic for headings (all caps, short lines, etc.)
                        if len(line) < 100 and (
                            line.isupper()
                            or line.endswith(":")
                            or any(word.isupper() for word in line.split()[:3])
                        ):
                            # Likely a heading
                            structured_lines.append(f"## {line}")
                            elements_data.append(
                                {
                                    "text": line,
                                    "type": "Title",
                                    "hierarchy_level": 2,
                                    "hierarchy_path": f"Page {page_num + 1} > {line}",
                                    "metadata": {"page_number": page_num + 1},
                                }
                            )
                        else:
                            # Regular text
                            structured_lines.append(f"  {line}")
                            elements_data.append(
                                {
                                    "text": line,
                                    "type": "Text",
                                    "hierarchy_level": 3,
                                    "hierarchy_path": f"Page {page_num + 1}",
                                    "metadata": {"page_number": page_num + 1},
                                }
                            )

                    page_text = "\n".join(structured_lines)

                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

            return ExtractionResult(
                full_text=full_text.strip(),
                elements=elements_data,
                metadata={
                    "method": "pypdf",
                    "total_pages": total_pages,
                    "pages_processed": pages_to_process,
                    "preserve_hierarchy": preserve_hierarchy,
                },
                success=True,
                method="pypdf",
                pages_processed=pages_to_process,
            )

    except Exception as e:
        logger.error(f"Error processing {pdf_path} with pypdf: {str(e)}", exc_info=True)
        return ExtractionResult(
            full_text="",
            elements=[],
            metadata={},
            success=False,
            method="pypdf",
            pages_processed=0,
            error_message=str(e),
        )


def extract_text_unstructured_fast(
    pdf_path: Path, max_pages: Optional[int] = None
) -> ExtractionResult:
    """
    Extract text using unstructured library with fast strategy.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None for all pages)

    Returns:
        ExtractionResult with structured text and metadata
    """
    return extract_text_unstructured(
        pdf_path=pdf_path,
        max_pages=max_pages,
        strategy="fast",
        preserve_hierarchy=False,
        method_name="unstructured_fast",
    )


def extract_text_unstructured_hierarchical(
    pdf_path: Path, max_pages: Optional[int] = None
) -> ExtractionResult:
    """
    Extract text using unstructured library with maximum hierarchy preservation.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None for all pages)

    Returns:
        ExtractionResult with structured text and metadata
    """
    return extract_text_unstructured(
        pdf_path=pdf_path,
        max_pages=max_pages,
        strategy="hi_res",
        preserve_hierarchy=True,
        method_name="unstructured_hierarchical",
    )


def extract_text_from_pdf(
    pdf_path: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED_HIERARCHICAL,
    max_pages: Optional[int] = None,
    **kwargs,
) -> ExtractionResult:
    """
    Extract text from PDF using the specified method.

    Args:
        pdf_path: Path to the PDF file
        method: Extraction method to use
        max_pages: Maximum number of pages to process (None for all pages)
        **kwargs: Additional arguments for the extraction method

    Returns:
        ExtractionResult with structured text and metadata
    """
    if isinstance(method, str):
        method = ExtractionMethod(method)

    method_map = {
        ExtractionMethod.UNSTRUCTURED: extract_text_unstructured,
        ExtractionMethod.PYPDF: extract_text_pypdf,
        ExtractionMethod.UNSTRUCTURED_FAST: extract_text_unstructured_fast,
        ExtractionMethod.UNSTRUCTURED_HIERARCHICAL: extract_text_unstructured_hierarchical,
    }

    extractor_func = method_map.get(method)
    if not extractor_func:
        raise ValueError(f"Unknown extraction method: {method}")

    return extractor_func(pdf_path, max_pages=max_pages, **kwargs)


def process_pdf_directory(
    pdf_dir: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED_HIERARCHICAL,
    max_pages: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Process all PDFs in a directory using the specified method.

    Args:
        pdf_dir: Directory containing PDF files
        method: Extraction method to use
        max_pages: Maximum number of pages to process per PDF (None for all pages)
        **kwargs: Additional arguments for the extraction method

    Returns:
        List of dictionaries containing processing results for each PDF
    """
    if not pdf_dir.is_dir():
        raise ValueError(f"{pdf_dir} is not a valid directory")

    results = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            result = extract_text_from_pdf(
                pdf_path=pdf_path, method=method, max_pages=max_pages, **kwargs
            )

            results.append(
                {
                    "file": str(pdf_path),
                    "text": result.full_text,
                    "elements": result.elements,
                    "metadata": result.metadata,
                    "success": result.success,
                    "method": result.method,
                    "pages_processed": result.pages_processed,
                    "error_message": result.error_message,
                }
            )
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}")
            results.append(
                {
                    "file": str(pdf_path),
                    "error": str(e),
                    "success": False,
                    "method": str(method),
                    "pages_processed": 0,
                }
            )

    return results


def compare_extraction_methods(
    pdf_path: Path, max_pages: Optional[int] = None
) -> Dict[str, ExtractionResult]:
    """
    Compare different extraction methods on the same PDF.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None for all pages)

    Returns:
        Dictionary mapping method names to their results
    """
    methods = [
        ExtractionMethod.UNSTRUCTURED_HIERARCHICAL,
        ExtractionMethod.UNSTRUCTURED_FAST,
        ExtractionMethod.PYPDF,
    ]

    results = {}
    for method in methods:
        try:
            result = extract_text_from_pdf(pdf_path, method=method, max_pages=max_pages)
            results[method.value] = result
        except Exception as e:
            logger.error(f"Failed to extract with {method.value}: {str(e)}")
            results[method.value] = ExtractionResult(
                full_text="",
                elements=[],
                metadata={},
                success=False,
                method=method.value,
                pages_processed=0,
                error_message=str(e),
            )

    return results


def extract_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file using PyPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing PDF metadata
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            # Get document info (metadata)
            metadata = {}

            if pdf_reader.metadata:
                # Standard PDF metadata fields
                standard_fields = [
                    "/Title",
                    "/Author",
                    "/Subject",
                    "/Keywords",
                    "/Creator",
                    "/Producer",
                    "/CreationDate",
                    "/ModDate",
                    "/Trapped",
                ]

                for field in standard_fields:
                    if field in pdf_reader.metadata:
                        value = pdf_reader.metadata[field]
                        # Remove the leading slash for cleaner keys
                        clean_key = field[1:] if field.startswith("/") else field
                        metadata[clean_key] = str(value)
            else:
                logger.warning(f"No metadata found in PDF: {pdf_path}")

            # Add document properties
            metadata["total_pages"] = len(pdf_reader.pages)
            metadata["file_size_bytes"] = pdf_path.stat().st_size
            metadata["file_path"] = str(pdf_path)
            metadata["file_name"] = pdf_path.name

            # Try to extract creation and modification dates
            if "CreationDate" in metadata:
                try:
                    # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
                    date_str = metadata["CreationDate"]
                    if date_str.startswith("D:"):
                        date_str = date_str[2:]  # Remove 'D:' prefix
                        # Parse year, month, day
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        creation_date = datetime(year, month, day)
                        metadata["creation_date_parsed"] = creation_date.isoformat()
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse CreationDate from PDF: {pdf_path}")
            else:
                logger.warning(f"No CreationDate found in PDF metadata: {pdf_path}")

            if "ModDate" in metadata:
                try:
                    date_str = metadata["ModDate"]
                    if date_str.startswith("D:"):
                        date_str = date_str[2:]
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        mod_date = datetime(year, month, day)
                        metadata["modification_date_parsed"] = mod_date.isoformat()
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse ModDate from PDF: {pdf_path}")
            else:
                logger.warning(f"No ModDate found in PDF metadata: {pdf_path}")

            # Check if PDF is encrypted
            metadata["is_encrypted"] = pdf_reader.is_encrypted

            # Get PDF version
            metadata["pdf_version"] = str(pdf_reader.pdf_header)

            # Warn if no standard metadata fields were found
            standard_metadata_count = sum(
                1
                for key in metadata.keys()
                if key
                in ["Title", "Author", "Subject", "Keywords", "Creator", "Producer"]
            )
            if standard_metadata_count == 0:
                logger.warning(
                    f"No standard metadata fields (Title, Author, Subject, etc.) found in PDF: {pdf_path}"
                )

            return {"success": True, "metadata": metadata, "file_path": str(pdf_path)}

    except Exception as e:
        logger.error(
            f"Error extracting metadata from {pdf_path}: {str(e)}", exc_info=True
        )
        return {
            "success": False,
            "error": str(e),
            "file_path": str(pdf_path),
            "metadata": {},
        }


def extract_pdf_metadata_with_unstructured(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file using unstructured library.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing PDF metadata
    """
    try:
        # Use unstructured to get document metadata
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",  # Use fast strategy for metadata extraction
            include_page_breaks=False,
            max_partition=1,  # Only need first page for metadata
        )

        metadata = {
            "total_elements": len(elements),
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
        }

        # Extract metadata from elements
        if elements:
            first_element = elements[0]
            if hasattr(first_element, "metadata"):
                element_metadata = first_element.metadata
                if hasattr(element_metadata, "filename"):
                    metadata["filename"] = element_metadata.filename
                if hasattr(element_metadata, "page_number"):
                    metadata["first_page"] = element_metadata.page_number
            else:
                logger.warning(
                    f"No element metadata found in unstructured extraction for PDF: {pdf_path}"
                )
        else:
            logger.warning(
                f"No elements extracted from PDF using unstructured: {pdf_path}"
            )

        return {"success": True, "metadata": metadata, "file_path": str(pdf_path)}

    except Exception as e:
        logger.error(
            f"Error extracting metadata with unstructured from {pdf_path}: {str(e)}",
            exc_info=True,
        )
        return {
            "success": False,
            "error": str(e),
            "file_path": str(pdf_path),
            "metadata": {},
        }


def compare_pdf_metadata_methods(pdf_path: Path) -> Dict[str, Any]:
    """
    Compare metadata extraction methods for a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with comparison results
    """
    results = {
        "file_path": str(pdf_path),
        "pypdf_metadata": extract_pdf_metadata(pdf_path),
        "unstructured_metadata": extract_pdf_metadata_with_unstructured(pdf_path),
    }

    # Add comparison summary
    pypdf_success = results["pypdf_metadata"]["success"]
    unstructured_success = results["unstructured_metadata"]["success"]

    results["summary"] = {
        "pypdf_success": pypdf_success,
        "unstructured_success": unstructured_success,
        "both_successful": pypdf_success and unstructured_success,
        "metadata_fields_pypdf": (
            len(results["pypdf_metadata"].get("metadata", {})) if pypdf_success else 0
        ),
        "metadata_fields_unstructured": (
            len(results["unstructured_metadata"].get("metadata", {}))
            if unstructured_success
            else 0
        ),
    }

    # Warn if neither method was successful
    if not pypdf_success and not unstructured_success:
        logger.warning(f"Both metadata extraction methods failed for PDF: {pdf_path}")
    elif not pypdf_success:
        logger.warning(f"PyPDF metadata extraction failed for PDF: {pdf_path}")
    elif not unstructured_success:
        logger.warning(f"Unstructured metadata extraction failed for PDF: {pdf_path}")

    # Warn if metadata fields are significantly different between methods
    pypdf_fields = results["summary"]["metadata_fields_pypdf"]
    unstructured_fields = results["summary"]["metadata_fields_unstructured"]

    if pypdf_success and unstructured_success:
        if abs(pypdf_fields - unstructured_fields) > 5:  # Significant difference
            logger.warning(
                f"Large difference in metadata fields between methods for PDF: {pdf_path} "
                f"(PyPDF: {pypdf_fields}, Unstructured: {unstructured_fields})"
            )

    return results


def create_supreme_court_case_from_pdf(
    extraction_result: ExtractionResult,
    pdf_metadata: Dict[str, Any],
    **case_kwargs,
) -> SupremeCourtCase:
    """
    Create a SupremeCourtCase instance from PDF processing results.

    Args:
        extraction_result: Result from extract_text_from_pdf()
        pdf_metadata: Result from extract_pdf_metadata()
        **case_kwargs: Additional case information

    Returns:
        SupremeCourtCase instance with all metadata populated
    """
    # Create PDF metadata object
    pdf_meta = PDFDocumentMetadata(
        title=pdf_metadata.get("metadata", {}).get("Title"),
        creator=pdf_metadata.get("metadata", {}).get("Creator"),
        producer=pdf_metadata.get("metadata", {}).get("Producer"),
        creation_date=pdf_metadata.get("metadata", {}).get("CreationDate"),
        mod_date=pdf_metadata.get("metadata", {}).get("ModDate"),
        total_pages=pdf_metadata.get("metadata", {}).get("total_pages", 0),
        file_size_bytes=pdf_metadata.get("metadata", {}).get("file_size_bytes", 0),
        file_path=pdf_metadata.get("metadata", {}).get("file_path", ""),
        file_name=pdf_metadata.get("metadata", {}).get("file_name", ""),
        creation_date_parsed=pdf_metadata.get("metadata", {}).get(
            "creation_date_parsed"
        ),
        modification_date_parsed=pdf_metadata.get("metadata", {}).get(
            "modification_date_parsed"
        ),
        is_encrypted=pdf_metadata.get("metadata", {}).get("is_encrypted", False),
        pdf_version=pdf_metadata.get("metadata", {}).get("pdf_version", ""),
    )

    # Create extraction metadata object
    extraction_meta = ExtractionMetadata(
        extraction_method=extraction_result.method,
        pages_processed=extraction_result.pages_processed,
        total_elements=len(extraction_result.elements),
        extraction_date=datetime.now().isoformat() + "Z",
        preserve_hierarchy=extraction_result.metadata.get("preserve_hierarchy", True),
        strategy=extraction_result.metadata.get("strategy", "hi_res"),
    )

    # Extract docket number and case year
    docket_number = None
    case_year = None

    if pdf_meta.title:
        # Try to extract from title like "24A1007 A. A. R. P. v. Trump (05/16/2025)"
        title_parts = pdf_meta.title.split(" ")
        if title_parts and len(title_parts[0]) >= 7:  # Likely a docket number
            docket_number = title_parts[0]
    elif pdf_meta.file_name:
        # Try to extract from filename like "24A1007_AARPvTrump_20250516.pdf"
        filename_parts = pdf_meta.file_name.split("_")
        if filename_parts:
            docket_number = filename_parts[0]

    # Extract year from filename or creation date
    if pdf_meta.file_name and "2025" in pdf_meta.file_name:
        case_year = 2025
    elif pdf_meta.creation_date_parsed:
        try:
            case_year = datetime.fromisoformat(
                pdf_meta.creation_date_parsed.replace("Z", "+00:00")
            ).year
        except:
            pass

    # Create case metadata object
    case_meta = CaseDocumentMetadata(
        docket_number=docket_number,
        case_type=case_kwargs.get("case_type"),
        jurisdiction="supreme_court",
        term=f"{case_year-1}-{case_year}" if case_year else None,
        case_year=case_year,
    )

    # Create the SupremeCourtCase instance
    return SupremeCourtCase(
        pdf_metadata=pdf_meta,
        extraction_metadata=extraction_meta,
        case_metadata=case_meta,
        **case_kwargs,
    )
