from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Title, NarrativeText, ListItem, Table
import pypdf
import logging
from datetime import datetime
from src.docs.models import (
    ExtractionMethod,
    UnstructuredStrategy,
    ExtractionResult,
    PDFExtractionResult,
)

logger = logging.getLogger(__name__)


def extract_text_unstructured(
    pdf_path: Path,
    max_pages: Optional[int] = None,
    strategy: str = "hi_res",
    preserve_hierarchy: bool = True,
    method_name: str = "unstructured",
) -> ExtractionResult:
    """
    Extract text using unstructured library with configurable strategy and hierarchy preservation.

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


def extract_text_from_pdf(
    pdf_path: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED,
    max_pages: Optional[int] = None,
    strategy: Optional[UnstructuredStrategy] = UnstructuredStrategy.FAST,
    preserve_hierarchy: bool = True,
    **kwargs,
) -> ExtractionResult:
    """
    Extract text from PDF using the specified method with configurable parameters.

    Args:
        pdf_path: Path to the PDF file
        method: Extraction method to use (default: unstructured)
        max_pages: Maximum number of pages to process (None for all pages)
        strategy: Strategy for unstructured extraction ("hi_res", "fast", "ocr_only")
        preserve_hierarchy: Whether to preserve document hierarchy
        **kwargs: Additional arguments for the extraction method

    Returns:
        ExtractionResult with structured text and metadata
    """
    if isinstance(method, str):
        method = ExtractionMethod(method)

    # Convert strategy string to UnstructuredStrategy enum if needed
    if method == ExtractionMethod.UNSTRUCTURED and isinstance(strategy, str):
        try:
            strategy = UnstructuredStrategy(strategy)
        except ValueError:
            # Default to fast if invalid strategy provided
            strategy = UnstructuredStrategy.FAST

    if method == ExtractionMethod.UNSTRUCTURED:
        if strategy is None:
            strategy = UnstructuredStrategy.FAST
        return extract_text_unstructured(
            pdf_path=pdf_path,
            max_pages=max_pages,
            strategy=strategy.value,
            preserve_hierarchy=preserve_hierarchy,
            method_name=(
                f"unstructured_{strategy.value}"
                if strategy.value != "hi_res"
                else "unstructured"
            )
            + ("_no_hierarchy" if not preserve_hierarchy else ""),
        )
    elif method == ExtractionMethod.PYPDF:
        return extract_text_pypdf(
            pdf_path=pdf_path,
            max_pages=max_pages,
            preserve_hierarchy=preserve_hierarchy,
        )
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def extract_pdf(
    pdf_path: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED,
    max_pages: Optional[int] = None,
    strategy: str = "fast",
    preserve_hierarchy: bool = True,
    extract_metadata: bool = True,
    **kwargs,
) -> PDFExtractionResult:
    """
    Extract text and metadata from PDF using the specified method.

    This is a unified function that handles both text and metadata extraction
    using the same method selection approach.

    Args:
        pdf_path: Path to the PDF file
        method: Extraction method to use (default: unstructured)
        max_pages: Maximum number of pages to process (None for all pages)
        strategy: Strategy for unstructured extraction ("hi_res", "fast", "ocr_only")
        preserve_hierarchy: Whether to preserve document hierarchy
        extract_metadata: Whether to extract metadata (default: True)
        **kwargs: Additional arguments for the extraction method

    Returns:
        PDFExtractionResult with both text and metadata results
    """
    if isinstance(method, str):
        method = ExtractionMethod(method)

    # Convert strategy string to UnstructuredStrategy enum
    strategy_enum = None
    if method == ExtractionMethod.UNSTRUCTURED:
        try:
            strategy_enum = UnstructuredStrategy(strategy)
        except ValueError:
            # Default to fast if invalid strategy provided
            strategy_enum = UnstructuredStrategy.FAST

    try:
        # Extract text
        text_result = extract_text_from_pdf(
            pdf_path=pdf_path,
            method=method,
            max_pages=max_pages,
            strategy=strategy_enum if strategy_enum else UnstructuredStrategy.FAST,
            preserve_hierarchy=preserve_hierarchy,
            **kwargs,
        )

        # Extract metadata if requested
        metadata_result = {}
        if extract_metadata:
            metadata_result = extract_pdf_metadata(
                pdf_path=pdf_path,
                method=method,
                strategy=strategy,
            )

        # Determine overall success
        success = text_result.success and (
            not extract_metadata or metadata_result.get("success", False)
        )

        # Use the method name from the text result since it already reflects the configuration
        method_name = text_result.method

        return PDFExtractionResult(
            text_result=text_result,
            metadata_result=metadata_result,
            success=success,
            method=method_name,
        )

    except Exception as e:
        logger.error(
            f"Error in unified PDF extraction for {pdf_path}: {str(e)}", exc_info=True
        )
        return PDFExtractionResult(
            text_result=ExtractionResult(
                full_text="",
                elements=[],
                metadata={},
                success=False,
                method=str(method),
                pages_processed=0,
                error_message=str(e),
            ),
            metadata_result={"success": False, "error": str(e)},
            success=False,
            method=str(method),
            error_message=str(e),
        )


def process_pdf_directory_unified(
    pdf_dir: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED,
    max_pages: Optional[int] = None,
    strategy: str = "fast",
    preserve_hierarchy: bool = True,
    extract_metadata: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Process all PDFs in a directory using the unified extraction method.

    Args:
        pdf_dir: Directory containing PDF files
        method: Extraction method to use
        max_pages: Maximum number of pages to process per PDF (None for all pages)
        strategy: Strategy for unstructured extraction ("hi_res", "fast", "ocr_only")
        preserve_hierarchy: Whether to preserve document hierarchy
        extract_metadata: Whether to extract metadata (default: True)
        **kwargs: Additional arguments for the extraction method

    Returns:
        List of dictionaries containing processing results for each PDF
    """
    if not pdf_dir.is_dir():
        raise ValueError(f"{pdf_dir} is not a valid directory")

    results = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            result = extract_pdf(
                pdf_path=pdf_path,
                method=method,
                max_pages=max_pages,
                strategy=strategy,
                preserve_hierarchy=preserve_hierarchy,
                extract_metadata=extract_metadata,
                **kwargs,
            )

            results.append(
                {
                    "file": str(pdf_path),
                    "text": result.text_result.full_text,
                    "elements": result.text_result.elements,
                    "text_metadata": result.text_result.metadata,
                    "pdf_metadata": result.metadata_result,
                    "success": result.success,
                    "method": result.method,
                    "pages_processed": result.text_result.pages_processed,
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


def process_pdf_directory(
    pdf_dir: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED,
    max_pages: Optional[int] = None,
    strategy: str = "fast",
    preserve_hierarchy: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Process all PDFs in a directory using the specified method and parameters.
    (Legacy function - consider using process_pdf_directory_unified for better results)

    Args:
        pdf_dir: Directory containing PDF files
        method: Extraction method to use
        max_pages: Maximum number of pages to process per PDF (None for all pages)
        strategy: Strategy for unstructured extraction ("hi_res", "fast", "ocr_only")
        preserve_hierarchy: Whether to preserve document hierarchy
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
                pdf_path=pdf_path,
                method=method,
                max_pages=max_pages,
                strategy=strategy,
                preserve_hierarchy=preserve_hierarchy,
                **kwargs,
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


def extract_pdf_metadata(
    pdf_path: Path,
    method: Union[ExtractionMethod, str] = ExtractionMethod.UNSTRUCTURED,
    strategy: str = "fast",
) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file using the specified method.

    Args:
        pdf_path: Path to the PDF file
        method: Extraction method to use (default: unstructured)
        strategy: Strategy for unstructured extraction ("hi_res", "fast", "ocr_only")

    Returns:
        Dictionary containing PDF metadata
    """
    if isinstance(method, str):
        method = ExtractionMethod(method)

    if method == ExtractionMethod.UNSTRUCTURED:
        return extract_pdf_metadata_unstructured(pdf_path, strategy)
    elif method == ExtractionMethod.PYPDF:
        return extract_pdf_metadata_pypdf(pdf_path)
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def extract_pdf_metadata_unstructured(
    pdf_path: Path, strategy: str = "fast"
) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file using unstructured library.

    Args:
        pdf_path: Path to the PDF file
        strategy: Strategy for unstructured extraction ("hi_res", "fast", "ocr_only")

    Returns:
        Dictionary containing PDF metadata
    """
    try:
        # Use unstructured to get document metadata
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=strategy,
            include_page_breaks=False,
            max_partition=1,  # Only need first page for metadata
        )

        metadata = {
            "total_elements": len(elements),
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
            "extraction_method": "unstructured",
            "extraction_strategy": strategy,
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

                # Extract additional metadata if available
                for attr in dir(element_metadata):
                    if not attr.startswith("_") and not callable(
                        getattr(element_metadata, attr)
                    ):
                        value = getattr(element_metadata, attr)
                        if value is not None:
                            metadata[f"element_{attr}"] = str(value)
            else:
                logger.warning(
                    f"No element metadata found in unstructured extraction for PDF: {pdf_path}"
                )
        else:
            logger.warning(
                f"No elements extracted from PDF using unstructured: {pdf_path}"
            )

        # Try to get additional metadata using PyPDF as fallback for standard PDF metadata
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                # Add basic PDF properties
                metadata["total_pages"] = len(pdf_reader.pages)
                metadata["is_encrypted"] = pdf_reader.is_encrypted
                metadata["pdf_version"] = str(pdf_reader.pdf_header)

                # Extract standard PDF metadata if available
                if pdf_reader.metadata:
                    standard_fields = [
                        "/Title",
                        "/Author",
                        "/Subject",
                        "/Keywords",
                        "/Creator",
                        "/Producer",
                        "/CreationDate",
                        "/ModDate",
                    ]

                    for field in standard_fields:
                        if field in pdf_reader.metadata:
                            value = pdf_reader.metadata[field]
                            clean_key = field[1:] if field.startswith("/") else field
                            metadata[f"pdf_{clean_key.lower()}"] = str(value)

                    # Parse dates if available
                    if "pdf_creationdate" in metadata:
                        try:
                            date_str = metadata["pdf_creationdate"]
                            if date_str.startswith("D:"):
                                date_str = date_str[2:]
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                creation_date = datetime(year, month, day)
                                metadata["creation_date_parsed"] = (
                                    creation_date.isoformat()
                                )
                        except (ValueError, IndexError):
                            pass

                    if "pdf_moddate" in metadata:
                        try:
                            date_str = metadata["pdf_moddate"]
                            if date_str.startswith("D:"):
                                date_str = date_str[2:]
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                mod_date = datetime(year, month, day)
                                metadata["modification_date_parsed"] = (
                                    mod_date.isoformat()
                                )
                        except (ValueError, IndexError):
                            pass

        except Exception as e:
            logger.warning(f"Could not extract PyPDF metadata as fallback: {str(e)}")

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


def extract_pdf_metadata_pypdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file using PyPDF (legacy method).

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing PDF metadata
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            # Get document info (metadata)
            metadata = {
                "extraction_method": "pypdf",
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "file_size_bytes": pdf_path.stat().st_size,
                "total_pages": len(pdf_reader.pages),
                "is_encrypted": pdf_reader.is_encrypted,
                "pdf_version": str(pdf_reader.pdf_header),
            }

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
                        metadata[clean_key.lower()] = str(value)
            else:
                logger.warning(f"No metadata found in PDF: {pdf_path}")

            # Try to extract creation and modification dates
            if "creationdate" in metadata:
                try:
                    # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
                    date_str = metadata["creationdate"]
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

            if "moddate" in metadata:
                try:
                    date_str = metadata["moddate"]
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

            # Warn if no standard metadata fields were found
            standard_metadata_count = sum(
                1
                for key in metadata.keys()
                if key
                in ["title", "author", "subject", "keywords", "creator", "producer"]
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
