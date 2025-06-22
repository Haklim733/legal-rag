"""
Data models for the knowledge graph.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ExtractionMethod(Enum):
    """Available PDF text extraction methods."""

    UNSTRUCTURED = "unstructured"
    PYPDF = "pypdf"


class UnstructuredStrategy(Enum):
    """Available PDF text extraction methods."""

    FAST = "fast"
    HI_RES = "hi_res"
    OCR_ONLY = "ocr_only"


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


@dataclass
class PDFExtractionResult:
    """Result of PDF extraction (text + metadata)."""

    text_result: ExtractionResult
    metadata_result: Dict[str, Any]
    success: bool
    method: str
    error_message: Optional[str] = None


class PDFDocumentMetadata(BaseModel):
    """PDF document metadata extracted from the PDF file."""

    title: Optional[str] = Field(None, description="PDF title")
    creator: Optional[str] = Field(None, description="Software that created the PDF")
    producer: Optional[str] = Field(None, description="Software that produced the PDF")
    creation_date: Optional[str] = Field(
        None, description="Original PDF creation date (D:YYYYMMDDHHmmSSOHH'mm' format)"
    )
    mod_date: Optional[str] = Field(
        None,
        description="Original PDF modification date (D:YYYYMMDDHHmmSSOHH'mm' format)",
    )
    total_pages: int = Field(..., description="Total number of pages in the PDF", gt=0)
    file_size_bytes: int = Field(..., description="File size in bytes", ge=0)
    file_path: str = Field(..., description="Full path to the PDF file")
    file_name: str = Field(..., description="Name of the PDF file")
    creation_date_parsed: Optional[str] = Field(
        None, description="Parsed creation date in ISO format"
    )
    modification_date_parsed: Optional[str] = Field(
        None, description="Parsed modification date in ISO format"
    )
    is_encrypted: bool = Field(..., description="Whether the PDF is encrypted")
    pdf_version: str = Field(..., description="PDF version string")

    @field_validator("total_pages")
    @classmethod
    def validate_total_pages(cls, v):
        if v <= 0:
            raise ValueError("total_pages must be greater than 0")
        return v

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(cls, v):
        if v < 0:
            raise ValueError("file_size_bytes must be non-negative")
        return v


class ExtractionMetadata(BaseModel):
    """Metadata about the text extraction process."""

    extraction_method: str = Field(
        ..., description="Method used for extraction (e.g., unstructured_hierarchical)"
    )
    pages_processed: int = Field(..., description="Number of pages actually processed")
    total_elements: int = Field(..., description="Total number of elements extracted")
    extraction_date: str = Field(
        ..., description="Date and time of extraction in ISO format"
    )
    preserve_hierarchy: bool = Field(
        True, description="Whether document hierarchy was preserved"
    )
    strategy: str = Field("hi_res", description="Extraction strategy used")


class CaseDocumentMetadata(BaseModel):
    """Metadata specific to the case document."""

    docket_number: Optional[str] = Field(
        None, description="Docket number extracted from title or filename"
    )
    case_type: Optional[str] = Field(
        None, description="Type of case (per_curiam, majority, etc.)"
    )
    jurisdiction: str = Field("supreme_court", description="Court jurisdiction")
    term: Optional[str] = Field(None, description="Court term (e.g., 2024-2025)")
    case_year: Optional[int] = Field(None, description="Year of the case")


class SupremeCourtCase(BaseModel):
    """Represents a U.S. Supreme Court case in the knowledge graph.

    Attributes:
        dcid: Unique identifier for the case
        name: Name of the case (e.g., "Roe v. Wade")
        description: Brief description of the case
        date_decided: Date the case was decided (YYYY-MM-DD format)
        citation: Official citation (e.g., "410 U.S. 113")
        parties: Parties involved in the case
        decision_direction: Direction of the decision (e.g., "affirmed", "reversed")
        opinion_author: Name of the justice who wrote the majority opinion
        cites: List of case DCIDs that this case cites
        pdf_metadata: PDF document metadata
        extraction_metadata: Text extraction metadata
        case_metadata: Case-specific metadata
    """

    dcid: str = Field(..., description="Unique identifier for the case")
    name: str = Field(..., description="Name of the case")
    description: str = Field(..., description="Brief description of the case")
    date_decided: str = Field(..., description="Date decided in YYYY-MM-DD format")
    citation: Optional[str] = Field(None, description="Official case citation")
    parties: Optional[str] = Field(None, description="Parties involved in the case")
    decision_direction: Optional[str] = Field(
        None, description="Direction of the decision"
    )
    opinion_author: Optional[str] = Field(
        None, description="Author of the majority opinion"
    )
    cites: List[str] = Field(
        default_factory=list, description="List of cited case DCIDs"
    )
    pdf_metadata: Optional[PDFDocumentMetadata] = Field(
        None, description="PDF document metadata"
    )
    extraction_metadata: Optional[ExtractionMetadata] = Field(
        None, description="Text extraction metadata"
    )
    case_metadata: Optional[CaseDocumentMetadata] = Field(
        None, description="Case-specific metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dcid": "dcid:case_24A1007",
                "name": "A. A. R. P., ET AL. v. DONALD J. TRUMP, PRESIDENT OF THE UNITED STATES, ET AL.",
                "description": "Case regarding Alien Enemies Act and removal of Venezuelan nationals who are members of Tren de Aragua",
                "date_decided": "2025-05-16",
                "citation": "24A1007",
                "parties": "A. A. R. P., ET AL. v. DONALD J. TRUMP, PRESIDENT OF THE UNITED STATES, ET AL.",
                "decision_direction": "per_curiam",
                "opinion_author": "Per Curiam",
                "cites": [],
                "pdf_metadata": {
                    "title": "24A1007 A. A. R. P. v. Trump (05/16/2025)",
                    "creator": "PScript5.dll Version 5.2.2",
                    "producer": "Acrobat Distiller 24.0 (Windows)",
                    "creation_date": "D:20250516150044-04'00'",
                    "mod_date": "D:20250516150157-04'00'",
                    "total_pages": 24,
                    "file_size_bytes": 151001,
                    "file_path": "/home/llee/projects/cases-rag/assets/supreme-court/2025/24A1007_AARPvTrump_20250516.pdf",
                    "file_name": "24A1007_AARPvTrump_20250516.pdf",
                    "creation_date_parsed": "2025-05-16T00:00:00",
                    "modification_date_parsed": "2025-05-16T00:00:00",
                    "is_encrypted": False,
                    "pdf_version": "%PDF-1.6",
                },
                "extraction_metadata": {
                    "extraction_method": "unstructured_hierarchical",
                    "pages_processed": 5,
                    "total_elements": 150,
                    "extraction_date": "2025-01-20T10:30:00Z",
                    "preserve_hierarchy": True,
                    "strategy": "hi_res",
                },
                "case_metadata": {
                    "docket_number": "24A1007",
                    "case_type": "per_curiam",
                    "jurisdiction": "supreme_court",
                    "term": "2024-2025",
                    "case_year": 2025,
                },
            }
        }

    @classmethod
    def from_pdf_metadata(
        cls, pdf_metadata: Dict[str, Any], **kwargs
    ) -> "SupremeCourtCase":
        """
        Create a SupremeCourtCase instance from PDF metadata.

        Args:
            pdf_metadata: Dictionary containing PDF metadata from extract_pdf_metadata()
            **kwargs: Additional case information

        Returns:
            SupremeCourtCase instance with populated metadata
        """
        # Extract docket number from title or filename
        docket_number = None
        if pdf_metadata.get("Title"):
            # Try to extract docket number from title like "24A1007 A. A. R. P. v. Trump (05/16/2025)"
            title = pdf_metadata["Title"]
            if " " in title:
                docket_number = title.split(" ")[0]
        elif pdf_metadata.get("file_name"):
            # Try to extract from filename like "24A1007_AARPvTrump_20250516.pdf"
            filename = pdf_metadata["file_name"]
            if "_" in filename:
                docket_number = filename.split("_")[0]

        # Extract case year from filename or creation date
        case_year = None
        if pdf_metadata.get("file_name"):
            filename = pdf_metadata["file_name"]
            if "2025" in filename:
                case_year = 2025
        elif pdf_metadata.get("creation_date_parsed"):
            try:
                case_year = datetime.fromisoformat(
                    pdf_metadata["creation_date_parsed"].replace("Z", "+00:00")
                ).year
            except:
                pass

        # Create PDF metadata object
        pdf_meta = PDFDocumentMetadata(
            title=pdf_metadata.get("Title"),
            creator=pdf_metadata.get("Creator"),
            producer=pdf_metadata.get("Producer"),
            creation_date=pdf_metadata.get("CreationDate"),
            mod_date=pdf_metadata.get("ModDate"),
            total_pages=pdf_metadata.get("total_pages", 0),
            file_size_bytes=pdf_metadata.get("file_size_bytes", 0),
            file_path=pdf_metadata.get("file_path", ""),
            file_name=pdf_metadata.get("file_name", ""),
            creation_date_parsed=pdf_metadata.get("creation_date_parsed"),
            modification_date_parsed=pdf_metadata.get("modification_date_parsed"),
            is_encrypted=pdf_metadata.get("is_encrypted", False),
            pdf_version=pdf_metadata.get("pdf_version", ""),
        )

        # Create case metadata object
        case_meta = CaseDocumentMetadata(
            docket_number=docket_number,
            case_type=kwargs.get("case_type"),
            jurisdiction="supreme_court",
            term=f"{case_year-1}-{case_year}" if case_year else None,
            case_year=case_year,
        )

        return cls(pdf_metadata=pdf_meta, case_metadata=case_meta, **kwargs)

    @staticmethod
    def create_supreme_court_case_from_pdf(
        extraction_result: ExtractionResult,
        pdf_metadata: Dict[str, Any],
        **case_kwargs,
    ) -> "SupremeCourtCase":
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
            preserve_hierarchy=extraction_result.metadata.get(
                "preserve_hierarchy", True
            ),
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
