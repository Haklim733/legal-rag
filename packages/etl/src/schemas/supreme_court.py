from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum

class CourtType(str, Enum):
    """Types of courts that can hear a case before it reaches the Supreme Court."""
    STATE = "state"
    FEDERAL_DISTRICT = "federal_district"
    FEDERAL_APPEALS = "federal_appeals"
    STATE_SUPREME = "state_supreme"
    OTHER = "other"

class CaseDisposition(str, Enum):
    """Possible dispositions of a Supreme Court case."""
    AFFIRMED = "affirmed"
    REVERSED = "reversed"
    REMANDED = "remanded"
    VACATED = "vacated"
    DISMISSED = "dismissed"
    REHEARING_DENIED = "rehearing_denied"
    OTHER = "other"

class OpinionType(str, Enum):
    """Types of opinions in a Supreme Court case."""
    MAJORITY = "majority"
    CONCURRING = "concurring"
    DISSENTING = "dissenting"
    PLURALITY = "plurality"
    PER_CURIAM = "per_curiam"

class Justice(BaseModel):
    """Information about a Supreme Court Justice."""
    name: str
    id: str  # Unique identifier (e.g., 'RBG' for Ruth Bader Ginsburg)
    appointed_by: str  # President who appointed
    year_appointed: int
    year_left_court: Optional[int] = None
    chief_justice: bool = False

class Opinion(BaseModel):
    """An opinion in a Supreme Court case."""
    type: OpinionType
    author: str  # Justice ID
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    joined_by: List[str] = Field(default_factory=list)  # List of Justice IDs

class Citation(BaseModel):
    """Legal citation for a case."""
    volume: int
    reporter: str  # e.g., "U.S.", "S. Ct.", "L. Ed. 2d"
    page: int
    year: int

class LowerCourt(BaseModel):
    """Information about the lower court decision."""
    name: str
    court_type: CourtType
    disposition: Optional[str] = None  # How the lower court ruled
    citation: Optional[Citation] = None

class SupremeCourtCase(BaseModel):
    """Complete schema for a Supreme Court case."""
    # Core identification
    id: str  # Unique identifier (e.g., citation as "410 U.S. 113")
    docket_number: str  # e.g., "70-18"
    name: str  # Short name (e.g., "Roe v. Wade")
    full_name: str  # Full case name
    
    # Procedural information
    date_argued: Optional[datetime] = None
    date_reargued: Optional[datetime] = None
    date_decided: datetime
    term: int  # The Supreme Court term (e.g., 2023 for October Term 2023)
    
    # Case content
    summary: Optional[str] = None
    facts: Optional[str] = None
    question: Optional[str] = None  # The legal question presented
    conclusion: Optional[str] = None
    
    # Legal analysis
    holding: Optional[str] = None
    majority_vote: int
    minority_vote: int
    
    # Citations and references
    us_cite: Optional[Citation] = None  # Official U.S. Reports citation
    scotus_blog_url: Optional[HttpUrl] = None
    oyez_url: Optional[HttpUrl] = None
    justia_url: Optional[HttpUrl] = None
    
    # Court composition and votes
    chief_justice: str  # Justice ID of Chief Justice during decision
    justices_in_majority: List[str] = Field(default_factory=list)  # Justice IDs
    justices_in_dissent: List[str] = Field(default_factory=list)  # Justice IDs
    
    # Case history
    lower_court: LowerCourt
    disposition: CaseDisposition
    
    # Legal categories and topics
    topics: List[str] = Field(default_factory=list)  # Broad legal topics
    constitutional_amendments: List[str] = Field(default_factory=list)  # e.g., ["First Amendment"]
    
    # Full text and opinions
    syllabus: Optional[str] = None  # Summary from the Court
    opinions: List[Opinion] = Field(default_factory=list)
    
    # Metadata
    source_file: Optional[str] = None  # Original source file if applicable
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "410 U.S. 113",
                "docket_number": "70-18",
                "name": "Roe v. Wade",
                "full_name": "Jane Roe, et al. v. Henry Wade, District Attorney of Dallas County",
                "date_decided": "1973-01-22T00:00:00",
                "term": 1972,
                "holding": "A right to privacy under the Due Process Clause of the 14th Amendment extends to a woman's decision to have an abortion...",
                "majority_vote": 7,
                "minority_vote": 2,
                "chief_justice": "WEBurger",
                "justices_in_majority": ["WEBurger", "WODouglas", "WJBrennan", "PStewart", "TCMarshall", "HABlackmun", "LFPowell"],
                "justices_in_dissent": ["WEBurger", "WHRehnquist"],
                "lower_court": {
                    "name": "United States District Court for the Northern District of Texas",
                    "court_type": "federal_district",
                    "disposition": "Affirmed in part, reversed in part, and remanded"
                },
                "disposition": "affirmed",
                "constitutional_amendments": ["Fourteenth Amendment"],
                "topics": ["Abortion", "Right to Privacy", "Reproductive Rights"],
                "source_file": "roe_v_wade_410_us_113.pdf"
            }
        }

class CaseChunk(BaseModel):
    """Schema for chunks of case text with metadata for RAG."""
    id: str  # Unique identifier for the chunk
    case_id: str  # Reference to the parent case
    chunk_index: int  # Position of this chunk in the sequence
    total_chunks: int  # Total number of chunks for this case
    text: str  # The actual chunk content
    page_number: Optional[int] = None  # Source page number if available
    section: Optional[str] = None  # e.g., "syllabus", "opinion", "dissent"
    author: Optional[str] = None  # For opinions, the justice who wrote it
    opinion_type: Optional[OpinionType] = None  # Type of opinion if applicable
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "410_U_S_113_chunk_1",
                "case_id": "410 U.S. 113",
                "chunk_index": 0,
                "total_chunks": 15,
                "text": "The Constitution does not explicitly mention any right of privacy...",
                "page_number": 1,
                "section": "majority_opinion",
                "author": "HABlackmun",
                "opinion_type": "majority",
                "metadata": {
                    "topic": "Right to Privacy",
                    "citations": ["Griswold v. Connecticut, 381 U.S. 479 (1965)"]
                }
            }
        }
