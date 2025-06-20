import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from docs.process_pdfs import (
    extract_text_from_pdf,
    process_pdf_directory,
    compare_extraction_methods,
    extract_pdf_metadata,
    extract_pdf_metadata_with_unstructured,
    compare_pdf_metadata_methods,
    create_supreme_court_case_from_pdf,
    ExtractionMethod,
    ExtractionResult,
)
from docs.models import (
    PDFDocumentMetadata,
    ExtractionMetadata,
    CaseDocumentMetadata,
    SupremeCourtCase,
)

# Get the path to the test PDF
TEST_PDF_PATH = (
    Path(__file__).parent.parent.parent
    / "assets"
    / "supreme-court"
    / "2025"
    / "24A1007_AARPvTrump_20250516.pdf"
)

# Set page limit for all tests to reduce I/O
PAGE_LIMIT = 5


def test_extract_text_from_pdf():
    """Test that we can extract text from a PDF file."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test text extraction with default method and 5-page limit
        result = extract_text_from_pdf(TEST_PDF_PATH, max_pages=PAGE_LIMIT)

        # Basic assertions for ExtractionResult
        assert isinstance(result, ExtractionResult), "Expected ExtractionResult object"
        assert result.success, "Expected successful extraction"
        assert isinstance(result.full_text, str), "Expected full_text to be a string"
        assert len(result.full_text) > 0, "Expected non-empty text"
        assert isinstance(result.elements, list), "Expected elements to be a list"
        assert result.method == "unstructured_hierarchical", "Expected default method"
        assert result.pages_processed > 0, "Expected pages to be processed"
        assert (
            result.pages_processed <= PAGE_LIMIT
        ), f"Expected no more than {PAGE_LIMIT} pages processed"

        # Skip elements check if no elements were extracted (might be an issue with test file)
        if len(result.elements) == 0:
            pytest.skip(
                "No elements extracted from PDF - check if the test file has extractable text"
            )

        # Test that elements have the expected structure
        for element in result.elements:
            assert "type" in element, "Element missing 'type' field"
            assert "text" in element, "Element missing 'text' field"
            assert isinstance(element["text"], str), "Element text should be a string"
            assert (
                "hierarchy_level" in element
            ), "Element missing 'hierarchy_level' field"
            assert "hierarchy_path" in element, "Element missing 'hierarchy_path' field"
            assert "metadata" in element, "Element missing 'metadata' field"
            assert isinstance(
                element["metadata"], dict
            ), "Element metadata should be a dictionary"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_extract_text_from_pdf_different_methods():
    """Test different extraction methods."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    methods_to_test = [
        ExtractionMethod.UNSTRUCTURED_HIERARCHICAL,
        ExtractionMethod.UNSTRUCTURED_FAST,
        ExtractionMethod.PYPDF,
    ]

    for method in methods_to_test:
        try:
            result = extract_text_from_pdf(
                TEST_PDF_PATH, method=method, max_pages=PAGE_LIMIT
            )

            assert isinstance(
                result, ExtractionResult
            ), f"Expected ExtractionResult for {method.value}"
            assert result.method == method.value, f"Expected method {method.value}"
            assert (
                result.pages_processed <= PAGE_LIMIT
            ), f"Expected no more than {PAGE_LIMIT} pages for {method.value}"

        except RuntimeError as e:
            if "poppler" in str(e).lower():
                pytest.skip(f"Skipping test due to missing system dependency: {e}")
            raise


@patch("docs.process_pdfs.partition_pdf")
def test_extract_text_from_pdf_error_handling(mock_partition):
    """Test error handling when processing a PDF that raises an error."""
    # Setup mock to raise an exception when partition_pdf is called
    mock_partition.side_effect = Exception("Test error")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            # Test that the function handles errors properly with page limit
            result = extract_text_from_pdf(temp_path, max_pages=PAGE_LIMIT)
            assert isinstance(
                result, ExtractionResult
            ), "Expected ExtractionResult object"
            assert not result.success, "Expected unsuccessful extraction"
            assert result.full_text == "", "Expected empty text on error"
            assert result.elements == [], "Expected empty elements on error"
            assert result.error_message is not None, "Expected error message"
        finally:
            if temp_path.exists():
                temp_path.unlink()


@patch("docs.process_pdfs.extract_text_from_pdf")
def test_process_pdf_directory(mock_extract):
    """Test processing a directory of PDFs."""
    # Setup mock return value
    mock_result = ExtractionResult(
        full_text="Test text",
        elements=[
            {
                "type": "Text",
                "text": "Test",
                "hierarchy_level": 1,
                "hierarchy_path": "Test",
                "metadata": {},
            }
        ],
        metadata={"method": "test"},
        success=True,
        method="test",
        pages_processed=PAGE_LIMIT,
    )
    mock_extract.return_value = mock_result

    # Create a temporary directory with a test PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf = Path(temp_dir) / "test.pdf"
        temp_pdf.touch()  # Create empty file

        try:
            # Test processing the directory with page limit
            results = process_pdf_directory(Path(temp_dir), max_pages=PAGE_LIMIT)

            # Verify results
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["file"] == str(temp_pdf)
            assert results[0]["success"] is True
            assert "text" in results[0]
            assert "elements" in results[0]
            assert "metadata" in results[0]
            assert "method" in results[0]
            assert "pages_processed" in results[0]
            assert (
                results[0]["pages_processed"] <= PAGE_LIMIT
            ), f"Expected no more than {PAGE_LIMIT} pages processed"
        except RuntimeError as e:
            if "poppler" in str(e).lower():
                pytest.skip(f"Skipping test due to missing system dependency: {e}")
            raise


def test_process_pdf_directory_error_handling():
    """Test error handling with a non-directory path."""
    try:
        with pytest.raises(ValueError, match="is not a valid directory"):
            process_pdf_directory(Path("nonexistent_dir"), max_pages=PAGE_LIMIT)
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


@patch("docs.process_pdfs.extract_text_from_pdf")
def test_process_pdf_directory_empty(mock_extract):
    """Test processing an empty directory."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test processing an empty directory with page limit
            results = process_pdf_directory(Path(temp_dir), max_pages=PAGE_LIMIT)
            assert results == []
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


@patch("docs.process_pdfs.extract_text_from_pdf")
def test_process_pdf_directory_with_errors(mock_extract):
    """Test processing a directory with a PDF that raises an error."""
    try:
        # Setup mock to raise an exception
        mock_extract.side_effect = Exception("Test error")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf = Path(temp_dir) / "test.pdf"
            temp_pdf.touch()  # Create empty file

            # Test processing the directory with page limit
            results = process_pdf_directory(Path(temp_dir), max_pages=PAGE_LIMIT)

            # Verify error handling
            assert len(results) == 1
            assert results[0]["file"] == str(temp_pdf)
            assert results[0]["success"] is False
            assert "error" in results[0]
    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_compare_extraction_methods():
    """Test comparing different extraction methods."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test comparison with PAGE_LIMIT
        results = compare_extraction_methods(TEST_PDF_PATH, max_pages=PAGE_LIMIT)

        # Verify results structure
        assert isinstance(results, dict), "Expected dictionary of results"
        expected_methods = ["unstructured_hierarchical", "unstructured_fast", "pypdf"]

        for method_name in expected_methods:
            assert method_name in results, f"Expected method {method_name} in results"
            result = results[method_name]
            assert isinstance(
                result, ExtractionResult
            ), f"Expected ExtractionResult for {method_name}"
            assert (
                result.pages_processed <= PAGE_LIMIT
            ), f"Expected no more than {PAGE_LIMIT} pages for {method_name}"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_extraction_method_enum():
    """Test ExtractionMethod enum values."""
    assert ExtractionMethod.UNSTRUCTURED.value == "unstructured"
    assert ExtractionMethod.PYPDF.value == "pypdf"
    assert ExtractionMethod.UNSTRUCTURED_FAST.value == "unstructured_fast"
    assert (
        ExtractionMethod.UNSTRUCTURED_HIERARCHICAL.value == "unstructured_hierarchical"
    )


def test_extraction_result_dataclass():
    """Test ExtractionResult dataclass creation."""
    result = ExtractionResult(
        full_text="Test text",
        elements=[
            {
                "type": "Text",
                "text": "Test",
                "hierarchy_level": 1,
                "hierarchy_path": "Test",
                "metadata": {},
            }
        ],
        metadata={"method": "test"},
        success=True,
        method="test",
        pages_processed=PAGE_LIMIT,
    )

    assert result.full_text == "Test text"
    assert len(result.elements) == 1
    assert result.success is True
    assert result.method == "test"
    assert result.pages_processed == PAGE_LIMIT
    assert result.error_message is None


def test_extraction_result_with_error():
    """Test ExtractionResult with error message."""
    result = ExtractionResult(
        full_text="",
        elements=[],
        metadata={},
        success=False,
        method="test",
        pages_processed=0,
        error_message="Test error",
    )

    assert result.full_text == ""
    assert len(result.elements) == 0
    assert result.success is False
    assert result.error_message == "Test error"


def test_extract_pdf_metadata():
    """Test extracting metadata from a PDF file and validating with metadata model."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test metadata extraction
        result = extract_pdf_metadata(TEST_PDF_PATH)

        # Basic assertions
        assert isinstance(result, dict), "Expected dictionary result"
        assert result["success"], "Expected successful metadata extraction"
        assert "metadata" in result, "Expected metadata field"
        assert result["file_path"] == str(TEST_PDF_PATH), "Expected correct file path"

        metadata = result["metadata"]

        # Check for expected metadata fields
        assert "total_pages" in metadata, "Expected total_pages field"
        assert "file_size_bytes" in metadata, "Expected file_size_bytes field"
        assert "file_name" in metadata, "Expected file_name field"
        assert "is_encrypted" in metadata, "Expected is_encrypted field"
        assert "pdf_version" in metadata, "Expected pdf_version field"

        # Verify data types
        assert isinstance(metadata["total_pages"], int), "total_pages should be integer"
        assert isinstance(
            metadata["file_size_bytes"], int
        ), "file_size_bytes should be integer"
        assert isinstance(
            metadata["is_encrypted"], bool
        ), "is_encrypted should be boolean"

        # Verify reasonable values
        assert metadata["total_pages"] > 0, "Should have at least 1 page"
        assert metadata["file_size_bytes"] > 0, "Should have positive file size"
        assert metadata["file_name"] == TEST_PDF_PATH.name, "Should match filename"

        # Test validation with PDFDocumentMetadata model
        try:
            pdf_metadata_model = PDFDocumentMetadata(
                title=metadata.get("Title"),
                creator=metadata.get("Creator"),
                producer=metadata.get("Producer"),
                creation_date=metadata.get("CreationDate"),
                mod_date=metadata.get("ModDate"),
                total_pages=metadata.get("total_pages", 0),
                file_size_bytes=metadata.get("file_size_bytes", 0),
                file_path=metadata.get("file_path", ""),
                file_name=metadata.get("file_name", ""),
                creation_date_parsed=metadata.get("creation_date_parsed"),
                modification_date_parsed=metadata.get("modification_date_parsed"),
                is_encrypted=metadata.get("is_encrypted", False),
                pdf_version=metadata.get("pdf_version", ""),
            )

            # Validate the model was created successfully
            assert (
                pdf_metadata_model.total_pages == metadata["total_pages"]
            ), "Model should preserve total_pages"
            assert (
                pdf_metadata_model.file_size_bytes == metadata["file_size_bytes"]
            ), "Model should preserve file_size_bytes"
            assert (
                pdf_metadata_model.file_name == metadata["file_name"]
            ), "Model should preserve file_name"
            assert (
                pdf_metadata_model.is_encrypted == metadata["is_encrypted"]
            ), "Model should preserve is_encrypted"
            assert (
                pdf_metadata_model.pdf_version == metadata["pdf_version"]
            ), "Model should preserve pdf_version"

            # Test that required fields are present
            assert (
                pdf_metadata_model.total_pages > 0
            ), "Model should have positive total_pages"
            assert (
                pdf_metadata_model.file_size_bytes > 0
            ), "Model should have positive file_size_bytes"
            assert pdf_metadata_model.file_path, "Model should have non-empty file_path"
            assert pdf_metadata_model.file_name, "Model should have non-empty file_name"
            assert (
                pdf_metadata_model.pdf_version
            ), "Model should have non-empty pdf_version"

        except Exception as e:
            pytest.fail(f"Failed to create PDFDocumentMetadata model: {str(e)}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_extract_pdf_metadata_with_unstructured():
    """Test extracting metadata using unstructured library."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test metadata extraction with unstructured
        result = extract_pdf_metadata_with_unstructured(TEST_PDF_PATH)

        # Basic assertions
        assert isinstance(result, dict), "Expected dictionary result"
        assert result["success"], "Expected successful metadata extraction"
        assert "metadata" in result, "Expected metadata field"
        assert result["file_path"] == str(TEST_PDF_PATH), "Expected correct file path"

        metadata = result["metadata"]

        # Check for expected metadata fields
        assert "total_elements" in metadata, "Expected total_elements field"
        assert "file_size_bytes" in metadata, "Expected file_size_bytes field"
        assert "file_name" in metadata, "Expected file_name field"

        # Verify data types
        assert isinstance(
            metadata["total_elements"], int
        ), "total_elements should be integer"
        assert isinstance(
            metadata["file_size_bytes"], int
        ), "file_size_bytes should be integer"

        # Verify reasonable values
        assert metadata["total_elements"] > 0, "Should have at least 1 element"
        assert metadata["file_size_bytes"] > 0, "Should have positive file size"

        print(f"\nUnstructured Metadata extracted:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_compare_pdf_metadata_methods():
    """Test comparing different metadata extraction methods."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test metadata comparison
        result = compare_pdf_metadata_methods(TEST_PDF_PATH)

        # Basic assertions
        assert isinstance(result, dict), "Expected dictionary result"
        assert "pypdf_metadata" in result, "Expected pypdf_metadata field"
        assert "unstructured_metadata" in result, "Expected unstructured_metadata field"
        assert "summary" in result, "Expected summary field"
        assert result["file_path"] == str(TEST_PDF_PATH), "Expected correct file path"

        # Check summary
        summary = result["summary"]
        assert "pypdf_success" in summary, "Expected pypdf_success field"
        assert "unstructured_success" in summary, "Expected unstructured_success field"
        assert "both_successful" in summary, "Expected both_successful field"
        assert (
            "metadata_fields_pypdf" in summary
        ), "Expected metadata_fields_pypdf field"
        assert (
            "metadata_fields_unstructured" in summary
        ), "Expected metadata_fields_unstructured field"

        # Print comparison results
        print(f"\nMetadata Comparison Results:")
        print(f"  PyPDF Success: {summary['pypdf_success']}")
        print(f"  Unstructured Success: {summary['unstructured_success']}")
        print(f"  Both Successful: {summary['both_successful']}")
        print(f"  PyPDF Fields: {summary['metadata_fields_pypdf']}")
        print(f"  Unstructured Fields: {summary['metadata_fields_unstructured']}")

        # Show detailed metadata from both methods
        if result["pypdf_metadata"]["success"]:
            print(f"\nPyPDF Metadata:")
            for key, value in result["pypdf_metadata"]["metadata"].items():
                print(f"  {key}: {value}")

        if result["unstructured_metadata"]["success"]:
            print(f"\nUnstructured Metadata:")
            for key, value in result["unstructured_metadata"]["metadata"].items():
                print(f"  {key}: {value}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_pdf_metadata_error_handling():
    """Test metadata extraction error handling with non-existent file."""
    non_existent_pdf = Path("non_existent_file.pdf")

    # Test PyPDF metadata extraction
    result = extract_pdf_metadata(non_existent_pdf)
    assert not result["success"], "Expected unsuccessful extraction"
    assert "error" in result, "Expected error field"
    assert result["file_path"] == str(non_existent_pdf), "Expected correct file path"

    # Test unstructured metadata extraction
    result = extract_pdf_metadata_with_unstructured(non_existent_pdf)
    assert not result["success"], "Expected unsuccessful extraction"
    assert "error" in result, "Expected error field"
    assert result["file_path"] == str(non_existent_pdf), "Expected correct file path"


def test_pdf_metadata_with_text_extraction():
    """Test combining metadata extraction with text extraction."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Extract both metadata and text
        metadata_result = extract_pdf_metadata(TEST_PDF_PATH)
        text_result = extract_text_from_pdf(TEST_PDF_PATH, max_pages=PAGE_LIMIT)

        # Verify both extractions were successful
        assert metadata_result["success"], "Expected successful metadata extraction"
        assert text_result.success, "Expected successful text extraction"

        # Verify consistency between metadata and text extraction
        metadata = metadata_result["metadata"]
        assert (
            metadata["total_pages"] >= text_result.pages_processed
        ), "Total pages should be >= processed pages"
        assert metadata["file_name"] == TEST_PDF_PATH.name, "Filenames should match"

        print(f"\nCombined Extraction Results:")
        print(f"  Total Pages (metadata): {metadata['total_pages']}")
        print(f"  Pages Processed (text): {text_result.pages_processed}")
        print(f"  File Size: {metadata['file_size_bytes']} bytes")
        print(f"  Text Length: {len(text_result.full_text)} characters")
        print(f"  Elements Extracted: {len(text_result.elements)}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_create_supreme_court_case_from_pdf():
    """Test creating SupremeCourtCase from PDF processing results."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Extract text and metadata
        extraction_result = extract_text_from_pdf(TEST_PDF_PATH, max_pages=PAGE_LIMIT)
        pdf_metadata_result = extract_pdf_metadata(TEST_PDF_PATH)

        # Verify extractions were successful
        assert extraction_result.success, "Text extraction should succeed"
        assert pdf_metadata_result["success"], "Metadata extraction should succeed"

        # Create SupremeCourtCase instance
        case = create_supreme_court_case_from_pdf(
            pdf_path=TEST_PDF_PATH,
            extraction_result=extraction_result,
            pdf_metadata=pdf_metadata_result,
            dcid="dcid:test_case_24A1007",
            name="A. A. R. P., ET AL. v. DONALD J. TRUMP, PRESIDENT OF THE UNITED STATES, ET AL.",
            description="Test case regarding Alien Enemies Act",
            date_decided="2025-05-16",
            citation="24A1007",
            parties="A. A. R. P., ET AL. v. DONALD J. TRUMP, PRESIDENT OF THE UNITED STATES, ET AL.",
            decision_direction="per_curiam",
            opinion_author="Per Curiam",
            case_type="per_curiam",
        )

        # Validate the case was created successfully
        assert isinstance(
            case, SupremeCourtCase
        ), "Should create SupremeCourtCase instance"
        assert case.dcid == "dcid:test_case_24A1007", "Should set correct dcid"
        assert case.name, "Should have case name"
        assert case.description, "Should have case description"
        assert case.date_decided == "2025-05-16", "Should have correct date"

        # Validate PDF metadata
        assert case.pdf_metadata is not None, "Should have PDF metadata"
        assert case.pdf_metadata.total_pages > 0, "Should have positive total pages"
        assert case.pdf_metadata.file_size_bytes > 0, "Should have positive file size"
        assert (
            case.pdf_metadata.file_name == TEST_PDF_PATH.name
        ), "Should have correct filename"
        assert case.pdf_metadata.is_encrypted is False, "Should not be encrypted"

        # Validate extraction metadata
        assert case.extraction_metadata is not None, "Should have extraction metadata"
        assert (
            case.extraction_metadata.extraction_method == "unstructured_hierarchical"
        ), "Should have correct method"
        assert (
            case.extraction_metadata.pages_processed <= PAGE_LIMIT
        ), f"Should process no more than {PAGE_LIMIT} pages"
        assert (
            case.extraction_metadata.total_elements > 0
        ), "Should have positive element count"
        assert (
            case.extraction_metadata.preserve_hierarchy is True
        ), "Should preserve hierarchy"

        # Validate case metadata
        assert case.case_metadata is not None, "Should have case metadata"
        assert (
            case.case_metadata.jurisdiction == "supreme_court"
        ), "Should have correct jurisdiction"
        assert (
            case.case_metadata.case_type == "per_curiam"
        ), "Should have correct case type"

        # Test docket number extraction
        if case.case_metadata.docket_number:
            assert (
                case.case_metadata.docket_number == "24A1007"
            ), "Should extract correct docket number"

        # Test case year extraction
        if case.case_metadata.case_year:
            assert (
                case.case_metadata.case_year == 2025
            ), "Should extract correct case year"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_pdf_metadata_model_validation():
    """Test PDFDocumentMetadata model validation with various inputs."""

    # Test with complete metadata
    valid_metadata = PDFDocumentMetadata(
        title="Test Case Title",
        creator="PScript5.dll Version 5.2.2",
        producer="Acrobat Distiller 24.0 (Windows)",
        creation_date="D:20250516150044-04'00'",
        mod_date="D:20250516150157-04'00'",
        total_pages=24,
        file_size_bytes=151001,
        file_path="/path/to/test.pdf",
        file_name="test.pdf",
        creation_date_parsed="2025-05-16T00:00:00",
        modification_date_parsed="2025-05-16T00:00:00",
        is_encrypted=False,
        pdf_version="%PDF-1.6",
    )

    assert valid_metadata.title == "Test Case Title"
    assert valid_metadata.total_pages == 24
    assert valid_metadata.file_size_bytes == 151001
    assert valid_metadata.is_encrypted is False

    # Test with minimal required fields
    minimal_metadata = PDFDocumentMetadata(
        total_pages=1,
        file_size_bytes=1000,
        file_path="/path/to/minimal.pdf",
        file_name="minimal.pdf",
        is_encrypted=False,
        pdf_version="%PDF-1.4",
    )

    assert minimal_metadata.total_pages == 1
    assert minimal_metadata.title is None
    assert minimal_metadata.creator is None

    # Test that validation fails with invalid data
    with pytest.raises(ValueError):
        PDFDocumentMetadata(
            total_pages=-1,  # Invalid: negative pages
            file_size_bytes=1000,
            file_path="/path/to/test.pdf",
            file_name="test.pdf",
            is_encrypted=False,
            pdf_version="%PDF-1.6",
        )


def test_extraction_metadata_model_validation():
    """Test ExtractionMetadata model validation."""

    # Test with valid extraction metadata
    valid_extraction = ExtractionMetadata(
        extraction_method="unstructured_hierarchical",
        pages_processed=5,
        total_elements=150,
        extraction_date="2025-01-20T10:30:00Z",
        preserve_hierarchy=True,
        strategy="hi_res",
    )

    assert valid_extraction.extraction_method == "unstructured_hierarchical"
    assert valid_extraction.pages_processed == 5
    assert valid_extraction.total_elements == 150
    assert valid_extraction.preserve_hierarchy is True

    # Test with minimal required fields
    minimal_extraction = ExtractionMetadata(
        extraction_method="pypdf",
        pages_processed=1,
        total_elements=10,
        extraction_date="2025-01-20T10:30:00Z",
    )

    assert minimal_extraction.extraction_method == "pypdf"
    assert minimal_extraction.preserve_hierarchy is True  # Default value
    assert minimal_extraction.strategy == "hi_res"  # Default value


def test_case_metadata_model_validation():
    """Test CaseDocumentMetadata model validation."""

    # Test with complete case metadata
    valid_case = CaseDocumentMetadata(
        docket_number="24A1007",
        case_type="per_curiam",
        jurisdiction="supreme_court",
        term="2024-2025",
        case_year=2025,
    )

    assert valid_case.docket_number == "24A1007"
    assert valid_case.case_type == "per_curiam"
    assert valid_case.jurisdiction == "supreme_court"
    assert valid_case.term == "2024-2025"
    assert valid_case.case_year == 2025

    # Test with minimal required fields
    minimal_case = CaseDocumentMetadata()

    assert minimal_case.jurisdiction == "supreme_court"  # Default value
    assert minimal_case.docket_number is None
    assert minimal_case.case_type is None


def test_supreme_court_case_metadata_integration():
    """Test full integration of metadata in SupremeCourtCase."""

    # Create a complete SupremeCourtCase with all metadata
    case = SupremeCourtCase(
        dcid="dcid:test_case",
        name="Test Case",
        description="Test case description",
        date_decided="2025-05-16",
        citation="24A1007",
        parties="Test Parties",
        decision_direction="affirmed",
        opinion_author="Test Justice",
        cites=[],
        pdf_metadata=PDFDocumentMetadata(
            title="Test PDF Title",
            total_pages=24,
            file_size_bytes=151001,
            file_path="/path/to/test.pdf",
            file_name="test.pdf",
            is_encrypted=False,
            pdf_version="%PDF-1.6",
        ),
        extraction_metadata=ExtractionMetadata(
            extraction_method="unstructured_hierarchical",
            pages_processed=5,
            total_elements=150,
            extraction_date="2025-01-20T10:30:00Z",
        ),
        case_metadata=CaseDocumentMetadata(
            docket_number="24A1007", case_type="per_curiam", case_year=2025
        ),
    )

    # Validate all metadata is properly integrated
    assert case.pdf_metadata.title == "Test PDF Title"
    assert case.extraction_metadata.extraction_method == "unstructured_hierarchical"
    assert case.case_metadata.docket_number == "24A1007"

    # Test JSON serialization
    case_json = case.model_dump()
    assert "pdf_metadata" in case_json
    assert "extraction_metadata" in case_json
    assert "case_metadata" in case_json
    assert case_json["pdf_metadata"]["title"] == "Test PDF Title"
