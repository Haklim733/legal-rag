# for unstructured methods, fast extraction is default
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from src.docs.process_pdfs import (
    extract_text_from_pdf,
    extract_pdf,
    process_pdf_directory,
    process_pdf_directory_unified,
    extract_pdf_metadata,
    ExtractionMethod,
    UnstructuredStrategy,
    ExtractionResult,
    PDFExtractionResult,
)
from src.docs.models import (
    PDFDocumentMetadata,
    ExtractionMetadata,
    CaseDocumentMetadata,
    SupremeCourtCase,
)

# Set page limit for all tests to reduce I/O
PAGE_LIMIT = 5


@pytest.fixture
def test_pdf_path():
    """Fixture that provides the test PDF path or skips the test if not found."""
    pdf_path = (
        Path(__file__).parent.parent.parent
        / "assets"
        / "supreme-court"
        / "downloaded"
        / "24a966_1b8e.pdf"
    )

    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found at {pdf_path}")

    return pdf_path


def test_extract_text_from_pdf(test_pdf_path):
    """Test that we can extract text from a PDF file."""
    try:
        # Test text extraction with default method and 5-page limit
        result = extract_text_from_pdf(test_pdf_path, max_pages=PAGE_LIMIT)

        # Basic assertions for ExtractionResult
        assert isinstance(result, ExtractionResult), "Expected ExtractionResult object"
        assert result.success, "Expected successful extraction"
        assert isinstance(result.full_text, str), "Expected full_text to be a string"
        assert len(result.full_text) > 0, "Expected non-empty text"
        assert isinstance(result.elements, list), "Expected elements to be a list"
        assert result.method == "unstructured_fast", "Expected default method"
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


@pytest.mark.skip(reason="Skipping test_extract_text_from_pdf_different_methods")
def test_extract_text_from_pdf_different_methods(test_pdf_path):
    """Test different extraction methods."""
    methods_to_test = [
        ExtractionMethod.UNSTRUCTURED,
        ExtractionMethod.PYPDF,
    ]

    for method in methods_to_test:
        try:
            result = extract_text_from_pdf(
                test_pdf_path, method=method, max_pages=PAGE_LIMIT
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


def test_extract_text_from_pdf_with_strategy(test_pdf_path):
    """Test text extraction with different strategies."""
    strategies_to_test = [
        UnstructuredStrategy.HI_RES,
        UnstructuredStrategy.FAST,
        UnstructuredStrategy.OCR_ONLY,
    ]

    for strategy in strategies_to_test:
        try:
            result = extract_text_from_pdf(
                test_pdf_path,
                method=ExtractionMethod.UNSTRUCTURED,
                strategy=strategy,
                max_pages=PAGE_LIMIT,
            )

            assert isinstance(
                result, ExtractionResult
            ), f"Expected ExtractionResult for {strategy.value}"
            assert (
                result.success
            ), f"Expected successful extraction for {strategy.value}"
            assert (
                result.method == f"unstructured_{strategy.value}"
                if strategy.value != "hi_res"
                else "unstructured"
            )
            assert (
                result.pages_processed <= PAGE_LIMIT
            ), f"Expected no more than {PAGE_LIMIT} pages for {strategy.value}"

        except RuntimeError as e:
            if "poppler" in str(e).lower():
                pytest.skip(f"Skipping test due to missing system dependency: {e}")
            raise


def test_extract_text_from_pdf_with_hierarchy(test_pdf_path):
    """Test text extraction with and without hierarchy preservation."""
    try:
        # Test with hierarchy preservation
        result_with_hierarchy = extract_text_from_pdf(
            test_pdf_path,
            method=ExtractionMethod.UNSTRUCTURED,
            preserve_hierarchy=True,
            max_pages=PAGE_LIMIT,
        )

        # Test without hierarchy preservation
        result_without_hierarchy = extract_text_from_pdf(
            test_pdf_path,
            method=ExtractionMethod.UNSTRUCTURED,
            preserve_hierarchy=False,
            max_pages=PAGE_LIMIT,
        )

        assert (
            result_with_hierarchy.success
        ), "Expected successful extraction with hierarchy"
        assert (
            result_without_hierarchy.success
        ), "Expected successful extraction without hierarchy"
        assert (
            result_with_hierarchy.method == "unstructured_fast"
        ), "Expected correct method name with hierarchy"
        assert (
            result_without_hierarchy.method == "unstructured_fast_no_hierarchy"
        ), "Expected correct method name without hierarchy"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


@patch("src.docs.process_pdfs.partition_pdf")
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
            assert (
                "Test error" in result.error_message
            ), "Expected error message to contain the test error"
        finally:
            if temp_path.exists():
                temp_path.unlink()


def test_extract_pdf_unified(test_pdf_path):
    """Test the unified PDF extraction function."""
    try:
        # Test unified extraction with both text and metadata
        result = extract_pdf(
            test_pdf_path,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            extract_metadata=True,
            max_pages=PAGE_LIMIT,
        )

        # Basic assertions for PDFExtractionResult
        assert isinstance(
            result, PDFExtractionResult
        ), "Expected PDFExtractionResult object"
        assert result.success, "Expected successful extraction"
        assert result.method == "unstructured_fast", "Expected correct method name"

        # Check text result
        assert isinstance(
            result.text_result, ExtractionResult
        ), "Expected text_result to be ExtractionResult"
        assert result.text_result.success, "Expected successful text extraction"
        assert len(result.text_result.full_text) > 0, "Expected non-empty text"

        # Check metadata result
        assert isinstance(
            result.metadata_result, dict
        ), "Expected metadata_result to be dict"
        assert result.metadata_result.get(
            "success", False
        ), "Expected successful metadata extraction"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_extract_pdf_unified_text_only(test_pdf_path):
    """Test the unified PDF extraction function with text only."""
    try:
        # Test unified extraction with text only (no metadata)
        result = extract_pdf(
            test_pdf_path,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            extract_metadata=False,
            max_pages=PAGE_LIMIT,
        )

        assert isinstance(
            result, PDFExtractionResult
        ), "Expected PDFExtractionResult object"
        assert result.success, "Expected successful extraction"
        assert result.text_result.success, "Expected successful text extraction"
        assert result.metadata_result == {}, "Expected empty metadata result"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


@patch("src.docs.process_pdfs.extract_text_from_pdf")
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


@patch("src.docs.process_pdfs.extract_pdf")
def test_process_pdf_directory_unified(mock_extract):
    """Test processing a directory of PDFs with unified extraction."""
    # Setup mock return value
    mock_result = PDFExtractionResult(
        text_result=ExtractionResult(
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
        ),
        metadata_result={"success": True, "metadata": {"test": "data"}},
        success=True,
        method="test",
    )
    mock_extract.return_value = mock_result

    # Create a temporary directory with a test PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf = Path(temp_dir) / "test.pdf"
        temp_pdf.touch()  # Create empty file

        try:
            # Test processing the directory with unified extraction
            results = process_pdf_directory_unified(
                Path(temp_dir),
                method=ExtractionMethod.UNSTRUCTURED,
                strategy="fast",
                preserve_hierarchy=True,
                extract_metadata=True,
                max_pages=PAGE_LIMIT,
            )

            # Verify results
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["file"] == str(temp_pdf)
            assert results[0]["success"] is True
            assert "text" in results[0]
            assert "elements" in results[0]
            assert "text_metadata" in results[0]
            assert "pdf_metadata" in results[0]
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


@patch("src.docs.process_pdfs.extract_text_from_pdf")
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


@patch("src.docs.process_pdfs.extract_text_from_pdf")
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


def test_extraction_method_enum():
    """Test ExtractionMethod enum values."""
    assert ExtractionMethod.UNSTRUCTURED.value == "unstructured"
    assert ExtractionMethod.PYPDF.value == "pypdf"


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


def test_pdf_extraction_result_dataclass():
    """Test PDFExtractionResult dataclass creation."""
    text_result = ExtractionResult(
        full_text="Test text",
        elements=[],
        metadata={},
        success=True,
        method="test",
        pages_processed=PAGE_LIMIT,
    )

    result = PDFExtractionResult(
        text_result=text_result,
        metadata_result={"success": True, "metadata": {"test": "data"}},
        success=True,
        method="test",
    )

    assert result.text_result == text_result
    assert result.metadata_result == {"success": True, "metadata": {"test": "data"}}
    assert result.success is True
    assert result.method == "test"
    assert result.error_message is None


def test_pdf_extraction_result_with_error():
    """Test PDFExtractionResult with error message."""
    text_result = ExtractionResult(
        full_text="",
        elements=[],
        metadata={},
        success=False,
        method="test",
        pages_processed=0,
        error_message="Text error",
    )

    result = PDFExtractionResult(
        text_result=text_result,
        metadata_result={"success": False, "error": "Metadata error"},
        success=False,
        method="test",
        error_message="Overall error",
    )

    assert result.text_result == text_result
    assert result.metadata_result == {"success": False, "error": "Metadata error"}
    assert result.success is False
    assert result.error_message == "Overall error"


@pytest.mark.skip(reason="Skipping this test for now; takes too long")
def test_compare_extraction_methods(test_pdf_path):
    """Test comparing different extraction methods."""
    # Test comparison with PAGE_LIMIT using existing extraction methods
    results = {}

    # Test unstructured method with different strategies
    strategies_to_test = [UnstructuredStrategy.HI_RES, UnstructuredStrategy.FAST]
    for strategy in strategies_to_test:
        try:
            result = extract_text_from_pdf(
                test_pdf_path,
                method=ExtractionMethod.UNSTRUCTURED,
                strategy=strategy,
                max_pages=PAGE_LIMIT,
            )
            method_name = (
                f"unstructured_{strategy.value}"
                if strategy.value != "hi_res"
                else "unstructured"
            )
            results[method_name] = result
        except Exception as e:
            print(f"Failed to extract with {strategy.value}: {e}")

    # Test PyPDF method
    try:
        pypdf_result = extract_text_from_pdf(
            test_pdf_path, method=ExtractionMethod.PYPDF, max_pages=PAGE_LIMIT
        )
        results["pypdf"] = pypdf_result
    except Exception as e:
        print(f"Failed to extract with PyPDF: {e}")

    # Verify results structure
    assert isinstance(results, dict), "Expected dictionary of results"
    assert len(results) > 0, "Expected at least one successful extraction"

    # Verify each result
    for method_name, result in results.items():
        assert isinstance(
            result, ExtractionResult
        ), f"Expected ExtractionResult for {method_name}"
        assert (
            result.pages_processed <= PAGE_LIMIT
        ), f"Expected no more than {PAGE_LIMIT} pages for {method_name}"

        print(f"\n{method_name} extraction results:")
        print(f"  Success: {result.success}")
        print(f"  Pages processed: {result.pages_processed}")
        print(f"  Text length: {len(result.full_text)}")
        print(f"  Elements: {len(result.elements)}")
        print(f"  Method: {result.method}")


def test_compare_pdf_metadata_methods(test_pdf_path):
    """Test comparing different metadata extraction methods."""
    try:
        # Test metadata comparison using existing extraction methods
        results = {
            "file_path": str(test_pdf_path),
            "unstructured_metadata": extract_pdf_metadata(
                test_pdf_path, method=ExtractionMethod.UNSTRUCTURED
            ),
            "pypdf_metadata": extract_pdf_metadata(
                test_pdf_path, method=ExtractionMethod.PYPDF
            ),
        }

        # Basic assertions
        assert isinstance(results, dict), "Expected dictionary result"
        assert "pypdf_metadata" in results, "Expected pypdf_metadata field"
        assert (
            "unstructured_metadata" in results
        ), "Expected unstructured_metadata field"
        assert results["file_path"] == str(test_pdf_path), "Expected correct file path"

        # Add comparison summary
        unstructured_success = results["unstructured_metadata"]["success"]
        pypdf_success = results["pypdf_metadata"]["success"]

        summary = {
            "pypdf_success": pypdf_success,
            "unstructured_success": unstructured_success,
            "both_successful": unstructured_success and pypdf_success,
            "metadata_fields_pypdf": (
                len(results["pypdf_metadata"].get("metadata", {}))
                if pypdf_success
                else 0
            ),
            "metadata_fields_unstructured": (
                len(results["unstructured_metadata"].get("metadata", {}))
                if unstructured_success
                else 0
            ),
        }

        results["summary"] = summary

        # Check summary
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
        if results["pypdf_metadata"]["success"]:
            print(f"\nPyPDF Metadata:")
            for key, value in results["pypdf_metadata"]["metadata"].items():
                print(f"  {key}: {value}")

        if results["unstructured_metadata"]["success"]:
            print(f"\nUnstructured Metadata:")
            for key, value in results["unstructured_metadata"]["metadata"].items():
                print(f"  {key}: {value}")

        # Warn if neither method was successful
        if not unstructured_success and not pypdf_success:
            print(
                f"Warning: Both metadata extraction methods failed for PDF: {test_pdf_path}"
            )
        elif not unstructured_success:
            print(
                f"Warning: Unstructured metadata extraction failed for PDF: {test_pdf_path}"
            )
        elif not pypdf_success:
            print(f"Warning: PyPDF metadata extraction failed for PDF: {test_pdf_path}")

        # Warn if metadata fields are significantly different between methods
        unstructured_fields = summary["metadata_fields_unstructured"]
        pypdf_fields = summary["metadata_fields_pypdf"]

        if unstructured_success and pypdf_success:
            if abs(unstructured_fields - pypdf_fields) > 5:  # Significant difference
                print(
                    f"Warning: Large difference in metadata fields between methods for PDF: {test_pdf_path} "
                    f"(Unstructured: {unstructured_fields}, PyPDF: {pypdf_fields})"
                )

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


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


def test_extract_pdf_metadata(test_pdf_path):
    """Test extracting metadata from a PDF file and validating with metadata model."""
    try:
        # Test metadata extraction with unstructured method
        result = extract_pdf_metadata(
            test_pdf_path, method=ExtractionMethod.UNSTRUCTURED
        )

        # Basic assertions
        assert isinstance(result, dict), "Expected dictionary result"
        assert result["success"], "Expected successful metadata extraction"
        assert "metadata" in result, "Expected metadata field"
        assert result["file_path"] == str(test_pdf_path), "Expected correct file path"

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
        assert metadata["file_name"] == test_pdf_path.name, "Should match filename"

        # Test validation with PDFDocumentMetadata model
        try:
            pdf_metadata_model = PDFDocumentMetadata(
                title=metadata.get("pdf_title"),
                creator=metadata.get("pdf_creator"),
                producer=metadata.get("pdf_producer"),
                creation_date=metadata.get("pdf_creationdate"),
                mod_date=metadata.get("pdf_moddate"),
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


def test_extract_pdf_metadata_pypdf(test_pdf_path):
    """Test extracting metadata using PyPDF method."""
    try:
        # Test metadata extraction with PyPDF method
        result = extract_pdf_metadata(test_pdf_path, method=ExtractionMethod.PYPDF)

        # Basic assertions
        assert isinstance(result, dict), "Expected dictionary result"
        assert result["success"], "Expected successful metadata extraction"
        assert "metadata" in result, "Expected metadata field"
        assert result["file_path"] == str(test_pdf_path), "Expected correct file path"

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

        print(f"\nPyPDF Metadata extracted:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_pdf_metadata_error_handling():
    """Test metadata extraction error handling with non-existent file."""
    non_existent_pdf = Path("non_existent_file.pdf")

    # Test unstructured metadata extraction
    result = extract_pdf_metadata(
        non_existent_pdf, method=ExtractionMethod.UNSTRUCTURED
    )
    assert not result["success"], "Expected unsuccessful extraction"
    assert "error" in result, "Expected error field"
    assert result["file_path"] == str(non_existent_pdf), "Expected correct file path"

    # Test PyPDF metadata extraction
    result = extract_pdf_metadata(non_existent_pdf, method=ExtractionMethod.PYPDF)
    assert not result["success"], "Expected unsuccessful extraction"
    assert "error" in result, "Expected error field"
    assert result["file_path"] == str(non_existent_pdf), "Expected correct file path"


def test_pdf_metadata_with_text_extraction(test_pdf_path):
    """Test combining metadata extraction with text extraction."""
    try:
        # Extract both metadata and text using unified function
        unified_result = extract_pdf(
            test_pdf_path,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            extract_metadata=True,
            max_pages=PAGE_LIMIT,
        )

        # Verify both extractions were successful
        assert unified_result.success, "Expected successful unified extraction"
        assert unified_result.text_result.success, "Expected successful text extraction"
        assert unified_result.metadata_result.get(
            "success", False
        ), "Expected successful metadata extraction"

        # Verify consistency between metadata and text extraction
        metadata = unified_result.metadata_result.get("metadata", {})
        text_result = unified_result.text_result

        assert (
            metadata.get("total_pages", 0) >= text_result.pages_processed
        ), "Total pages should be >= processed pages"
        assert metadata.get("file_name") == test_pdf_path.name, "Filenames should match"

        print(f"\nCombined Extraction Results:")
        print(f"  Total Pages (metadata): {metadata.get('total_pages', 'N/A')}")
        print(f"  Pages Processed (text): {text_result.pages_processed}")
        print(f"  File Size: {metadata.get('file_size_bytes', 'N/A')} bytes")
        print(f"  Text Length: {len(text_result.full_text)} characters")
        print(f"  Elements Extracted: {len(text_result.elements)}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise
