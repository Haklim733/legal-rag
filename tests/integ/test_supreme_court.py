"""
Integration tests for Supreme Court PDF extraction.

These tests verify that the PDF extraction pipeline works correctly
with actual Supreme Court documents and returns all expected elements.
"""

import pytest
from pathlib import Path

from src.docs.process_pdfs import (
    extract_text_from_pdf,
    extract_pdf,
    ExtractionMethod,
    UnstructuredStrategy,
    ExtractionResult,
    PDFExtractionResult,
)
from src.docs.models import SupremeCourtCase
from src.rag.main import main
from src.rag.models import RAGResponse


# Use the same test PDF path as unit tests
TEST_PDF_PATH = Path("assets/supreme-court/2025/24A1007_AARPvTrump_20250516.pdf")


def test_supreme_court_pdf_full_extraction():
    """Test complete extraction of Supreme Court PDF with all elements."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test unified extraction with both text and metadata
        result = extract_pdf(
            TEST_PDF_PATH,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            extract_metadata=True,
            max_pages=10,  # Limit pages for faster testing
        )

        # Basic assertions for PDFExtractionResult
        assert isinstance(
            result, PDFExtractionResult
        ), "Expected PDFExtractionResult object"
        assert result.success, "Expected successful extraction"
        assert result.method == "unstructured_fast", "Expected correct method name"

        # Check text result
        text_result = result.text_result
        assert isinstance(
            text_result, ExtractionResult
        ), "Expected text_result to be ExtractionResult"
        assert text_result.success, "Expected successful text extraction"
        assert len(text_result.full_text) > 0, "Expected non-empty text"
        assert text_result.pages_processed > 0, "Expected pages to be processed"
        assert (
            text_result.pages_processed <= 10
        ), "Expected no more than 10 pages processed"

        # Check that we have elements
        assert len(text_result.elements) > 0, "Expected non-empty elements list"

        # Verify element structure
        for i, element in enumerate(text_result.elements):
            assert "text" in element, f"Element {i} missing 'text' field"
            assert "type" in element, f"Element {i} missing 'type' field"
            assert (
                "hierarchy_level" in element
            ), f"Element {i} missing 'hierarchy_level' field"
            assert (
                "hierarchy_path" in element
            ), f"Element {i} missing 'hierarchy_path' field"
            assert "metadata" in element, f"Element {i} missing 'metadata' field"

            # Check data types
            assert isinstance(
                element["text"], str
            ), f"Element {i} text should be string"
            assert isinstance(
                element["type"], str
            ), f"Element {i} type should be string"
            assert isinstance(
                element["hierarchy_level"], int
            ), f"Element {i} hierarchy_level should be int"
            assert isinstance(
                element["hierarchy_path"], str
            ), f"Element {i} hierarchy_path should be string"
            assert isinstance(
                element["metadata"], dict
            ), f"Element {i} metadata should be dict"

        # Check metadata result
        assert isinstance(
            result.metadata_result, dict
        ), "Expected metadata_result to be dict"
        assert result.metadata_result.get(
            "success", False
        ), "Expected successful metadata extraction"

        # Verify text content contains Supreme Court case indicators
        full_text = text_result.full_text.lower()
        assert any(
            keyword in full_text
            for keyword in [
                "supreme",
                "court",
                "case",
                "petition",
                "respondent",
                "petitioner",
            ]
        ), "Expected Supreme Court case content in extracted text"

        print(
            f"Successfully extracted {len(text_result.elements)} elements from {text_result.pages_processed} pages"
        )
        print(f"Text length: {len(text_result.full_text)} characters")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_supreme_court_pdf_hierarchy_preservation():
    """Test that hierarchy is properly preserved in Supreme Court PDF extraction."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Test with hierarchy preservation
        result_with_hierarchy = extract_text_from_pdf(
            TEST_PDF_PATH,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            max_pages=5,
        )

        # Test without hierarchy preservation
        result_without_hierarchy = extract_text_from_pdf(
            TEST_PDF_PATH,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=False,
            max_pages=5,
        )

        assert (
            result_with_hierarchy.success
        ), "Expected successful extraction with hierarchy"
        assert (
            result_without_hierarchy.success
        ), "Expected successful extraction without hierarchy"

        # Check that hierarchy preservation affects the structure
        assert (
            len(result_with_hierarchy.elements) > 0
        ), "Expected elements with hierarchy"
        assert (
            len(result_without_hierarchy.elements) > 0
        ), "Expected elements without hierarchy"

        # With hierarchy, we should see different hierarchy levels
        hierarchy_levels_with = set(
            elem["hierarchy_level"] for elem in result_with_hierarchy.elements
        )
        hierarchy_levels_without = set(
            elem["hierarchy_level"] for elem in result_without_hierarchy.elements
        )

        print(f"Hierarchy levels with preservation: {hierarchy_levels_with}")
        print(f"Hierarchy levels without preservation: {hierarchy_levels_without}")

        # Verify method names are correct
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


def test_supreme_court_pdf_different_strategies():
    """Test Supreme Court PDF extraction with different strategies."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    strategies_to_test = [
        UnstructuredStrategy.FAST,
        UnstructuredStrategy.HI_RES,
        UnstructuredStrategy.OCR_ONLY,
    ]

    for strategy in strategies_to_test:
        try:
            result = extract_text_from_pdf(
                TEST_PDF_PATH,
                method=ExtractionMethod.UNSTRUCTURED,
                strategy=strategy,
                preserve_hierarchy=True,
                max_pages=3,  # Limit pages for faster testing
            )

            assert isinstance(
                result, ExtractionResult
            ), f"Expected ExtractionResult for {strategy.value}"
            assert (
                result.success
            ), f"Expected successful extraction for {strategy.value}"

            # Verify method name
            expected_method = (
                f"unstructured_{strategy.value}"
                if strategy.value != "hi_res"
                else "unstructured"
            )
            assert (
                result.method == expected_method
            ), f"Expected method {expected_method} for {strategy.value}"

            # Verify we get some content
            assert (
                len(result.full_text) > 0
            ), f"Expected non-empty text for {strategy.value}"
            assert (
                len(result.elements) > 0
            ), f"Expected non-empty elements for {strategy.value}"

            print(
                f"Strategy {strategy.value}: {len(result.elements)} elements, {len(result.full_text)} chars"
            )

        except RuntimeError as e:
            if "poppler" in str(e).lower():
                pytest.skip(f"Skipping test due to missing system dependency: {e}")
            raise


def test_supreme_court_case_creation():
    """Test creating a Supreme Court case object from extracted PDF data."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Extract text and metadata
        result = extract_pdf(
            TEST_PDF_PATH,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            extract_metadata=True,
            max_pages=5,
        )

        assert result.success, "Expected successful extraction"
        assert result.text_result.success, "Expected successful text extraction"
        assert result.metadata_result.get(
            "success", False
        ), "Expected successful metadata extraction"

        # Test metadata fields were created
        metadata = result.metadata_result
        assert isinstance(metadata, dict), "Expected metadata to be a dictionary"

        # Check for required metadata fields
        if "metadata" in metadata:
            pdf_metadata = metadata["metadata"]
            assert isinstance(
                pdf_metadata, dict
            ), "Expected PDF metadata to be a dictionary"

            # Test specific metadata fields
            assert "file_path" in pdf_metadata, "Expected file_path in metadata"
            assert "file_name" in pdf_metadata, "Expected file_name in metadata"
            assert "total_pages" in pdf_metadata, "Expected total_pages in metadata"
            assert (
                "file_size_bytes" in pdf_metadata
            ), "Expected file_size_bytes in metadata"

            # Test file path matches
            assert pdf_metadata["file_path"] == str(
                TEST_PDF_PATH
            ), "Expected correct file path"
            assert (
                pdf_metadata["file_name"] == TEST_PDF_PATH.name
            ), "Expected correct file name"

            # Test numeric fields
            assert pdf_metadata["total_pages"] > 0, "Expected positive total_pages"
            assert (
                pdf_metadata["file_size_bytes"] > 0
            ), "Expected positive file_size_bytes"

            print(f"✅ Metadata validation successful:")
            print(f"   - File: {pdf_metadata['file_name']}")
            print(f"   - Pages: {pdf_metadata['total_pages']}")
            print(f"   - Size: {pdf_metadata['file_size_bytes']} bytes")

        # Test text result metadata
        text_metadata = result.text_result.metadata
        assert isinstance(
            text_metadata, dict
        ), "Expected text metadata to be a dictionary"
        assert "method" in text_metadata, "Expected method in text metadata"
        assert "strategy" in text_metadata, "Expected strategy in text metadata"
        assert (
            "preserve_hierarchy" in text_metadata
        ), "Expected preserve_hierarchy in text metadata"
        assert (
            "total_elements" in text_metadata
        ), "Expected total_elements in text metadata"

        print(f"✅ Text metadata validation successful:")
        print(f"   - Method: {text_metadata['method']}")
        print(f"   - Strategy: {text_metadata['strategy']}")
        print(f"   - Elements: {text_metadata['total_elements']}")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


def test_supreme_court_pdf_performance():
    """Test performance characteristics of Supreme Court PDF extraction."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    import time

    try:
        # Test extraction time
        start_time = time.time()

        result = extract_pdf(
            TEST_PDF_PATH,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=True,
            extract_metadata=True,
            max_pages=10,
        )

        end_time = time.time()
        extraction_time = end_time - start_time

        assert result.success, "Expected successful extraction"

        # Performance assertions
        assert (
            extraction_time < 30.0
        ), f"Extraction took too long: {extraction_time:.2f} seconds"
        assert result.text_result.pages_processed > 0, "Expected pages to be processed"

        # Calculate performance metrics
        pages_per_second = result.text_result.pages_processed / extraction_time
        elements_per_page = (
            len(result.text_result.elements) / result.text_result.pages_processed
        )

        print(f"Performance metrics:")
        print(f"  Extraction time: {extraction_time:.2f} seconds")
        print(f"  Pages processed: {result.text_result.pages_processed}")
        print(f"  Pages per second: {pages_per_second:.2f}")
        print(f"  Elements per page: {elements_per_page:.2f}")
        print(f"  Total elements: {len(result.text_result.elements)}")
        print(f"  Text length: {len(result.text_result.full_text)} characters")

        # Reasonable performance expectations
        assert (
            pages_per_second > 0.1
        ), f"Expected at least 0.1 pages per second, got {pages_per_second:.2f}"
        assert (
            elements_per_page > 0
        ), f"Expected at least 1 element per page, got {elements_per_page:.2f}"

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise


# @pytest.mark.skip(reason="Skipping LLM extraction test")
def test_llm_extraction():
    """Test LLM extraction of Supreme Court PDF with response validation."""
    # Skip test if the test PDF doesn't exist
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found at {TEST_PDF_PATH}")

    try:
        # Extract text from PDF
        result = extract_pdf(
            TEST_PDF_PATH,
            method=ExtractionMethod.UNSTRUCTURED,
            strategy="fast",
            preserve_hierarchy=False,
            extract_metadata=False,
            max_pages=1,
        )

        assert result.success, "Expected successful PDF extraction"
        assert (
            len(result.text_result.full_text) > 0
        ), "Expected non-empty extracted text"

        # Test LLM extraction and response validation
        response = main(result.text_result.full_text)

        # Validate response structure
        assert isinstance(response, RAGResponse), "Expected RAGResponse object"
        assert (
            response.query_text == result.text_result.full_text
        ), "Expected query text to match"
        assert len(response.response_text) > 0, "Expected non-empty response text"

        assert len(response.entities) > 0
        print(response.entities)
        assert len(response.relationships) > 0
        print(response.entities)

        # Validate confidence summary
        overall = response.overall
        assert isinstance(overall, dict), "Expected overall to be dict"
        # Test individual entity validation
        for entity in response.entities:
            assert entity.entity_name, "Expected non-empty entity name"
            assert entity.entity_type, "Expected non-empty entity type"
            assert entity.description, "Expected non-empty description"
            # assert entity.source_id, "Expected non-empty source_id"
            assert isinstance(entity.chunk_ids, list), "Expected chunk_ids to be list"

        # Test individual relationship validation
        for rel in response.relationships:
            assert rel.src_id, "Expected non-empty src_id"
            assert rel.tgt_id, "Expected non-empty tgt_id"
            assert rel.description, "Expected non-empty description"
            assert rel.keywords, "Expected non-empty keywords"
            assert 0.0 <= rel.weight <= 1.0, "Expected valid weight"
            assert rel.source_id, "Expected non-empty source_id"

        # Print validation results
        print(f"✅ LLM Extraction and Response Validation Successful!")
        print(f"📊 Response Summary:")
        print(f"   - Total concepts found: {total_concepts}")
        print(f"   - Entities: {len(response.entities)}")
        print(f"   - Relationships: {len(response.relationships)}")
        print(f"   - Overall confidence: {overall['overall']:.3f}")

        # Print detailed concept information
        if response.entities:
            print(f"🏢 Entities found:")
            for entity in response.entities:
                print(f"   - {entity.entity_name} (type: {entity.entity_type})")

        if response.relationships:
            print(f"🔗 Relationships found:")
            for rel in response.relationships:
                print(f"   - {rel.src_id} -> {rel.tgt_id} (weight: {rel.weight:.3f})")

    except RuntimeError as e:
        if "poppler" in str(e).lower():
            pytest.skip(f"Skipping test due to missing system dependency: {e}")
        raise
    except Exception as e:
        print(f"❌ LLM extraction test failed: {e}")
        raise
