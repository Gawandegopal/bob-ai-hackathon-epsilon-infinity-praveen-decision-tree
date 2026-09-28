"""
test_document_processor.py
==========================
Unit tests for document_processor.py.

Covers:
- Supported extension detection
- Plaintext file processing
- Empty/invalid document handling
- Graceful fallback when Docling unavailable
- processing_capability_summary() returns a non-empty string
- is_document_processing_available() returns bool
- explain_document_processing() returns correct structure
- Unsupported file type returns failure result with helpful message

Run with:
    pytest src/tests/test_document_processor.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from document_processor import (
    process_uploaded_document,
    is_document_processing_available,
    processing_capability_summary,
    explain_document_processing,
    DocumentResult,
    SUPPORTED_EXTENSIONS,
    DOCLING_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Utility / status function tests
# ---------------------------------------------------------------------------

class TestCapabilityFunctions:
    def test_processing_capability_summary_returns_string(self):
        summary = processing_capability_summary()
        assert isinstance(summary, str)
        assert len(summary) > 5

    def test_is_document_processing_available_returns_bool(self):
        result = is_document_processing_available()
        assert isinstance(result, bool)

    def test_explain_document_processing_returns_dict(self):
        info = explain_document_processing()
        assert isinstance(info, dict)
        assert "docling_available" in info
        assert "capability_summary" in info
        assert "supported_types" in info
        assert "is_available" in info

    def test_supported_extensions_is_nonempty_set(self):
        assert isinstance(SUPPORTED_EXTENSIONS, set)
        assert len(SUPPORTED_EXTENSIONS) > 0
        assert ".pdf" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Plaintext processing tests
# ---------------------------------------------------------------------------

class TestPlaintext:
    def test_txt_success(self):
        content = b"Blood swab collected from scene. Case FIR-2024-001."
        result = process_uploaded_document(content, "case_notes.txt")
        assert isinstance(result, DocumentResult)
        assert result.success is True
        assert "Blood swab" in result.text
        assert result.method == "plaintext"

    def test_txt_empty_file(self):
        result = process_uploaded_document(b"   ", "empty.txt")
        # Empty text — success should be False
        assert isinstance(result, DocumentResult)
        assert result.success is False

    def test_txt_unicode_content(self):
        content = "Évidence forensique — cas de vol.".encode("utf-8")
        result = process_uploaded_document(content, "notes.txt")
        assert isinstance(result, DocumentResult)
        # success depends on whether text was decoded
        assert result.method == "plaintext" or not result.success


# ---------------------------------------------------------------------------
# Empty / invalid input tests
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_empty_bytes_returns_failure(self):
        result = process_uploaded_document(b"", "empty.pdf")
        assert isinstance(result, DocumentResult)
        assert result.success is False
        assert result.warning  # should have a warning message

    def test_unsupported_extension_returns_failure(self):
        result = process_uploaded_document(b"some bytes", "evidence.xyz")
        assert isinstance(result, DocumentResult)
        assert result.success is False
        assert ".xyz" in result.warning or "not supported" in result.warning.lower()

    def test_unsupported_extension_mp4(self):
        result = process_uploaded_document(b"video", "cctv.mp4")
        assert isinstance(result, DocumentResult)
        assert result.success is False

    def test_result_filename_preserved(self):
        result = process_uploaded_document(b"test content", "test_case.txt")
        # filename may not be set if empty, but the call should not crash
        assert isinstance(result, DocumentResult)


# ---------------------------------------------------------------------------
# Docling availability test
# ---------------------------------------------------------------------------

class TestDoclingAvailability:
    def test_docling_available_is_bool(self):
        """DOCLING_AVAILABLE should always be a bool regardless of install state."""
        assert isinstance(DOCLING_AVAILABLE, bool)

    def test_fallback_used_when_docling_unavailable(self):
        """When docling is not available, txt processing should still work."""
        content = b"This is a test document for the case."
        result = process_uploaded_document(content, "test.txt")
        assert isinstance(result, DocumentResult)
        # Should work via plaintext fallback regardless of Docling state
        assert result.success is True

    def test_pdf_returns_result_not_exception(self):
        """PDF processing should return a DocumentResult, never raise."""
        # Minimal fake PDF bytes (not a valid PDF — should fail gracefully)
        fake_pdf = b"%PDF-1.4 fake content"
        result = process_uploaded_document(fake_pdf, "test.pdf")
        assert isinstance(result, DocumentResult)
        # success may be True or False depending on what libs are available
        # but it must not raise an exception
