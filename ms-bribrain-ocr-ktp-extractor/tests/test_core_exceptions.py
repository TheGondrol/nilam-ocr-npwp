"""Unit tests for src.core.exceptions module"""

import pytest
from src.core.exceptions import (
    OCRServiceError,
    ImageValidationError,
    OCRProcessingError,
    OCRInitializationError
)


class TestOCRServiceError:
    """Test cases for OCRServiceError base class"""

    def test_error_with_message_only(self):
        """Test creating error with message only"""
        error = OCRServiceError("Test error message")
        assert error.message == "Test error message"
        assert error.details is None
        assert str(error) == "Test error message"

    def test_error_with_message_and_details(self):
        """Test creating error with message and details"""
        error = OCRServiceError("Test error", details="Additional details")
        assert error.message == "Test error"
        assert error.details == "Additional details"

    def test_error_inheritance(self):
        """Test that OCRServiceError inherits from Exception"""
        error = OCRServiceError("Test")
        assert isinstance(error, Exception)


class TestImageValidationError:
    """Test cases for ImageValidationError"""

    def test_inherits_from_ocr_service_error(self):
        """Test that ImageValidationError inherits from OCRServiceError"""
        error = ImageValidationError("Invalid image")
        assert isinstance(error, OCRServiceError)
        assert isinstance(error, Exception)

    def test_error_creation(self):
        """Test creating ImageValidationError"""
        error = ImageValidationError(
            "Invalid file type",
            details="Only JPEG and PNG supported"
        )
        assert error.message == "Invalid file type"
        assert error.details == "Only JPEG and PNG supported"


class TestOCRProcessingError:
    """Test cases for OCRProcessingError"""

    def test_inherits_from_ocr_service_error(self):
        """Test that OCRProcessingError inherits from OCRServiceError"""
        error = OCRProcessingError("Processing failed")
        assert isinstance(error, OCRServiceError)
        assert isinstance(error, Exception)

    def test_error_creation(self):
        """Test creating OCRProcessingError"""
        error = OCRProcessingError(
            "OCR prediction failed",
            details="Paddle engine error"
        )
        assert error.message == "OCR prediction failed"
        assert error.details == "Paddle engine error"


class TestOCRInitializationError:
    """Test cases for OCRInitializationError"""

    def test_inherits_from_ocr_service_error(self):
        """Test that OCRInitializationError inherits from OCRServiceError"""
        error = OCRInitializationError("Init failed")
        assert isinstance(error, OCRServiceError)
        assert isinstance(error, Exception)

    def test_error_creation(self):
        """Test creating OCRInitializationError"""
        error = OCRInitializationError(
            "Failed to initialize OCR",
            details="Missing model files"
        )
        assert error.message == "Failed to initialize OCR"
        assert error.details == "Missing model files"

    def test_error_catching(self):
        """Test catching specific error types"""
        def raise_init_error():
            raise OCRInitializationError("Test error")

        # Should catch specific error type
        with pytest.raises(OCRInitializationError):
            raise_init_error()

        # Should also catch base class
        with pytest.raises(OCRServiceError):
            raise_init_error()

        # Should also catch Exception
        with pytest.raises(Exception):
            raise_init_error()


class TestErrorHierarchy:
    """Test error hierarchy and catching behavior"""

    def test_catch_all_ocr_errors(self):
        """Test catching all OCR errors using base class"""
        errors = [
            ImageValidationError("test"),
            OCRProcessingError("test"),
            OCRInitializationError("test")
        ]

        for error in errors:
            with pytest.raises(OCRServiceError):
                raise error

    def test_specific_error_handling(self):
        """Test handling specific vs general errors"""
        def handle_error(error):
            if isinstance(error, ImageValidationError):
                return "validation_error"
            elif isinstance(error, OCRProcessingError):
                return "processing_error"
            elif isinstance(error, OCRInitializationError):
                return "init_error"
            elif isinstance(error, OCRServiceError):
                return "general_error"
            return "unknown"

        assert handle_error(ImageValidationError("test")) == "validation_error"
        assert handle_error(OCRProcessingError("test")) == "processing_error"
        assert handle_error(OCRInitializationError("test")) == "init_error"
