"""Custom exceptions for OCR service"""


class OCRServiceError(Exception):
    """
    Base exception for all OCR service errors.
    
    All custom exceptions in this module inherit from this class,
    allowing for catch-all error handling when needed.
    """
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ImageValidationError(OCRServiceError):
    """
    Raised when image validation fails.
    
    Examples:
        - Invalid file type (not JPEG/PNG)
        - File too large
        - Corrupted image data
        - Unable to decode image
    """
    pass


class OCRProcessingError(OCRServiceError):
    """
    Raised when OCR processing fails.
    
    Examples:
        - OCR engine initialization failed
        - OCR prediction failed
        - Result transformation failed
    """
    pass


class OCRInitializationError(OCRServiceError):
    """
    Raised when OCR engine fails to initialize.
    
    Examples:
        - Missing model files
        - Invalid configuration
        - GPU initialization failed
    """
    pass


class DatabaseError(OCRServiceError):
    """
    Raised when database operations fail.
    
    Examples:
        - Connection failed
        - Insert/update failed
        - Transaction rollback
    """
    pass
