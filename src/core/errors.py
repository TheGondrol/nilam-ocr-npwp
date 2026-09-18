class ServiceError(Exception):
    """
    Raised by the service layer when a request can't be fulfilled. Carries the
    HTTP status the API layer should respond with. Same role as OcrServiceError
    in the ocr-* mocks, shared here because this monolith has several services.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)
