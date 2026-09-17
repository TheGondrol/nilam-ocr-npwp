"""Unit tests for src.api.routes module"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import UploadFile
from fastapi.testclient import TestClient
from io import BytesIO
from src.core.logging import request_id_ctx


class TestExtractTextLinesEndpoint:
    """Test cases for /v1/ocr_extract endpoint"""

    @pytest.mark.asyncio
    async def test_successful_ocr_extraction(self, sample_image_bytes, mock_request_id):
        """Test successful OCR extraction"""
        from src.api.routes import extract_text_lines, router
        
        # Set request_id in context
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            mock_request.request_id = mock_request_id
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=sample_image_bytes)
            
            mock_ocr_result = [
                ([[10, 10], [100, 10], [100, 30], [10, 30]], ("Test Text", 0.95))
            ]
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                mock_settings.ocr_max_size_mb = 10
                
                with patch('src.api.routes.perform_ocr_async', return_value=mock_ocr_result):
                    with patch('src.api.routes.insert_log'):
                        response = await extract_text_lines(mock_request, mock_file)
                        
                        assert response.status_code == 200
                        content = response.body.decode()
                        assert "ocr_result" in content
                        assert "processing_time" in content
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_invalid_file_type(self, mock_request_id):
        """Test rejection of invalid file types"""
        from src.api.routes import extract_text_lines
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.pdf"
            mock_file.content_type = "application/pdf"
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                
                with patch('src.api.routes.insert_log'):
                    response = await extract_text_lines(mock_request, mock_file)
                    
                    assert response.status_code == 400
                    content = response.body.decode()
                    assert "Invalid file type" in content
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_file_too_large(self, mock_request_id):
        """Test rejection of files that are too large"""
        from src.api.routes import extract_text_lines
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            # Create bytes larger than 10MB
            large_bytes = b"x" * (11 * 1024 * 1024)  # 11MB
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "large.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=large_bytes)
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                mock_settings.ocr_max_size_mb = 10
                
                with patch('src.api.routes.insert_log'):
                    response = await extract_text_lines(mock_request, mock_file)
                    
                    assert response.status_code == 400
                    content = response.body.decode()
                    assert "File too large" in content
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_failed_to_read_file(self, mock_request_id):
        """Test handling of file read errors"""
        from src.api.routes import extract_text_lines
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(side_effect=Exception("Read error"))
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                
                with patch('src.api.routes.insert_log'):
                    response = await extract_text_lines(mock_request, mock_file)
                    
                    assert response.status_code == 400
                    content = response.body.decode()
                    assert "Failed to read uploaded file" in content
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_image_validation_error(self, sample_image_bytes, mock_request_id):
        """Test handling of image validation errors"""
        from src.api.routes import extract_text_lines
        from src.core.exceptions import ImageValidationError
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "corrupt.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=sample_image_bytes)
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                mock_settings.ocr_max_size_mb = 10
                
                with patch('src.api.routes.perform_ocr_async', side_effect=ImageValidationError("Invalid image")):
                    with patch('src.api.routes.insert_log'):
                        response = await extract_text_lines(mock_request, mock_file)
                        
                        assert response.status_code == 400
                        content = response.body.decode()
                        assert "Image processing failed" in content
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_ocr_processing_error(self, sample_image_bytes, mock_request_id):
        """Test handling of OCR processing errors"""
        from src.api.routes import extract_text_lines
        from src.core.exceptions import OCRProcessingError
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=sample_image_bytes)
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                mock_settings.ocr_max_size_mb = 10
                
                with patch('src.api.routes.perform_ocr_async', side_effect=OCRProcessingError("OCR failed")):
                    with patch('src.api.routes.insert_log'):
                        response = await extract_text_lines(mock_request, mock_file)
                        
                        assert response.status_code == 500
                        content = response.body.decode()
                        assert "OCR processing failed" in content
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_ocr_initialization_error(self, sample_image_bytes, mock_request_id):
        """Test handling of OCR initialization errors"""
        from src.api.routes import extract_text_lines
        from src.core.exceptions import OCRInitializationError
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=sample_image_bytes)
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                mock_settings.ocr_max_size_mb = 10
                
                with patch('src.api.routes.perform_ocr_async', side_effect=OCRInitializationError("Init failed")):
                    with patch('src.api.routes.insert_log'):
                        response = await extract_text_lines(mock_request, mock_file)
                        
                        assert response.status_code == 500
        finally:
            request_id_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_unexpected_error(self, sample_image_bytes, mock_request_id):
        """Test handling of unexpected errors"""
        from src.api.routes import extract_text_lines
        
        token = request_id_ctx.set(mock_request_id)
        
        try:
            mock_request = MagicMock()
            
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read = AsyncMock(return_value=sample_image_bytes)
            
            with patch('src.api.routes.settings') as mock_settings:
                mock_settings.ocr_allowed_types = ["image/jpeg", "image/png"]
                mock_settings.ocr_max_size_mb = 10
                
                with patch('src.api.routes.perform_ocr_async', side_effect=RuntimeError("Unexpected")):
                    with patch('src.api.routes.insert_log'):
                        response = await extract_text_lines(mock_request, mock_file)
                        
                        assert response.status_code == 500
        finally:
            request_id_ctx.reset(token)


@pytest.mark.asyncio
class TestLogAndRespondError:
    """Test cases for _log_and_respond_error helper function"""

    async def test_log_and_respond_error_with_request_id(self, mock_request_id):
        """Test error response includes request_id in envelope"""
        from src.api.routes import _log_and_respond_error

        with patch('src.api.routes.insert_log', new_callable=AsyncMock):
            response = await _log_and_respond_error(
                request_id=mock_request_id,
                status_code=400,
                error_message="Test error",
                filename="test.jpg",
                processing_time=1.0,
                error_code="INVALID_FILE_TYPE"
            )

            assert response.status_code == 400
            content = response.body.decode()
            assert "Test error" in content
            assert mock_request_id in content
            assert "INVALID_FILE_TYPE" in content

    async def test_log_and_respond_error_default_error_code(self, mock_request_id):
        """Test error response uses default INTERNAL_ERROR error code"""
        from src.api.routes import _log_and_respond_error

        with patch('src.api.routes.insert_log', new_callable=AsyncMock):
            response = await _log_and_respond_error(
                request_id=mock_request_id,
                status_code=500,
                error_message="Internal error",
                filename="test.jpg",
                processing_time=1.5,
            )

            assert response.status_code == 500
            content = response.body.decode()
            assert "Internal error" in content
            assert "INTERNAL_ERROR" in content

    async def test_log_and_respond_error_calls_insert_log(self, mock_request_id):
        """Test that insert_log is called with correct parameters"""
        from src.api.routes import _log_and_respond_error

        with patch('src.api.routes.insert_log', new_callable=AsyncMock) as mock_insert:
            await _log_and_respond_error(
                request_id=mock_request_id,
                status_code=400,
                error_message="Test error",
                filename="test.jpg",
                processing_time=2.0
            )

            mock_insert.assert_called_once()
            call_args = mock_insert.call_args
            assert call_args[1]['request_id'] == mock_request_id
            assert call_args[1]['response_code'] == 400
            assert call_args[1]['error_message'] == "Test error"
            assert call_args[1]['payload'] == "test.jpg"
            assert call_args[1]['processing_time'] == 2.0


class TestEarlyRejectionEnvelope:
    """401/403/422 use the unified envelope shape and are logged to OcrKtpLog."""

    _ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}

    def _client(self):
        from fastapi import FastAPI
        from src.api.routes import router, register_exception_handlers
        from src.middleware.add_requestid import RequestIdMiddleware
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)
        app.include_router(router)
        register_exception_handlers(app)
        return TestClient(app)

    def test_wrong_api_key_uses_envelope_and_logs(self, sample_image_bytes):
        with patch.dict("os.environ", {"API_KEY": "secret"}), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
            client = self._client()
            resp = client.post(
                "/v1/ocr_extract",
                files={"file": ("t.jpg", BytesIO(sample_image_bytes), "image/jpeg")},
                headers={"X-API-Key": "wrong"},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 401

    def test_missing_file_uses_envelope_and_logs(self):
        with patch.dict("os.environ", {"API_KEY": "secret"}), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
            client = self._client()
            resp = client.post(
                "/v1/ocr_extract",
                headers={"X-API-Key": "secret"},  # file field omitted
            )
        assert resp.status_code == 422
        body = resp.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 422
