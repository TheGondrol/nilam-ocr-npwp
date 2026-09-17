"""
Unit tests for middleware.add_requestid module
"""
from unittest.mock import MagicMock, patch
import pytest
from fastapi import Request, Response
from starlette.datastructures import Headers

from src.middleware.add_requestid import RequestIdMiddleware


@pytest.mark.unit
@pytest.mark.asyncio
class TestRequestIdMiddleware:
    """Test RequestIdMiddleware"""
    
    async def test_middleware_with_existing_request_id(self):
        """Test middleware when request has x-request-id header"""
        middleware = RequestIdMiddleware(app=MagicMock())
        
        # Mock request with x-request-id
        mock_request = MagicMock(spec=Request)
        mock_request.headers = Headers({"x-request-id": "existing-id-123"})
        
        # Mock response
        mock_response = Response(content="test", status_code=200)
        
        # Mock call_next
        async def call_next(request):
            return mock_response
        
        with patch('src.middleware.add_requestid.request_id_ctx') as mock_ctx:
            mock_token = MagicMock()
            mock_ctx.set.return_value = mock_token
            mock_ctx.get.return_value = "existing-id-123"
            
            response = await middleware.dispatch(mock_request, call_next)
            
            # Should set context with existing ID
            mock_ctx.set.assert_called_once_with("existing-id-123")
            
            # Should reset context
            mock_ctx.reset.assert_called_once_with(mock_token)
            
            # Response should have x-request-id header
            assert "x-request-id" in response.headers
            assert response.headers["x-request-id"] == "existing-id-123"
    
    async def test_middleware_without_request_id(self):
        """Test middleware when request has no x-request-id header"""
        middleware = RequestIdMiddleware(app=MagicMock())
        
        # Mock request without x-request-id
        mock_request = MagicMock(spec=Request)
        mock_request.headers = Headers({})
        
        # Mock response
        mock_response = Response(content="test", status_code=200)
        
        # Mock call_next
        async def call_next(request):
            return mock_response
        
        with patch('src.middleware.add_requestid.request_id_ctx') as mock_ctx:
            with patch('src.middleware.add_requestid.uuid') as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "generated-id-456"
                mock_token = MagicMock()
                mock_ctx.set.return_value = mock_token
                
                response = await middleware.dispatch(mock_request, call_next)
                
                # Should generate and set new ID
                mock_ctx.set.assert_called_once()
                call_args = mock_ctx.set.call_args[0][0]
                assert isinstance(call_args, str)
                
                # Should reset context
                mock_ctx.reset.assert_called_once_with(mock_token)
                
                # Response should have x-request-id header
                assert "x-request-id" in response.headers
    
    async def test_middleware_context_cleanup_on_exception(self):
        """Test middleware cleans up context even on exception"""
        middleware = RequestIdMiddleware(app=MagicMock())
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.headers = Headers({"x-request-id": "test-id"})
        
        # Mock call_next that raises exception
        async def call_next(request):
            raise Exception("Test error")
        
        with patch('src.middleware.add_requestid.request_id_ctx') as mock_ctx:
            mock_token = MagicMock()
            mock_ctx.set.return_value = mock_token
            
            with pytest.raises(Exception, match="Test error"):
                await middleware.dispatch(mock_request, call_next)
            
            # Should still reset context
            mock_ctx.reset.assert_called_once_with(mock_token)
    
    async def test_middleware_preserves_response_status(self):
        """Test middleware preserves response status code"""
        middleware = RequestIdMiddleware(app=MagicMock())
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.headers = Headers({"x-request-id": "test-id"})
        
        # Mock response with custom status
        mock_response = Response(content="error", status_code=404)
        
        async def call_next(request):
            return mock_response
        
        with patch('src.middleware.add_requestid.request_id_ctx') as mock_ctx:
            mock_token = MagicMock()
            mock_ctx.set.return_value = mock_token
            
            response = await middleware.dispatch(mock_request, call_next)
            
            # Should preserve status code
            assert response.status_code == 404
    
    async def test_middleware_preserves_response_content(self):
        """Test middleware preserves response content"""
        middleware = RequestIdMiddleware(app=MagicMock())
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.headers = Headers({"x-request-id": "test-id"})
        
        # Mock response with content
        expected_content = b"test response content"
        mock_response = Response(content=expected_content, status_code=200)
        
        async def call_next(request):
            return mock_response
        
        with patch('src.middleware.add_requestid.request_id_ctx') as mock_ctx:
            mock_token = MagicMock()
            mock_ctx.set.return_value = mock_token
            
            response = await middleware.dispatch(mock_request, call_next)
            
            # Should preserve content
            assert response.body == expected_content
    
    async def test_middleware_uuid_generation(self):
        """Test that middleware generates valid UUID when no request ID"""
        middleware = RequestIdMiddleware(app=MagicMock())
        
        # Mock request without x-request-id
        mock_request = MagicMock(spec=Request)
        mock_request.headers = Headers({})
        
        mock_response = Response(content="test", status_code=200)
        
        async def call_next(request):
            return mock_response
        
        with patch('src.middleware.add_requestid.request_id_ctx') as mock_ctx:
            mock_token = MagicMock()
            mock_ctx.set.return_value = mock_token
            
            await middleware.dispatch(mock_request, call_next)
            
            # Should have called set with some string value
            mock_ctx.set.assert_called_once()
            request_id = mock_ctx.set.call_args[0][0]
            assert isinstance(request_id, str)
            assert len(request_id) > 0
