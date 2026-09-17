"""
Unit tests for RequestId middleware
"""

import pytest
from unittest.mock import Mock
from fastapi import Request, Response
from starlette.datastructures import Headers

from src.middleware.add_requestid import RequestIdMiddleware
from src.core.logging import request_id_ctx


@pytest.mark.unit
class TestRequestIdMiddleware:
    """Tests for RequestIdMiddleware"""
    
    @pytest.mark.asyncio
    async def test_uses_existing_request_id(self):
        """Test that existing request ID from header is used"""
        middleware = RequestIdMiddleware(app=Mock())
        
        # Mock request with x-request-id header
        request = Mock(spec=Request)
        request.headers = Headers({"x-request-id": "existing-id"})
        
        # Mock response
        response = Mock(spec=Response)
        response.headers = {}
        
        # Mock call_next
        async def mock_call_next(req):
            # Verify request_id is set in context
            assert request_id_ctx.get() == "existing-id"
            return response
        
        result = await middleware.dispatch(request, mock_call_next)
        
        assert result.headers["x-request-id"] == "existing-id"
    
    @pytest.mark.asyncio
    async def test_generates_new_request_id(self):
        """Test that new request ID is generated when not provided"""
        middleware = RequestIdMiddleware(app=Mock())
        
        # Mock request without x-request-id header
        request = Mock(spec=Request)
        request.headers = Headers({})
        
        # Mock response
        response = Mock(spec=Response)
        response.headers = {}
        
        # Track the generated request_id
        generated_id = None
        
        async def mock_call_next(req):
            nonlocal generated_id
            generated_id = request_id_ctx.get()
            return response
        
        result = await middleware.dispatch(request, mock_call_next)
        
        # Verify a UUID was generated
        assert generated_id is not None
        assert len(generated_id) > 0
        assert result.headers["x-request-id"] == generated_id
    
    @pytest.mark.asyncio
    async def test_context_is_reset_after_request(self):
        """Test that context is reset after request processing"""
        middleware = RequestIdMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.headers = Headers({"x-request-id": "test-id"})
        
        response = Mock(spec=Response)
        response.headers = {}
        
        async def mock_call_next(req):
            return response
        
        # Set a different value in context before
        initial_token = request_id_ctx.set("initial-id")
        
        try:
            await middleware.dispatch(request, mock_call_next)
            
            # After dispatch, context should be reset (not 'test-id')
            # Because the middleware resets the token
            assert request_id_ctx.get() == "initial-id"
        finally:
            request_id_ctx.reset(initial_token)
    
    @pytest.mark.asyncio
    async def test_response_includes_request_id_header(self):
        """Test that response includes x-request-id header"""
        middleware = RequestIdMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.headers = Headers({"x-request-id": "header-id"})
        
        response = Mock(spec=Response)
        response.headers = {}
        
        async def mock_call_next(req):
            return response
        
        result = await middleware.dispatch(request, mock_call_next)
        
        assert "x-request-id" in result.headers
        assert result.headers["x-request-id"] == "header-id"
    
    @pytest.mark.asyncio
    async def test_middleware_handles_exception(self):
        """Test that middleware properly handles exceptions"""
        middleware = RequestIdMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.headers = Headers({"x-request-id": "test-id"})
        
        async def mock_call_next_with_error(req):
            raise ValueError("Test error")
        
        # Context should still be reset even if exception occurs
        with pytest.raises(ValueError):
            await middleware.dispatch(request, mock_call_next_with_error)
        
        # Verify context was reset (should be default value "-")
        assert request_id_ctx.get() == "-"
    
    @pytest.mark.asyncio
    async def test_request_id_available_during_request(self):
        """Test that request_id is available in context during request"""
        middleware = RequestIdMiddleware(app=Mock())
        
        request = Mock(spec=Request)
        request.headers = Headers({"x-request-id": "test-123"})
        
        response = Mock(spec=Response)
        response.headers = {}
        
        context_value_during_request = None
        
        async def mock_call_next(req):
            nonlocal context_value_during_request
            context_value_during_request = request_id_ctx.get()
            return response
        
        await middleware.dispatch(request, mock_call_next)
        
        assert context_value_during_request == "test-123"
