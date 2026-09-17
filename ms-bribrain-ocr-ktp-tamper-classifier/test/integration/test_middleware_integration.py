"""
Integration tests for middleware components
Tests middleware interaction with API
"""

from typing import Any, cast

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def test_app():
    """Create a minimal FastAPI app with middleware for testing"""
    from fastapi import FastAPI, Request
    from src.middleware.add_requestid import RequestIdMiddleware
    from src.core.logging import request_id_ctx
    
    app = FastAPI()
    app.add_middleware(cast(Any, RequestIdMiddleware))
    
    @app.get("/test")
    async def test_endpoint(request: Request):
        return {
            "message": "test",
            "request_id": request_id_ctx.get()
        }
    
    return app


@pytest.mark.integration
class TestRequestIdMiddlewareIntegration:
    """Integration tests for RequestId middleware"""
    
    def test_request_id_flow_through_middleware(self, test_app):
        """Test request ID flows through middleware to endpoint"""
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"x-request-id": "test-123"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "test-123"
        assert response.headers["x-request-id"] == "test-123"
    
    def test_request_id_generation_when_not_provided(self, test_app):
        """Test request ID is generated when not provided"""
        client = TestClient(test_app)
        
        response = client.get("/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] != "-"  # Should have generated ID
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0
    
    def test_different_requests_have_different_ids(self, test_app):
        """Test that different requests get different IDs"""
        client = TestClient(test_app)
        
        response1 = client.get("/test")
        response2 = client.get("/test")
        
        id1 = response1.json()["request_id"]
        id2 = response2.json()["request_id"]
        
        assert id1 != id2


@pytest.mark.integration
class TestMiddlewareStackIntegration:
    """Integration tests for full middleware stack"""
    
    def test_cors_middleware_integration(self):
        """Test CORS middleware is properly integrated"""
        from src.main import app
        
        # Mock services to avoid initialization errors
        with patch("src.services.tamper_detection.get_tamper_service") as mock_service:
            mock_service.return_value = Mock(is_ready=Mock(return_value=True), device="cpu")
            
            client = TestClient(app)
            response = client.get("/")
            
            assert response.status_code == 200
            # CORS headers should be present (depends on config)
    
    def test_multiple_middleware_interaction(self):
        """Test multiple middleware work together"""
        from src.main import app
        
        with patch("src.services.tamper_detection.get_tamper_service") as mock_service:
            mock_service.return_value = Mock(is_ready=Mock(return_value=True), device="cpu")
            
            client = TestClient(app)
            
            # Make request with request ID
            response = client.get("/", headers={"x-request-id": "integration-test"})
            
            assert response.status_code == 200
            assert "x-request-id" in response.headers
            assert response.headers["x-request-id"] == "integration-test"


@pytest.mark.integration
class TestLoggingIntegration:
    """Integration tests for logging with middleware"""
    
    def test_logging_with_request_id(self, test_app):
        """Test that logging captures request ID from middleware"""
        from src.core.logging import get_logger
        
        get_logger(__name__)
        client = TestClient(test_app)
        
        # The middleware should set request_id_ctx
        response = client.get("/test", headers={"x-request-id": "log-test-123"})
        
        assert response.status_code == 200
        # Request ID should have been available during request processing
        assert response.json()["request_id"] == "log-test-123"
