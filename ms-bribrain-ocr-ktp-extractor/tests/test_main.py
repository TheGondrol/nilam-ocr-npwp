"""Unit tests for src.main module"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


class TestApplicationLifespan:
    """Test cases for application lifespan"""

    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self):
        """Test successful application startup"""
        from src.main import lifespan
        
        mock_app = MagicMock()
        
        with patch('src.main.detect_device', return_value="cpu"):
            async with lifespan(mock_app):
                # Startup code runs here
                pass
            # Cleanup runs after yield

    @pytest.mark.asyncio
    async def test_lifespan_startup_error(self):
        """Test application startup with device detection error"""
        from src.main import lifespan
        
        mock_app = MagicMock()
        
        with patch('src.main.detect_device', side_effect=Exception("Device error")):
            # Should not raise error, just log it
            async with lifespan(mock_app):
                pass

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_cleanup(self):
        """Test that cleanup_ocr is called during shutdown"""
        from src.main import lifespan
        
        mock_app = MagicMock()
        
        with patch('src.main.detect_device', return_value="gpu"):
            with patch('src.main.cleanup_ocr') as mock_cleanup:
                async with lifespan(mock_app):
                    pass
                
                # Verify cleanup was called
                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_error(self):
        """Test handling of cleanup errors during shutdown"""
        from src.main import lifespan
        
        mock_app = MagicMock()
        
        with patch('src.main.detect_device', return_value="cpu"):
            with patch('src.main.cleanup_ocr', side_effect=Exception("Cleanup error")):
                # Should not raise error, just log it
                async with lifespan(mock_app):
                    pass


class TestApplicationEndpoints:
    """Test cases for application endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('src.main.setup_logging'):
            with patch('src.main.lifespan'):
                from src.main import app
                with TestClient(app) as test_client:
                    yield test_client

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "OCR Extract API"
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]

    def test_liveness_returns_200(self, client):
        """Test liveness probe always returns 200"""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "version" in data

    @patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"})
    @patch("src.main.is_ocr_ready", return_value=True)
    def test_readiness_ready(self, mock_ocr, mock_db, client):
        """Test readiness probe when service is ready"""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "not initialised"})
    @patch("src.main.is_ocr_ready", return_value=False)
    def test_readiness_not_ready(self, mock_ocr, mock_db, client):
        """Test readiness probe when service is not ready"""
        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"

    @patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"})
    @patch("src.main.is_ocr_ready", return_value=True)
    def test_health_check_healthy(self, mock_ocr, mock_db, client):
        """Test health check when all components are up"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ocr_loaded"] is True
        assert "checks" in data

    @patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "connection refused"})
    @patch("src.main.is_ocr_ready", return_value=True)
    def test_health_check_degraded(self, mock_ocr, mock_db, client):
        """Test health check when OCR up but DB down"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ocr_loaded"] is True

    @patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"})
    @patch("src.main.is_ocr_ready", return_value=False)
    def test_health_check_unhealthy(self, mock_ocr, mock_db, client):
        """Test health check when OCR is not ready"""
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["ocr_loaded"] is False


class TestApplicationConfiguration:
    """Test cases for application configuration"""

    def test_app_has_cors_middleware(self):
        """Test that CORS middleware is configured"""
        with patch('src.main.setup_logging'):
            with patch('src.main.lifespan'):
                from src.main import app
                
                # Check middleware exists by examining middleware cls attribute
                middleware_classes = [str(m.cls) for m in app.user_middleware]
                cors_found = any('CORSMiddleware' in cls for cls in middleware_classes)
                assert cors_found, f"CORSMiddleware not found in {middleware_classes}"

    def test_app_has_request_id_middleware(self):
        """Test that RequestIdMiddleware is configured"""
        with patch('src.main.setup_logging'):
            with patch('src.main.lifespan'):
                from src.main import app
                
                # Check middleware exists by examining middleware cls attribute
                middleware_classes = [str(m.cls) for m in app.user_middleware]
                request_id_found = any('RequestIdMiddleware' in cls for cls in middleware_classes)
                assert request_id_found, f"RequestIdMiddleware not found in {middleware_classes}"

    def test_app_has_router_included(self):
        """Test that API router is included"""
        with patch('src.main.setup_logging'):
            with patch('src.main.lifespan'):
                from src.main import app
                
                # Check that routes exist
                routes = [route.path for route in app.routes]
                # Should have at least the main endpoints
                assert "/" in routes
                assert "/health" in routes
                assert "/health/live" in routes
                assert "/health/ready" in routes

    def test_app_metadata(self):
        """Test application metadata"""
        with patch('src.main.setup_logging'):
            with patch('src.main.lifespan'):
                from src.main import app
                
                assert app.title == "OCR Extract API"
                assert app.version == "1.0.0"
                assert app.docs_url == "/docs"
                assert app.redoc_url == "/redoc"


class TestMainModule:
    """Test cases for main module execution"""

    def test_version_constant(self):
        """Test that VERSION constant is defined"""
        from src.main import VERSION
        
        assert VERSION == "1.0.0"

    def test_app_object_exists(self):
        """Test that app object is created"""
        with patch('src.main.setup_logging'):
            with patch('src.main.lifespan'):
                from src.main import app
                
                assert app is not None
                from fastapi import FastAPI
                assert isinstance(app, FastAPI)
