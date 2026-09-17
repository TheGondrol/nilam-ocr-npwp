"""
Unit tests for rate limiter middleware.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
import time

from fastapi import Request
from starlette.responses import Response

from src.middleware.rate_limiter import (
    SlidingWindowRateLimiter,
    RateLimitExceeded,
    RateLimitMiddleware
)


class TestSlidingWindowRateLimiter:
    """Tests for sliding window rate limiter."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = SlidingWindowRateLimiter(
            requests_per_minute=60,
            requests_per_second=10,
            burst_size=20
        )
        
        assert limiter.requests_per_minute == 60
        assert limiter.requests_per_second == 10
        assert limiter.burst_size == 20

    def test_get_client_id_from_ip(self):
        """Test client ID extraction from IP."""
        limiter = SlidingWindowRateLimiter()
        
        # Mock request with client
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = None
        
        client_id = limiter._get_client_id(request)
        
        assert client_id == "192.168.1.100"

    def test_get_client_id_from_forwarded_header(self):
        """X-Forwarded-For is honored only when the peer is a trusted proxy,
        and then the right-most hop is used (BUG-10)."""
        limiter = SlidingWindowRateLimiter(trusted_proxies={"192.168.1.1"})

        # Peer is the trusted proxy; XFF carries the real client on the right.
        request = MagicMock(spec=Request)
        request.client.host = "192.168.1.1"
        request.headers.get.return_value = "10.0.0.1, 203.0.113.5"

        client_id = limiter._get_client_id(request)

        assert client_id == "203.0.113.5"

    def test_get_client_id_unknown(self):
        """Test client ID extraction when no info available."""
        limiter = SlidingWindowRateLimiter()
        
        # Mock request with no client info
        request = MagicMock(spec=Request)
        request.client = None
        request.headers.get.return_value = None
        
        client_id = limiter._get_client_id(request)
        
        assert client_id == "unknown"

    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self):
        """Test rate limiting when within limits."""
        limiter = SlidingWindowRateLimiter(
            requests_per_minute=60,
            requests_per_second=10
        )
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = None
        
        # First request should pass
        allowed, retry_after = limiter.is_allowed(request)
        assert allowed is True
        assert retry_after is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceed_per_second(self):
        """Test rate limiting when exceeding per-second limit."""
        limiter = SlidingWindowRateLimiter(
            requests_per_minute=100,
            requests_per_second=2  # Very low limit for testing
        )
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = None
        
        # First 2 requests should pass
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True
        
        # Third request in same second should fail
        allowed, retry_after = limiter.is_allowed(request)
        assert allowed is False
        assert retry_after == 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceed_per_minute(self):
        """Test rate limiting when exceeding per-minute limit."""
        limiter = SlidingWindowRateLimiter(
            requests_per_minute=3,  # Very low limit for testing
            requests_per_second=10
        )
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = None
        
        # First 3 requests should pass
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True
        await asyncio.sleep(0.1)  # Small delay to avoid per-second limit
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True
        await asyncio.sleep(0.1)
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True

        # Fourth request should fail
        await asyncio.sleep(0.1)
        allowed, retry_after = limiter.is_allowed(request)
        assert allowed is False
        assert retry_after is not None

    def test_cleanup_old_entries(self):
        """Test cleanup of old request entries."""
        limiter = SlidingWindowRateLimiter()
        
        # Add old timestamps
        current_time = time.time()
        old_time = current_time - 120  # 2 minutes ago
        
        limiter._requests["test_client"] = [old_time, current_time]
        limiter._last_cleanup = 0  # Force cleanup
        
        limiter._cleanup_old_entries(current_time)
        
        # Old timestamp should be removed
        assert len(limiter._requests["test_client"]) == 1
        assert limiter._requests["test_client"][0] == current_time


class TestRateLimitMiddleware:
    """Tests for rate limit middleware."""

    @pytest.mark.asyncio
    async def test_middleware_allows_request(self):
        """Test middleware allows request within rate limit."""
        app = MagicMock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=100)
        middleware = RateLimitMiddleware(app, limiter)
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = None
        request.url.path = "/v1/ppocr"
        
        async def mock_call_next(req):
            return Response(content="OK", status_code=200)
        
        with patch.object(limiter, "is_allowed", return_value=(True, None)):
            response = await middleware.dispatch(request, mock_call_next)
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_middleware_blocks_request(self):
        """Test middleware blocks request exceeding rate limit."""
        app = MagicMock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=1)
        middleware = RateLimitMiddleware(app, limiter)
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = None
        request.url.path = "/v1/ppocr"
        
        async def mock_call_next(req):
            return Response(content="OK", status_code=200)
        
        with patch.object(limiter, "is_allowed", return_value=(False, 60)):
            response = await middleware.dispatch(request, mock_call_next)
            assert response.status_code == 429
            body = response.body
            assert isinstance(body, bytes)
            assert "RATE_LIMIT_EXCEEDED" in body.decode()

    @pytest.mark.asyncio
    async def test_middleware_skips_health_check(self):
        """Test middleware skips rate limiting for health check."""
        app = MagicMock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=1)
        middleware = RateLimitMiddleware(app, limiter)
        
        request = MagicMock(spec=Request)
        request.url.path = "/health"
        
        async def mock_call_next(req):
            return Response(content="OK", status_code=200)
        
        # Should not call rate limiter for health endpoint
        response = await middleware.dispatch(request, mock_call_next)
        
        assert response.status_code == 200


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_exception_initialization(self):
        """Test RateLimitExceeded exception initialization."""
        exc = RateLimitExceeded(retry_after=60)
        
        assert exc.status_code == 429
        assert exc.retry_after == 60
        detail = exc.detail
        assert isinstance(detail, dict)
        assert detail["error_code"] == "RATE_LIMIT_EXCEEDED"

    def test_exception_default_retry_after(self):
        """Test RateLimitExceeded with default retry_after."""
        exc = RateLimitExceeded()

        assert exc.retry_after == 60


class TestRateLimiterEdgeCases:

    def test_cleanup_removes_empty_clients(self):
        """Cleanup removes clients with all expired timestamps."""
        limiter = SlidingWindowRateLimiter()
        current_time = time.time()
        old_time = current_time - 120

        limiter._requests["expired_client"] = [old_time]
        limiter._requests["active_client"] = [current_time]
        limiter._last_cleanup = 0

        limiter._cleanup_old_entries(current_time)

        assert "expired_client" not in limiter._requests
        assert "active_client" in limiter._requests
        assert len(limiter._requests["active_client"]) == 1

    def test_cleanup_skips_when_recent(self):
        """Cleanup is skipped when interval hasn't elapsed."""
        limiter = SlidingWindowRateLimiter(cleanup_interval=300)
        current_time = time.time()
        limiter._last_cleanup = current_time - 10
        limiter._requests["client"] = [current_time - 120]

        limiter._cleanup_old_entries(current_time)

        assert "client" in limiter._requests
        assert len(limiter._requests["client"]) == 1
