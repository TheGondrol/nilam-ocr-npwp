"""
Rate limiting middleware for FastAPI.
Implements a simple sliding window rate limiter.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Callable, Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

import json

from src.core.logging import get_logger

logger = get_logger(__name__)


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=429,
            detail={
                "status_code": 429,
                "status_desc": "Too Many Requests",
                "message": "Too many requests. Please try again later.",
                "data": "",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "errors": {"retry_after": retry_after},
                "request_id": None,
            }
        )
        self.retry_after = retry_after


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation.
    
    Tracks request timestamps per client and rejects requests
    that exceed the configured rate limit.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_second: int = 10,
        burst_size: int = 20,
        cleanup_interval: int = 60,
        trusted_proxies: Optional[set] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per client
            requests_per_second: Maximum requests allowed per second per client
            burst_size: Maximum burst of requests allowed
            cleanup_interval: Interval in seconds for cleaning up expired request history
            trusted_proxies: IPs of trusted reverse proxies/LBs whose
                X-Forwarded-For header may be used to identify the client.
                Empty by default, so untrusted XFF is ignored.
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        # Trusted-proxy allowlist for X-Forwarded-For (BUG-10).
        self.trusted_proxies = set(trusted_proxies or [])

        # Store request timestamps per client: {client_id: [timestamps]}
        self._requests: dict = defaultdict(list)
        self._lock = Lock()

        # Cleanup old entries periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request.

        X-Forwarded-For is attacker-controlled, so it is only honored when the
        immediate peer is a configured trusted proxy; otherwise a client could
        rotate the header to mint a fresh rate-limit bucket per request and
        bypass the per-IP limit (BUG-10). When trusted, take the right-most hop
        (the address the trusted proxy actually observed).
        """
        peer = request.client.host if request.client else "unknown"
        if peer in self.trusted_proxies:
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[-1].strip()
        return peer
    
    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove expired request timestamps."""
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            cutoff = current_time - 60  # Keep only last minute
            clients_to_remove = []
            
            for client_id, timestamps in self._requests.items():
                # Remove old timestamps
                self._requests[client_id] = [
                    ts for ts in timestamps if ts > cutoff
                ]
                # Mark empty clients for removal
                if not self._requests[client_id]:
                    clients_to_remove.append(client_id)
            
            for client_id in clients_to_remove:
                del self._requests[client_id]
            
            self._last_cleanup = current_time
    
    def is_allowed(self, request: Request) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            request: The incoming request
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        current_time = time.time()
        client_id = self._get_client_id(request)
        
        # Periodic cleanup
        self._cleanup_old_entries(current_time)
        
        with self._lock:
            timestamps = self._requests[client_id]
            
            # Calculate requests in last second
            second_ago = current_time - 1
            requests_last_second = sum(1 for ts in timestamps if ts > second_ago)
            
            # Calculate requests in last minute
            minute_ago = current_time - 60
            requests_last_minute = sum(1 for ts in timestamps if ts > minute_ago)
            
            # Check burst limit (per second)
            if requests_last_second >= self.requests_per_second:
                logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{requests_last_second} req/s (limit: {self.requests_per_second})"
                )
                return False, 1
            
            # Check per-minute limit
            if requests_last_minute >= self.requests_per_minute:
                # Calculate when client can retry
                oldest_in_window = min(ts for ts in timestamps if ts > minute_ago)
                retry_after = int(60 - (current_time - oldest_in_window)) + 1
                
                logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{requests_last_minute} req/min (limit: {self.requests_per_minute})"
                )
                return False, retry_after
            
            # Request allowed - record timestamp
            self._requests[client_id].append(current_time)
            
            # Clean up old timestamps for this client
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > minute_ago
            ]
            
            return True, None
    
    def get_stats(self, request: Request) -> dict:
        """Get rate limit stats for a client."""
        current_time = time.time()
        client_id = self._get_client_id(request)
        
        with self._lock:
            timestamps = self._requests.get(client_id, [])
            minute_ago = current_time - 60
            second_ago = current_time - 1
            
            return {
                "client_id": client_id,
                "requests_last_minute": sum(1 for ts in timestamps if ts > minute_ago),
                "requests_last_second": sum(1 for ts in timestamps if ts > second_ago),
                "limit_per_minute": self.requests_per_minute,
                "limit_per_second": self.requests_per_second
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    """
    
    def __init__(
        self,
        app,
        limiter: SlidingWindowRateLimiter,
        exclude_paths: Optional[list[str]] = None
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            limiter: Rate limiter instance
            exclude_paths: Paths to exclude from rate limiting (e.g., /health)
        """
        super().__init__(app)
        self.limiter = limiter
        self.exclude_paths = exclude_paths or ["/health", "/", "/docs", "/openapi.json", "/redoc"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiter."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Check rate limit
        is_allowed, retry_after = self.limiter.is_allowed(request)
        
        if not is_allowed:
            return Response(
                content=json.dumps({
                    "status_code": 429,
                    "status_desc": "Too Many Requests",
                    "message": "Too many requests. Please try again later.",
                    "data": "",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "errors": {"retry_after": retry_after},
                    "request_id": None,
                }),
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.limiter.requests_per_minute),
                    "Content-Type": "application/json"
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        stats = self.limiter.get_stats(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.limiter.requests_per_minute - stats["requests_last_minute"])
        )
        
        return response
