"""
Shared HTTP client module.

Provides a global aiohttp.ClientSession for connection pooling and reuse.
"""

import aiohttp
from typing import Optional

_http_client: Optional[aiohttp.ClientSession] = None


async def get_http_client() -> aiohttp.ClientSession:
    """
    Get the shared HTTP client session.
    
    Returns:
        The global aiohttp.ClientSession instance.
        
    Raises:
        RuntimeError: If the HTTP client hasn't been initialized.
    """
    if _http_client is None:
        raise RuntimeError(
            "HTTP client not initialized. Ensure the FastAPI lifespan context is set up."
        )
    return _http_client


async def init_http_client() -> None:
    """
    Initialize the shared HTTP client session.

    Creates a new aiohttp.ClientSession with connection pooling enabled.
    Should be called during application startup.
    """
    global _http_client

    from src.core.config import get_settings
    http_config = get_settings().http_client

    # Configure connector for connection pooling
    connector = aiohttp.TCPConnector(
        limit=http_config.limit,
        limit_per_host=http_config.limit_per_host,
        ttl_dns_cache=http_config.ttl_dns_cache,
        keepalive_timeout=http_config.keepalive_timeout,
    )

    _http_client = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=None),  # Per-request timeouts handled individually
    )


async def close_http_client() -> None:
    """
    Close the shared HTTP client session.
    
    Should be called during application shutdown to properly close all connections.
    """
    global _http_client
    
    if _http_client is not None:
        await _http_client.close()
        _http_client = None
