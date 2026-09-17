"""
Unit tests for src/core/http_client module.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp


class TestHttpClient:

    @pytest.mark.asyncio
    async def test_get_http_client_raises_before_init(self):
        """get_http_client raises RuntimeError if not initialized."""
        import src.core.http_client as mod
        original = mod._http_client
        try:
            mod._http_client = None
            with pytest.raises(RuntimeError, match="HTTP client not initialized"):
                await mod.get_http_client()
        finally:
            mod._http_client = original

    @pytest.mark.asyncio
    async def test_init_http_client_creates_session(self, mock_settings):
        """init_http_client creates an aiohttp.ClientSession."""
        import src.core.http_client as mod
        original = mod._http_client
        try:
            mod._http_client = None
            with patch("src.core.config.get_settings", return_value=mock_settings):
                mock_settings.http_client = MagicMock()
                mock_settings.http_client.limit = 100
                mock_settings.http_client.limit_per_host = 10
                mock_settings.http_client.ttl_dns_cache = 300
                mock_settings.http_client.keepalive_timeout = 30
                await mod.init_http_client()
                assert mod._http_client is not None
                assert isinstance(mod._http_client, aiohttp.ClientSession)
                await mod._http_client.close()
        finally:
            mod._http_client = original

    @pytest.mark.asyncio
    async def test_get_http_client_returns_session_after_init(self, mock_settings):
        """get_http_client returns session after initialization."""
        import src.core.http_client as mod
        original = mod._http_client
        try:
            mod._http_client = None
            with patch("src.core.config.get_settings", return_value=mock_settings):
                mock_settings.http_client = MagicMock()
                mock_settings.http_client.limit = 100
                mock_settings.http_client.limit_per_host = 10
                mock_settings.http_client.ttl_dns_cache = 300
                mock_settings.http_client.keepalive_timeout = 30
                await mod.init_http_client()
                client = await mod.get_http_client()
                assert client is mod._http_client
                await mod._http_client.close()
        finally:
            mod._http_client = original

    @pytest.mark.asyncio
    async def test_close_http_client(self, mock_settings):
        """close_http_client closes session and sets to None."""
        import src.core.http_client as mod
        original = mod._http_client
        try:
            mod._http_client = None
            with patch("src.core.config.get_settings", return_value=mock_settings):
                mock_settings.http_client = MagicMock()
                mock_settings.http_client.limit = 100
                mock_settings.http_client.limit_per_host = 10
                mock_settings.http_client.ttl_dns_cache = 300
                mock_settings.http_client.keepalive_timeout = 30
                await mod.init_http_client()
                assert mod._http_client is not None
                await mod.close_http_client()
                assert mod._http_client is None
        finally:
            mod._http_client = original

    @pytest.mark.asyncio
    async def test_close_http_client_when_already_none(self):
        """close_http_client is a no-op when client is already None."""
        import src.core.http_client as mod
        original = mod._http_client
        try:
            mod._http_client = None
            await mod.close_http_client()
            assert mod._http_client is None
        finally:
            mod._http_client = original
