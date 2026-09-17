"""
Unit tests for src/services/minio_service module.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from io import BytesIO


class TestMinioService:

    def test_singleton_pattern(self):
        """MinioService returns same instance."""
        from src.services.minio_service import MinioService
        a = MinioService()
        b = MinioService()
        assert a is b

    @pytest.mark.asyncio
    async def test_get_client_missing_endpoint_raises(self):
        """get_client raises RuntimeError when MINIO_ENDPOINT is missing."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        svc._client = None
        svc._bucket_verified = False

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="MINIO_ENDPOINT"):
                await svc.get_client()

    @pytest.mark.asyncio
    async def test_get_client_creates_minio_client(self):
        """get_client creates Minio client with env vars."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        svc._client = None
        svc._bucket_verified = False

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
            "MINIO_SECURE": "false",
        }
        with patch.dict("os.environ", env, clear=False), \
             patch("src.services.minio_service.Minio") as mock_minio:
            mock_minio.return_value = MagicMock()
            client = await svc.get_client()
            assert client is not None
            mock_minio.assert_called_once_with(
                endpoint="localhost:9000",
                access_key="key",
                secret_key="secret",
                secure=False,
            )

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """get_client returns existing client when already created."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        mock_client = MagicMock()
        svc._client = mock_client
        svc._bucket_verified = False

        client = await svc.get_client()
        assert client is mock_client

    @pytest.mark.asyncio
    async def test_ensure_bucket_creates_if_not_exists(self):
        """ensure_bucket creates bucket when it doesn't exist."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        mock_client = AsyncMock()
        mock_client.bucket_exists.return_value = False
        svc._client = mock_client
        svc._bucket_verified = False

        await svc.ensure_bucket("test-bucket")
        mock_client.make_bucket.assert_called_once_with("test-bucket")
        assert svc._bucket_verified is True

    @pytest.mark.asyncio
    async def test_ensure_bucket_skips_when_exists(self):
        """ensure_bucket skips creation when bucket exists."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        mock_client = AsyncMock()
        mock_client.bucket_exists.return_value = True
        svc._client = mock_client
        svc._bucket_verified = False

        await svc.ensure_bucket("test-bucket")
        mock_client.make_bucket.assert_not_called()
        assert svc._bucket_verified is True

    @pytest.mark.asyncio
    async def test_ensure_bucket_handles_s3_error(self):
        """ensure_bucket handles S3Error gracefully."""
        from src.services.minio_service import MinioService
        from miniopy_async.error import S3Error

        svc = MinioService.__new__(MinioService)
        mock_client = AsyncMock()
        mock_client.bucket_exists.side_effect = S3Error("test", "test", "test", "test", "test", MagicMock())
        svc._client = mock_client
        svc._bucket_verified = False

        await svc.ensure_bucket("test-bucket")
        assert svc._bucket_verified is True

    @pytest.mark.asyncio
    async def test_ensure_bucket_skips_when_already_verified(self):
        """ensure_bucket is a no-op if already verified."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        mock_client = AsyncMock()
        svc._client = mock_client
        svc._bucket_verified = True

        await svc.ensure_bucket("test-bucket")
        mock_client.bucket_exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        """close resets client and bucket verified flag."""
        from src.services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        svc._client = MagicMock()
        svc._bucket_verified = True

        await svc.close()
        assert svc._client is None
        assert svc._bucket_verified is False


class TestResizeImage:

    def test_resize_image_sync_small_image(self):
        """Small image is returned at high quality without resize."""
        from PIL import Image
        from src.services.minio_service import _resize_image_sync

        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        small_bytes = buf.getvalue()

        result = _resize_image_sync(small_bytes, max_size=1048576)
        assert len(result) <= 1048576

    def test_resize_image_sync_large_image(self):
        """Large image is compressed to fit max_size."""
        from PIL import Image
        from src.services.minio_service import _resize_image_sync

        img = Image.new("RGB", (4000, 4000), color="blue")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        large_bytes = buf.getvalue()

        result = _resize_image_sync(large_bytes, max_size=50000)
        assert len(result) <= 50000

    def test_resize_image_sync_rgba_conversion(self):
        """RGBA image is converted to RGB before saving as JPEG."""
        from PIL import Image
        from src.services.minio_service import _resize_image_sync

        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        result = _resize_image_sync(png_bytes, max_size=1048576)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_async(self):
        """Async resize delegates to thread pool."""
        from PIL import Image
        from src.services.minio_service import resize_image

        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        small_bytes = buf.getvalue()

        result = await resize_image(small_bytes, max_size=1048576)
        assert len(result) > 0


class TestSaveImage:

    @pytest.mark.asyncio
    async def test_save_image_success(self):
        """save_image uploads to MinIO."""
        from src.services.minio_service import save_image
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        file_bytes = buf.getvalue()

        with patch("src.services.minio_service._minio_service") as mock_svc:
            mock_svc.ensure_bucket = AsyncMock()
            mock_svc.get_client = AsyncMock()
            mock_client = AsyncMock()
            mock_svc.get_client.return_value = mock_client

            result = await save_image(file_bytes, "test_file")
            assert result == "test_file"
            mock_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_image_s3_error(self):
        """save_image raises on S3Error."""
        from src.services.minio_service import save_image
        from miniopy_async.error import S3Error

        with patch("src.services.minio_service._minio_service") as mock_svc:
            mock_svc.ensure_bucket = AsyncMock()
            mock_svc.get_client = AsyncMock()
            mock_client = AsyncMock()
            mock_client.put_object.side_effect = S3Error("test", "test", "test", "test", "test", MagicMock())
            mock_svc.get_client.return_value = mock_client

            with pytest.raises(RuntimeError, match="MinIO save failed"):
                await save_image(b"\xff\xd8\xff\xe0test", "test_file")

    @pytest.mark.asyncio
    async def test_save_image_resizes_large_file(self):
        """save_image resizes image when it exceeds max size."""
        from src.services.minio_service import save_image
        from PIL import Image

        img = Image.new("RGB", (2000, 2000), color="green")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=99)
        large_bytes = buf.getvalue()

        with patch("src.services.minio_service._minio_service") as mock_svc, \
             patch("src.services.minio_service.MAX_SIZE_BYTES", 1000), \
             patch("src.services.minio_service.resize_image", new_callable=AsyncMock, return_value=b"small"):
            mock_svc.ensure_bucket = AsyncMock()
            mock_svc.get_client = AsyncMock()
            mock_client = AsyncMock()
            mock_svc.get_client.return_value = mock_client

            await save_image(large_bytes, "test_file")
            mock_client.put_object.assert_called_once()


class TestCloseMinioService:

    @pytest.mark.asyncio
    async def test_close_minio_service(self):
        """close_minio_service calls close on singleton."""
        from src.services.minio_service import close_minio_service

        with patch("src.services.minio_service._minio_service") as mock_svc:
            mock_svc.close = AsyncMock()
            await close_minio_service()
            mock_svc.close.assert_called_once()
