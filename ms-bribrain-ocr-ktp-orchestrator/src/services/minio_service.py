import os
import asyncio
from io import BytesIO
from typing import Optional

from miniopy_async import Minio
from miniopy_async.error import S3Error
from PIL import Image

from src.core.logging import get_logger
from src.core.config import config

logger = get_logger(__name__)

# Connection pool settings for high RPS
MAX_SIZE_BYTES = config.get('minio.max_size', 1048576)  # 1MB in bytes
POOL_SIZE = config.get('minio.pool_size', 100)
POOL_TIMEOUT = config.get('minio.pool_timeout', 30)
TTL_DNS_CACHE = config.get('minio.ttl_dns_cache', 300)

# Image processing settings
IMAGE_FORMAT = config.get('minio.image.format', 'JPEG')
IMAGE_CONTENT_TYPE = config.get('minio.image.content_type', 'image/jpeg')
INITIAL_QUALITY = config.get('minio.image.initial_quality', 95)
MIN_QUALITY = config.get('minio.image.min_quality', 20)
QUALITY_STEP = config.get('minio.image.quality_step', 10)
SCALE_STEP = config.get('minio.image.scale_step', 0.1)
MIN_SCALE = config.get('minio.image.min_scale', 0.1)
PATH = config.get('minio.path', 'ocr_ktp_image_log')


class MinioService:
    """
    Async MinIO service for high RPS.
    Uses singleton pattern to reuse client across requests.
    miniopy_async manages its own HTTP sessions internally.
    """
    
    _instance: Optional['MinioService'] = None
    _client: Optional[Minio] = None
    _bucket_verified: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_client(self) -> Minio:
        """Get or create async MinIO client."""
        if self._client is None:
            endpoint = os.getenv("MINIO_ENDPOINT")
            if not endpoint:
                raise RuntimeError("MINIO_ENDPOINT environment variable is required")
            self._client = Minio(
                endpoint=endpoint,
                access_key=os.getenv("MINIO_ACCESS_KEY"),
                secret_key=os.getenv("MINIO_SECRET_KEY"),
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            )
        return self._client
    
    async def ensure_bucket(self, bucket_name: str) -> None:
        """Ensure bucket exists (cached check)."""
        if not self._bucket_verified:
            client = await self.get_client()
            try:
                if not await client.bucket_exists(bucket_name):
                    await client.make_bucket(bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
            except S3Error as e:
                # Bucket might already exist or we don't have permission to create
                logger.warning(f"Bucket check/create warning: {e}")
            self._bucket_verified = True
    
    async def close(self) -> None:
        """Cleanup resources."""
        self._client = None
        self._bucket_verified = False


# Global singleton instance
_minio_service = MinioService()


def _resize_image_sync(file_bytes: bytes, max_size: int = MAX_SIZE_BYTES) -> bytes:
    """
    Synchronous image resize (CPU-bound operation).
    Resize image to ensure it's under max_size (default 1MB).
    Uses binary search on JPEG quality for fast convergence (3-4 iterations).
    Falls back to dimension scaling if quality alone isn't enough.
    """
    # Open image from bytes
    img = Image.open(BytesIO(file_bytes))
    original_size = img.size
    
    # Convert to RGB only if necessary (for PNG with transparency, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Reuse buffer for iterations
    output_buffer = BytesIO()
    
    # Binary search on quality parameter (much faster than linear)
    low_quality = MIN_QUALITY
    high_quality = INITIAL_QUALITY
    best_quality = high_quality
    best_result = None
    
    # Phase 1: Binary search on quality at original dimensions
    while low_quality <= high_quality:
        mid_quality = (low_quality + high_quality) // 2
        
        # Reset buffer
        output_buffer.seek(0)
        output_buffer.truncate()
        
        # Encode with mid quality
        img.save(output_buffer, format=IMAGE_FORMAT, quality=mid_quality, optimize=True)
        current_size = output_buffer.tell()
        
        if current_size <= max_size:
            # Found a valid quality, try higher
            best_quality = mid_quality
            best_result = output_buffer.getvalue()
            low_quality = mid_quality + 1
        else:
            # Too large, try lower quality
            high_quality = mid_quality - 1
    
    # If binary search found a solution, return it
    if best_result is not None:
        logger.info(f"Image resized: {len(file_bytes)} -> {len(best_result)} bytes (quality={best_quality})")
        return best_result
    
    # Phase 2: Quality alone wasn't enough, scale dimensions
    scale_factor = 0.9  # Start with 90% of original
    
    while scale_factor >= MIN_SCALE:
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)
        scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Binary search on quality for this scale
        low_quality = MIN_QUALITY
        high_quality = INITIAL_QUALITY
        
        while low_quality <= high_quality:
            mid_quality = (low_quality + high_quality) // 2
            
            output_buffer.seek(0)
            output_buffer.truncate()
            
            scaled_img.save(output_buffer, format=IMAGE_FORMAT, quality=mid_quality, optimize=True)
            current_size = output_buffer.tell()
            
            if current_size <= max_size:
                best_quality = mid_quality
                best_result = output_buffer.getvalue()
                low_quality = mid_quality + 1
            else:
                high_quality = mid_quality - 1
        
        if best_result is not None:
            logger.info(f"Image resized: {len(file_bytes)} -> {len(best_result)} bytes (quality={best_quality}, scale={scale_factor:.2f})")
            return best_result
        
        # Reduce scale and try again
        scale_factor -= SCALE_STEP
    
    # Last resort: return smallest possible
    logger.warning(f"Could not reduce image to {max_size} bytes, returning smallest version")
    output_buffer.seek(0)
    output_buffer.truncate()
    img.resize((int(original_size[0] * MIN_SCALE), int(original_size[1] * MIN_SCALE)), Image.Resampling.LANCZOS).save(
        output_buffer, format=IMAGE_FORMAT, quality=MIN_QUALITY, optimize=True
    )
    return output_buffer.getvalue()


async def resize_image(file_bytes: bytes, max_size: int = MAX_SIZE_BYTES) -> bytes:
    """
    Async wrapper for image resize.
    Runs CPU-bound operation in thread pool to avoid blocking event loop.
    """
    return await asyncio.to_thread(_resize_image_sync, file_bytes, max_size)


async def save_image(file_bytes: bytes, filename: str) -> str:
    """
    Async save image to MinIO. If image size > 1MB, resize it first.
    Uses connection pooling for high RPS performance.
    
    Args:
        file_bytes: Image data as bytes
        filename: Object name/path in MinIO bucket
        
    Returns:
        The object name in MinIO
    """
    # 1. Check file size and resize if needed (runs in thread pool)
    if len(file_bytes) > MAX_SIZE_BYTES:
        logger.info(f"Image size ({len(file_bytes)} bytes) exceeds {MAX_SIZE_BYTES} bytes. Resizing...")
        file_bytes = await resize_image(file_bytes)
    
    # 2. Get bucket name from config
    bucket_name = config.get('minio.bucket')
    
    # 3. Ensure bucket exists (cached)
    await _minio_service.ensure_bucket(bucket_name)
    
    # 4. Get client (reuses connection pool)
    client = await _minio_service.get_client()

    try:
        # 5. Upload to MinIO using BytesIO
        data = BytesIO(file_bytes)
        data_length = len(file_bytes)
        
        await client.put_object(
            bucket_name=bucket_name,
            object_name=f"{PATH}/{filename}.jpg",
            data=data,
            length=data_length,
            content_type=IMAGE_CONTENT_TYPE
        )
        logger.info(f"Saved image to MinIO: {bucket_name}/{filename} ({data_length} bytes)")
        return filename

    except S3Error as e:
        logger.error(f"MinIO save failed: {e}")
        raise RuntimeError(f"MinIO save failed: {e}") from e


async def close_minio_service() -> None:
    """
    Cleanup function to close MinIO connections.
    Call this on application shutdown.
    """
    await _minio_service.close()
    logger.info("MinIO service connections closed")