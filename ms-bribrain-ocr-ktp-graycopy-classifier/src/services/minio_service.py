import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from src.core.config import get_config

config = get_config()
from src.core.logging import get_logger

logger = get_logger()

def download_model_minio():
    output_path = Path(config.model.path)

    # 1. Check local file first
    if output_path.exists():
        logger.info(f"File already exists, skipping download: {output_path}")
        return

    # 2. Check if MinIO is configured
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    if not minio_endpoint:
        logger.warning("MINIO_ENDPOINT not configured, skipping model download")
        logger.warning(f"Please ensure model file exists at: {output_path}")
        return

    # 3. Init MinIO client
    client = Minio(
        endpoint=minio_endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.fget_object(
            bucket_name=config.minio.bucket,
            object_name=config.minio.object,
            file_path=str(output_path),
        )
        logger.info(f"Downloaded file to: {output_path}")

    except S3Error as e:
        logger.error(f"MinIO download failed: {e}")
        raise RuntimeError(f"MinIO download failed: {e}") from e