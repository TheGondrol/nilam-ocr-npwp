import os
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from ..core.logging import logger
from ..core.config import config

def download_model_minio():
    output_path = Path(config.get('model.path', "./src/models/recapture_model.pth"))

    # 1. Check local file first
    if output_path.exists():
        logger.info(f"File already exists, skipping download: {output_path}")
        return

    # 2. Init MinIO client
    endpoint = os.getenv("MINIO_ENDPOINT")
    if not endpoint:
        raise RuntimeError("MINIO_ENDPOINT environment variable is not set")
    client = Minio(
        endpoint=endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.fget_object(
            bucket_name=config.get('minio.bucket'),
            object_name=config.get("minio.object"),
            file_path=str(output_path),
        )
        logger.info(f"Downloaded file to: {output_path}")

    except S3Error as e:
        logger.info(f"MinIO download failed: {e}")
        raise RuntimeError(f"MinIO download failed: {e}") from e