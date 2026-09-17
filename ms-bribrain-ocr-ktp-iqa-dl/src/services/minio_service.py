"""MinIO service for downloading model weights."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from minio import Minio

from src.core.config import config

logger = logging.getLogger(__name__)


def download_model_minio() -> None:
    """
    Download model from MinIO if not present locally.

    Checks if model file exists at configured path. If not, downloads
    from MinIO using credentials from environment variables.

    Environment Variables:
        MINIO_ENDPOINT: MinIO server endpoint
        MINIO_ACCESS_KEY: Access key for authentication
        MINIO_SECRET_KEY: Secret key for authentication
        MINIO_SECURE: Whether to use HTTPS (true/false)

    Raises:
        Exception: If download fails
    """
    model_path = Path(config.model_path)

    # Check if model already exists
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        return

    logger.info(f"Model not found at {model_path}, attempting download from MinIO")

    # Get MinIO configuration
    bucket_name = config.get("minio.bucket")
    object_name = config.get("minio.object")

    if not bucket_name or not object_name:
        raise ValueError("MinIO bucket and object must be configured in config.yaml")

    # Get credentials from environment
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "true").lower() == "true"

    if not endpoint or not access_key or not secret_key:
        raise ValueError(
            "MinIO credentials not found in environment. "
            "Set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
        )

    try:
        # Initialize MinIO client
        logger.info(f"Connecting to MinIO at {endpoint}")
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        # Create directory if needed
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download model
        logger.info(f"Downloading {object_name} from bucket {bucket_name}")
        client.fget_object(bucket_name, object_name, str(model_path))

        logger.info(f"Model downloaded successfully to {model_path}")

    except Exception as e:
        logger.error(f"Failed to download model from MinIO: {e}")
        raise
