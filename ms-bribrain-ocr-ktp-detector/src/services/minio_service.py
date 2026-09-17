import os
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from src.core.logging import logger
from src.core.config import config


def download_model_minio():
    # Use absolute path to ensure consistent resolution in Docker container
    config_path = config.get("model.path", "./src/models/ktp_detection_model.pt")

    # Convert to absolute path relative to /app if not already absolute
    # if not os.path.isabs(config_path):
    #     # In Docker, WORKDIR is /app
    #     output_path = Path("/app") / config_path.lstrip("./")
    # else:
    output_path = Path(config_path)

    logger.info(f"Model download path (resolved): {output_path}")

    # 1. Check local file first
    if output_path.exists():
        logger.info(f"File already exists, skipping download: {output_path}")
        return

    # 2. Init MinIO client
    endpoint = os.getenv("MINIO_ENDPOINT")
    if not endpoint:
        raise RuntimeError("MINIO_ENDPOINT environment variable is required")

    client = Minio(
        endpoint=endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )

    # Ensure directory exists (using absolute path)
    model_dir = output_path.parent
    logger.info(f"Creating directory: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Verify directory was created successfully
    if not model_dir.exists():
        raise RuntimeError(f"Failed to create directory: {model_dir}")
    if not model_dir.is_dir():
        raise RuntimeError(f"Path exists but is not a directory: {model_dir}")

    # Log directory contents before download
    logger.info(f"Directory exists: {model_dir.exists()}, is_dir: {model_dir.is_dir()}")
    logger.info(f"Target file path: {output_path}")
    logger.info(f"Target file path (str): {str(output_path)}")

    try:
        bucket_name = config.get("minio.bucket")
        object_name = config.get("minio.object")
        logger.info(
            f"Downloading from MinIO: bucket={bucket_name}, object={object_name}"
        )

        # Use get_object instead of fget_object to avoid .part.minio temp file issues
        # get_object returns a urllib3.response.HTTPResponse object
        response = client.get_object(
            bucket_name=bucket_name,
            object_name=object_name,
        )

        try:
            # Write the content directly to the target file
            with open(str(output_path), "wb") as file:
                # Read in chunks to handle large files efficiently
                while True:
                    chunk = response.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    file.write(chunk)
            logger.info(f"Downloaded file to: {output_path}")
        finally:
            response.close()
            response.release_conn()

    except S3Error as e:
        logger.error(f"MinIO download failed: {e}")
        raise RuntimeError(f"MinIO download failed: {e}") from e
    except IOError as e:
        logger.error(f"Failed to write file to {output_path}: {e}")
        raise RuntimeError(f"Failed to write file: {e}") from e
