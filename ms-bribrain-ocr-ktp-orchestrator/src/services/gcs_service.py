"""
GCS (Google Cloud Storage) Service Module

Handles GCS credential management, client lifecycle, image uploads,
and model downloads.

All blocking I/O from the synchronous ``google-cloud-storage`` library
is wrapped with ``asyncio.to_thread()`` to avoid blocking the event loop.
"""

import asyncio
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import google.auth.transport.requests
import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.oauth2 import service_account

from ..core.config import config
from ..core.logging import get_logger
from ..services.minio_service import resize_image

logger = get_logger(__name__)

# ============================================================================
# Credentials helpers (sync — called rarely, invoked via to_thread)
# ============================================================================

_GCS_SCOPES = ["https://www.googleapis.com/auth/devstorage.read_write"]


def _load_credentials():
    """Load GCS credentials, preferring Application Default Credentials (e.g. the
    GCE VM's attached service account) and falling back to the inline SA_* env
    vars. ``GCS_AUTH_MODE`` may force a source: auto (default) | adc | sa_env.
    Returns a ``(credentials, project)`` tuple.
    """
    mode = os.getenv("GCS_AUTH_MODE", "auto").strip().lower()
    if mode != "sa_env":
        try:
            creds, project = google.auth.default(scopes=_GCS_SCOPES)
            logger.info("GCS auth: using Application Default Credentials")
            return creds, project
        except DefaultCredentialsError:
            if mode == "adc":
                raise
            logger.info(
                "GCS auth: ADC unavailable, falling back to SA_* env-var credentials"
            )
    creds = service_account.Credentials.from_service_account_info(
        _build_credentials_info(), scopes=_GCS_SCOPES
    )
    return creds, os.environ.get("SA_PROJECT_ID")


def _build_credentials_info() -> dict:
    """
    Build service account credentials info from environment variables.

    Returns:
        dict suitable for ``Credentials.from_service_account_info``.

    Raises:
        KeyError: If a required environment variable is missing.
    """
    return {
        "type": os.environ["SA_TYPE"],
        "project_id": os.environ["SA_PROJECT_ID"],
        "private_key": os.environ["SA_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": os.environ["SA_CLIENT_MAIL"],
        "client_id": os.environ["SA_CLIENT_ID"],
        "auth_uri": os.environ["SA_AUTH_URI"],
        "token_uri": os.environ["SA_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["SA_AUTH_PROVIDER"],
        "client_x509_cert_url": os.environ["SA_CERT_URL"],
    }


def _parse_version(filename: str) -> tuple:
    """
    Extract a version tuple from a filename like ``graycopy_model_v1.1.pth``.

    Returns a tuple of ints, e.g. ``(1, 1)`` so that versions can be
    compared with normal tuple ordering.
    Returns ``(0,)`` if no version is found.
    """
    match = re.search(r"_v(\d+(?:\.\d+)*)", filename)
    if match:
        return tuple(int(p) for p in match.group(1).split("."))
    return (0,)


# ============================================================================
# GCS Service (async singleton)
# ============================================================================


class GcsService:
    """
    Async GCS service for high RPS.
    Uses singleton pattern to reuse client across requests.
    Wraps synchronous google-cloud-storage calls with asyncio.to_thread().
    """

    _instance: Optional['GcsService'] = None
    _credentials: Optional[service_account.Credentials] = None
    _client = None  # google.cloud.storage.Client
    _project: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_credentials_sync(self) -> service_account.Credentials:
        """Get or create/refresh credentials (synchronous)."""
        if self._credentials is None:
            try:
                self._credentials, self._project = _load_credentials()
                logger.debug("Created GCS credentials")
            except Exception as exc:
                logger.error("Failed to create GCS credentials: %s", exc, exc_info=True)
                raise RuntimeError(f"Failed to create GCS credentials: {exc}") from exc

        if not self._credentials.valid:
            try:
                session = requests.Session()
                auth_request = google.auth.transport.requests.Request(session=session)
                self._credentials.refresh(auth_request)
                logger.debug(
                    "Refreshed GCS credentials (expires: %s)", self._credentials.expiry
                )
            except RefreshError as exc:
                logger.error(
                    "Failed to refresh GCS credentials: %s", exc, exc_info=True
                )
                raise RuntimeError(f"Failed to refresh GCS credentials: {exc}") from exc

        return self._credentials

    def _get_client_sync(self):
        """Get or create GCS storage client (synchronous)."""
        if self._client is None:
            from google.cloud import storage  # heavy import — defer

            creds = self._get_credentials_sync()
            self._client = storage.Client(credentials=creds, project=self._project)
            logger.info("Initialized GCS storage client")
        return self._client

    async def get_client(self):
        """Get GCS client, initializing in thread pool if needed."""
        if self._client is None:
            await asyncio.to_thread(self._get_client_sync)
        return self._client

    async def close(self) -> None:
        """Cleanup resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._credentials = None
        logger.info("GCS service resources cleaned up")


# Global singleton instance
_gcs_service = GcsService()


# ============================================================================
# Public async API
# ============================================================================


def _upload_sync(bucket_name: str, object_name: str, file_bytes: bytes) -> None:
    """Synchronous GCS upload (runs in thread pool)."""
    client = _gcs_service._get_client_sync()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_file(BytesIO(file_bytes), content_type="image/jpeg", size=len(file_bytes))


async def save_image_gcs(file_bytes: bytes, filename: str) -> str:
    """
    Upload an image to GCS, resizing if it exceeds the size limit.

    Runs the blocking GCS upload in a thread pool to avoid blocking
    the event loop.

    Args:
        file_bytes: Image data as bytes.
        filename: Object name/path in GCS bucket.

    Returns:
        The GCS object name.

    Raises:
        RuntimeError: If GCS credentials or upload fail.
    """
    start = time.time()

    max_size = config.get("minio.max_size", 1_048_576)  # reuse same config key

    if len(file_bytes) > max_size:
        logger.info(
            "Image size (%d bytes) exceeds %d bytes. Resizing...",
            len(file_bytes), max_size,
        )
        file_bytes = await resize_image(file_bytes)

    bucket_name = config.get("gcs.bucket_name")
    prefix = config.get("gcs.path", "ocr_ktp_image_log")
    object_name = f"{prefix}/{filename}.jpg" if prefix else f"{filename}.jpg"

    await asyncio.to_thread(_upload_sync, bucket_name, object_name, file_bytes)

    logger.info(
        "Uploaded image to GCS: gs://%s/%s (%d bytes), (Total Time: %.2f s)",
        bucket_name, object_name, len(file_bytes), time.time() - start,
    )

    return object_name


def _download_model_sync() -> None:
    """Synchronous model download from GCS (runs in thread pool)."""
    bucket_name = config.get("gcs.bucket_name")
    prefix = config.get("gcs.prefix", "")
    model_prefix = config.get("gcs.model_prefix", "laminate_model")

    model_path = Path(config.get("model.path", "./src/models/laminate_model.pt"))
    output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _gcs_service._get_client_sync()
    bucket = client.bucket(bucket_name)

    # --- list and filter blobs ---
    blobs = list(bucket.list_blobs(prefix=prefix))
    model_blobs = [
        b for b in blobs
        if b.name.split("/")[-1].startswith(model_prefix)
    ]

    if not model_blobs:
        raise RuntimeError(
            f"No blobs matching '{model_prefix}*' found in "
            f"gs://{bucket_name}/{prefix}"
        )

    # --- pick the newest version ---
    best_blob = max(model_blobs, key=lambda b: _parse_version(b.name))

    logger.info(
        "Selected latest model: %s (version %s)",
        best_blob.name,
        ".".join(str(v) for v in _parse_version(best_blob.name)),
    )

    # --- skip if already downloaded ---
    if model_path.exists():
        logger.info("Model already exists locally, skipping download: %s", model_path)
        return

    # --- download ---
    logger.info("Downloading model to %s ...", model_path)
    best_blob.download_to_filename(str(model_path))
    logger.info("Download complete: %s", model_path)


async def download_model_gcs() -> None:
    """
    Download the latest model from GCS asynchronously.

    Runs the blocking GCS operations (list blobs, download) in a thread
    pool to avoid blocking the event loop.
    """
    await asyncio.to_thread(_download_model_sync)


async def close_gcs_service() -> None:
    """
    Cleanup function to close GCS resources.
    Call this on application shutdown.
    """
    await _gcs_service.close()
    logger.info("GCS service connections closed")
