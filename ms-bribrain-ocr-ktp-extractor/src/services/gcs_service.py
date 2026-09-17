"""
GCS (Google Cloud Storage) Service Module

Handles GCS credential management, client lifecycle, and model downloads.
"""

import os
import re
import threading
from pathlib import Path

import requests
import google.auth.transport.requests
import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.oauth2 import service_account

import logging

from src.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level state (thread-safe singletons)
# ============================================================================

_credentials: service_account.Credentials | None = None
_credentials_lock = threading.Lock()

# GCP project resolved alongside credentials (from ADC or SA_PROJECT_ID)
_gcs_project: str | None = None

_gcs_client = None
_gcs_client_lock = threading.Lock()


# ============================================================================
# Credentials
# ============================================================================

def _strip_env_quotes(value: str) -> str:
    """Strip a single pair of surrounding quotes from a value.

    docker ``--env-file`` keeps quote characters as part of the value, so a
    ``.env`` line like ``SA_PRIVATE_KEY="-----BEGIN..."`` arrives in the
    container with the literal quotes wrapping the PEM, which breaks the
    cryptography PEM parser with ``MalformedFraming``.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
        return value[1:-1]
    return value


def _build_credentials_info() -> dict:
    """
    Build service account credentials info from environment variables.

    Returns:
        dict suitable for ``Credentials.from_service_account_info``.

    Raises:
        KeyError: If a required environment variable is missing.
    """
    private_key = _strip_env_quotes(os.environ["SA_PRIVATE_KEY"]).replace("\\n", "\n")
    return {
        "type": os.environ["SA_TYPE"],
        "project_id": os.environ["SA_PROJECT_ID"],
        "private_key": private_key,
        "client_email": os.environ["SA_CLIENT_MAIL"],
        "client_id": os.environ["SA_CLIENT_ID"],
        "auth_uri": os.environ["SA_AUTH_URI"],
        "token_uri": os.environ["SA_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["SA_AUTH_PROVIDER"],
        "client_x509_cert_url": os.environ["SA_CERT_URL"],
    }


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


def _get_credentials() -> service_account.Credentials:
    """
    Return valid GCS credentials, creating or refreshing them as needed.

    This function is thread-safe.  Credential refresh is synchronous but
    only happens roughly once per hour.

    Returns:
        Valid ``service_account.Credentials``.

    Raises:
        RuntimeError: If credentials cannot be created or refreshed.
    """
    global _credentials, _gcs_project

    with _credentials_lock:
        # --- create on first call ---
        if _credentials is None:
            try:
                _credentials, _gcs_project = _load_credentials()
                logger.debug("Created GCS credentials")
            except Exception as exc:
                logger.error("Failed to create GCS credentials: %s", exc, exc_info=True)
                raise RuntimeError(f"Failed to create GCS credentials: {exc}") from exc

        # --- refresh if expired ---
        if not _credentials.valid:
            try:
                session = requests.Session()
                auth_request = google.auth.transport.requests.Request(session=session)
                _credentials.refresh(auth_request)
                logger.debug(
                    "Refreshed GCS credentials (expires: %s)", _credentials.expiry
                )
            except RefreshError as exc:
                logger.error(
                    "Failed to refresh GCS credentials: %s", exc, exc_info=True
                )
                raise RuntimeError(f"Failed to refresh GCS credentials: {exc}") from exc

    return _credentials


# ============================================================================
# Client helpers
# ============================================================================

def get_gcs_client():
    """
    Return a singleton GCS storage client, creating it on first call.

    Thread-safe via a module-level lock.
    """
    global _gcs_client

    with _gcs_client_lock:
        if _gcs_client is None:
            from google.cloud import storage  # heavy import – defer

            creds = _get_credentials()
            _gcs_client = storage.Client(credentials=creds, project=_gcs_project)
            logger.info("Initialized GCS storage client")

    return _gcs_client


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


def _download_latest_versioned(
    bucket_name: str,
    gcs_prefix: str,
    filename_prefix: str,
    dest_path: Path,
) -> None:
    """
    List blobs under ``gs://<bucket_name>/<gcs_prefix>``, pick the one whose
    basename starts with ``filename_prefix`` and has the highest ``_v<N>``
    suffix (e.g. ``server_det_v3.pth`` > ``server_det_v2.pth``), and download
    it to ``dest_path``.

    Skips the download if ``dest_path`` already exists.
    """
    # Short-circuit before any network call if the model is already on disk,
    # mirroring download_model_minio's exists-first ordering (BUG-17).
    if dest_path.exists():
        logger.info("Model already exists locally, skipping download: %s", dest_path)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    model_blobs = [
        b for b in blobs
        if b.name.split("/")[-1].startswith(filename_prefix)
    ]

    if not model_blobs:
        raise RuntimeError(
            f"No blobs matching '{filename_prefix}*' found in "
            f"gs://{bucket_name}/{gcs_prefix}"
        )

    best_blob = max(model_blobs, key=lambda b: _parse_version(b.name))

    logger.info(
        "Selected latest model: %s (version %s)",
        best_blob.name,
        ".".join(str(v) for v in _parse_version(best_blob.name)),
    )

    logger.info("Downloading model to %s ...", dest_path)
    best_blob.download_to_filename(str(dest_path))
    logger.info("Download complete: %s", dest_path)


def download_model_gcs(device: str) -> None:
    """
    Download the latest OCR model(s) from GCS.

    - ``device == "cpu"``: legacy mobile flow — pulls a single PaddleOCR
      ``inference.pdiparams`` under ``gcs_mobile_prefix``.
    - otherwise (server / GPU): pulls **two** PyTorch checkpoints, the
      detection and recognition models, from the configured detection /
      recognition GCS folders. Each is the highest-versioned file matching
      ``server_det_*.pth`` / ``server_rec_*.pth`` (latest of v1/v2/v3/...).
      They land at the paths the autokernel OCR backend expects
      (``autokernel/workspace/ppocrv5/server_det.pth`` and ``server_rec.pth``).

    Each download is skipped if the destination already exists locally.
    """
    bucket_name = settings.gcs_bucket_name

    if device == "cpu":
        _download_latest_versioned(
            bucket_name=bucket_name,
            gcs_prefix=settings.gcs_mobile_prefix,
            filename_prefix=settings.gcs_model_prefix,
            dest_path=Path(settings.gcs_mobile_model_path),
        )
        return

    # Server / GPU: detection + recognition .pth files
    _download_latest_versioned(
        bucket_name=bucket_name,
        gcs_prefix=settings.gcs_server_det_prefix,
        filename_prefix=settings.gcs_server_det_filename_prefix,
        dest_path=Path(settings.gcs_server_det_local_path),
    )
    _download_latest_versioned(
        bucket_name=bucket_name,
        gcs_prefix=settings.gcs_server_rec_prefix,
        filename_prefix=settings.gcs_server_rec_filename_prefix,
        dest_path=Path(settings.gcs_server_rec_local_path),
    )
