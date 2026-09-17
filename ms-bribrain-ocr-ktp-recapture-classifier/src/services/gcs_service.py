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

from ..core.config import config
from ..core.logging import logger

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
            # Run the Session as a context manager so the socket is always
            # released, and wrap ANY refresh failure (not just RefreshError —
            # e.g. TransportError / Timeout) in RuntimeError (BUG-23).
            try:
                with requests.Session() as session:
                    auth_request = google.auth.transport.requests.Request(session=session)
                    _credentials.refresh(auth_request)
                logger.debug(
                    "Refreshed GCS credentials (expires: %s)", _credentials.expiry
                )
            except Exception as exc:
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


def download_model_gcs() -> None:
    """
    Download the latest ``graycopy_model`` from GCS.

    Lists all blobs under the configured GCS path, selects the one
    whose filename starts with ``graycopy_model`` and has the highest
    version number (e.g. ``graycopy_model_v2.0.pth`` > ``…_v1.1.pth``),
    and downloads it to the local model directory.

    Skips the download if a file with the same name already exists locally.
    """
    bucket_name = config.gcs_bucket_name
    prefix = config.gcs_prefix
    model_prefix = config.gcs_model_prefix

    model_path = Path(config.get("model.path", "./src/models/recapture_model.pth"))
    output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_gcs_client()
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

    # Always (re)download the selected newest version. The previous skip-if-
    # exists check on the fixed local filename made the version selection dead
    # after the first download, so a newer published version was never fetched
    # (BUG-07). Re-downloading on startup keeps the served model fresh.
    logger.info("Downloading model to %s ...", model_path)
    best_blob.download_to_filename(str(model_path))
    logger.info("Download complete: %s", model_path)
