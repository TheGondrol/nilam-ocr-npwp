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

from src.core.config import config
from src.core.logging import logger

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


_OPENVINO_EXTENSIONS = (".bin", ".xml", ".yaml")


def _parse_folder_version(folder_name: str) -> tuple:
    """
    Extract a version tuple from a folder name like
    ``ktp_detection_model_openvino_v2`` or ``..._v2.3``.

    Returns ``(0,)`` if no version suffix is found.
    """
    match = re.search(r"_v(\d+(?:\.\d+)*)$", folder_name)
    if match:
        return tuple(int(p) for p in match.group(1).split("."))
    return (0,)


def download_openvino_gcs() -> None:
    """
    Download the latest versioned OpenVINO model folder from GCS.

    Looks under ``gs://<bucket>/<prefix>`` for folders named
    ``<model_prefix>_openvino_v<version>/`` (e.g.
    ``ktp_detection_model_openvino_v2/``), picks the highest
    version, and downloads every ``.bin``, ``.xml``, and ``.yaml``
    blob inside it to ``model.openvino_path`` locally.

    Skips files that already exist locally with the same name.
    """
    bucket_name = config.gcs_bucket_name
    prefix = config.gcs_prefix
    openvino_folder_prefix = config.gcs_openvino_prefix
    if not openvino_folder_prefix:
        raise RuntimeError("gcs.openvino_prefix is not configured")

    output_dir = Path(
        config.get("model.openvino_path", "./src/models/best_openvino_model")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    # List everything under the parent prefix, then group blobs by the
    # first path segment after the prefix (the openvino folder name).
    blobs = list(bucket.list_blobs(prefix=prefix))
    folder_blobs: dict[str, list] = {}
    for blob in blobs:
        relative = blob.name[len(prefix):] if blob.name.startswith(prefix) else blob.name
        if "/" not in relative:
            continue
        folder = relative.split("/", 1)[0]
        if not folder.startswith(openvino_folder_prefix):
            continue
        folder_blobs.setdefault(folder, []).append(blob)

    if not folder_blobs:
        raise RuntimeError(
            f"No folders matching '{openvino_folder_prefix}*' found in "
            f"gs://{bucket_name}/{prefix}"
        )

    latest_folder = max(folder_blobs.keys(), key=_parse_folder_version)
    logger.info(
        "Selected latest OpenVINO folder: %s (version %s)",
        latest_folder,
        ".".join(str(v) for v in _parse_folder_version(latest_folder)),
    )

    target_blobs = [
        b for b in folder_blobs[latest_folder]
        if b.name.lower().endswith(_OPENVINO_EXTENSIONS)
    ]

    if not target_blobs:
        raise RuntimeError(
            f"No .bin/.xml/.yaml files found in "
            f"gs://{bucket_name}/{prefix}{latest_folder}/"
        )

    for blob in target_blobs:
        filename = blob.name.rsplit("/", 1)[-1]
        local_path = output_dir / filename

        if local_path.exists():
            logger.info("Skipping existing file: %s", local_path)
            continue

        logger.info("Downloading %s -> %s", blob.name, local_path)
        blob.download_to_filename(str(local_path))

    logger.info("OpenVINO model ready in %s", output_dir)


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

    model_path = Path(config.get("model.path", "./src/models/ktp_detection_model.pt"))
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

    # --- skip if already downloaded ---
    if model_path.exists():
        logger.info("Model already exists locally, skipping download: %s", model_path)
        return

    # --- download ---
    logger.info("Downloading model to %s ...", model_path)
    best_blob.download_to_filename(str(model_path))
    logger.info("Download complete: %s", model_path)
