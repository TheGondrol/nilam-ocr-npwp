"""Where a service loads a model file from: its local path (baked in the image, or `weights/` on a laptop),
or, with a `gs://` URI configured, the object downloaded from GCS at start and verified (ocr_common.clients.gcs).

The download uses GCP Workload Identity Federation with Entra ID (ocr_common.clients.gcp): AZURE_TENANT_ID,
AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, GCP_PROJECT_NUMBER, GCP_POOL_ID, GCP_PROVIDER_ID, GCP_SERVICE_ACCOUNT_EMAIL
(Tim SEA's GCS identity). A URI without them, or a download that fails or does not match its SHA-256, stops the
start: a service never runs with a model other than the one configured.
"""

import logging
from pathlib import Path

from ocr_common.config import BaseServiceSettings

logger = logging.getLogger(__name__)


def model_file(local_path: str, gcs_uri: str | None, sha256: str | None, settings: BaseServiceSettings) -> str:
    """`local_path` without `gcs_uri`; with it, the object downloaded into `MODELS_DIR` (a file there with the
    expected SHA-256 is reused) and verified against `sha256`, or the SHA-256 recorded at upload."""
    if not gcs_uri:
        return local_path
    from ocr_common.clients import gcs
    from ocr_common.clients.gcp import ENV_NAMES, wif_config, wif_credentials

    config = wif_config({name: getattr(settings, name.lower(), None) for name in ENV_NAMES.values()})
    if config is None:
        raise RuntimeError(
            f"{gcs_uri} needs GCP credentials to download: set {', '.join(ENV_NAMES.values())} (the GCS identity)"
        )
    obj = gcs.GcsObject.parse(gcs_uri)
    target = Path(settings.models_dir) / obj.name.rsplit("/", 1)[-1]
    logger.info("model from %s as %s", obj.uri, config.service_account_email)
    return str(gcs.download(obj.uri, target, wif_credentials(config), sha256=sha256))
