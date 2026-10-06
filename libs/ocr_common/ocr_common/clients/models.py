"""Where a service loads a model file from: its local path (baked in the image, or `weights/` on a laptop),
or, with a `gs://` URI configured, the object downloaded from GCS at start and verified (ocr_common.clients.gcs).

Normally the model is baked into the image at build (deploy/helm/deploy.sh fetches it from GCS first), so this
is the optional way to take it at start instead. The download uses ocr_common.clients.gcp.google_credentials: Tim
SEA's Entra ID workload identity (AZURE_* / GCP_*) or, without it, Application Default Credentials (a Compute
Engine VM's service account, GKE Workload Identity). No credentials, or a download that fails or does not match
its SHA-256, stops the
start: a service never runs with a half-downloaded or corrupted model. The object at the URI is replaced when a
new model is uploaded; pods take it when they restart.
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
    from ocr_common.clients.gcp import ENV_NAMES, google_credentials

    credentials, who = google_credentials({name: getattr(settings, name.lower(), None) for name in ENV_NAMES.values()})
    obj = gcs.GcsObject.parse(gcs_uri)
    target = Path(settings.models_dir) / obj.name.rsplit("/", 1)[-1]
    logger.info("model from %s as %s", obj.uri, who)
    return str(gcs.download(obj.uri, target, credentials, sha256=sha256))
