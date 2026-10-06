"""Google Cloud credentials through Workload Identity Federation with Microsoft Entra ID, as Tim SEA set it up
("User Guide - GCP workload identity federation (OIDC)"): no service account key file.

    Entra ID token (client credentials) -> GCP STS (token exchange) -> impersonate the service account

The result is an ordinary google-auth credential that renews itself (every token lives an hour), so a
long-running service keeps working. The settings are the guide's environment variables: AZURE_TENANT_ID,
AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, GCP_PROJECT_NUMBER, GCP_POOL_ID, GCP_PROVIDER_ID, GCP_SERVICE_ACCOUNT_EMAIL.

Without them, `google_credentials` falls back to Application Default Credentials: the service account attached to
a Compute Engine VM, GKE Workload Identity, or `gcloud auth application-default login` on a laptop. Inside GCP
that is the better way: no client secret to keep anywhere.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from google.auth import exceptions, identity_pool

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
# Entra answers this for a scope the app registration does not expose; the next form of the scope is tried.
INVALID_RESOURCE = "AADSTS500011"


@dataclass(frozen=True)
class WifConfig:
    """Who to be: the Entra ID app (from Tim SEA) and the GCP pool, provider and service account it maps to."""

    tenant_id: str
    client_id: str
    client_secret: str
    project_number: str
    pool_id: str
    provider_id: str
    service_account_email: str

    @property
    def audience(self) -> str:
        return (
            f"//iam.googleapis.com/projects/{self.project_number}/locations/global"
            f"/workloadIdentityPools/{self.pool_id}/providers/{self.provider_id}"
        )


class EntraTokenSupplier(identity_pool.SubjectTokenSupplier):
    """Hands GCP STS a fresh Entra ID access token (client credentials grant) whenever google-auth refreshes."""

    def __init__(self, config: WifConfig):
        self._config = config

    def get_subject_token(self, context: Any, request: Any) -> str:
        url = f"https://login.microsoftonline.com/{self._config.tenant_id}/oauth2/v2.0/token"
        # The guide uses `<client_id>/.default`; an app registration that exposes an API answers `api://...`.
        response = None
        for scope in (f"api://{self._config.client_id}/.default", f"{self._config.client_id}/.default"):
            response = request(
                url=url,
                method="POST",
                body=urlencode(
                    {
                        "grant_type": "client_credentials",
                        "client_id": self._config.client_id,
                        "client_secret": self._config.client_secret,
                        "scope": scope,
                    }
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status == 200:
                return json.loads(response.data)["access_token"]
            if INVALID_RESOURCE not in _text(response.data):
                break
        detail = _entra_error(response.data) if response is not None else "no answer"
        raise exceptions.RefreshError(f"Entra ID token request failed ({getattr(response, 'status', '-')}): {detail}")


ENV_NAMES = {
    "tenant_id": "AZURE_TENANT_ID",
    "client_id": "AZURE_CLIENT_ID",
    "client_secret": "AZURE_CLIENT_SECRET",
    "project_number": "GCP_PROJECT_NUMBER",
    "pool_id": "GCP_POOL_ID",
    "provider_id": "GCP_PROVIDER_ID",
    "service_account_email": "GCP_SERVICE_ACCOUNT_EMAIL",
}


def wif_config(values: Mapping[str, str | None]) -> WifConfig | None:
    """The config from the guide's variables (`ENV_NAMES`, e.g. `os.environ` or a parsed .env file); None when
    none is set. Raises `ValueError` naming the missing ones when only some are."""
    found = {field: (values.get(name) or "").strip() for field, name in ENV_NAMES.items()}
    if not any(found.values()):
        return None
    missing = [ENV_NAMES[field] for field, value in found.items() if not value]
    if missing:
        raise ValueError(f"GCP Workload Identity Federation is half configured, missing: {', '.join(missing)}")
    return WifConfig(**found)


def read_env_file(path: Path) -> dict[str, str]:
    """`KEY=VALUE` lines of a .env file such as Tim SEA's wif.gcs.env; comments and blank lines skipped."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def google_credentials(values: Mapping[str, str | None]) -> tuple[Any, str]:
    """The credentials to call Google APIs with, and whose they are: Entra ID workload identity federation when the
    variables of `ENV_NAMES` are set in `values`, else Application Default Credentials. `RuntimeError` when there
    are neither."""
    config = wif_config(values)
    if config is not None:
        return wif_credentials(config), f"{config.service_account_email} (Entra ID workload identity federation)"
    import google.auth

    try:
        credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
    except exceptions.DefaultCredentialsError as exc:
        raise RuntimeError(
            f"no Google credentials: set {', '.join(ENV_NAMES.values())} (Tim SEA's workload identity, e.g. "
            "wif.gcs.env), or run where Application Default Credentials exist (a VM's service account, GKE "
            "Workload Identity, `gcloud auth application-default login`)"
        ) from exc
    who = getattr(credentials, "service_account_email", None) or "application default credentials"
    return credentials, f"{who} (application default credentials)"


def credentials_from_env(env_file: Path | None = None) -> tuple[Any, str]:
    """For scripts: `google_credentials` from `env_file` (when it exists) over `os.environ`; `SystemExit` when there
    are none."""
    values = {**os.environ, **(read_env_file(env_file) if env_file and env_file.is_file() else {})}
    try:
        return google_credentials(values)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def wif_credentials(config: WifConfig, scopes: tuple[str, ...] = (CLOUD_PLATFORM_SCOPE,)) -> identity_pool.Credentials:
    """Credentials of `config.service_account_email`, renewed on their own; use with any Google client."""
    return identity_pool.Credentials(
        audience=config.audience,
        subject_token_type=JWT_TOKEN_TYPE,
        subject_token_supplier=EntraTokenSupplier(config),
        service_account_impersonation_url=(
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
            f"{config.service_account_email}:generateAccessToken"
        ),
        scopes=list(scopes),
    )


def _text(data: Any) -> str:
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)


def _entra_error(data: Any) -> str:
    """Entra's `error` and the first line of its description; never the request (it holds the secret)."""
    try:
        body = json.loads(_text(data))
    except ValueError:
        return _text(data)[:200]
    description = str(body.get("error_description", "")).splitlines()[0] if body.get("error_description") else ""
    return f"{body.get('error', 'error')}: {description}".strip(": ")
