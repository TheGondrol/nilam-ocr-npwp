from functools import lru_cache
from typing import Self

from pydantic import model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8032

    structuring_backend: str = "npwp_rules"

    # Reference data of the ML team's rules (backend `npwp_rules`). Each is optional: a missing file
    # turns the check it feeds into "no signal" instead of failing the request. The defaults are the
    # same file names under app/vendor/npwp_rules/data/.
    #   kode_wilayah.json  -> invalid_kecamatan_prefix (16-digit NIK-based numbers)
    #   kpp_codes.json     -> invalid_kpp_prefix (15-digit numbers)
    #   name_lnmast.xlsx   -> "recognised name" tie-break between name candidates (internal data)
    #   list_name_npwp.xlsx-> per-document name correction; unused here (no file_id: the orchestrator
    #                         does the fuzzy name match), kept so the vendored code has its path
    wilayah_codes_path: str | None = None
    kpp_codes_path: str | None = None
    name_master_path: str | None = None
    npwp_name_list_path: str | None = None

    scoring_service_url: str = "http://127.0.0.1:8033"
    scoring_api_key: str | None = None
    scoring_timeout_seconds: float = 10.0

    @model_validator(mode="after")
    def _guard_structuring(self) -> Self:
        self.reject_localhost_outside_local(scoring_service_url=self.scoring_service_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
