from functools import lru_cache

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    # DATABASE_URL (scoring.jobs/scoring.results), ORCHESTRATION_*, PIPELINE_*:
    # lihat PipelineSettings. Scoring tahap terakhir: tidak ada handoff, dan
    # callback-nya yang membawa hasil akhir ke Orkestrasi.
    port: int = 8033

    # Model kepercayaan per field dari ML engineer (pipeline + /v1/scoring/confidence).
    scoring_model_path: str = "weights/trust_model.joblib"

    # HANYA endpoint lama /v1/scoring/score (dipakai kontrak lama extract-ocr): skor dokumen
    # heuristik + ambang approve/review/reject. Bukan dari ML engineer; pipeline tidak memakainya.
    # Nama harus terdaftar di src/models/scoring.py.
    scoring_backend: str = "heuristic"

    # Ambang batas keputusan (aturan bisnis, bukan bagian dari model)
    scoring_approve_threshold: float = 0.8
    scoring_review_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
