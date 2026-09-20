"""
Model kepercayaan per field dari ML engineer: weights/trust_model.joblib.

Kontraknya (diberikan ML engineer, bukan rancangan kami):

    payload  {"npwp", "npwp_score", "npwp_has_homoglyph", "npwp_candidate_count",
              "name", "name_score", "name_corrected",
              "n_boxes", "num_pages", "avg_doc_score", "min_doc_score",
              "flag", "guardrail_probability"}
    keluaran {"npwp_confidence": 0..1, "name_confidence": 0..1}

Isi file (dibaca dari file-nya, bukan diasumsikan): dict dengan `pipeline` =
SimpleImputer(median) -> StandardScaler -> LogisticRegression, `feature_cols` (10 fitur),
`train_report` (n=1760 = 880 dokumen x 2 field, auc 0.96), dan jejak dataset training-nya.
Modelnya PER FIELD: satu baris fitur untuk nomor NPWP (`field_is_npwp=1`), satu untuk nama (`=0`);
kelas 1 = field itu benar, jadi confidence = P(kelas 1).

Pemetaan payload -> fitur. Semua kecuali satu terbaca langsung dari nama kuncinya:

    field_is_npwp              1 untuk baris nomor, 0 untuk baris nama
    field_score                npwp_score | name_score
    field_corrected            npwp_has_homoglyph | name_corrected
    field_multiple_candidates  npwp_candidate_count > 1. Payload hanya punya SATU hitungan kandidat,
                               dan payload adalah seluruh masukan model, jadi nilai yang sama dipakai
                               untuk kedua baris.
    n_boxes_per_page           n_boxes / num_pages
    avg_doc_score, min_doc_score, flag, guardrail_probability   kunci bernama sama

    shape_confidence           TIDAK ADA di payload dan definisinya tidak ada di file mana pun yang
                               kami terima. Di data training nilainya 0/1/2 (median 2). Dikirim sebagai
                               nilai kosong, sehingga SimpleImputer milik model mengisinya dengan median
                               training. Akibatnya confidence cenderung terlalu tinggi untuk field yang
                               shape_confidence aslinya 1 (pada sampel kartu asli: 0.74 -> 0.05).
                               Minta rumusnya ke ML engineer, lalu isi di _shape_confidence().
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Urutan kolom yang diharapkan pipeline; dicocokkan dengan `feature_cols` di file saat dimuat.
FEATURES = (
    "field_is_npwp",
    "field_score",
    "field_corrected",
    "shape_confidence",
    "field_multiple_candidates",
    "n_boxes_per_page",
    "avg_doc_score",
    "min_doc_score",
    "flag",
    "guardrail_probability",
)
# Fitur yang tidak bisa kami hitung dari payload (lihat docstring modul).
UNDEFINED_FEATURES = ("shape_confidence",)
MISSING = float("nan")


def _number(value: Any) -> float:
    """None / bukan angka -> nilai kosong, yang diisi imputer milik model. bool -> 0/1."""
    if value is None:
        return MISSING
    try:
        return float(value)
    except (TypeError, ValueError):
        return MISSING


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class TrustModel:
    name = "trust_model"

    def __init__(self, model_path: str):
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"Trust model not found at {path} (SCORING_MODEL_PATH)")

        # Versi scikit-learn yang berbeda dari saat training hanya memicu peringatan, padahal hasilnya
        # bisa bergeser diam-diam. requirements.txt mem-pin versi yang sama; kalau tetap beda, gagal start.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(path)

        if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_cols" not in bundle:
            raise RuntimeError(f"{path} is not the expected trust-model bundle (dict with pipeline + feature_cols)")
        columns = tuple(bundle["feature_cols"])
        if columns != FEATURES:
            raise RuntimeError(f"Unexpected feature_cols in {path}: {columns}")
        self._pipeline = bundle["pipeline"]
        classes = [int(c) for c in self._pipeline.named_steps["clf"].classes_]
        if classes != [0, 1]:
            raise RuntimeError(f"Unexpected classes in {path}: {classes}")

        self.metadata: dict[str, Any] = {
            "steps": [type(step).__name__ for _, step in self._pipeline.steps],
            "features": list(FEATURES),
            "undefined_features": list(UNDEFINED_FEATURES),
            "train_report": {key: _plain(value) for key, value in (bundle.get("train_report") or {}).items()},
        }
        logger.info("trust model loaded from %s: %s", path, self.metadata)

    @staticmethod
    def _shape_confidence(payload: dict[str, Any], *, is_npwp: bool) -> float:
        """Definisinya belum diterima dari ML engineer (lihat docstring modul). Kosong = diisi imputer."""
        return MISSING

    def feature_rows(self, payload: dict[str, Any]) -> dict[str, list[float]]:
        """-> {"npwp": [10 fitur], "name": [10 fitur]} dalam urutan FEATURES."""
        n_boxes, num_pages = _number(payload.get("n_boxes")), _number(payload.get("num_pages"))
        boxes_per_page = n_boxes / num_pages if num_pages and num_pages > 0 else MISSING
        candidates = payload.get("npwp_candidate_count")
        multiple = MISSING if candidates is None else float(candidates > 1)
        document = [
            boxes_per_page,
            _number(payload.get("avg_doc_score")),
            _number(payload.get("min_doc_score")),
            _number(payload.get("flag")),
            _number(payload.get("guardrail_probability")),
        ]
        return {
            "npwp": [
                1.0,
                _number(payload.get("npwp_score")),
                _number(payload.get("npwp_has_homoglyph")),
                self._shape_confidence(payload, is_npwp=True),
                multiple,
                *document,
            ],
            "name": [
                0.0,
                _number(payload.get("name_score")),
                _number(payload.get("name_corrected")),
                self._shape_confidence(payload, is_npwp=False),
                multiple,
                *document,
            ],
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, float | None]:
        rows = self.feature_rows(payload)
        with warnings.catch_warnings():
            # Pipeline dilatih dengan DataFrame; kami mengirim array dalam urutan kolom yang sama
            # (sudah dicocokkan saat dimuat), jadi peringatan "no valid feature names" tidak relevan.
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            probabilities = self._pipeline.predict_proba(np.array([rows["npwp"], rows["name"]], dtype=float))[:, 1]

        # Field yang tidak ditemukan tidak punya confidence: tanpa ini imputer mengisi skor OCR-nya
        # dengan median training (0.998) dan field KOSONG mendapat confidence tinggi.
        return {
            "npwp_confidence": round(float(probabilities[0]), 4) if _has_text(payload.get("npwp")) else None,
            "name_confidence": round(float(probabilities[1]), 4) if _has_text(payload.get("name")) else None,
        }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
