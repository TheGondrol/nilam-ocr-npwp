from fastapi import APIRouter

from src.core.config import get_settings
from src.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Reports service liveness and the active model backend per app. Does not require an API key.",
)
async def health():
    settings = get_settings()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "device": "cpu",
        "backends": {
            "guardrails": settings.guardrails_backend,
            "ekstraksi": settings.ekstraksi_backend,
            "structuring": settings.structuring_backend,
            "scoring": settings.scoring_backend,
            "storage": "postgres" if settings.database_url else "memory",
        },
    }
