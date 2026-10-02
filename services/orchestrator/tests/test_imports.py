"""The 0.2.0 deploy of 2026-09-23 crashed at startup because the app imported SQLAlchemy while the image did not
install it. Since nilam_guardrails_results the orchestrator does use it, so the image must: the database drivers the
app imports have to be pinned in the lock the Dockerfile installs from."""

from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]


def test_the_database_drivers_the_app_imports_are_in_the_lock():
    lock = (SERVICE / "requirements.lock").read_text(encoding="utf-8")
    pinned = {line.split("==")[0] for line in lock.splitlines() if "==" in line and not line.startswith(" ")}

    assert {"sqlalchemy", "asyncpg", "greenlet"} <= pinned
